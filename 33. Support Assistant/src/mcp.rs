//! The `crm` MCP server: Notely's users and support tickets over Streamable
//! HTTP at `/mcp/crm`. Built like lesson 31's `project` server (rmcp macros,
//! a tool error is a result the model can read, not a protocol error). Five
//! tools read the CRM and two change it. Any MCP client can use it, not only
//! this lesson's assistant.

use std::sync::Arc;

use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::crm::{Crm, CrmData, LoginEvent, Note, STATUSES, Ticket, User, now_utc, ticket_key};

/// Sign-in attempts shown with a user (newest first).
pub const RECENT_EVENTS: usize = 5;
const DEFAULT_EVENTS: u32 = 10;
const MAX_EVENTS: u32 = 50;
const DEFAULT_TICKETS: u32 = 20;
const MAX_TICKETS: u32 = 100;
const MAX_NOTE_CHARS: usize = 2_000;
/// Who the notes written through MCP are signed by.
pub const NOTE_AUTHOR: &str = "assistant";

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong inside a tool (an unknown id, a bad argument, a failed
/// write) is a tool-level error the model can read and react to.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

fn in_range(name: &str, value: Option<u32>, default: u32, max: u32) -> Result<u32, String> {
    let value = value.unwrap_or(default);
    if (1..=max).contains(&value) {
        Ok(value)
    } else {
        Err(format!("`{name}` must be 1-{max}, got {value}."))
    }
}

// ---------------------------------------------------------------------------
// What the model reads: plain text for every record.
// ---------------------------------------------------------------------------

fn event_line(e: &LoginEvent) -> String {
    let mut line = format!("{} {} via {}: {}", e.at, e.method, e.client, e.result);
    if let Some(code) = &e.code {
        line.push_str(&format!(" {code}"));
    }
    if let Some(detail) = &e.detail {
        line.push_str(&format!(" ({detail})"));
    }
    line
}

/// The newest `n` sign-in attempts, newest first.
fn recent_events(user: &User, n: usize) -> Vec<&LoginEvent> {
    user.login_events.iter().rev().take(n).collect()
}

fn security_text(user: &User) -> String {
    let s = &user.security;
    let verified = match (s.email_verified, &s.verification_email_sent_at) {
        (true, _) => "yes".to_string(),
        (false, Some(at)) => format!("NO (confirmation email sent {at})"),
        (false, None) => "NO".to_string(),
    };
    let two_factor = s
        .two_factor
        .as_ref()
        .map_or("off".to_string(), |m| format!("on ({m})"));
    let sso = match &s.sso {
        Some(sso) if sso.enforced => format!(
            "{} for {}, ENFORCED since {} (password sign-in is off)",
            sso.provider,
            sso.domain,
            sso.enforced_since.as_deref().unwrap_or("?")
        ),
        Some(sso) => format!("{} for {}, not enforced", sso.provider, sso.domain),
        None => "none".to_string(),
    };
    let locked = match (s.locked, &s.locked_until) {
        (false, _) => "no".to_string(),
        (true, Some(until)) => format!("YES, until {until}"),
        (true, None) => "YES, until the password is reset".to_string(),
    };
    let mut text = format!(
        "Security: email verified: {verified} · two-factor: {two_factor} · SSO: {sso} · \
         failed sign-ins in a row: {} · locked: {locked}",
        s.failed_logins
    );
    if let Some(reason) = &s.lock_reason {
        text.push_str(&format!(" ({reason})"));
    }
    text
}

fn user_text(user: &User, tickets: &[&Ticket]) -> String {
    let mut text = format!(
        "User {}: {} <{}>\nPlan: {} · status: {} · company: {} · devices: {} · customer since {}\n{}",
        user.id,
        user.name,
        user.email,
        user.plan,
        user.status,
        user.company.as_deref().unwrap_or("—"),
        user.devices,
        user.created_at,
        security_text(user)
    );
    let events = recent_events(user, RECENT_EVENTS);
    if events.is_empty() {
        text.push_str("\nNo sign-in attempts recorded.");
    } else {
        text.push_str(&format!(
            "\nLast {} sign-in attempts (newest first, of {}):",
            events.len(),
            user.login_events.len()
        ));
        for e in events {
            text.push_str("\n- ");
            text.push_str(&event_line(e));
        }
    }
    if !tickets.is_empty() {
        let list: Vec<String> = tickets
            .iter()
            .map(|t| format!("{} ({}, \"{}\")", t.id, t.status, t.subject))
            .collect();
        text.push_str(&format!("\nTickets: {}", list.join(", ")));
    }
    text
}

/// The user as JSON, with only the recent sign-in attempts (newest first)
/// and their tickets.
fn user_json(user: &User, tickets: &[&Ticket]) -> Value {
    let mut value = json!(user);
    value["login_events"] = json!(recent_events(user, RECENT_EVENTS));
    value["login_events_total"] = json!(user.login_events.len());
    value["tickets"] = json!(
        tickets
            .iter()
            .map(|t| json!({ "id": t.id, "subject": t.subject, "status": t.status }))
            .collect::<Vec<_>>()
    );
    value
}

fn ticket_text(ticket: &Ticket, user: Option<&User>) -> String {
    let customer = match user {
        Some(u) => format!("{} <{}> ({}, plan {})", u.name, u.email, u.id, u.plan),
        None => format!("{} (not in the CRM)", ticket.user_id),
    };
    let mut text = format!(
        "Ticket {}: \"{}\"\nStatus: {} · priority: {} · channel: {} · tags: {}\nCustomer: {customer}\n\
         Created {}, updated {}\nMessages:",
        ticket.id,
        ticket.subject,
        ticket.status,
        ticket.priority,
        ticket.channel,
        if ticket.tags.is_empty() {
            "—".to_string()
        } else {
            ticket.tags.join(", ")
        },
        ticket.created_at,
        ticket.updated_at
    );
    for m in &ticket.messages {
        text.push_str(&format!("\n- {} {}: {}", m.at, m.from, m.text));
    }
    if !ticket.notes.is_empty() {
        text.push_str("\nInternal notes:");
        for n in &ticket.notes {
            text.push_str(&format!("\n- {} {}: {}", n.at, n.author, n.text));
        }
    }
    text
}

fn ticket_row(data: &CrmData, t: &Ticket) -> Value {
    let user = data.user(&t.user_id);
    json!({
        "id": t.id,
        "subject": t.subject,
        "status": t.status,
        "priority": t.priority,
        "channel": t.channel,
        "tags": t.tags,
        "updated_at": t.updated_at,
        "user_id": t.user_id,
        "customer": user.map(|u| u.name.clone()),
        "email": user.map(|u| u.email.clone()),
        "plan": user.map(|u| u.plan.clone()),
    })
}

// ---------------------------------------------------------------------------
// Tool arguments.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct TicketArgs {
    /// The ticket number, e.g. "T-1001" (or just "1001").
    ticket_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UserArgs {
    /// The user's id, e.g. "U-101". Give this or `email`.
    user_id: Option<String>,
    /// The user's email address, e.g. "anna.petrova@acme.io".
    email: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LoginEventsArgs {
    /// The user's id, e.g. "U-101".
    user_id: String,
    /// How many attempts to return, newest first: 1-50, default 10.
    limit: Option<u32>,
    /// Only the failed attempts.
    failed_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListTicketsArgs {
    /// Only tickets with this status: "open", "pending", "resolved" or "closed".
    status: Option<String>,
    /// Only the tickets of this user, e.g. "U-101".
    user_id: Option<String>,
    /// How many tickets to return: 1-100, default 20.
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchUsersArgs {
    /// Text to look for in the name, email, company or id, e.g. "bigcorp" or "Petrova".
    query: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NoteArgs {
    /// The ticket number, e.g. "T-1001".
    ticket_id: String,
    /// The note: what was found and what the customer was told. Internal, never shown to the customer.
    text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StatusArgs {
    /// The ticket number, e.g. "T-1001".
    ticket_id: String,
    /// The new status: "open", "pending" (waiting for the customer), "resolved" or "closed".
    status: String,
    /// Optional internal note saying why, added to the ticket together with the change.
    note: Option<String>,
}

fn note_text(text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("The note is empty.".to_string());
    }
    if text.chars().count() > MAX_NOTE_CHARS {
        return Err(format!("A note can be at most {MAX_NOTE_CHARS} characters."));
    }
    Ok(text.to_string())
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CrmServer {
    crm: Arc<Crm>,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl CrmServer {
    pub fn new(crm: Arc<Crm>) -> Self {
        Self {
            crm,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "get_ticket",
        description = "A support ticket: subject, status, priority, tags, the customer it \
                       belongs to, the whole conversation and the internal notes."
    )]
    async fn get_ticket(
        &self,
        Parameters(args): Parameters<TicketArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let data = self.crm.snapshot();
        let Some(ticket) = data.ticket(&args.ticket_id) else {
            return tool_error(format!(
                "There is no ticket `{}`. Use `list_tickets` to see the tickets.",
                ticket_key(&args.ticket_id)
            ));
        };
        tool_ok(
            ticket_text(ticket, data.user(&ticket.user_id)),
            json!(ticket),
        )
    }

    #[tool(
        name = "get_user",
        description = "A Notely user's account by id or email: plan, devices, security \
                       settings (email confirmed, two-factor, SSO, lock), the last sign-in \
                       attempts with their error codes, and their tickets."
    )]
    async fn get_user(
        &self,
        Parameters(args): Parameters<UserArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let data = self.crm.snapshot();
        let user_id = args.user_id.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let email = args.email.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let user = match (user_id, email) {
            (Some(id), _) => data.user(id),
            (None, Some(email)) => data.user_by_email(email),
            (None, None) => return tool_error("Give `user_id` or `email`.".to_string()),
        };
        let Some(user) = user else {
            return tool_error(format!(
                "There is no user {}. Use `search_users` to find one.",
                user_id.or(email).unwrap_or_default()
            ));
        };
        let tickets = data.tickets_of(&user.id);
        tool_ok(user_text(user, &tickets), user_json(user, &tickets))
    }

    #[tool(
        name = "get_login_events",
        description = "A user's sign-in history, newest first: when, how (password, SSO, \
                       two-factor), from which client, and the error code and detail of every \
                       failed attempt."
    )]
    async fn get_login_events(
        &self,
        Parameters(args): Parameters<LoginEventsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = match in_range("limit", args.limit, DEFAULT_EVENTS, MAX_EVENTS) {
            Ok(limit) => limit as usize,
            Err(e) => return tool_error(e),
        };
        let data = self.crm.snapshot();
        let Some(user) = data.user(&args.user_id) else {
            return tool_error(format!("There is no user `{}`.", args.user_id.trim()));
        };
        let failed_only = args.failed_only.unwrap_or(false);
        let events: Vec<&LoginEvent> = user
            .login_events
            .iter()
            .rev()
            .filter(|e| !failed_only || e.result != "ok")
            .take(limit)
            .collect();
        let mut text = format!(
            "{} {}sign-in attempts of {} ({}), newest first:",
            events.len(),
            if failed_only { "failed " } else { "" },
            user.name,
            user.id
        );
        for e in &events {
            text.push_str("\n- ");
            text.push_str(&event_line(e));
        }
        tool_ok(text, json!({ "user_id": user.id, "events": events }))
    }

    #[tool(
        name = "list_tickets",
        description = "Support tickets, most recently updated first, optionally only one \
                       status or one user's: id, subject, status, priority, customer."
    )]
    async fn list_tickets(
        &self,
        Parameters(args): Parameters<ListTicketsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = match in_range("limit", args.limit, DEFAULT_TICKETS, MAX_TICKETS) {
            Ok(limit) => limit as usize,
            Err(e) => return tool_error(e),
        };
        let status = args
            .status
            .as_deref()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty());
        if let Some(status) = &status
            && !STATUSES.contains(&status.as_str())
        {
            return tool_error(format!(
                "Unknown status `{status}`; use one of {}.",
                STATUSES.join(", ")
            ));
        }
        let user_id = args
            .user_id
            .as_deref()
            .map(crate::crm::user_key)
            .filter(|s| !s.is_empty());
        let data = self.crm.snapshot();
        let mut tickets: Vec<&Ticket> = data
            .tickets
            .iter()
            .filter(|t| status.as_ref().is_none_or(|s| t.status == *s))
            .filter(|t| user_id.as_ref().is_none_or(|u| t.user_id == *u))
            .collect();
        tickets.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        let total = tickets.len();
        tickets.truncate(limit);
        let mut text = format!("{total} tickets, showing {}:", tickets.len());
        for t in &tickets {
            let customer = data
                .user(&t.user_id)
                .map_or(t.user_id.clone(), |u| format!("{} <{}>", u.name, u.email));
            text.push_str(&format!(
                "\n- {} [{}, {}] \"{}\" from {customer}, updated {}",
                t.id, t.status, t.priority, t.subject, t.updated_at
            ));
        }
        let rows: Vec<Value> = tickets.iter().map(|t| ticket_row(&data, t)).collect();
        tool_ok(text, json!({ "total": total, "tickets": rows }))
    }

    #[tool(
        name = "search_users",
        description = "Find users whose name, email, company or id contains a text."
    )]
    async fn search_users(
        &self,
        Parameters(args): Parameters<SearchUsersArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let query = args.query.trim().to_lowercase();
        if query.is_empty() {
            return tool_error("The query is empty.".to_string());
        }
        let data = self.crm.snapshot();
        let users: Vec<&User> = data
            .users
            .iter()
            .filter(|u| {
                [
                    u.id.as_str(),
                    u.name.as_str(),
                    u.email.as_str(),
                    u.company.as_deref().unwrap_or_default(),
                ]
                .iter()
                .any(|field| field.to_lowercase().contains(&query))
            })
            .collect();
        if users.is_empty() {
            return tool_error(format!("No user matches `{query}`."));
        }
        let mut text = format!("{} users match `{query}`:", users.len());
        for u in &users {
            text.push_str(&format!(
                "\n- {} {} <{}>, plan {}, {}",
                u.id,
                u.name,
                u.email,
                u.plan,
                u.company.as_deref().unwrap_or("no company")
            ));
        }
        let rows: Vec<Value> = users
            .iter()
            .map(|u| {
                json!({ "id": u.id, "name": u.name, "email": u.email, "plan": u.plan, "company": u.company })
            })
            .collect();
        tool_ok(text, json!({ "users": rows }))
    }

    #[tool(
        name = "add_ticket_note",
        description = "WRITE. Add an internal note to a ticket (what was found, what the \
                       customer was told). The customer never sees notes. Saved to the CRM \
                       at once."
    )]
    async fn add_ticket_note(
        &self,
        Parameters(args): Parameters<NoteArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = match note_text(&args.text) {
            Ok(text) => text,
            Err(e) => return tool_error(e),
        };
        let id = ticket_key(&args.ticket_id);
        let saved = self.crm.update(|data| {
            let ticket = data
                .ticket_mut(&id)
                .ok_or_else(|| format!("There is no ticket `{id}`."))?;
            let note = Note {
                at: now_utc(),
                author: NOTE_AUTHOR.to_string(),
                text,
            };
            ticket.updated_at = note.at.clone();
            ticket.notes.push(note.clone());
            Ok((note, ticket.notes.len()))
        });
        match saved {
            Ok((note, count)) => tool_ok(
                format!(
                    "Added a note to {id} at {} ({count} notes now): {}",
                    note.at, note.text
                ),
                json!({ "ticket_id": id, "note": note, "notes": count }),
            ),
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "set_ticket_status",
        description = "WRITE. Change a ticket's status to open, pending (waiting for the \
                       customer), resolved or closed, optionally with an internal note saying \
                       why. Saved to the CRM at once."
    )]
    async fn set_ticket_status(
        &self,
        Parameters(args): Parameters<StatusArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let status = args.status.trim().to_lowercase();
        if !STATUSES.contains(&status.as_str()) {
            return tool_error(format!(
                "Unknown status `{status}`; use one of {}.",
                STATUSES.join(", ")
            ));
        }
        let note = match args.note.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
            Some(note) => match note_text(note) {
                Ok(note) => Some(note),
                Err(e) => return tool_error(e),
            },
            None => None,
        };
        let id = ticket_key(&args.ticket_id);
        let saved = self.crm.update(|data| {
            let ticket = data
                .ticket_mut(&id)
                .ok_or_else(|| format!("There is no ticket `{id}`."))?;
            let from = std::mem::replace(&mut ticket.status, status.clone());
            let at = now_utc();
            ticket.updated_at = at.clone();
            if let Some(text) = note {
                ticket.notes.push(Note {
                    at,
                    author: NOTE_AUTHOR.to_string(),
                    text,
                });
            }
            Ok(from)
        });
        match saved {
            Ok(from) => tool_ok(
                format!("{id}: status changed from {from} to {status}."),
                json!({ "ticket_id": id, "from": from, "to": status }),
            ),
            Err(e) => tool_error(e),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CrmServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("crm", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Notely's support CRM. Read: `get_ticket`, `get_user`, `get_login_events`, \
                 `list_tickets`, `search_users`. Write: `add_ticket_note`, `set_ticket_status`.",
            )
    }
}

pub fn crm_service(crm: Arc<Crm>) -> StreamableHttpService<CrmServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(CrmServer::new(crm.clone())),
        Default::default(),
        // Reached on the VDS by IP, like lesson 31's server, so any Host.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

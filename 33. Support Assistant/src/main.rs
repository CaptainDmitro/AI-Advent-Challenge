mod crm;
mod index;
mod mcp;
mod rerank;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::{Query, State},
    response::Html,
    routing::{get, post},
};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult, Tool},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crm::{Crm, now_utc};
use index::{Hit, Index, kb_documents, local};
use mcp::crm_service;
use rerank::Heuristic;

/// Candidates from the vector search, before reranking.
const TOP_K: usize = 20;
/// At most this many fragments go into the prompt.
const TOP_N: usize = 5;
/// The reranker score a fragment needs (lessons 23-25, 31).
const THRESHOLD: f32 = 0.30;
const MAX_TOOL_ROUNDS: usize = 8;
/// A tool result is clipped to this before it goes to the model.
const MAX_RESULT_CHARS: usize = 12_000;
const MAX_QUESTION_CHARS: usize = 2_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const RUN_TIMEOUT: Duration = Duration::from_secs(180);
/// The tools that change the CRM.
const WRITE_TOOLS: [&str; 2] = ["add_ticket_note", "set_ticket_status"];

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("\n… (clipped)");
    out
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

// ---------------------------------------------------------------------------
// The knowledge base: Notely's FAQ and docs, indexed once at start.
// ---------------------------------------------------------------------------

struct Kb {
    index: Index,
    heuristic: Heuristic,
    built_ms: u128,
}

#[derive(Serialize, Clone, Debug)]
struct KbInfo {
    documents: usize,
    chunks: usize,
    sources: Vec<String>,
    embedder: String,
    built_ms: u128,
}

impl Kb {
    fn build() -> Kb {
        let started = Instant::now();
        let index = Index::build(kb_documents(), local());
        let heuristic = Heuristic::new(&index.chunks);
        let built_ms = started.elapsed().as_millis();
        println!(
            "indexed {} documents, {} chunks in {built_ms} ms",
            index.documents,
            index.chunks.len()
        );
        Kb {
            index,
            heuristic,
            built_ms,
        }
    }

    fn info(&self) -> KbInfo {
        KbInfo {
            documents: self.index.documents,
            chunks: self.index.chunks.len(),
            sources: self.index.titles.iter().map(|(s, _)| s.clone()).collect(),
            embedder: self.index.embedder.id(),
            built_ms: self.built_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// The MCP client side: the assistant's connection to the `crm` server.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct ToolCallTrace {
    step: usize,
    /// 0 for the calls the assistant makes itself before the model runs.
    round: usize,
    tool: String,
    arguments: Value,
    /// Made by the assistant (`get_ticket`, `get_user` for the selected
    /// ticket), not chosen by the model.
    auto: bool,
    is_error: bool,
    result: String,
    structured: Option<Value>,
    ms: u128,
}

struct CrmClient {
    client: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

/// `initialize` + `tools/list` over Streamable HTTP, exactly as with a remote
/// server.
async fn connect(url: &str) -> Result<CrmClient, String> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.to_string()),
    );
    let client = ()
        .serve(transport)
        .await
        .map_err(|e| format!("MCP handshake with {url} failed: {e}"))?;
    match client.list_all_tools().await {
        Ok(tools) => Ok(CrmClient { client, tools }),
        Err(e) => {
            let _ = client.cancel().await;
            Err(format!("tools/list failed: {e}"))
        }
    }
}

async fn connect_with_timeout(url: &str) -> Result<CrmClient, String> {
    tokio::time::timeout(CONNECT_TIMEOUT, connect(url))
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "The MCP server at {url} didn't answer within {}s.",
                CONNECT_TIMEOUT.as_secs()
            ))
        })
}

fn tool_result_text(result: &CallToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        result
            .structured_content
            .as_ref()
            .map(Value::to_string)
            .unwrap_or_default()
    } else {
        text
    }
}

impl CrmClient {
    fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.to_string()).collect()
    }

    /// The server's tools as OpenAI function definitions.
    fn llm_tools(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|tool| {
                let mut parameters = (*tool.input_schema).clone();
                parameters.remove("$schema");
                parameters.insert("type".to_string(), json!("object"));
                parameters
                    .entry("properties")
                    .or_insert_with(|| json!({}));
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description.as_deref().unwrap_or_default(),
                        "parameters": parameters,
                    }
                })
            })
            .collect()
    }

    async fn call(
        &self,
        step: usize,
        round: usize,
        name: &str,
        arguments: Value,
        auto: bool,
    ) -> ToolCallTrace {
        let started = Instant::now();
        let outcome = if !self.tools.iter().any(|t| &*t.name == name) {
            Err(format!(
                "The crm server has no tool `{name}` (it has: {}).",
                self.tool_names().join(", ")
            ))
        } else {
            match &arguments {
                Value::Object(map) => self
                    .client
                    .call_tool(
                        CallToolRequestParams::new(name.to_string()).with_arguments(map.clone()),
                    )
                    .await
                    .map_err(|e| format!("tools/call failed: {e}")),
                _ => Err(format!("Arguments aren't a JSON object: {arguments}")),
            }
        };
        let (is_error, result, structured) = match outcome {
            Ok(r) => (
                r.is_error.unwrap_or(false),
                tool_result_text(&r),
                r.structured_content,
            ),
            Err(e) => (true, e, None),
        };
        println!(
            "mcp {step}: {name}({arguments}) -> {} in {} ms",
            if is_error { "error" } else { "ok" },
            started.elapsed().as_millis()
        );
        ToolCallTrace {
            step,
            round,
            tool: name.to_string(),
            arguments,
            auto,
            is_error,
            result: clip(&result, MAX_RESULT_CHARS),
            structured,
            ms: started.elapsed().as_millis(),
        }
    }

    async fn close(self) {
        let _ = self.client.cancel().await;
    }
}

// ---------------------------------------------------------------------------
// The ticket context: what the assistant knows before the model runs.
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct TicketContext {
    /// `get_ticket`'s structured result.
    ticket: Option<Value>,
    /// `get_user`'s structured result for the ticket's customer.
    user: Option<Value>,
    /// What goes into the system prompt.
    text: String,
    /// Added to the knowledge base search: the ticket's tags and the error
    /// codes of the customer's recent failed sign-ins.
    hints: Vec<String>,
    error: Option<String>,
}

const NO_TICKET: &str = "No ticket is selected. If the message names a ticket number or an \
    email address, look it up first (get_ticket, get_user, search_users). Otherwise answer from \
    the knowledge base and say which detail (the account's email or the ticket number) would let \
    you check the customer's account.";

/// `get_ticket`, then `get_user` for its customer, over MCP, every time a
/// ticket is selected, before the model runs.
async fn ticket_context(
    crm: &CrmClient,
    ticket_id: &str,
    calls: &mut Vec<ToolCallTrace>,
) -> TicketContext {
    let mut ctx = TicketContext::default();
    let trace = crm
        .call(calls.len() + 1, 0, "get_ticket", json!({ "ticket_id": ticket_id }), true)
        .await;
    if trace.is_error {
        ctx.text = format!("get_ticket failed: {}", trace.result);
        ctx.error = Some(trace.result.clone());
        calls.push(trace);
        return ctx;
    }
    ctx.text = trace.result.clone();
    ctx.ticket = trace.structured.clone();
    calls.push(trace);

    let user_id = ctx
        .ticket
        .as_ref()
        .and_then(|t| t["user_id"].as_str())
        .map(str::to_string);
    if let Some(tags) = ctx.ticket.as_ref().and_then(|t| t["tags"].as_array()) {
        ctx.hints
            .extend(tags.iter().filter_map(Value::as_str).map(str::to_string));
    }
    let Some(user_id) = user_id else {
        return ctx;
    };
    let trace = crm
        .call(calls.len() + 1, 0, "get_user", json!({ "user_id": user_id }), true)
        .await;
    if trace.is_error {
        ctx.text.push_str(&format!("\n\nget_user failed: {}", trace.result));
    } else {
        ctx.text.push_str("\n\n");
        ctx.text.push_str(&trace.result);
        if let Some(events) = trace
            .structured
            .as_ref()
            .and_then(|u| u["login_events"].as_array())
        {
            for e in events {
                if let Some(code) = e["code"].as_str()
                    && !ctx.hints.iter().any(|h| h == code)
                {
                    ctx.hints.push(code.to_string());
                }
            }
        }
        ctx.user = trace.structured.clone();
    }
    calls.push(trace);
    ctx
}

// ---------------------------------------------------------------------------
// The LLM.
// ---------------------------------------------------------------------------

struct Llm {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    /// Kept as raw JSON so `tool_calls` (and provider extras such as
    /// DeepSeek's `reasoning_content`) go back to the model exactly as they came.
    message: Value,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

impl Llm {
    async fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Value, String> {
        let mut body = json!({ "model": self.model, "messages": messages, "temperature": 0.2 });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("LLM request failed: {e}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<ErrorResponse>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(format!("LLM API error ({status}): {detail}"));
        }
        serde_json::from_str::<ChatCompletionResponse>(&text)
            .map_err(|e| format!("Failed to parse LLM response: {e}"))?
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| "The model returned no choices.".to_string())
    }
}

fn message_text(message: &Value) -> Option<String> {
    let text = message.get("content")?.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

// ---------------------------------------------------------------------------
// Answering a question.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct Source {
    /// The `[n]` the model cites it by.
    n: usize,
    chunk_id: String,
    source: String,
    section: String,
    score: f32,
    /// The answer has `[n]`.
    cited: bool,
    text: String,
}

#[derive(Serialize, Clone, Debug)]
struct Candidate {
    chunk_id: String,
    source: String,
    section: String,
    cosine: f32,
    score: f32,
    passed: bool,
}

#[derive(Serialize, Default, Debug)]
struct Reply {
    message: String,
    ticket_id: Option<String>,
    /// What the knowledge base was searched with.
    search_query: Option<String>,
    answer: Option<String>,
    error: Option<String>,
    /// From the `crm` MCP server's `get_ticket`, never from the model.
    ticket: Option<Value>,
    /// From `get_user`, for the ticket's customer.
    user: Option<Value>,
    /// The tags and error codes the search was given from the ticket.
    context_hints: Vec<String>,
    sources: Vec<Source>,
    candidates: Vec<Candidate>,
    calls: Vec<ToolCallTrace>,
    /// The tools the MCP server listed.
    tools: Vec<String>,
    /// A write tool succeeded: the CRM changed.
    changed: bool,
    mcp_url: String,
    mcp_error: Option<String>,
    llm_rounds: usize,
    ms: u128,
}

#[derive(Clone)]
struct Settings {
    model: String,
    mcp_url: String,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<Llm>,
    kb: Arc<Kb>,
    crm: Arc<Crm>,
    settings: Arc<Settings>,
}

const TRANSLATE_PROMPT: &str = "You turn a customer's support question about Notely, a \
    note-taking app, into a short English search query for its help center (sign-in, \
    two-factor authentication, SSO, sync, plans and billing, account). Keep error codes, \
    product names and numbers exactly as written. Reply with the query only, one line, no quotes.";

/// The knowledge base is in English and the index is lexical, so a question
/// in Cyrillic is searched with an English rewrite of it.
fn needs_translation(question: &str) -> bool {
    question
        .chars()
        .any(|c| matches!(c, '\u{0400}'..='\u{04FF}'))
}

fn system_prompt(now: &str, context: &str, fragments: &[Source], tools: &[String]) -> String {
    let mut docs = String::new();
    if fragments.is_empty() {
        docs.push_str(
            "(No knowledge base fragment passed the relevance threshold for this question.)\n",
        );
    }
    for f in fragments {
        docs.push_str(&format!(
            "[{}] {} › {}\n{}\n\n",
            f.n,
            f.source,
            f.section,
            f.text.trim()
        ));
    }
    let tools = if tools.is_empty() {
        "The crm MCP server is not reachable right now, so there are no tools: answer from the \
         knowledge base only and say that the account couldn't be checked."
            .to_string()
    } else {
        format!(
            "The `crm` MCP server: {}. get_ticket, get_user, get_login_events, list_tickets and \
             search_users read the CRM; add_ticket_note and set_ticket_status change it. Don't \
             call a read tool whose result is already above.",
            tools.join(", ")
        )
    };
    format!(
        "You are the support assistant of Notely, a note-taking app with sync across devices \
         (plans Free, Pro and Team). You answer the questions of customers and of the support \
         agents who work their tickets.\n\n\
         CURRENT TIME (UTC): {now}\n\n\
         TICKET CONTEXT (live, from the `crm` MCP server, fetched before you were asked):\n\
         {context}\n\n\
         KNOWLEDGE BASE fragments retrieved for this question (Notely's FAQ and help center):\n\
         {docs}\
         TOOLS: {tools}\n\n\
         Rules:\n\
         - First work out what is actually happening from the ticket context: the customer's \
         plan, security settings and recent sign-in attempts (their error codes and details) \
         usually say why something fails. State the cause plainly and name the data it comes \
         from, e.g. \"(CRM: 10 failed sign-ins, account locked)\".\n\
         - Then give the steps that fix it for this customer, from the knowledge base. Cite a \
         fragment as [n] right after the fact it supports (one number per bracket). Leave out \
         causes that the data rules out.\n\
         - If neither the knowledge base nor the CRM answers the question, say so and offer to \
         hand the ticket to a human agent. Never invent features, settings, prices, limits or \
         account data.\n\
         - add_ticket_note and set_ticket_status change the CRM: call them only when the \
         message asks for it (e.g. \"запиши в тикет\", \"закрой тикет\", \"add a note\"). Say \
         what you changed only after the tool succeeded.\n\
         - Answer in the language of the question, in Markdown, concisely: the cause first, \
         then the steps. Address the customer by name as \"you\", unless the message is \
         clearly from a support agent."
    )
}

async fn ask(app: &AppState, req: AskRequest) -> Reply {
    let started = Instant::now();
    let message = req.message.trim().to_string();
    let ticket_id = req
        .ticket_id
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string);
    let mut reply = Reply {
        message: message.clone(),
        ticket_id: ticket_id.clone(),
        mcp_url: app.settings.mcp_url.clone(),
        ..Default::default()
    };
    if message.is_empty() {
        reply.error = Some("Type a question.".to_string());
    } else if message.chars().count() > MAX_QUESTION_CHARS {
        reply.error = Some(format!(
            "The question is longer than {MAX_QUESTION_CHARS} characters."
        ));
    } else {
        match tokio::time::timeout(
            RUN_TIMEOUT,
            run(app, &message, ticket_id.as_deref(), &mut reply),
        )
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => reply.error = Some(e),
            Err(_) => reply.error = Some(format!("Timed out after {}s.", RUN_TIMEOUT.as_secs())),
        }
    }
    reply.ms = started.elapsed().as_millis();
    if let Some(e) = &reply.error {
        eprintln!("ask failed: {e}");
    }
    reply
}

async fn run(
    app: &AppState,
    question: &str,
    ticket_id: Option<&str>,
    reply: &mut Reply,
) -> Result<(), String> {
    // MCP: connect to the crm server, and when a ticket is selected, read it
    // and its customer first, so the answer always starts from the data.
    let crm = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(crm) => Some(crm),
        Err(e) => {
            reply.mcp_error = Some(e);
            None
        }
    };
    let context = match (&crm, ticket_id) {
        (Some(crm), Some(id)) => {
            reply.tools = crm.tool_names();
            ticket_context(crm, id, &mut reply.calls).await
        }
        (Some(crm), None) => {
            reply.tools = crm.tool_names();
            TicketContext {
                text: NO_TICKET.to_string(),
                ..Default::default()
            }
        }
        (None, _) => TicketContext {
            text: format!(
                "Unknown: the crm MCP server is unreachable ({}).",
                reply.mcp_error.as_deref().unwrap_or_default()
            ),
            ..Default::default()
        },
    };
    reply.ticket = context.ticket.clone();
    reply.user = context.user.clone();
    reply.context_hints = context.hints.clone();
    let result = answer(app, crm.as_ref(), question, &context, reply).await;
    if let Some(crm) = crm {
        crm.close().await;
    }
    result
}

async fn answer(
    app: &AppState,
    crm: Option<&CrmClient>,
    question: &str,
    context: &TicketContext,
    reply: &mut Reply,
) -> Result<(), String> {
    // The knowledge base is English only, so Cyrillic words can't match it:
    // a Russian question is searched by its English rewrite. The ticket's
    // tags and error codes are added either way, so "why can't I sign in?"
    // finds the section about this customer's error.
    let mut search_query = question.to_string();
    if needs_translation(question) {
        let messages = [
            json!({ "role": "system", "content": TRANSLATE_PROMPT }),
            json!({ "role": "user", "content": question }),
        ];
        if let Some(english) = app
            .llm
            .chat(&messages, &[])
            .await
            .ok()
            .and_then(|m| message_text(&m))
        {
            search_query = english;
        }
    }
    if !context.hints.is_empty() {
        search_query = format!("{search_query} {}", context.hints.join(" "));
    }
    reply.search_query = Some(search_query.clone());

    // RAG: vector search, heuristic rerank, threshold (lessons 21-25).
    let kb = &app.kb;
    let hits = kb.index.search(&search_query, TOP_K);
    let terms = kb.heuristic.terms(&search_query);
    let mut scored: Vec<(f32, Hit)> = hits
        .into_iter()
        .map(|h| (kb.heuristic.score(&terms, &h).score, h))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let round3 = |x: f32| (x * 1000.0).round() / 1000.0;
    reply.candidates = scored
        .iter()
        .map(|(score, h)| Candidate {
            chunk_id: h.chunk.chunk_id.clone(),
            source: h.chunk.source.clone(),
            section: h.chunk.section.clone(),
            cosine: h.score,
            score: round3(*score),
            passed: *score >= THRESHOLD,
        })
        .collect();
    reply.sources = scored
        .iter()
        .filter(|(score, _)| *score >= THRESHOLD)
        .take(TOP_N)
        .enumerate()
        .map(|(i, (score, h))| Source {
            n: i + 1,
            chunk_id: h.chunk.chunk_id.clone(),
            source: h.chunk.source.clone(),
            section: h.chunk.section.clone(),
            score: round3(*score),
            cited: false,
            text: h.chunk.text.clone(),
        })
        .collect();

    let tools = crm.map(CrmClient::llm_tools).unwrap_or_default();
    let mut messages = vec![
        json!({
            "role": "system",
            "content": system_prompt(&now_utc(), &context.text, &reply.sources, &reply.tools),
        }),
        json!({ "role": "user", "content": question }),
    ];
    for round in 1..=MAX_TOOL_ROUNDS {
        reply.llm_rounds = round;
        let message = app.llm.chat(&messages, &tools).await?;
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            let text = message_text(&message)
                .ok_or_else(|| "The model returned an empty answer.".to_string())?;
            for source in reply.sources.iter_mut() {
                source.cited = text.contains(&format!("[{}]", source.n));
            }
            reply.answer = Some(text);
            return Ok(());
        }
        messages.push(message);
        for call in tool_calls {
            let id = call["id"].as_str().unwrap_or_default().to_string();
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let raw = call["function"]["arguments"].as_str().unwrap_or("{}");
            let arguments: Value = if raw.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            let step = reply.calls.len() + 1;
            let trace = match crm {
                Some(crm) => crm.call(step, round, name, arguments, false).await,
                None => ToolCallTrace {
                    step,
                    round,
                    tool: name.to_string(),
                    arguments,
                    auto: false,
                    is_error: true,
                    result: "The crm MCP server is unreachable.".to_string(),
                    structured: None,
                    ms: 0,
                },
            };
            if !trace.is_error && WRITE_TOOLS.contains(&trace.tool.as_str()) {
                reply.changed = true;
            }
            let content = if trace.is_error {
                format!("Error: {}", trace.result)
            } else {
                trace.result.clone()
            };
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": content }));
            reply.calls.push(trace);
        }
    }
    Err(format!(
        "Gave up after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer."
    ))
}

// ---------------------------------------------------------------------------
// HTTP.
// ---------------------------------------------------------------------------

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[derive(Deserialize)]
struct AskRequest {
    message: String,
    #[serde(default)]
    ticket_id: Option<String>,
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Reply> {
    Json(ask(&app, req).await)
}

/// The ticket list, read through the MCP server like everything else.
async fn tickets_api(State(app): State<AppState>) -> Json<Value> {
    let crm = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(crm) => crm,
        Err(e) => return Json(json!({ "tickets": [], "error": e })),
    };
    let trace = crm
        .call(1, 0, "list_tickets", json!({ "limit": 100 }), true)
        .await;
    crm.close().await;
    if trace.is_error {
        return Json(json!({ "tickets": [], "error": trace.result }));
    }
    let tickets = trace
        .structured
        .as_ref()
        .map(|s| s["tickets"].clone())
        .unwrap_or_else(|| json!([]));
    Json(json!({ "tickets": tickets, "error": null }))
}

#[derive(Deserialize)]
struct ContextQuery {
    ticket_id: String,
}

/// What the assistant will know about a ticket: the same automatic
/// `get_ticket` + `get_user` an answer starts with.
async fn context_api(
    State(app): State<AppState>,
    Query(query): Query<ContextQuery>,
) -> Json<Value> {
    let crm = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(crm) => crm,
        Err(e) => {
            return Json(json!({
                "ticket": null, "user": null, "hints": [], "calls": [], "error": e,
            }));
        }
    };
    let mut calls = Vec::new();
    let ctx = ticket_context(&crm, &query.ticket_id, &mut calls).await;
    crm.close().await;
    Json(json!({
        "ticket": ctx.ticket,
        "user": ctx.user,
        "hints": ctx.hints,
        "calls": calls,
        "error": ctx.error,
    }))
}

async fn status_api(State(app): State<AppState>) -> Json<Value> {
    let tools = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(crm) => {
            let names = crm.tool_names();
            crm.close().await;
            Ok(names)
        }
        Err(e) => Err(e),
    };
    let data = app.crm.snapshot();
    Json(json!({
        "kb": app.kb.info(),
        "mcp": {
            "url": app.settings.mcp_url,
            "tools": tools.as_ref().ok(),
            "error": tools.as_ref().err(),
        },
        "crm": {
            "file": app.crm.path().display().to_string(),
            "users": data.users.len(),
            "tickets": data.tickets.len(),
        },
        "model": app.settings.model,
        "threshold": THRESHOLD,
    }))
}

/// Puts `data/crm.json` back, so the demo can be run again.
async fn reset_api(State(app): State<AppState>) -> Json<Value> {
    match app.crm.reset() {
        Ok(()) => Json(json!({ "ok": true, "error": null })),
        Err(e) => Json(json!({ "ok": false, "error": e })),
    }
}

fn app(state: AppState) -> Router {
    let crm = state.crm.clone();
    Router::new()
        .route("/", get(index_page))
        .route("/api/status", get(status_api))
        .route("/api/tickets", get(tickets_api))
        .route("/api/context", get(context_api))
        .route("/api/ask", post(ask_api))
        .route("/api/reset", post(reset_api))
        .with_state(state)
        .nest_service("/mcp/crm", crm_service(crm))
}

#[tokio::main]
async fn main() {
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let base_url =
        env_non_empty("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.deepseek.com".to_string());
    let model = env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key = env_non_empty("OPENAI_API_KEY").unwrap_or_else(|| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    // Relative to the working directory: `~/apps/lesson-33/crm.json` on the
    // VDS, created from data/crm.json on the first start.
    let crm_file = env_non_empty("CRM_FILE").unwrap_or_else(|| "crm.json".to_string());
    let crm = Arc::new(Crm::open(PathBuf::from(&crm_file)).unwrap_or_else(|e| {
        eprintln!("Error: can't open the CRM: {e}");
        std::process::exit(1);
    }));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    let local_port = listener.local_addr().unwrap().port();
    let mcp_url = env_non_empty("MCP_URL")
        .unwrap_or_else(|| format!("http://127.0.0.1:{local_port}/mcp/crm"));

    let state = AppState {
        llm: Arc::new(Llm {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model: model.clone(),
        }),
        kb: Arc::new(Kb::build()),
        crm,
        settings: Arc::new(Settings { model, mcp_url }),
    };
    println!(
        "CRM in {crm_file}. Listening on http://localhost:{port} (MCP server at /mcp/crm, \
         questions at /api/ask)"
    );
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    /// A fresh copy of the seed CRM in the temp folder.
    fn temp_crm(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lesson33-{}-{name}-{}.json",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// A scripted OpenAI-compatible LLM. A request without tools is the
    /// search-query translation and gets `translation`. Otherwise the n-th
    /// request of a conversation (n = assistant messages so far) gets the
    /// n-th round of tool calls, and the final `answer` after the last one.
    fn fake_llm(
        rounds: Vec<Vec<(&'static str, Value)>>,
        answer: &'static str,
        translation: &'static str,
        requests: Arc<StdMutex<Vec<Value>>>,
    ) -> Router {
        let rounds = Arc::new(rounds);
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let rounds = rounds.clone();
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body.clone());
                    if body.get("tools").is_none() {
                        return Json(json!({ "choices": [{ "message": {
                            "role": "assistant", "content": translation
                        } }] }));
                    }
                    let done = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|m| m["role"] == "assistant")
                        .count();
                    let Some(calls) = rounds.get(done) else {
                        return Json(json!({ "choices": [{ "message": {
                            "role": "assistant", "content": answer
                        } }] }));
                    };
                    let tool_calls: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(i, (name, arguments))| {
                            json!({
                                "id": format!("call_{done}_{i}"),
                                "type": "function",
                                "function": { "name": name, "arguments": arguments.to_string() }
                            })
                        })
                        .collect();
                    Json(json!({ "choices": [{ "message": {
                        "role": "assistant", "content": null, "tool_calls": tool_calls
                    } }] }))
                }
            }),
        )
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// The real app on a random port, over a CRM file at `crm_path`.
    async fn spawn_app(llm_base: &str, crm_path: PathBuf) -> (AppState, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = AppState {
            llm: Arc::new(Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{llm_base}/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            }),
            kb: Arc::new(Kb::build()),
            crm: Arc::new(Crm::open(crm_path).unwrap()),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                mcp_url: format!("{base}/mcp/crm"),
            }),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (state, base)
    }

    async fn post_ask(base: &str, message: &str, ticket_id: Option<&str>) -> Value {
        reqwest::Client::new()
            .post(format!("{base}/api/ask"))
            .json(&json!({ "message": message, "ticket_id": ticket_id }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    fn system_of(request: &Value) -> String {
        request["messages"][0]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    fn tools_of(reply: &Value) -> Vec<String> {
        reply["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["tool"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn russian_questions_are_translated_for_the_search() {
        assert!(needs_translation("Почему не работает авторизация?"));
        assert!(!needs_translation("Why can't I sign in?"));
    }

    #[tokio::test]
    async fn crm_mcp_server_reads_and_writes_the_crm() {
        let path = temp_crm("mcp");
        let (state, _) = spawn_app("http://127.0.0.1:9", path.clone()).await;
        let crm = connect(&state.settings.mcp_url).await.unwrap();

        let mut names = crm.tool_names();
        names.sort();
        assert_eq!(
            names,
            [
                "add_ticket_note",
                "get_login_events",
                "get_ticket",
                "get_user",
                "list_tickets",
                "search_users",
                "set_ticket_status"
            ]
        );
        for tool in &crm.tools {
            assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
        }
        assert_eq!(crm.llm_tools().len(), 7);

        // Reads.
        let ticket = crm.call(1, 0, "get_ticket", json!({ "ticket_id": "1001" }), true).await;
        assert!(!ticket.is_error, "{}", ticket.result);
        assert_eq!(ticket.structured.as_ref().unwrap()["id"], "T-1001");
        assert!(ticket.result.contains("Anna Petrova") && ticket.result.contains("С утра"));
        let missing = crm.call(2, 0, "get_ticket", json!({ "ticket_id": "T-9999" }), true).await;
        assert!(missing.is_error && missing.result.contains("T-9999"));

        let user = crm
            .call(3, 1, "get_user", json!({ "email": "Chen.Wei@bigcorp.com" }), false)
            .await;
        assert!(!user.is_error, "{}", user.result);
        assert!(user.result.contains("ENFORCED") && user.result.contains("SSO_REQUIRED"));
        let u = user.structured.unwrap();
        assert_eq!(u["id"], "U-103");
        assert_eq!(u["login_events"][0]["code"], "SSO_REQUIRED", "newest first");
        assert_eq!(u["tickets"][0]["id"], "T-1003");
        let anna = crm.call(4, 1, "get_user", json!({ "user_id": "U-101" }), false).await;
        assert!(anna.result.contains("YES, until the password is reset"), "{}", anna.result);
        assert_eq!(anna.structured.unwrap()["login_events"].as_array().unwrap().len(), 5);
        assert!(crm.call(5, 1, "get_user", json!({}), false).await.is_error);

        let failed = crm
            .call(6, 1, "get_login_events", json!({ "user_id": "U-101", "limit": 50, "failed_only": true }), false)
            .await;
        assert_eq!(failed.structured.unwrap()["events"].as_array().unwrap().len(), 12);
        assert!(
            crm.call(7, 1, "get_login_events", json!({ "user_id": "U-101", "limit": 0 }), false)
                .await
                .is_error
        );

        let open = crm.call(8, 1, "list_tickets", json!({ "status": "open" }), false).await;
        assert_eq!(open.structured.as_ref().unwrap()["total"], 5);
        let first = &open.structured.as_ref().unwrap()["tickets"][0];
        assert_eq!(first["id"], "T-1004", "most recently updated first");
        assert_eq!(first["customer"], "Elena Smirnova");
        let of_anna = crm.call(9, 1, "list_tickets", json!({ "user_id": "u101" }), false).await;
        assert_eq!(of_anna.structured.unwrap()["total"], 2);
        assert!(crm.call(10, 1, "list_tickets", json!({ "status": "lost" }), false).await.is_error);

        let bigcorp = crm.call(11, 1, "search_users", json!({ "query": "BigCorp" }), false).await;
        assert_eq!(bigcorp.structured.unwrap()["users"].as_array().unwrap().len(), 2);
        assert!(crm.call(12, 1, "search_users", json!({ "query": "zzz" }), false).await.is_error);

        // Writes: saved to the file at once.
        let note = crm
            .call(13, 1, "add_ticket_note", json!({ "ticket_id": "T-1002", "text": "Email not confirmed." }), false)
            .await;
        assert!(!note.is_error, "{}", note.result);
        let status = crm
            .call(14, 1, "set_ticket_status", json!({ "ticket_id": "T-1002", "status": "Pending", "note": "Waiting for the customer." }), false)
            .await;
        assert!(!status.is_error, "{}", status.result);
        assert_eq!(status.structured.unwrap()["from"], "open");
        let on_disk = crm::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let t = on_disk.ticket("T-1002").unwrap();
        assert_eq!(t.status, "pending");
        assert_eq!(t.notes.len(), 2);
        assert_eq!(t.notes[0].author, mcp::NOTE_AUTHOR);
        assert_eq!(t.notes[0].text, "Email not confirmed.");

        for (args, why) in [
            (json!({ "ticket_id": "T-1002", "text": "   " }), "empty note"),
            (json!({ "ticket_id": "T-9999", "text": "x" }), "unknown ticket"),
        ] {
            let r = crm.call(15, 1, "add_ticket_note", args, false).await;
            assert!(r.is_error, "{why}: {}", r.result);
        }
        let bad = crm
            .call(16, 1, "set_ticket_status", json!({ "ticket_id": "T-1002", "status": "deleted" }), false)
            .await;
        assert!(bad.is_error);
        let unknown = crm.call(17, 1, "delete_user", json!({}), false).await;
        assert!(unknown.is_error && unknown.result.contains("no tool `delete_user`"));
        crm.close().await;
        std::fs::remove_file(&path).unwrap();
    }

    /// The task's example: the same question about a different ticket gets a
    /// different context, a different search and different fragments.
    #[tokio::test]
    async fn the_same_question_is_answered_from_each_tickets_data() {
        let path = temp_crm("same-question");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![],
            "Ваш аккаунт заблокирован (CRM: 10 неудачных входов) [1].",
            "why does sign-in not work",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, path.clone()).await;

        for (ticket, code, fact) in [
            ("T-1001", "ACCOUNT_LOCKED", "YES, until the password is reset"),
            ("T-1002", "EMAIL_NOT_VERIFIED", "email verified: NO"),
            ("T-1003", "SSO_REQUIRED", "ENFORCED since"),
            ("T-1004", "INVALID_2FA_CODE", "clock skew"),
        ] {
            requests.lock().unwrap().clear();
            let reply = post_ask(&base, "Почему не работает авторизация?", Some(ticket)).await;
            assert!(reply["error"].is_null(), "{ticket}: {reply}");

            // MCP: the ticket and its customer, before the model.
            assert_eq!(tools_of(&reply), ["get_ticket", "get_user"], "{ticket}");
            assert!(reply["calls"].as_array().unwrap().iter().all(|c| c["auto"] == true));
            assert_eq!(reply["ticket"]["id"], ticket);
            assert_eq!(reply["user"]["id"], reply["ticket"]["user_id"]);

            // RAG: the English rewrite plus the ticket's error code finds
            // the section about exactly that error.
            let hints: Vec<&str> = reply["context_hints"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h.as_str().unwrap())
                .collect();
            assert!(hints.contains(&code), "{ticket}: {hints:?}");
            let query = reply["search_query"].as_str().unwrap();
            assert!(query.starts_with("why does sign-in not work") && query.contains(code), "{query}");
            let sources = reply["sources"].as_array().unwrap();
            assert!(
                sources.iter().any(|s| s["section"].as_str().unwrap().contains(code)),
                "{ticket}: {sources:?}"
            );

            // The model saw the customer's data and the fragments.
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 2, "translation + answer");
            assert!(requests[0].get("tools").is_none());
            let system = system_of(&requests[1]);
            assert!(system.contains(&format!("Ticket {ticket}")), "{system}");
            assert!(system.contains(fact), "{ticket}: {system}");
            assert!(system.contains(code));
            assert!(system.contains("[1] kb/"), "{system}");
            assert_eq!(requests[1]["tools"].as_array().unwrap().len(), 7);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[tokio::test]
    async fn the_assistant_writes_to_the_ticket_when_asked() {
        let path = temp_crm("writes");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![vec![
                (
                    "add_ticket_note",
                    json!({ "ticket_id": "T-1003", "text": "SSO enforced for bigcorp.com; told the customer to use Sign in with SSO." }),
                ),
                ("set_ticket_status", json!({ "ticket_id": "T-1003", "status": "pending" })),
            ]],
            "Записал диагноз в тикет и перевёл его в pending.",
            "write the diagnosis into the ticket",
            requests.clone(),
        ))
        .await;
        let (state, base) = spawn_app(&llm, path.clone()).await;

        let reply = post_ask(&base, "Запиши диагноз в тикет и переведи его в pending", Some("T-1003")).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(
            tools_of(&reply),
            ["get_ticket", "get_user", "add_ticket_note", "set_ticket_status"]
        );
        assert_eq!(reply["changed"], true);
        assert_eq!(reply["llm_rounds"], 2);
        let data = state.crm.snapshot();
        let ticket = data.ticket("T-1003").unwrap();
        assert_eq!(ticket.status, "pending");
        assert!(ticket.notes[0].text.contains("SSO enforced"));
        // The results went back to the model.
        let requests = requests.lock().unwrap();
        let tool_messages: Vec<&Value> = requests
            .last()
            .unwrap()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .collect();
        assert_eq!(tool_messages.len(), 2);
        assert!(tool_messages[1]["content"].as_str().unwrap().contains("from open to pending"));
        drop(requests);

        // The ticket list (through MCP) shows the change.
        let tickets: Value = reqwest::get(format!("{base}/api/tickets")).await.unwrap().json().await.unwrap();
        let t1003 = tickets["tickets"].as_array().unwrap().iter().find(|t| t["id"] == "T-1003").unwrap().clone();
        assert_eq!(t1003["status"], "pending");

        // And the demo can start over.
        let reset: Value = reqwest::Client::new()
            .post(format!("{base}/api/reset"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(reset["ok"], true);
        assert_eq!(state.crm.snapshot().ticket("T-1003").unwrap().status, "open");
        std::fs::remove_file(&path).unwrap();
    }

    #[tokio::test]
    async fn without_a_ticket_it_answers_from_the_knowledge_base() {
        let path = temp_crm("no-ticket");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![],
            "The Free plan syncs on 2 devices [1].",
            "unused",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, path.clone()).await;

        let reply = post_ask(&base, "How many devices can sync on the Free plan?", None).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert!(reply["calls"].as_array().unwrap().is_empty());
        assert!(reply["ticket"].is_null() && reply["user"].is_null());
        assert_eq!(reply["search_query"], "How many devices can sync on the Free plan?");
        assert_eq!(reply["tools"].as_array().unwrap().len(), 7);
        let sources = reply["sources"].as_array().unwrap();
        assert!(!sources.is_empty(), "{reply}");
        assert!(
            sources.iter().any(|s| s["source"] == "kb/sync.md" || s["source"] == "kb/plans-and-billing.md"),
            "{sources:?}"
        );
        assert_eq!(sources[0]["cited"], true);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "an English question needs no translation");
        assert!(system_of(&requests[0]).contains("No ticket is selected"));
        drop(requests);
        std::fs::remove_file(&path).unwrap();
    }

    #[tokio::test]
    async fn an_unknown_ticket_is_reported_to_the_model() {
        let path = temp_crm("unknown");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(vec![], "There is no such ticket.", "unused", requests.clone())).await;
        let (_, base) = spawn_app(&llm, path.clone()).await;

        let reply = post_ask(&base, "Why can't I sign in?", Some("T-4242")).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(tools_of(&reply), ["get_ticket"]);
        assert_eq!(reply["calls"][0]["is_error"], true);
        assert!(reply["ticket"].is_null());
        assert!(system_of(&requests.lock().unwrap()[0]).contains("get_ticket failed"));
        std::fs::remove_file(&path).unwrap();
    }

    #[tokio::test]
    async fn status_tickets_and_context_endpoints() {
        let path = temp_crm("status");
        let (_, base) = spawn_app("http://127.0.0.1:9", path.clone()).await;
        let status: Value = reqwest::get(format!("{base}/api/status")).await.unwrap().json().await.unwrap();
        assert_eq!(status["mcp"]["tools"].as_array().unwrap().len(), 7);
        assert_eq!(status["kb"]["documents"], index::KB.len());
        assert_eq!(status["crm"]["tickets"], 7);
        assert_eq!(status["crm"]["users"], 6);

        let tickets: Value = reqwest::get(format!("{base}/api/tickets")).await.unwrap().json().await.unwrap();
        assert!(tickets["error"].is_null(), "{tickets}");
        assert_eq!(tickets["tickets"].as_array().unwrap().len(), 7);

        let ctx: Value = reqwest::get(format!("{base}/api/context?ticket_id=T-1004"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(ctx["error"].is_null(), "{ctx}");
        assert_eq!(ctx["user"]["name"], "Elena Smirnova");
        assert_eq!(ctx["user"]["security"]["two_factor"], "totp");
        assert!(ctx["hints"].as_array().unwrap().contains(&json!("INVALID_2FA_CODE")));
        assert_eq!(ctx["calls"].as_array().unwrap().len(), 2);

        let page = reqwest::get(&base).await.unwrap().text().await.unwrap();
        assert!(page.contains("/api/ask") && page.contains("/api/tickets"));
        std::fs::remove_file(&path).unwrap();
    }
}

//! The CRM: Notely's users and support tickets, kept in one JSON file. The
//! file is created from `data/crm.json` on the first start, and every change
//! is written back with write-then-rename, so a reader never sees half a
//! file. The `crm` MCP server (`mcp.rs`) is the only way the assistant reads
//! or changes it.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// The demo data: six users and seven tickets, four of them about sign-in
/// failing for four different reasons.
pub const SEED: &str = include_str!("../data/crm.json");

/// The statuses a ticket can have.
pub const STATUSES: [&str; 4] = ["open", "pending", "resolved", "closed"];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CrmData {
    pub users: Vec<User>,
    pub tickets: Vec<Ticket>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct User {
    /// `U-101`.
    pub id: String,
    pub name: String,
    pub email: String,
    /// `free`, `pro` or `team`.
    pub plan: String,
    #[serde(default)]
    pub company: Option<String>,
    /// `active` or `suspended`.
    pub status: String,
    pub created_at: String,
    /// Devices signed in to the account.
    #[serde(default)]
    pub devices: u32,
    pub security: Security,
    /// Sign-in attempts, oldest first.
    #[serde(default)]
    pub login_events: Vec<LoginEvent>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Security {
    pub email_verified: bool,
    #[serde(default)]
    pub verification_email_sent_at: Option<String>,
    /// The two-factor method (`totp`), or none when 2FA is off.
    #[serde(default)]
    pub two_factor: Option<String>,
    #[serde(default)]
    pub sso: Option<Sso>,
    /// Failed sign-ins in a row.
    #[serde(default)]
    pub failed_logins: u32,
    #[serde(default)]
    pub locked: bool,
    /// When a temporary lock ends; none with `locked` means until the
    /// password is reset.
    #[serde(default)]
    pub locked_until: Option<String>,
    #[serde(default)]
    pub lock_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sso {
    pub provider: String,
    pub domain: String,
    pub enforced: bool,
    #[serde(default)]
    pub enforced_since: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoginEvent {
    pub at: String,
    /// `password`, `password+totp`, `sso`, `password_reset`.
    pub method: String,
    /// `web`, `desktop`, `ios`, `android`.
    pub client: String,
    /// `ok` or `failed`.
    pub result: String,
    /// The error code of a failed attempt, as in the knowledge base
    /// (`ACCOUNT_LOCKED`, `SSO_REQUIRED`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ticket {
    /// `T-1001`.
    pub id: String,
    pub user_id: String,
    pub subject: String,
    /// One of `STATUSES`.
    pub status: String,
    /// `low`, `normal` or `high`.
    pub priority: String,
    /// `email` or `chat`.
    pub channel: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// The conversation with the customer, oldest first.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Internal notes, never shown to the customer.
    #[serde(default)]
    pub notes: Vec<Note>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub at: String,
    /// `customer` or `agent`.
    pub from: String,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Note {
    pub at: String,
    pub author: String,
    pub text: String,
}

/// `1001`, `#1001`, `t1001` and `T-1001` are the same ticket; the same for
/// users with `U`.
fn normalize_id(id: &str, prefix: &str) -> String {
    let id = id.trim().trim_start_matches('#').to_uppercase();
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if digits(id.as_str()) {
        return format!("{prefix}-{id}");
    }
    if let Some(rest) = id.strip_prefix(prefix)
        && digits(rest)
    {
        return format!("{prefix}-{rest}");
    }
    id
}

pub fn ticket_key(id: &str) -> String {
    normalize_id(id, "T")
}

pub fn user_key(id: &str) -> String {
    normalize_id(id, "U")
}

impl CrmData {
    pub fn ticket(&self, id: &str) -> Option<&Ticket> {
        let key = ticket_key(id);
        self.tickets.iter().find(|t| t.id == key)
    }

    pub fn ticket_mut(&mut self, id: &str) -> Option<&mut Ticket> {
        let key = ticket_key(id);
        self.tickets.iter_mut().find(|t| t.id == key)
    }

    pub fn user(&self, id: &str) -> Option<&User> {
        let key = user_key(id);
        self.users.iter().find(|u| u.id == key)
    }

    pub fn user_by_email(&self, email: &str) -> Option<&User> {
        let email = email.trim();
        self.users
            .iter()
            .find(|u| u.email.eq_ignore_ascii_case(email))
    }

    /// A user's tickets, newest first.
    pub fn tickets_of(&self, user_id: &str) -> Vec<&Ticket> {
        let mut tickets: Vec<&Ticket> = self
            .tickets
            .iter()
            .filter(|t| t.user_id == user_id)
            .collect();
        tickets.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        tickets
    }
}

pub fn parse(text: &str) -> Result<CrmData, String> {
    serde_json::from_str(text).map_err(|e| format!("invalid CRM JSON: {e}"))
}

pub fn seed() -> CrmData {
    parse(SEED).expect("data/crm.json is valid")
}

fn save(path: &Path, data: &CrmData) -> Result<(), String> {
    let text = serde_json::to_string_pretty(data).map_err(|e| e.to_string())?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("Can't create {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("Can't write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("Can't replace {}: {e}", path.display()))
}

#[derive(Debug)]
pub struct Crm {
    path: PathBuf,
    data: Mutex<CrmData>,
}

impl Crm {
    /// Opens the CRM file, creating it from the seed when it doesn't exist.
    pub fn open(path: PathBuf) -> Result<Crm, String> {
        let data = if path.exists() {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("Can't read {}: {e}", path.display()))?;
            parse(&text).map_err(|e| format!("{}: {e}", path.display()))?
        } else {
            let data = seed();
            save(&path, &data)?;
            data
        };
        Ok(Crm {
            path,
            data: Mutex::new(data),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn snapshot(&self) -> CrmData {
        self.data.lock().unwrap().clone()
    }

    /// Applies `change` to a copy and writes it to the file. Only when the
    /// write succeeded does the copy become the CRM, so a failed write
    /// changes nothing.
    pub fn update<T>(
        &self,
        change: impl FnOnce(&mut CrmData) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self.data.lock().unwrap();
        let mut next = guard.clone();
        let out = change(&mut next)?;
        save(&self.path, &next)?;
        *guard = next;
        Ok(out)
    }

    /// Back to `data/crm.json`, for the next demo run.
    pub fn reset(&self) -> Result<(), String> {
        self.update(|data| {
            *data = seed();
            Ok(())
        })
    }
}

/// The current UTC time as `2026-10-16T10:07:30Z`, the format of every
/// timestamp in the CRM (so they compare as strings).
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_normalized() {
        for id in ["T-1001", "t-1001", "1001", "#1001", " T1001 "] {
            assert_eq!(ticket_key(id), "T-1001", "{id}");
        }
        assert_eq!(user_key("u101"), "U-101");
        assert_eq!(user_key("101"), "U-101");
        assert_eq!(ticket_key("nope"), "NOPE");
    }

    #[test]
    fn the_seed_has_four_ways_for_sign_in_to_fail() {
        let data = seed();
        assert_eq!(data.users.len(), 6);
        assert_eq!(data.tickets.len(), 7);
        for ticket in &data.tickets {
            assert!(data.user(&ticket.user_id).is_some(), "{}", ticket.id);
            assert!(STATUSES.contains(&ticket.status.as_str()), "{}", ticket.id);
        }
        let codes = |ticket: &str| -> Vec<String> {
            let user = data.user(&data.ticket(ticket).unwrap().user_id).unwrap();
            user.login_events
                .iter()
                .filter_map(|e| e.code.clone())
                .collect()
        };
        assert!(codes("T-1001").contains(&"ACCOUNT_LOCKED".to_string()));
        assert!(codes("T-1002").contains(&"EMAIL_NOT_VERIFIED".to_string()));
        assert!(codes("T-1003").contains(&"SSO_REQUIRED".to_string()));
        assert!(codes("T-1004").contains(&"INVALID_2FA_CODE".to_string()));
        assert!(data.user("U-101").unwrap().security.locked);
        assert_eq!(data.user_by_email("CHEN.WEI@bigcorp.com").unwrap().id, "U-103");
        let anna: Vec<&str> = data.tickets_of("U-101").iter().map(|t| t.id.as_str()).collect();
        assert_eq!(anna, ["T-1001", "T-0998"]);
    }

    #[test]
    fn changes_are_saved_and_a_failed_change_is_not() {
        let path = std::env::temp_dir().join(format!("lesson33-crm-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let crm = Crm::open(path.clone()).unwrap();
        assert!(path.exists(), "created from the seed");

        crm.update(|d| {
            d.ticket_mut("1002").unwrap().status = "pending".to_string();
            Ok(())
        })
        .unwrap();
        let err = crm.update(|d| -> Result<(), String> {
            d.ticket_mut("1002").unwrap().status = "broken".to_string();
            Err("nope".to_string())
        });
        assert!(err.is_err());
        assert_eq!(crm.snapshot().ticket("T-1002").unwrap().status, "pending");

        let reopened = Crm::open(path.clone()).unwrap();
        assert_eq!(reopened.snapshot().ticket("T-1002").unwrap().status, "pending");
        reopened.reset().unwrap();
        assert_eq!(reopened.snapshot().ticket("T-1002").unwrap().status, "open");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn now_is_an_iso_timestamp() {
        let now = now_utc();
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.starts_with("20") && now.ends_with('Z'), "{now}");
        assert_eq!(&now[10..11], "T");
        assert!(now.as_str() > "2026-01-01T00:00:00Z");
    }
}

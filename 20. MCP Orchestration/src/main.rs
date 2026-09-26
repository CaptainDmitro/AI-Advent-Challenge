use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use reqwest::header::{ACCEPT, USER_AGENT};
use rmcp::{
    ErrorData, RoleClient, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolRequestParams, CallToolResult, ContentBlock, Implementation, ServerCapabilities,
        ServerConfig, Tool,
    },
    service::RunningService,
    tool, tool_handler, tool_router,
    transport::{
        StreamableHttpClientTransport,
        streamable_http_client::StreamableHttpClientTransportConfig,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

const CRATES_API: &str = "https://crates.io";
const GITHUB_API: &str = "https://api.github.com";
/// crates.io's crawler policy asks for a User-Agent that says who is calling.
const USER_AGENT_VALUE: &str =
    "ai-advent-orchestration (https://github.com/CaptainDmitro/AI-Advent-Challenge)";
const DEFAULT_LIMIT: u32 = 5;
const MAX_LIMIT: u32 = 10;
const MAX_QUERY_CHARS: usize = 200;
const MAX_DESCRIPTION_CHARS: usize = 200;
const MAX_FILENAME_CHARS: usize = 80;
const ALLOWED_EXTENSIONS: [&str; 3] = ["md", "txt", "json"];
const MAX_NOTE_BYTES: usize = 64 * 1024;
/// A public, always-on instance can't be allowed to fill the disk.
const MAX_NOTES: usize = 100;
/// What the model gets back from one tool call, at most.
const MAX_RESULT_CHARS: usize = 8_000;
/// A long flow needs room: search, a few lookups per item, write, read back.
const MAX_TOOL_ROUNDS: usize = 16;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const RUN_TIMEOUT: Duration = Duration::from_secs(300);
/// Joins a server's name and a tool's name into what the LLM sees:
/// `crates__crate_info`. Server names can't contain it, so it splits back
/// unambiguously.
const SEP: &str = "__";

// ---------------------------------------------------------------------------
// Small helpers: time, checksums, names.
// ---------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Whole days between an ISO date (`YYYY-MM-DD...`) and `now` (unix seconds).
/// Howard Hinnant's days-from-civil, so no date crate is needed.
fn days_since(iso: &str, now: u64) -> Option<i64> {
    let date = iso.get(..10)?;
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(now as i64 / 86_400 - days)
}

/// FNV-1a over the UTF-8 bytes: enough to show that the note read back is
/// the note that was written.
fn checksum(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn shorten(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut short: String = text.chars().take(max).collect();
    short.push('…');
    short
}

/// What the model gets back from a tool: the whole text, or its start.
fn clip(text: &str) -> String {
    if text.chars().count() <= MAX_RESULT_CHARS {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(MAX_RESULT_CHARS).collect();
    clipped.push_str("\n…(truncated)");
    clipped
}

/// `owner/repo` from anything that names a GitHub repository: `owner/repo`,
/// `github.com/owner/repo`, or a full URL, with or without `.git`.
fn normalize_repo(input: &str) -> Option<String> {
    let mut s = input.trim();
    for prefix in ["https://", "http://", "www.", "github.com/"] {
        s = s.strip_prefix(prefix).unwrap_or(s);
    }
    let mut parts = s.split('/').filter(|p| !p.is_empty());
    let owner = parts.next()?;
    let repo = parts.next()?;
    let repo = repo.strip_suffix(".git").unwrap_or(repo);
    let owner_ok = owner.len() <= 39 && owner.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
    let repo_ok = !repo.is_empty()
        && repo.len() <= 100
        && !repo.starts_with('.')
        && repo
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    (!owner.is_empty() && owner_ok && repo_ok).then(|| format!("{owner}/{repo}"))
}

/// The GitHub repository a crate's `repository` URL points at, if any.
fn github_repo_of(url: &str) -> Option<String> {
    let start = url.find("github.com/")?;
    normalize_repo(&url[start + "github.com/".len()..])
}

fn valid_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

/// Whether `needle` occurs in `haystack` as a whole name: `octo/alpha` is in
/// "see github.com/octo/alpha." but not in "octo/alpha-cli".
fn mentions(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let is_name_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_');
    haystack.match_indices(needle).any(|(start, _)| {
        let before = haystack[..start].chars().next_back();
        let after = haystack[start + needle.len()..].chars().next();
        !before.is_some_and(is_name_char) && !after.is_some_and(is_name_char)
    })
}

// ---------------------------------------------------------------------------
// Upstream HTTP APIs: crates.io and GitHub.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Upstream {
    label: &'static str,
    http: reqwest::Client,
    base: String,
    token: Option<String>,
}

/// GitHub says `{"message": ...}`, crates.io says `{"errors": [{"detail": ...}]}`.
fn error_detail(body: &str) -> String {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    parsed["message"]
        .as_str()
        .or_else(|| parsed["errors"][0]["detail"].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| shorten(body, 200))
}

impl Upstream {
    async fn get<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, String)],
        what: &str,
    ) -> Result<T, String> {
        let url = format!("{}{path}", self.base.trim_end_matches('/'));
        let mut request = self
            .http
            .get(&url)
            .query(query)
            .header(ACCEPT, "application/json")
            .header(USER_AGENT, USER_AGENT_VALUE);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("{} request failed: {e}", self.label))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = error_detail(&body);
            return Err(match status.as_u16() {
                404 => format!("{what} was not found on {} ({detail}).", self.label),
                403 | 429 => format!(
                    "{} refused the request ({status}): {detail}. It is probably rate \
                     limiting; try again in a minute.",
                    self.label
                ),
                _ => format!("{} API error ({status}): {detail}", self.label),
            });
        }
        serde_json::from_str(&body).map_err(|e| format!("Unexpected {} response: {e}", self.label))
    }
}

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong *inside* a tool (a bad argument, a failed upstream call) is
/// a tool-level error the model can read and react to, not a protocol error.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

fn check_limit(limit: Option<u32>) -> Result<u32, String> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if (1..=MAX_LIMIT).contains(&limit) {
        Ok(limit)
    } else {
        Err(format!("`limit` must be 1-{MAX_LIMIT}, got {limit}."))
    }
}

fn check_query(query: &str) -> Result<&str, String> {
    let query = query.trim();
    if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
        Err(format!("`query` must be 1-{MAX_QUERY_CHARS} characters long."))
    } else {
        Ok(query)
    }
}

// ---------------------------------------------------------------------------
// MCP server 1: `crates`, the crates.io registry.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct CioMeta {
    #[serde(default)]
    total: u64,
}

#[derive(Debug, Deserialize)]
struct CioSearch {
    crates: Vec<CioCrate>,
    #[serde(default)]
    meta: CioMeta,
}

#[derive(Debug, Deserialize)]
struct CioCrate {
    name: String,
    description: Option<String>,
    #[serde(default)]
    downloads: u64,
    recent_downloads: Option<u64>,
    #[serde(default)]
    max_version: String,
    max_stable_version: Option<String>,
    repository: Option<String>,
    homepage: Option<String>,
    documentation: Option<String>,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    updated_at: String,
}

#[derive(Debug, Deserialize)]
struct CioCrateResponse {
    #[serde(rename = "crate")]
    krate: CioCrate,
    #[serde(default)]
    versions: Vec<CioVersion>,
}

#[derive(Debug, Deserialize)]
struct CioVersion {
    license: Option<String>,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    yanked: bool,
    rust_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CrateSummary {
    name: String,
    description: String,
    version: String,
    downloads: u64,
    recent_downloads: Option<u64>,
    repository: Option<String>,
    github_repo: Option<String>,
    updated_at: String,
}

impl From<CioCrate> for CrateSummary {
    fn from(c: CioCrate) -> Self {
        let github_repo = c.repository.as_deref().and_then(github_repo_of);
        CrateSummary {
            description: shorten(&c.description.unwrap_or_default(), MAX_DESCRIPTION_CHARS),
            version: c.max_stable_version.unwrap_or(c.max_version),
            name: c.name,
            downloads: c.downloads,
            recent_downloads: c.recent_downloads,
            repository: c.repository,
            github_repo,
            updated_at: c.updated_at,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchCratesArgs {
    /// Keywords to search crates.io for, e.g. "mcp server" or "http client".
    query: String,
    /// How many crates to return, most relevant first. 1-10, default 5.
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CrateInfoArgs {
    /// Exact crate name as it appears on crates.io, e.g. "rmcp".
    name: String,
}

#[derive(Debug, Clone)]
struct CratesServer {
    api: Upstream,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl CratesServer {
    fn new(api: Upstream) -> Self {
        Self {
            api,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "search_crates",
        description = "Search the crates.io registry of Rust packages by keyword. Returns each \
                       crate's name, latest version, total and recent downloads, and its source \
                       repository."
    )]
    async fn search_crates(
        &self,
        Parameters(args): Parameters<SearchCratesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let (query, limit) = match (check_query(&args.query), check_limit(args.limit)) {
            (Ok(query), Ok(limit)) => (query, limit),
            (Err(e), _) | (_, Err(e)) => return tool_error(e),
        };
        let found: CioSearch = match self
            .api
            .get(
                "/api/v1/crates",
                &[("q", query.to_string()), ("per_page", limit.to_string())],
                "The search",
            )
            .await
        {
            Ok(found) => found,
            Err(e) => return tool_error(e),
        };
        let crates: Vec<CrateSummary> = found
            .crates
            .into_iter()
            .take(limit as usize)
            .map(CrateSummary::from)
            .collect();

        let mut text = format!(
            "crates.io has {} crates matching {query:?}; the top {}:\n",
            found.meta.total,
            crates.len()
        );
        for (i, c) in crates.iter().enumerate() {
            text.push_str(&format!(
                "\n{}. {} {} · {} downloads ({} recent) · repository: {}",
                i + 1,
                c.name,
                c.version,
                c.downloads,
                c.recent_downloads.map_or("?".to_string(), |d| d.to_string()),
                c.repository.as_deref().unwrap_or("none")
            ));
            if !c.description.is_empty() {
                text.push_str(&format!(" — {}", c.description));
            }
        }
        tool_ok(
            text,
            json!({ "query": query, "total": found.meta.total, "crates": crates }),
        )
    }

    #[tool(
        name = "crate_info",
        description = "Details of one crate from crates.io: latest version, downloads, license, \
                       how many versions and when they were released, minimum Rust version, and \
                       the source repository (as `owner/repo` when it is on GitHub)."
    )]
    async fn crate_info(
        &self,
        Parameters(args): Parameters<CrateInfoArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let name = args.name.trim();
        if !valid_crate_name(name) {
            return tool_error(format!(
                "`name` must be a crate name: letters, digits, '-' or '_' (got {name:?})."
            ));
        }
        let found: CioCrateResponse = match self
            .api
            .get(&format!("/api/v1/crates/{name}"), &[], &format!("Crate {name:?}"))
            .await
        {
            Ok(found) => found,
            Err(e) => return tool_error(e),
        };
        let versions = &found.versions;
        let latest = versions.iter().find(|v| !v.yanked).or(versions.first());
        let license = latest.and_then(|v| v.license.clone());
        let rust_version = latest.and_then(|v| v.rust_version.clone());
        let yanked = versions.iter().filter(|v| v.yanked).count();
        let released: Vec<&str> = versions
            .iter()
            .map(|v| v.created_at.get(..10).unwrap_or(""))
            .filter(|d| !d.is_empty())
            .collect();
        let first_release = released.iter().min().map(|d| d.to_string());
        let last_release = released.iter().max().map(|d| d.to_string());
        let homepage = found.krate.homepage.clone();
        let documentation = found.krate.documentation.clone();
        let created_at = found.krate.created_at.clone();
        let summary = CrateSummary::from(found.krate);

        let mut text = format!("{} {}", summary.name, summary.version);
        if !summary.description.is_empty() {
            text.push_str(&format!(" — {}", summary.description));
        }
        text.push_str(&format!(
            "\nDownloads: {} total, {} in the last 90 days",
            summary.downloads,
            summary
                .recent_downloads
                .map_or("?".to_string(), |d| d.to_string())
        ));
        text.push_str(&format!(
            "\nLicense: {} · {} versions ({yanked} yanked) · first release {} · latest release {}",
            license.as_deref().unwrap_or("not specified"),
            versions.len(),
            first_release.as_deref().unwrap_or("?"),
            last_release.as_deref().unwrap_or("?"),
        ));
        if let Some(msrv) = &rust_version {
            text.push_str(&format!(" · minimum Rust {msrv}"));
        }
        text.push_str(&format!(
            "\nRepository: {}",
            summary.repository.as_deref().unwrap_or("none")
        ));
        if let Some(repo) = &summary.github_repo {
            text.push_str(&format!(" (GitHub: {repo})"));
        }
        tool_ok(
            text,
            json!({
                "name": summary.name,
                "description": summary.description,
                "version": summary.version,
                "downloads": summary.downloads,
                "recent_downloads": summary.recent_downloads,
                "license": license,
                "rust_version": rust_version,
                "versions": versions.len(),
                "yanked": yanked,
                "first_release": first_release,
                "last_release": last_release,
                "created_at": created_at,
                "repository": summary.repository,
                "github_repo": summary.github_repo,
                "homepage": homepage,
                "documentation": documentation,
            }),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CratesServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("crates", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Rust package registry data from crates.io: `search_crates` finds crates by \
                 keyword, `crate_info` gives one crate's downloads, versions, license and \
                 source repository.",
            )
    }
}

// ---------------------------------------------------------------------------
// MCP server 2: `github`, repositories on GitHub.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GhSearch {
    total_count: u64,
    items: Vec<GhRepo>,
}

#[derive(Debug, Deserialize)]
struct GhLicense {
    spdx_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GhRepo {
    full_name: String,
    html_url: String,
    description: Option<String>,
    stargazers_count: u64,
    #[serde(default)]
    forks_count: u64,
    #[serde(default)]
    open_issues_count: u64,
    language: Option<String>,
    #[serde(default)]
    topics: Vec<String>,
    license: Option<GhLicense>,
    #[serde(default)]
    archived: bool,
    pushed_at: Option<String>,
    created_at: Option<String>,
    default_branch: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RepoSummary {
    full_name: String,
    url: String,
    description: String,
    stars: u64,
    forks: u64,
    open_issues: u64,
    language: Option<String>,
    topics: Vec<String>,
    license: Option<String>,
    archived: bool,
    pushed_at: Option<String>,
    created_at: Option<String>,
    default_branch: Option<String>,
}

impl From<GhRepo> for RepoSummary {
    fn from(r: GhRepo) -> Self {
        RepoSummary {
            full_name: r.full_name,
            url: r.html_url,
            description: shorten(&r.description.unwrap_or_default(), MAX_DESCRIPTION_CHARS),
            stars: r.stargazers_count,
            forks: r.forks_count,
            open_issues: r.open_issues_count,
            language: r.language,
            topics: r.topics.into_iter().take(8).collect(),
            license: r
                .license
                .and_then(|l| l.spdx_id)
                .filter(|id| id != "NOASSERTION"),
            archived: r.archived,
            pushed_at: r.pushed_at,
            created_at: r.created_at,
            default_branch: r.default_branch,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchReposArgs {
    /// What to look for, in GitHub repository search syntax, e.g. "mcp server language:rust".
    query: String,
    /// How many repositories to return, most-starred first. 1-10, default 5.
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RepoInfoArgs {
    /// The repository as "owner/repo" (e.g. "tokio-rs/axum") or its github.com URL.
    repo: String,
}

#[derive(Debug, Clone)]
struct GithubServer {
    api: Upstream,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl GithubServer {
    fn new(api: Upstream) -> Self {
        Self {
            api,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "search_repositories",
        description = "Search public GitHub repositories, most-starred first. Returns each \
                       repository's owner/repo name, stars, language and description."
    )]
    async fn search_repositories(
        &self,
        Parameters(args): Parameters<SearchReposArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let (query, limit) = match (check_query(&args.query), check_limit(args.limit)) {
            (Ok(query), Ok(limit)) => (query, limit),
            (Err(e), _) | (_, Err(e)) => return tool_error(e),
        };
        let found: GhSearch = match self
            .api
            .get(
                "/search/repositories",
                &[
                    ("q", query.to_string()),
                    ("sort", "stars".to_string()),
                    ("order", "desc".to_string()),
                    ("per_page", limit.to_string()),
                ],
                "The search",
            )
            .await
        {
            Ok(found) => found,
            Err(e) => return tool_error(e),
        };
        let repos: Vec<RepoSummary> = found
            .items
            .into_iter()
            .take(limit as usize)
            .map(RepoSummary::from)
            .collect();
        let mut text = format!(
            "GitHub has {} repositories matching {query:?}; the top {} by stars:\n",
            found.total_count,
            repos.len()
        );
        for (i, r) in repos.iter().enumerate() {
            text.push_str(&format!(
                "\n{}. {} ★ {} · {}",
                i + 1,
                r.full_name,
                r.stars,
                r.language.as_deref().unwrap_or("—")
            ));
            if !r.description.is_empty() {
                text.push_str(&format!(" — {}", r.description));
            }
        }
        tool_ok(
            text,
            json!({ "query": query, "total_count": found.total_count, "repositories": repos }),
        )
    }

    #[tool(
        name = "repo_info",
        description = "Details of one GitHub repository: stars, forks, open issues and pull \
                       requests, license, whether it is archived, and when it was last pushed \
                       to (with the number of days since)."
    )]
    async fn repo_info(
        &self,
        Parameters(args): Parameters<RepoInfoArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(repo) = normalize_repo(&args.repo) else {
            return tool_error(format!(
                "`repo` must be \"owner/repo\" or a github.com URL (got {:?}).",
                args.repo
            ));
        };
        let found: GhRepo = match self
            .api
            .get(&format!("/repos/{repo}"), &[], &format!("Repository {repo}"))
            .await
        {
            Ok(found) => found,
            Err(e) => return tool_error(e),
        };
        let r = RepoSummary::from(found);
        let days = r
            .pushed_at
            .as_deref()
            .and_then(|p| days_since(p, unix_now()));

        let mut text = format!(
            "{} ★ {} · {} forks · {} open issues and PRs · license {}",
            r.full_name,
            r.stars,
            r.forks,
            r.open_issues,
            r.license.as_deref().unwrap_or("not specified")
        );
        text.push_str(&format!(
            "\nLast push: {}{} · created {} · default branch {}{}",
            r.pushed_at
                .as_deref()
                .and_then(|p| p.get(..10))
                .unwrap_or("?"),
            days.map_or(String::new(), |d| format!(" ({d} days ago)")),
            r.created_at
                .as_deref()
                .and_then(|p| p.get(..10))
                .unwrap_or("?"),
            r.default_branch.as_deref().unwrap_or("?"),
            if r.archived { " · ARCHIVED" } else { "" }
        ));
        if !r.description.is_empty() {
            text.push_str(&format!("\n{}", r.description));
        }
        let mut structured = serde_json::to_value(&r).unwrap_or_default();
        structured["days_since_push"] = json!(days);
        tool_ok(text, structured)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for GithubServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("github", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "GitHub repository data: `search_repositories` finds repositories, `repo_info` \
                 gives one repository's stars, forks, open issues, license and last push.",
            )
    }
}

// ---------------------------------------------------------------------------
// MCP server 3: `notes`, a small directory of saved files.
// ---------------------------------------------------------------------------

fn validate_filename(name: &str) -> Result<(), String> {
    let valid_chars = !name.is_empty()
        && name.len() <= MAX_FILENAME_CHARS
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !valid_chars {
        return Err(format!(
            "`filename` must be 1-{MAX_FILENAME_CHARS} letters, digits, '-', '_' or '.', \
             not start with '.', and name no directory (got {name:?})."
        ));
    }
    let extension = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    if !ALLOWED_EXTENSIONS.contains(&extension) {
        return Err(format!(
            "`filename` must end in .{} (got {name:?}).",
            ALLOWED_EXTENSIONS.join(", .")
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct NoteFile {
    filename: String,
    bytes: u64,
    modified: u64,
}

fn list_note_files(dir: &Path) -> Vec<NoteFile> {
    let mut files: Vec<NoteFile> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let filename = entry.file_name().into_string().ok()?;
                    validate_filename(&filename).ok()?;
                    let meta = entry.metadata().ok().filter(|m| m.is_file())?;
                    let modified = meta
                        .modified()
                        .ok()?
                        .duration_since(UNIX_EPOCH)
                        .ok()?
                        .as_secs();
                    Some(NoteFile {
                        filename,
                        bytes: meta.len(),
                        modified,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.filename.cmp(&b.filename))
    });
    files
}

/// Write-then-rename, so a reader never sees a half-written file.
fn write_file(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)?;
    std::fs::rename(&tmp, path)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteNoteArgs {
    /// File name, ending in .md, .txt or .json, e.g. "rust-mcp.md". An existing note with this name is replaced.
    filename: String,
    /// The full text to save, e.g. a Markdown report.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadNoteArgs {
    /// Name of a saved note, e.g. "rust-mcp.md".
    filename: String,
}

#[derive(Debug, Clone)]
struct NotesServer {
    dir: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl NotesServer {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "write_note",
        description = "Save text (Markdown, plain text or JSON) as a note in the server's notes \
                       directory. Returns the file name, size and checksum."
    )]
    async fn write_note(
        &self,
        Parameters(args): Parameters<WriteNoteArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let filename = args.filename.trim();
        if let Err(e) = validate_filename(filename) {
            return tool_error(e);
        }
        if args.content.trim().is_empty() || args.content.len() > MAX_NOTE_BYTES {
            return tool_error(format!(
                "`content` must be non-empty and at most {MAX_NOTE_BYTES} bytes (got {}).",
                args.content.len()
            ));
        }
        let path = self.dir.join(filename);
        let overwritten = path.exists();
        if !overwritten && list_note_files(&self.dir).len() >= MAX_NOTES {
            return tool_error(format!(
                "The notes directory already holds {MAX_NOTES} notes; overwrite an existing one."
            ));
        }
        if let Err(e) = write_file(&path, &args.content) {
            return tool_error(format!("Could not write {filename}: {e}"));
        }
        let sum = checksum(&args.content);
        tool_ok(
            format!(
                "Saved note {filename} ({} bytes, {sum}){}.",
                args.content.len(),
                if overwritten { ", replacing the old one" } else { "" }
            ),
            json!({
                "filename": filename,
                "bytes": args.content.len(),
                "checksum": sum,
                "overwritten": overwritten,
            }),
        )
    }

    #[tool(
        name = "read_note",
        description = "Read a saved note back from the notes directory: its full text, size and \
                       checksum."
    )]
    async fn read_note(
        &self,
        Parameters(args): Parameters<ReadNoteArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let filename = args.filename.trim();
        if let Err(e) = validate_filename(filename) {
            return tool_error(e);
        }
        let content = match std::fs::read_to_string(self.dir.join(filename)) {
            Ok(content) => content,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return tool_error(format!(
                    "There is no note named {filename}. Use `list_notes` to see what exists."
                ));
            }
            Err(e) => return tool_error(format!("Could not read {filename}: {e}")),
        };
        let sum = checksum(&content);
        tool_ok(
            format!(
                "Note {filename} ({} bytes, {sum}):\n\n{content}",
                content.len()
            ),
            json!({
                "filename": filename,
                "bytes": content.len(),
                "checksum": sum,
                "content": content,
            }),
        )
    }

    #[tool(
        name = "list_notes",
        description = "List the saved notes, newest first, with their sizes."
    )]
    async fn list_notes(&self) -> Result<CallToolResult, ErrorData> {
        let files = list_note_files(&self.dir);
        let text = if files.is_empty() {
            "There are no notes yet.".to_string()
        } else {
            let mut text = format!("{} notes, newest first:", files.len());
            for f in &files {
                text.push_str(&format!("\n- {} ({} bytes)", f.filename, f.bytes));
            }
            text
        };
        tool_ok(text, json!({ "notes": files }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for NotesServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("notes", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "A notes directory on the server: `write_note` saves Markdown, text or JSON, \
                 `read_note` reads a note back, `list_notes` lists what is saved.",
            )
    }
}

// Each server is its own Streamable HTTP endpoint with its own sessions. They
// share a process only so the lesson deploys as one binary on one port; the
// orchestrator reaches them by URL exactly as it would reach remote servers.

fn http_config() -> StreamableHttpServerConfig {
    // Meant to be reached on the VDS by IP (see lesson 17), so any Host.
    StreamableHttpServerConfig::default().disable_allowed_hosts()
}

fn crates_service(api: Upstream) -> StreamableHttpService<CratesServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(CratesServer::new(api.clone())),
        Default::default(),
        http_config(),
    )
}

fn github_service(api: Upstream) -> StreamableHttpService<GithubServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(GithubServer::new(api.clone())),
        Default::default(),
        http_config(),
    )
}

fn notes_service(dir: PathBuf) -> StreamableHttpService<NotesServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(NotesServer::new(dir.clone())),
        Default::default(),
        http_config(),
    )
}

// ---------------------------------------------------------------------------
// The server registry: which MCP servers the orchestrator connects to.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq)]
struct ServerSpec {
    name: String,
    url: String,
}

fn valid_server_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 20
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// `MCP_SERVERS`: comma-separated `name=url` pairs.
fn parse_servers(spec: &str) -> Result<Vec<ServerSpec>, String> {
    let mut servers: Vec<ServerSpec> = Vec::new();
    for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let Some((name, url)) = entry.split_once('=') else {
            return Err(format!("{entry:?} is not `name=url`."));
        };
        let (name, url) = (name.trim(), url.trim());
        if !valid_server_name(name) {
            return Err(format!(
                "server name {name:?} must be 1-20 lowercase letters, digits or '-'."
            ));
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(format!("server {name:?}: {url:?} is not an http(s) URL."));
        }
        if servers.iter().any(|s| s.name == name) {
            return Err(format!("server name {name:?} is used twice."));
        }
        servers.push(ServerSpec {
            name: name.to_string(),
            url: url.to_string(),
        });
    }
    if servers.is_empty() {
        return Err("no servers listed.".to_string());
    }
    Ok(servers)
}

/// The three servers this binary serves itself.
fn default_servers(base: &str) -> Vec<ServerSpec> {
    ["crates", "github", "notes"]
        .into_iter()
        .map(|name| ServerSpec {
            name: name.to_string(),
            url: format!("{}/mcp/{name}", base.trim_end_matches('/')),
        })
        .collect()
}

fn qualified(server: &str, tool: &str) -> String {
    format!("{server}{SEP}{tool}")
}

/// OpenAI-style function names: `^[a-zA-Z0-9_-]{1,64}$`.
fn valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

#[derive(Serialize, Debug, Clone)]
struct ToolInfo {
    name: String,
    /// What the LLM calls it: `<server>__<tool>`.
    qualified: String,
    description: Option<String>,
    input_schema: Value,
}

#[derive(Serialize, Debug, Clone)]
struct ServerStatus {
    name: String,
    url: String,
    connected: bool,
    /// `serverInfo` from the `initialize` handshake.
    server: Option<String>,
    instructions: Option<String>,
    tools: Vec<ToolInfo>,
    error: Option<String>,
    ms: u128,
}

struct Connected {
    spec: ServerSpec,
    client: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

#[derive(Default)]
struct Registry {
    connected: Vec<Connected>,
    status: Vec<ServerStatus>,
}

async fn open(url: &str) -> Result<(RunningService<RoleClient, ()>, Vec<Tool>), String> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.to_string()),
    );
    let client = ()
        .serve(transport)
        .await
        .map_err(|e| format!("MCP handshake failed: {e}"))?;
    let listed = client.list_all_tools().await;
    match listed {
        Ok(tools) => Ok((client, tools)),
        Err(e) => {
            let _ = client.cancel().await;
            Err(format!("tools/list failed: {e}"))
        }
    }
}

/// `serverInfo` ("name vX") and `instructions` from the handshake.
fn server_identity(client: &RunningService<RoleClient, ()>) -> (Option<String>, Option<String>) {
    let Some(info) = client
        .peer_info()
        .and_then(|info| serde_json::to_value(&*info).ok())
    else {
        return (None, None);
    };
    let server = &info["serverInfo"];
    let name = server["name"].as_str().map(|name| match server["version"].as_str() {
        Some(version) => format!("{name} v{version}"),
        None => name.to_string(),
    });
    (name, info["instructions"].as_str().map(str::to_string))
}

impl Registry {
    /// Connects to every registered server. One that can't be reached is
    /// reported and skipped; the others still work.
    async fn connect(specs: &[ServerSpec]) -> Registry {
        let mut registry = Registry::default();
        for spec in specs {
            let started = Instant::now();
            let attempt = match tokio::time::timeout(CONNECT_TIMEOUT, open(&spec.url)).await {
                Ok(result) => result,
                Err(_) => Err(format!("no answer within {}s", CONNECT_TIMEOUT.as_secs())),
            };
            let ms = started.elapsed().as_millis();
            match attempt {
                Ok((client, tools)) => {
                    let (server, instructions) = server_identity(&client);
                    println!(
                        "connected to {} at {} ({} tools) in {ms} ms",
                        spec.name,
                        spec.url,
                        tools.len()
                    );
                    registry.status.push(ServerStatus {
                        name: spec.name.clone(),
                        url: spec.url.clone(),
                        connected: true,
                        server,
                        instructions,
                        tools: tools
                            .iter()
                            .map(|t| ToolInfo {
                                name: t.name.to_string(),
                                qualified: qualified(&spec.name, &t.name),
                                description: t.description.as_ref().map(|d| d.to_string()),
                                input_schema: Value::Object((*t.input_schema).clone()),
                            })
                            .collect(),
                        error: None,
                        ms,
                    });
                    registry.connected.push(Connected {
                        spec: spec.clone(),
                        client,
                        tools,
                    });
                }
                Err(e) => {
                    eprintln!("could not connect to {} at {}: {e}", spec.name, spec.url);
                    registry.status.push(ServerStatus {
                        name: spec.name.clone(),
                        url: spec.url.clone(),
                        connected: false,
                        server: None,
                        instructions: None,
                        tools: Vec::new(),
                        error: Some(e),
                        ms,
                    });
                }
            }
        }
        registry
    }

    async fn close(self) {
        for c in self.connected {
            let _ = c.client.cancel().await;
        }
    }

    /// Every tool of every connected server, as OpenAI-style functions named
    /// `<server>__<tool>`, so two servers may even have tools of the same name.
    fn llm_tools(&self) -> Vec<Value> {
        self.connected
            .iter()
            .flat_map(|c| {
                c.tools
                    .iter()
                    .filter_map(move |t| openai_tool(&c.spec.name, t))
            })
            .collect()
    }

    /// `crates__crate_info` -> the `crates` server and its `crate_info` tool.
    fn route<'a>(&'a self, requested: &'a str) -> Result<(&'a Connected, &'a str), String> {
        let Some((server, tool)) = requested.split_once(SEP) else {
            return Err(format!(
                "`{requested}` is not a `<server>{SEP}<tool>` name, so it can't be routed."
            ));
        };
        let Some(connected) = self.connected.iter().find(|c| c.spec.name == server) else {
            let known: Vec<&str> = self.connected.iter().map(|c| c.spec.name.as_str()).collect();
            return Err(format!(
                "No connected MCP server named `{server}` (connected: {}).",
                known.join(", ")
            ));
        };
        if !connected.tools.iter().any(|t| &*t.name == tool) {
            let tools: Vec<&str> = connected.tools.iter().map(|t| &*t.name).collect();
            return Err(format!(
                "The `{server}` server has no tool `{tool}` (it has: {}).",
                tools.join(", ")
            ));
        }
        Ok((connected, tool))
    }

    /// Routes one call to the server that owns the tool and runs it there.
    async fn call(&self, step: usize, round: usize, requested: &str, arguments: Value) -> ToolCallTrace {
        let started = Instant::now();
        let (server, tool, routed, outcome) = match self.route(requested) {
            Ok((connected, tool)) => {
                let outcome = match &arguments {
                    Value::Object(map) => connected
                        .client
                        .call_tool(
                            CallToolRequestParams::new(tool.to_string())
                                .with_arguments(map.clone()),
                        )
                        .await
                        .map_err(|e| format!("tools/call failed: {e}")),
                    _ => Err(format!("Arguments aren't a JSON object: {arguments}")),
                };
                (Some(connected.spec.name.clone()), tool.to_string(), true, outcome)
            }
            Err(e) => (None, requested.to_string(), false, Err(e)),
        };
        let (is_error, result_text, structured) = match outcome {
            Ok(result) => (
                result.is_error.unwrap_or(false),
                tool_result_text(&result),
                result.structured_content,
            ),
            Err(e) => (true, e, None),
        };
        println!(
            "step {step}: {requested} -> {} {} in {} ms",
            server.as_deref().unwrap_or("(unrouted)"),
            if is_error { "error" } else { "ok" },
            started.elapsed().as_millis()
        );
        ToolCallTrace {
            step,
            round,
            requested: requested.to_string(),
            server,
            tool,
            routed,
            arguments,
            is_error,
            result_text,
            structured,
            ms: started.elapsed().as_millis(),
        }
    }
}

fn openai_tool(server: &str, tool: &Tool) -> Option<Value> {
    let name = qualified(server, &tool.name);
    if !valid_function_name(&name) {
        return None;
    }
    let mut parameters = (*tool.input_schema).clone();
    parameters.remove("$schema");
    parameters.insert("type".to_string(), json!("object"));
    parameters
        .entry("properties")
        .or_insert_with(|| json!({}));
    Some(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!(
                "[{server} server] {}",
                tool.description.as_deref().unwrap_or_default()
            ),
            "parameters": parameters,
        }
    }))
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

// ---------------------------------------------------------------------------
// The LLM.
// ---------------------------------------------------------------------------

#[derive(Debug)]
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
        let mut body = json!({ "model": self.model, "messages": messages });
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
// The orchestrator: one conversation, many servers.
// ---------------------------------------------------------------------------

/// One `tools/call`, and which server it was routed to.
#[derive(Serialize, Debug, Clone)]
struct ToolCallTrace {
    step: usize,
    /// Which LLM round asked for it; one round may ask for several calls.
    round: usize,
    /// The name the LLM used, e.g. `crates__crate_info`.
    requested: String,
    /// The server it was routed to; `None` if it couldn't be routed.
    server: Option<String>,
    tool: String,
    routed: bool,
    arguments: Value,
    is_error: bool,
    result_text: String,
    structured: Option<Value>,
    ms: u128,
}

#[derive(Serialize, Debug, Clone)]
struct Check {
    name: String,
    ok: bool,
    detail: String,
}

impl Check {
    fn new(name: &str, ok: bool, detail: String) -> Self {
        Check {
            name: name.to_string(),
            ok,
            detail,
        }
    }
}

#[derive(Serialize, Debug, Default)]
struct Run {
    request: String,
    servers: Vec<ServerStatus>,
    calls: Vec<ToolCallTrace>,
    llm_rounds: usize,
    answer: Option<String>,
    checks: Vec<Check>,
    ok: bool,
    error: Option<String>,
}

/// Built from what the servers said about themselves in the handshake, so a
/// newly registered server is described to the model without code changes.
fn system_prompt(status: &[ServerStatus]) -> String {
    let mut servers = String::new();
    for s in status {
        if s.connected {
            let tools: Vec<&str> = s.tools.iter().map(|t| t.qualified.as_str()).collect();
            servers.push_str(&format!(
                "\n- `{}`: {} Tools: {}.",
                s.name,
                s.instructions.as_deref().unwrap_or("(no description)"),
                tools.join(", ")
            ));
        } else {
            servers.push_str(&format!(
                "\n- `{}`: unavailable right now ({}); none of its tools can be called.",
                s.name,
                s.error.as_deref().unwrap_or("not connected")
            ));
        }
    }
    format!(
        "You are an orchestrator agent connected to several MCP servers. Every tool is named \
         `<server>{SEP}<tool>`; the prefix is the server that runs it. The servers:{servers}\n\n\
         How to work:\n\
         - For each step, pick the server whose domain fits: package registry data comes from \
         one server, repository data from another, saving and reading files from a third. Use \
         only the tools listed above.\n\
         - Work step by step. Take crate names, repositories and file names from the user's \
         request or from earlier tool results; never invent them. When a crate's repository is \
         on GitHub, use that `owner/repo` for repository lookups.\n\
         - Independent lookups may be requested together in one round.\n\
         - When asked to save something, first collect all the data, then write the note once \
         with the complete content, then read it back once to confirm it was saved.\n\
         - If a tool fails, say so, and continue with what you have when you can.\n\
         Finish with a short answer (3-6 sentences) in the user's language: the result, and \
         which servers you used for what."
    )
}

async fn converse(
    llm: &Llm,
    registry: &Registry,
    request: &str,
    run: &mut Run,
) -> Result<Option<String>, String> {
    let tools = registry.llm_tools();
    if tools.is_empty() {
        return Err("None of the registered MCP servers is reachable, so there are no tools \
                    to call."
            .to_string());
    }
    let mut messages = vec![
        json!({ "role": "system", "content": system_prompt(&registry.status) }),
        json!({ "role": "user", "content": request }),
    ];
    for round in 1..=MAX_TOOL_ROUNDS {
        run.llm_rounds = round;
        let message = llm.chat(&messages, &tools).await?;
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            return message_text(&message)
                .map(Some)
                .ok_or_else(|| "The model returned an empty answer.".to_string());
        }

        messages.push(message);
        for call in tool_calls {
            let id = call["id"].as_str().unwrap_or_default().to_string();
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let raw_arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
            let arguments: Value = if raw_arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(raw_arguments)
                    .unwrap_or_else(|_| Value::String(raw_arguments.to_string()))
            };
            let trace = registry
                .call(run.calls.len() + 1, round, name, arguments)
                .await;
            let content = if trace.is_error {
                format!("Error: {}", trace.result_text)
            } else {
                trace.result_text.clone()
            };
            messages.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": clip(&content),
            }));
            run.calls.push(trace);
        }
    }
    Err(format!(
        "Gave up after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer."
    ))
}

async fn orchestrate(app: &AppState, request: &str) -> Run {
    let mut run = Run {
        request: request.to_string(),
        ..Default::default()
    };
    let outcome = tokio::time::timeout(RUN_TIMEOUT, async {
        let registry = Registry::connect(&app.settings.servers).await;
        run.servers = registry.status.clone();
        let result = converse(&app.llm, &registry, request, &mut run).await;
        registry.close().await;
        result
    })
    .await;
    match outcome {
        Ok(Ok(answer)) => run.answer = answer,
        Ok(Err(e)) => run.error = Some(e),
        Err(_) => run.error = Some(format!("Timed out after {}s", RUN_TIMEOUT.as_secs())),
    }
    run.checks = flow_checks(request, &run.servers, &run.calls);
    run.ok = run.error.is_none() && run.checks.iter().all(|c| c.ok);
    if let Some(e) = &run.error {
        eprintln!("run failed: {e}");
    }
    run
}

// ---------------------------------------------------------------------------
// Checking the flow: right servers, right tools, right order.
// ---------------------------------------------------------------------------

/// An argument that names something the agent must have learned before the
/// call: from the user's request or from an earlier tool result.
fn dependency(call: &ToolCallTrace) -> Option<(String, String)> {
    fn arg<'a>(call: &'a ToolCallTrace, key: &str) -> Option<&'a str> {
        call.arguments[key].as_str().map(str::trim)
    }
    match call.tool.as_str() {
        "crate_info" => arg(call, "name").map(|n| (format!("crate_info({n})"), n.to_lowercase())),
        "repo_info" => arg(call, "repo").map(|r| {
            let repo = normalize_repo(r).unwrap_or_else(|| r.to_string());
            (format!("repo_info({repo})"), repo.to_lowercase())
        }),
        "read_note" => arg(call, "filename").map(|f| (format!("read_note({f})"), f.to_lowercase())),
        _ => None,
    }
}

fn output_of(call: &ToolCallTrace) -> String {
    let structured = call
        .structured
        .as_ref()
        .map(Value::to_string)
        .unwrap_or_default();
    format!("{}\n{structured}", call.result_text).to_lowercase()
}

fn flow_checks(request: &str, servers: &[ServerStatus], calls: &[ToolCallTrace]) -> Vec<Check> {
    if calls.is_empty() {
        return vec![Check::new(
            "Tools were called",
            false,
            "The agent answered without calling any tool.".to_string(),
        )];
    }
    let mut checks = Vec::new();

    // 1. The flow really spans several servers.
    let mut per_server: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for c in calls.iter().filter(|c| !c.is_error) {
        if let Some(server) = &c.server {
            per_server.entry(server).or_default().push(&c.tool);
        }
    }
    let breakdown = per_server
        .iter()
        .map(|(server, tools)| format!("{server} ← {}", tools.join(", ")))
        .collect::<Vec<_>>()
        .join("; ");
    checks.push(Check::new(
        "Tools from several MCP servers",
        per_server.len() >= 2,
        if per_server.is_empty() {
            "No call succeeded on any server.".to_string()
        } else {
            format!(
                "{} of {} servers answered successful calls: {breakdown}.",
                per_server.len(),
                servers.iter().filter(|s| s.connected).count()
            )
        },
    ));

    // 2. Every call went to the server that actually lists the tool.
    let misrouted: Vec<String> = calls
        .iter()
        .filter_map(|c| {
            let owner = c.server.as_deref().and_then(|name| {
                servers
                    .iter()
                    .find(|s| s.name == name && s.tools.iter().any(|t| t.name == c.tool))
            });
            match owner {
                Some(_) => None,
                None => Some(format!(
                    "step {} `{}`: {}",
                    c.step,
                    c.requested,
                    if c.routed { "not listed by that server" } else { c.result_text.as_str() }
                )),
            }
        })
        .collect();
    checks.push(Check::new(
        "Each call routed to the server that owns the tool",
        misrouted.is_empty(),
        if misrouted.is_empty() {
            format!(
                "All {} calls went to a server whose tools/list includes the tool.",
                calls.len()
            )
        } else {
            format!("{}.", misrouted.join("; "))
        },
    ));

    // 3. Order: what a call needs was already known when it was made.
    let request_text = request.to_lowercase();
    let mut grounded = Vec::new();
    let mut ungrounded = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        if !call.routed {
            continue;
        }
        let Some((label, needle)) = dependency(call) else {
            continue;
        };
        // The most recent earlier step that mentions it.
        let source = calls[..i]
            .iter()
            .rev()
            .filter(|c| !c.is_error)
            .find(|c| mentions(&output_of(c), &needle))
            .map(|c| format!("step {} {}", c.step, c.tool))
            .or_else(|| mentions(&request_text, &needle).then(|| "the request".to_string()));
        match source {
            Some(source) => grounded.push(format!("step {} {label} ← {source}", call.step)),
            None => ungrounded.push(format!("step {} {label}", call.step)),
        }
    }
    if !grounded.is_empty() || !ungrounded.is_empty() {
        checks.push(Check::new(
            "Each step uses what an earlier step found",
            ungrounded.is_empty(),
            if ungrounded.is_empty() {
                format!("{}.", grounded.join("; "))
            } else {
                format!(
                    "Not found in the request or any earlier result: {}.",
                    ungrounded.join("; ")
                )
            },
        ));
    }

    // 4. The note is written after the data it's built from, and reads back intact.
    for (i, write) in calls.iter().enumerate() {
        if write.tool != "write_note" || write.is_error {
            continue;
        }
        let Some(saved) = &write.structured else { continue };
        let filename = saved["filename"].as_str().unwrap_or_default();
        let data_steps: Vec<String> = calls[..i]
            .iter()
            .filter(|c| !c.is_error && c.server.is_some() && c.server != write.server)
            .map(|c| c.step.to_string())
            .collect();
        let read_back = calls[i + 1..]
            .iter()
            .filter(|c| c.tool == "read_note" && !c.is_error && c.server == write.server)
            .filter_map(|c| c.structured.as_ref().map(|s| (c.step, s)))
            .find(|(_, s)| s["filename"].as_str() == Some(filename));
        let (ok, detail) = match (data_steps.is_empty(), read_back) {
            (true, _) => (
                false,
                format!(
                    "step {} wrote {filename} before any data was collected from another server.",
                    write.step
                ),
            ),
            (false, Some((step, read))) if read["checksum"] == saved["checksum"] => (
                true,
                format!(
                    "step {} wrote {filename} after data steps {}; step {step} read it back with \
                     the same checksum ({}).",
                    write.step,
                    data_steps.join(", "),
                    saved["checksum"].as_str().unwrap_or_default()
                ),
            ),
            (false, Some((step, read))) => (
                false,
                format!(
                    "step {step} read {filename} back as {}, but step {} wrote {}.",
                    read["checksum"], write.step, saved["checksum"]
                ),
            ),
            (false, None) => (
                true,
                format!(
                    "step {} wrote {filename} after data steps {} (not read back in this run).",
                    write.step,
                    data_steps.join(", ")
                ),
            ),
        };
        checks.push(Check::new(
            &format!("{filename}: written after the data, read back intact"),
            ok,
            detail,
        ));
    }

    // 5. Nothing failed along the way.
    let failed: Vec<String> = calls
        .iter()
        .filter(|c| c.is_error)
        .map(|c| format!("step {} `{}`", c.step, c.requested))
        .collect();
    checks.push(Check::new(
        "Every tool call succeeded",
        failed.is_empty(),
        if failed.is_empty() {
            format!("{} calls, no errors.", calls.len())
        } else {
            format!("Failed: {}.", failed.join(", "))
        },
    ));
    checks
}

// ---------------------------------------------------------------------------
// HTTP API and page.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Settings {
    model: String,
    servers: Vec<ServerSpec>,
    notes_dir: String,
}

/// What this binary's own three servers talk to.
#[derive(Debug)]
struct Backends {
    crates: Upstream,
    github: Upstream,
    notes_dir: PathBuf,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<Llm>,
    backends: Arc<Backends>,
    settings: Arc<Settings>,
}

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

async fn ask(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Run> {
    let question = req.question.trim();
    if question.is_empty() {
        return Json(Run {
            error: Some("Ask something first.".to_string()),
            ..Default::default()
        });
    }
    Json(orchestrate(&app, question).await)
}

/// Connects to every registered server (initialize + tools/list) and reports.
async fn servers_api(State(app): State<AppState>) -> Json<Vec<ServerStatus>> {
    let registry = Registry::connect(&app.settings.servers).await;
    let status = registry.status.clone();
    registry.close().await;
    Json(status)
}

async fn notes_api(State(app): State<AppState>) -> Json<Vec<NoteFile>> {
    Json(list_note_files(&app.backends.notes_dir))
}

async fn note_api(
    State(app): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<Value> {
    let content = validate_filename(&name).and_then(|_| {
        std::fs::read_to_string(app.backends.notes_dir.join(&name)).map_err(|e| e.to_string())
    });
    Json(match content {
        Ok(content) => json!({
            "name": name,
            "bytes": content.len(),
            "checksum": checksum(&content),
            "content": content,
        }),
        Err(e) => json!({ "error": format!("Can't read {name:?}: {e}") }),
    })
}

async fn config(State(app): State<AppState>) -> Json<Settings> {
    Json((*app.settings).clone())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: AppState) -> Router {
    let backends = state.backends.clone();
    Router::new()
        .route("/", get(index))
        .route("/api/config", get(config))
        .route("/api/servers", get(servers_api))
        .route("/api/ask", post(ask))
        .route("/api/notes", get(notes_api))
        .route("/api/notes/{name}", get(note_api))
        .with_state(state)
        .nest_service("/mcp/crates", crates_service(backends.crates.clone()))
        .nest_service("/mcp/github", github_service(backends.github.clone()))
        .nest_service("/mcp/notes", notes_service(backends.notes_dir.clone()))
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build HTTP client")
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
    let servers = match env_non_empty("MCP_SERVERS") {
        Some(spec) => parse_servers(&spec).unwrap_or_else(|e| {
            eprintln!("Error: MCP_SERVERS: {e}");
            std::process::exit(1);
        }),
        None => default_servers(&format!("http://127.0.0.1:{port}")),
    };
    // Relative to the working directory: the lesson folder under `cargo run`,
    // `~/apps/lesson-20` on the VDS (the systemd unit's WorkingDirectory).
    let notes_dir = env_non_empty("NOTES_DIR").unwrap_or_else(|| "notes".to_string());

    let llm = Arc::new(Llm {
        http: http_client(Duration::from_secs(120)),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model: model.clone(),
    });
    let backends = Arc::new(Backends {
        crates: Upstream {
            label: "crates.io",
            http: http_client(Duration::from_secs(30)),
            base: CRATES_API.to_string(),
            token: None,
        },
        github: Upstream {
            label: "GitHub",
            http: http_client(Duration::from_secs(30)),
            base: GITHUB_API.to_string(),
            token: env_non_empty("GITHUB_TOKEN"),
        },
        notes_dir: PathBuf::from(&notes_dir),
    });
    for s in &servers {
        println!("registered MCP server {} at {}", s.name, s.url);
    }
    let state = AppState {
        llm,
        backends,
        settings: Arc::new(Settings {
            model,
            servers,
            notes_dir,
        }),
    };

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port} (MCP servers at /mcp/crates, /mcp/github, /mcp/notes)");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson20-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn upstream(label: &'static str, base: &str) -> Upstream {
        Upstream {
            label,
            http: reqwest::Client::new(),
            base: base.to_string(),
            token: None,
        }
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn crate_fixture(name: &str) -> Option<Value> {
        match name {
            "alpha-mcp" => Some(json!({
                "name": "alpha-mcp",
                "description": "Build MCP servers fast",
                "downloads": 50000,
                "recent_downloads": 9000,
                "max_version": "0.4.0",
                "max_stable_version": "0.4.0",
                "repository": "https://github.com/octo/alpha",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2026-09-10T00:00:00Z"
            })),
            "beta-mcp" => Some(json!({
                "name": "beta-mcp",
                "description": null,
                "downloads": 1200,
                "recent_downloads": 100,
                "max_version": "0.1.0-rc.1",
                "max_stable_version": null,
                "repository": "https://github.com/octo/beta.git",
                "created_at": "2026-02-01T00:00:00Z",
                "updated_at": "2026-03-01T00:00:00Z"
            })),
            _ => None,
        }
    }

    /// Stands in for crates.io: search, and one crate by name.
    fn fake_crates() -> Router {
        Router::new()
            .route(
                "/api/v1/crates",
                get(|| async {
                    Json(json!({
                        "crates": [crate_fixture("alpha-mcp"), crate_fixture("beta-mcp")],
                        "meta": { "total": 2 }
                    }))
                }),
            )
            .route(
                "/api/v1/crates/{name}",
                get(|axum::extract::Path(name): axum::extract::Path<String>| async move {
                    match crate_fixture(&name) {
                        Some(krate) => (
                            axum::http::StatusCode::OK,
                            Json(json!({
                                "crate": krate,
                                "versions": [
                                    { "num": "0.4.0", "license": "MIT", "created_at": "2026-09-10T00:00:00Z", "yanked": false, "rust_version": "1.85" },
                                    { "num": "0.3.0", "license": "MIT", "created_at": "2025-01-01T00:00:00Z", "yanked": true, "rust_version": null }
                                ]
                            })),
                        ),
                        None => (
                            axum::http::StatusCode::NOT_FOUND,
                            Json(json!({ "errors": [{ "detail": format!("crate `{name}` does not exist") }] })),
                        ),
                    }
                }),
            )
    }

    fn repo_fixture(full_name: &str) -> Option<Value> {
        let stars = match full_name {
            "octo/alpha" => 900,
            "octo/beta" => 12,
            _ => return None,
        };
        Some(json!({
            "full_name": full_name,
            "html_url": format!("https://github.com/{full_name}"),
            "description": "An MCP framework",
            "stargazers_count": stars,
            "forks_count": 3,
            "open_issues_count": 4,
            "language": "Rust",
            "topics": ["mcp"],
            "license": { "spdx_id": "MIT" },
            "archived": false,
            "pushed_at": "2026-09-20T10:00:00Z",
            "created_at": "2025-01-01T00:00:00Z",
            "default_branch": "main"
        }))
    }

    /// Stands in for api.github.com.
    fn fake_github() -> Router {
        Router::new()
            .route(
                "/repos/{owner}/{repo}",
                get(|axum::extract::Path((owner, repo)): axum::extract::Path<(String, String)>| async move {
                    match repo_fixture(&format!("{owner}/{repo}")) {
                        Some(r) => (axum::http::StatusCode::OK, Json(r)),
                        None => (
                            axum::http::StatusCode::NOT_FOUND,
                            Json(json!({ "message": "Not Found" })),
                        ),
                    }
                }),
            )
            .route(
                "/search/repositories",
                get(|| async {
                    Json(json!({
                        "total_count": 1,
                        "items": [repo_fixture("octo/alpha")]
                    }))
                }),
            )
    }

    /// Plays the model: returns the scripted tool calls for each round (several
    /// calls in one round are allowed), then `answer`. Records every request.
    fn scripted_llm(
        rounds: Vec<Vec<(&'static str, Value)>>,
        answer: &'static str,
        requests: Arc<Mutex<Vec<Value>>>,
    ) -> Router {
        let rounds = Arc::new(rounds);
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let rounds = rounds.clone();
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body.clone());
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

    /// The real app with its three MCP endpoints, on a random port, plus any
    /// `extra` servers registered next to them.
    async fn spawn_app(llm_base: &str, notes_dir: PathBuf, extra: &[(&str, &str)]) -> AppState {
        let crates = spawn(fake_crates()).await;
        let github = spawn(fake_github()).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut servers = default_servers(&format!("http://{addr}"));
        for (name, url) in extra {
            servers.push(ServerSpec {
                name: name.to_string(),
                url: url.to_string(),
            });
        }
        let state = AppState {
            llm: Arc::new(Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{llm_base}/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            }),
            backends: Arc::new(Backends {
                crates: upstream("crates.io", &crates),
                github: upstream("GitHub", &github),
                notes_dir: notes_dir.clone(),
            }),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                servers,
                notes_dir: notes_dir.display().to_string(),
            }),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        state
    }

    fn failed_checks(run: &Run) -> Vec<String> {
        run.checks
            .iter()
            .filter(|c| !c.ok)
            .map(|c| format!("{}: {}", c.name, c.detail))
            .collect()
    }

    fn tool_names(status: &ServerStatus) -> Vec<&str> {
        let mut names: Vec<&str> = status.tools.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        names
    }

    #[tokio::test]
    async fn each_server_is_its_own_mcp_endpoint_with_its_own_tools() {
        let notes_dir = temp_dir("endpoints");
        let state = spawn_app("http://127.0.0.1:9", notes_dir.clone(), &[]).await;
        let registry = Registry::connect(&state.settings.servers).await;

        let names: Vec<&str> = registry.status.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["crates", "github", "notes"]);
        for status in &registry.status {
            assert!(status.connected, "{}: {:?}", status.name, status.error);
            // Each endpoint introduces itself as a different server.
            let server = status.server.as_deref().unwrap();
            assert!(server.starts_with(&format!("{} v", status.name)), "{server}");
            assert!(status.instructions.as_deref().is_some_and(|i| !i.is_empty()));
            for tool in &status.tools {
                assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
                let properties = tool.input_schema["properties"].as_object();
                for (property, spec) in properties.into_iter().flatten() {
                    assert!(
                        spec["description"].as_str().is_some_and(|d| !d.is_empty()),
                        "{}.{property} has no description",
                        tool.qualified
                    );
                }
            }
        }
        assert_eq!(tool_names(&registry.status[0]), ["crate_info", "search_crates"]);
        assert_eq!(tool_names(&registry.status[1]), ["repo_info", "search_repositories"]);
        assert_eq!(tool_names(&registry.status[2]), ["list_notes", "read_note", "write_note"]);

        // What the LLM sees: every tool once, namespaced, as a valid function.
        let mut functions: Vec<String> = registry
            .llm_tools()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect();
        functions.sort_unstable();
        assert_eq!(
            functions,
            [
                "crates__crate_info",
                "crates__search_crates",
                "github__repo_info",
                "github__search_repositories",
                "notes__list_notes",
                "notes__read_note",
                "notes__write_note",
            ]
        );
        for tool in registry.llm_tools() {
            assert_eq!(tool["function"]["parameters"]["type"], "object");
            assert!(tool["function"]["parameters"]["properties"].is_object());
        }

        // The notes server validates, writes, and reads back.
        let bad = registry
            .call(1, 1, "notes__write_note", json!({ "filename": "../x.md", "content": "x" }))
            .await;
        assert!(bad.is_error && bad.routed);
        let written = registry
            .call(2, 1, "notes__write_note", json!({ "filename": "a.md", "content": "# A\n" }))
            .await;
        assert!(!written.is_error, "{}", written.result_text);
        let read = registry
            .call(3, 1, "notes__read_note", json!({ "filename": "a.md" }))
            .await;
        assert_eq!(read.structured.as_ref().unwrap()["content"], "# A\n");
        assert_eq!(read.structured.unwrap()["checksum"], written.structured.unwrap()["checksum"]);
        let listed = registry.call(4, 1, "notes__list_notes", json!({})).await;
        assert!(listed.result_text.contains("a.md"), "{}", listed.result_text);
        registry.close().await;
    }

    #[tokio::test]
    async fn registry_routes_by_prefix_and_refuses_the_rest() {
        let state = spawn_app("http://127.0.0.1:9", temp_dir("route"), &[]).await;
        let registry = Registry::connect(&state.settings.servers).await;

        let (server, tool) = registry.route("crates__crate_info").unwrap();
        assert_eq!((server.spec.name.as_str(), tool), ("crates", "crate_info"));
        let (server, tool) = registry.route("github__repo_info").unwrap();
        assert_eq!((server.spec.name.as_str(), tool), ("github", "repo_info"));

        let wrong_server = registry.route("github__crate_info").unwrap_err();
        assert!(wrong_server.contains("has no tool `crate_info`"), "{wrong_server}");
        let unknown = registry.route("weather__forecast").unwrap_err();
        assert!(unknown.contains("No connected MCP server named `weather`"), "{unknown}");
        assert!(registry.route("crate_info").is_err());

        // Routed for real: the crates server answers from (fake) crates.io.
        let info = registry
            .call(1, 1, "crates__crate_info", json!({ "name": "alpha-mcp" }))
            .await;
        assert_eq!(info.server.as_deref(), Some("crates"));
        let structured = info.structured.unwrap();
        assert_eq!(structured["github_repo"], "octo/alpha");
        assert_eq!(structured["license"], "MIT");
        assert_eq!(structured["yanked"], 1);
        let missing = registry
            .call(2, 1, "crates__crate_info", json!({ "name": "nope" }))
            .await;
        assert!(missing.is_error && missing.result_text.contains("does not exist"));
        registry.close().await;
    }

    const REPORT: &str = "# MCP crates\n\n\
        | crate | downloads | repo | stars |\n|---|---|---|---|\n\
        | alpha-mcp | 50000 | octo/alpha | 900 |\n| beta-mcp | 1200 | octo/beta | 12 |\n";

    #[tokio::test]
    async fn agent_runs_a_long_flow_across_three_servers() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = spawn(scripted_llm(
            vec![
                vec![("crates__search_crates", json!({ "query": "mcp", "limit": 2 }))],
                vec![
                    ("crates__crate_info", json!({ "name": "alpha-mcp" })),
                    ("crates__crate_info", json!({ "name": "beta-mcp" })),
                ],
                vec![("github__repo_info", json!({ "repo": "octo/alpha" }))],
                vec![("github__repo_info", json!({ "repo": "https://github.com/octo/beta.git" }))],
                vec![("notes__write_note", json!({ "filename": "report.md", "content": REPORT }))],
                vec![("notes__read_note", json!({ "filename": "report.md" }))],
            ],
            "alpha-mcp leads; saved to report.md using crates, github and notes.",
            requests.clone(),
        ))
        .await;
        let notes_dir = temp_dir("flow");
        let state = spawn_app(&llm, notes_dir.clone(), &[]).await;

        let run = orchestrate(
            &state,
            "Compare Rust MCP crates on crates.io and GitHub, save to report.md",
        )
        .await;

        assert_eq!(run.error, None);
        let order: Vec<&str> = run.calls.iter().map(|c| c.requested.as_str()).collect();
        assert_eq!(
            order,
            [
                "crates__search_crates",
                "crates__crate_info",
                "crates__crate_info",
                "github__repo_info",
                "github__repo_info",
                "notes__write_note",
                "notes__read_note",
            ]
        );
        let servers: Vec<&str> = run.calls.iter().map(|c| c.server.as_deref().unwrap()).collect();
        assert_eq!(servers, ["crates", "crates", "crates", "github", "github", "notes", "notes"]);
        let rounds: Vec<usize> = run.calls.iter().map(|c| c.round).collect();
        assert_eq!(rounds, [1, 2, 2, 3, 4, 5, 6]);
        assert_eq!(run.llm_rounds, 7);
        assert!(run.calls.iter().all(|c| c.routed && !c.is_error));
        assert!(run.answer.as_deref().unwrap().contains("report.md"));
        assert!(run.ok, "failed checks: {:?}", failed_checks(&run));
        assert_eq!(run.checks.len(), 5, "{:?}", run.checks);
        let order_check = &run.checks[2].detail;
        assert!(order_check.contains("crate_info(alpha-mcp) ← step 1 search_crates"), "{order_check}");
        assert!(order_check.contains("repo_info(octo/beta) ← step 3 crate_info"), "{order_check}");
        assert!(order_check.contains("read_note(report.md) ← step 6 write_note"), "{order_check}");

        // The saved note is on disk.
        assert_eq!(std::fs::read_to_string(notes_dir.join("report.md")).unwrap(), REPORT);

        // The model was told about all three servers and got every result back.
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 7);
        let system = requests[0]["messages"][0]["content"].as_str().unwrap();
        for server in ["`crates`", "`github`", "`notes`", "crates.io", "notes directory"] {
            assert!(system.contains(server), "system prompt lacks {server}");
        }
        assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 7);
        let last = requests[6]["messages"].as_array().unwrap();
        let tool_messages: Vec<&Value> = last.iter().filter(|m| m["role"] == "tool").collect();
        assert_eq!(tool_messages.len(), 7);
        assert!(tool_messages[1]["content"].as_str().unwrap().contains("GitHub: octo/alpha"));
        assert!(tool_messages[3]["content"].as_str().unwrap().contains("octo/alpha ★ 900"));
    }

    #[tokio::test]
    async fn unknown_tools_and_invented_arguments_are_flagged() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = spawn(scripted_llm(
            vec![
                vec![("weather__forecast", json!({ "city": "Paris" }))],
                vec![("github__repo_info", json!({ "repo": "someone/invented" }))],
                vec![("notes__read_note", json!({ "filename": "never.md" }))],
            ],
            "I could not do it.",
            requests.clone(),
        ))
        .await;
        let state = spawn_app(&llm, temp_dir("bad"), &[]).await;

        let run = orchestrate(&state, "Tell me the weather").await;

        assert_eq!(run.error, None);
        assert_eq!(run.calls.len(), 3);
        assert!(!run.calls[0].routed && run.calls[0].server.is_none());
        assert_eq!(run.calls[1].server.as_deref(), Some("github"));
        assert!(run.calls.iter().all(|c| c.is_error));
        // The refusal went back to the model as the tool result.
        let requests = requests.lock().unwrap();
        let reply = requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap()["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(reply.contains("No connected MCP server named `weather`"), "{reply}");

        assert!(!run.ok);
        let failed = failed_checks(&run).join("\n");
        assert!(failed.contains("Each call routed"), "{failed}");
        assert!(failed.contains("step 2 repo_info(someone/invented)"), "{failed}");
        assert!(failed.contains("step 3 read_note(never.md)"), "{failed}");
        assert!(failed.contains("Every tool call succeeded"), "{failed}");
    }

    #[tokio::test]
    async fn an_unreachable_server_is_reported_and_the_rest_still_work() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = spawn(scripted_llm(
            vec![vec![("crates__crate_info", json!({ "name": "alpha-mcp" }))]],
            "alpha-mcp is at 0.4.0.",
            requests.clone(),
        ))
        .await;
        let state = spawn_app(&llm, temp_dir("ghost"), &[("ghost", "http://127.0.0.1:9/mcp")]).await;

        let run = orchestrate(&state, "What version is alpha-mcp?").await;

        assert_eq!(run.error, None);
        assert_eq!(run.servers.len(), 4);
        let ghost = &run.servers[3];
        assert!(!ghost.connected && ghost.error.is_some() && ghost.tools.is_empty());
        assert!(run.servers[..3].iter().all(|s| s.connected));
        assert_eq!(run.calls.len(), 1);
        assert!(!run.calls[0].is_error);

        let requests = requests.lock().unwrap();
        let tools = requests[0]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 7);
        assert!(tools.iter().all(|t| !t["function"]["name"].as_str().unwrap().starts_with("ghost")));
        let system = requests[0]["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("`ghost`: unavailable"), "{system}");
    }

    fn trace(step: usize, server: &str, tool: &str, arguments: Value, structured: Value) -> ToolCallTrace {
        ToolCallTrace {
            step,
            round: step,
            requested: qualified(server, tool),
            server: Some(server.to_string()),
            tool: tool.to_string(),
            routed: true,
            arguments,
            is_error: false,
            result_text: String::new(),
            structured: Some(structured),
            ms: 0,
        }
    }

    fn status(name: &str, tools: &[&str]) -> ServerStatus {
        ServerStatus {
            name: name.to_string(),
            url: String::new(),
            connected: true,
            server: None,
            instructions: None,
            tools: tools
                .iter()
                .map(|t| ToolInfo {
                    name: t.to_string(),
                    qualified: qualified(name, t),
                    description: None,
                    input_schema: json!({}),
                })
                .collect(),
            error: None,
            ms: 0,
        }
    }

    #[test]
    fn checks_catch_wrong_order_and_a_note_that_changed() {
        let servers = [
            status("crates", &["search_crates", "crate_info"]),
            status("notes", &["write_note", "read_note"]),
        ];
        // Read before write, and the write came before any data.
        let calls = [
            trace(1, "notes", "read_note", json!({ "filename": "r.md" }), json!({ "filename": "r.md", "checksum": "b" })),
            trace(2, "notes", "write_note", json!({ "filename": "r.md" }), json!({ "filename": "r.md", "checksum": "a" })),
            trace(3, "crates", "crate_info", json!({ "name": "alpha-mcp" }), json!({ "name": "alpha-mcp" })),
        ];
        let checks = flow_checks("tell me about crates", &servers, &calls);
        let failed: Vec<&str> = checks.iter().filter(|c| !c.ok).map(|c| c.detail.as_str()).collect();
        assert_eq!(failed.len(), 2, "{checks:?}");
        assert!(failed[0].contains("step 1 read_note(r.md)"), "{failed:?}");
        assert!(failed[0].contains("step 3 crate_info(alpha-mcp)"), "{failed:?}");
        assert!(failed[1].contains("before any data"), "{failed:?}");

        // Right order, but the note read back differs from what was written.
        let calls = [
            trace(1, "crates", "search_crates", json!({ "query": "mcp" }), json!({ "crates": [{ "name": "alpha-mcp" }] })),
            trace(2, "notes", "write_note", json!({ "filename": "r.md" }), json!({ "filename": "r.md", "checksum": "a" })),
            trace(3, "notes", "read_note", json!({ "filename": "r.md" }), json!({ "filename": "r.md", "checksum": "b" })),
        ];
        let checks = flow_checks("mcp crates", &servers, &calls);
        let failed: Vec<&Check> = checks.iter().filter(|c| !c.ok).collect();
        assert_eq!(failed.len(), 1, "{checks:?}");
        assert!(failed[0].detail.contains("read r.md back as"), "{}", failed[0].detail);
    }

    #[test]
    fn server_list_is_parsed_and_validated() {
        let servers = parse_servers(" crates=http://a/mcp, docs=https://mcp.deepwiki.com/mcp ").unwrap();
        assert_eq!(
            servers,
            [
                ServerSpec { name: "crates".to_string(), url: "http://a/mcp".to_string() },
                ServerSpec { name: "docs".to_string(), url: "https://mcp.deepwiki.com/mcp".to_string() },
            ]
        );
        for bad in ["", "crates", "Crates=http://a", "a__b=http://a", "a=ftp://x", "a=http://x,a=http://y"] {
            assert!(parse_servers(bad).is_err(), "{bad:?} was accepted");
        }
        assert_eq!(default_servers("http://h:1/")[2].url, "http://h:1/mcp/notes");
    }

    #[test]
    fn repositories_are_normalized() {
        for input in [
            "octo/alpha",
            "https://github.com/octo/alpha",
            "github.com/octo/alpha.git",
            "https://github.com/octo/alpha/tree/main",
        ] {
            assert_eq!(normalize_repo(input).as_deref(), Some("octo/alpha"), "{input}");
        }
        for bad in ["", "octo", "https://gitlab.com/x/y", "octo/.hidden", "a b/c"] {
            assert_eq!(normalize_repo(bad), None, "{bad}");
        }
        assert_eq!(github_repo_of("https://github.com/tokio-rs/axum").as_deref(), Some("tokio-rs/axum"));
        assert_eq!(github_repo_of("https://gitlab.com/a/b"), None);
    }

    #[test]
    fn mentions_matches_whole_names_only() {
        assert!(mentions("see github.com/octo/alpha.", "octo/alpha"));
        assert!(mentions("\"name\":\"alpha-mcp\"", "alpha-mcp"));
        assert!(!mentions("octo/alpha-cli", "octo/alpha"));
        assert!(!mentions("tokio-util", "tokio"));
        assert!(!mentions("anything", ""));
    }

    #[test]
    fn helpers_behave() {
        assert_eq!(days_since("1970-01-01", 0), Some(0));
        assert_eq!(days_since("2026-09-20T10:00:00Z", 1_790_380_800), Some(6));
        assert_eq!(days_since("garbage", 0), None);
        assert_eq!(checksum(""), "fnv1a64:cbf29ce484222325");
        for good in ["report.md", "a_b-2.json", "notes.txt"] {
            assert!(validate_filename(good).is_ok(), "{good}");
        }
        for bad in ["", "../x.md", "a/b.md", ".env", "x.sh"] {
            assert!(validate_filename(bad).is_err(), "{bad}");
        }
        assert!(valid_function_name("crates__crate_info"));
        assert!(!valid_function_name("docs__read.wiki"));
    }
}

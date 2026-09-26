use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const GITHUB_API: &str = "https://api.github.com";
const DEFAULT_SEARCH_LIMIT: u32 = 5;
const MAX_SEARCH_LIMIT: u32 = 10;
const MAX_QUERY_CHARS: usize = 200;
const MAX_DESCRIPTION_CHARS: usize = 200;
const MAX_FOCUS_CHARS: usize = 200;
const MAX_LANGUAGE_CHARS: usize = 40;
const MAX_FILENAME_CHARS: usize = 80;
const ALLOWED_EXTENSIONS: [&str; 3] = ["md", "txt", "json"];
/// A public, always-on instance can't be allowed to fill the disk.
const MAX_OUTPUT_FILES: usize = 100;
/// Artifacts live in memory only; older ones are dropped.
const KEEP_ARTIFACTS: usize = 50;
const MAX_TOOL_ROUNDS: usize = 6;
const RUN_TIMEOUT: Duration = Duration::from_secs(180);

// ---------------------------------------------------------------------------
// Time and checksums, without pulling in extra crates.
// ---------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ`.
fn iso(ts: u64) -> String {
    let secs = ts % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        civil_from_days((ts / 86_400) as i64),
        secs / 3_600,
        secs % 3_600 / 60,
        secs % 60
    )
}

/// Days since 1970-01-01 -> proleptic Gregorian date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// FNV-1a over the UTF-8 bytes. Not cryptographic: it only has to show that
/// the bytes one tool handed on are the bytes the next one received.
fn checksum(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{hash:016x}")
}

// ---------------------------------------------------------------------------
// Artifacts: what each tool produces, kept on the server and passed on by id.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ArtifactKind {
    SearchResults,
    Summary,
}

impl ArtifactKind {
    fn name(self) -> &'static str {
        match self {
            ArtifactKind::SearchResults => "search_results",
            ArtifactKind::Summary => "summary",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Artifact {
    id: String,
    kind: ArtifactKind,
    produced_by: &'static str,
    /// The artifact this one was computed from, e.g. a summary's search results.
    derived_from: Option<String>,
    content: String,
    checksum: String,
    bytes: usize,
    created_at: u64,
}

#[derive(Debug, Default)]
struct Artifacts {
    next_id: u64,
    items: Vec<Artifact>,
}

impl Artifacts {
    fn add(&mut self, kind: ArtifactKind, derived_from: Option<&str>, content: String) -> Artifact {
        self.next_id += 1;
        let (prefix, produced_by) = match kind {
            ArtifactKind::SearchResults => ("search", "search"),
            ArtifactKind::Summary => ("summary", "summarize"),
        };
        let artifact = Artifact {
            id: format!("{prefix}-{}", self.next_id),
            kind,
            produced_by,
            derived_from: derived_from.map(str::to_string),
            checksum: checksum(&content),
            bytes: content.len(),
            content,
            created_at: unix_now(),
        };
        self.items.push(artifact.clone());
        if self.items.len() > KEEP_ARTIFACTS {
            let excess = self.items.len() - KEEP_ARTIFACTS;
            self.items.drain(..excess);
        }
        artifact
    }

    fn get(&self, id: &str) -> Option<Artifact> {
        self.items.iter().find(|a| a.id == id.trim()).cloned()
    }

    /// The chain of artifacts that led to `id`, oldest first.
    fn lineage(&self, id: &str) -> Vec<Value> {
        let mut chain = Vec::new();
        let mut next = Some(id.to_string());
        while let Some(id) = next {
            let Some(artifact) = self.items.iter().find(|a| a.id == id) else {
                break;
            };
            chain.push(json!({
                "artifact_id": artifact.id,
                "produced_by": artifact.produced_by,
                "checksum": artifact.checksum,
            }));
            next = artifact.derived_from.clone();
            if chain.len() > KEEP_ARTIFACTS {
                break;
            }
        }
        chain.reverse();
        chain
    }
}

// ---------------------------------------------------------------------------
// Step 1's data source: GitHub repository search.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Repo {
    full_name: String,
    url: String,
    description: String,
    stars: u64,
    language: Option<String>,
    topics: Vec<String>,
    updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SearchResults {
    query: String,
    total_count: u64,
    items: Vec<Repo>,
}

#[derive(Debug, Clone)]
struct GitHub {
    client: reqwest::Client,
    api_base: String,
    token: Option<String>,
}

#[derive(Deserialize)]
struct GhSearch {
    total_count: u64,
    items: Vec<GhRepo>,
}

#[derive(Deserialize)]
struct GhRepo {
    full_name: String,
    html_url: String,
    description: Option<String>,
    stargazers_count: u64,
    language: Option<String>,
    #[serde(default)]
    topics: Vec<String>,
    updated_at: Option<String>,
}

#[derive(Deserialize)]
struct GhError {
    message: String,
}

impl From<GhRepo> for Repo {
    fn from(r: GhRepo) -> Self {
        let description = r.description.unwrap_or_default();
        let mut short: String = description.trim().chars().take(MAX_DESCRIPTION_CHARS).collect();
        if description.trim().chars().count() > MAX_DESCRIPTION_CHARS {
            short.push('…');
        }
        Repo {
            full_name: r.full_name,
            url: r.html_url,
            description: short,
            stars: r.stargazers_count,
            language: r.language,
            topics: r.topics.into_iter().take(8).collect(),
            updated_at: r.updated_at.unwrap_or_default(),
        }
    }
}

impl GitHub {
    async fn search_repos(&self, query: &str, limit: u32) -> Result<SearchResults, String> {
        let url = format!(
            "{}/search/repositories",
            self.api_base.trim_end_matches('/')
        );
        let mut request = self
            .client
            .get(&url)
            .query(&[
                ("q", query.to_string()),
                ("sort", "stars".to_string()),
                ("order", "desc".to_string()),
                ("per_page", limit.to_string()),
            ])
            .header(ACCEPT, "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            // GitHub rejects requests without a User-Agent.
            .header(USER_AGENT, "ai-advent-pipeline");
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("GitHub request failed: {e}"))?;
        let status = response.status();
        let rate_limited = response
            .headers()
            .get("x-ratelimit-remaining")
            .is_some_and(|v| v.as_bytes() == b"0");
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            let detail = serde_json::from_str::<GhError>(&body)
                .map(|e| e.message)
                .unwrap_or(body);
            return Err(match status.as_u16() {
                422 => format!("GitHub rejected the query {query:?}: {detail}"),
                403 | 429 if rate_limited => "GitHub search rate limit exceeded \
                     (10 requests/minute without a token); try again in a minute."
                    .to_string(),
                _ => format!("GitHub API error ({status}): {detail}"),
            });
        }

        let found: GhSearch =
            serde_json::from_str(&body).map_err(|e| format!("Unexpected GitHub response: {e}"))?;
        Ok(SearchResults {
            query: query.to_string(),
            total_count: found.total_count,
            items: found
                .items
                .into_iter()
                .take(limit as usize)
                .map(Repo::from)
                .collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Step 2's processing: deterministic stats plus an LLM-written overview.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct LanguageCount {
    language: String,
    repositories: usize,
}

#[derive(Debug, Serialize)]
struct Stats {
    repositories: usize,
    total_matches: u64,
    total_stars: u64,
    most_starred: Option<String>,
    languages: Vec<LanguageCount>,
}

fn stats(results: &SearchResults) -> Stats {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for repo in &results.items {
        let language = repo.language.clone().unwrap_or_else(|| "not specified".to_string());
        *counts.entry(language).or_default() += 1;
    }
    let mut languages: Vec<LanguageCount> = counts
        .into_iter()
        .map(|(language, repositories)| LanguageCount {
            language,
            repositories,
        })
        .collect();
    languages.sort_by(|a, b| b.repositories.cmp(&a.repositories));
    Stats {
        repositories: results.items.len(),
        total_matches: results.total_count,
        total_stars: results.items.iter().map(|r| r.stars).sum(),
        most_starred: most_starred(results).map(|r| r.full_name.clone()),
        languages,
    }
}

fn most_starred(results: &SearchResults) -> Option<&Repo> {
    // Ties go to the earlier (higher-ranked) result.
    results
        .items
        .iter()
        .rev()
        .max_by_key(|r| r.stars)
}

fn summarize_prompt(language: &str, focus: Option<&str>) -> String {
    let focus = focus
        .map(|f| format!(" Pay particular attention to: {f}."))
        .unwrap_or_default();
    format!(
        "You are the processing step of a search -> summarize -> save pipeline. The user message \
         is JSON search results from GitHub: repositories with their stars, language, topics and \
         description. Write the overview section of a short report: 4-7 sentences of plain prose \
         (no headings, no lists, no links) saying what these projects are, how they differ, and \
         which stand out and why. Use only facts from the data and never invent anything. Refer \
         to repositories by their full name (owner/repo).{focus} Write in {language}."
    )
}

/// The Markdown report. The overview is the LLM's; everything else is built
/// straight from the search results, so the sources list can't drift.
fn render_report(
    source: &Artifact,
    results: &SearchResults,
    stats: &Stats,
    overview: &str,
    focus: Option<&str>,
    now: u64,
) -> String {
    let mut out = format!("# Search summary: {}\n\n", results.query);
    out.push_str(&format!(
        "Generated by the `summarize` MCP tool from `{}` ({}) at {}.\n",
        source.id,
        source.checksum,
        iso(now)
    ));
    if let Some(focus) = focus {
        out.push_str(&format!("Focus: {focus}\n"));
    }
    out.push_str("\n## Overview\n\n");
    out.push_str(overview.trim());
    out.push_str("\n\n## Stats\n\n");
    out.push_str(&format!(
        "- Repositories summarized: {} of {} matches on GitHub\n",
        stats.repositories, stats.total_matches
    ));
    if let Some(top) = most_starred(results) {
        out.push_str(&format!(
            "- Stars: {} in total; most starred: [{}]({}) ★ {}\n",
            stats.total_stars, top.full_name, top.url, top.stars
        ));
    }
    let languages = stats
        .languages
        .iter()
        .map(|l| format!("{} {}", l.language, l.repositories))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("- Languages: {languages}\n\n## Sources\n\n"));
    for (i, repo) in results.items.iter().enumerate() {
        out.push_str(&format!(
            "{}. [{}]({}) ★ {} · {}",
            i + 1,
            repo.full_name,
            repo.url,
            repo.stars,
            repo.language.as_deref().unwrap_or("—")
        ));
        if !repo.description.is_empty() {
            out.push_str(&format!(" — {}", repo.description));
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Step 3's output: files in one directory, never outside it.
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

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().is_file())
                .count()
        })
        .unwrap_or(0)
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

// ---------------------------------------------------------------------------
// The MCP server: search -> summarize -> save_to_file.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    /// What to look for, in GitHub repository search syntax, e.g. "mcp server language:rust".
    query: String,
    /// How many repositories to keep, most-starred first. 1-10, default 5.
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SummarizeArgs {
    /// Id of the search results to summarize: the `artifact_id` that `search` returned, e.g. "search-1".
    source_id: String,
    /// Optional angle for the overview, e.g. "maturity and maintenance activity".
    focus: Option<String>,
    /// Language to write the overview in, e.g. "English" or "Russian". Default English.
    language: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SaveToFileArgs {
    /// Id of the artifact to save: usually the `artifact_id` that `summarize` returned, e.g. "summary-2".
    source_id: String,
    /// File name inside the server's output directory, ending in .md, .txt or .json. Default "<source_id>.md".
    filename: Option<String>,
}

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong *inside* a tool (bad argument, unknown artifact, a failed
/// upstream call) is a tool-level error the model can explain, not a
/// JSON-RPC protocol error.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

/// What the tools share across MCP sessions: the artifacts and the clients.
#[derive(Debug)]
struct Toolbox {
    artifacts: Mutex<Artifacts>,
    github: GitHub,
    llm: Arc<Llm>,
    output_dir: PathBuf,
}

impl Toolbox {
    fn new(github: GitHub, llm: Arc<Llm>, output_dir: PathBuf) -> Self {
        Self {
            artifacts: Mutex::new(Artifacts::default()),
            github,
            llm,
            output_dir,
        }
    }

    fn store(&self, kind: ArtifactKind, derived_from: Option<&str>, content: String) -> Artifact {
        self.artifacts
            .lock()
            .unwrap()
            .add(kind, derived_from, content)
    }

    fn artifact(&self, id: &str) -> Option<Artifact> {
        self.artifacts.lock().unwrap().get(id)
    }

    fn lineage(&self, id: &str) -> Vec<Value> {
        self.artifacts.lock().unwrap().lineage(id)
    }
}

fn format_search(artifact: &Artifact, results: &SearchResults) -> String {
    let mut text = format!(
        "GitHub has {} repositories matching {:?}; kept the top {} by stars.\n\
         Stored as artifact {} ({}). Next step: `summarize` with source_id \"{}\".\n",
        results.total_count,
        results.query,
        results.items.len(),
        artifact.id,
        artifact.checksum,
        artifact.id
    );
    for (i, repo) in results.items.iter().enumerate() {
        text.push_str(&format!(
            "\n{}. {} ★ {} · {} — {}",
            i + 1,
            repo.full_name,
            repo.stars,
            repo.language.as_deref().unwrap_or("—"),
            if repo.description.is_empty() {
                "(no description)"
            } else {
                repo.description.as_str()
            }
        ));
    }
    text
}

#[derive(Debug, Clone)]
struct PipelineServer {
    toolbox: Arc<Toolbox>,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl PipelineServer {
    fn new(toolbox: Arc<Toolbox>) -> Self {
        Self {
            toolbox,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "search",
        description = "Pipeline step 1, get the data: search public GitHub repositories, \
                       most-starred first. The results are stored on this server as an artifact; \
                       pass the returned `artifact_id` to `summarize` as `source_id`, so the data \
                       moves to the next tool by reference instead of being retyped."
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let query = args.query.trim();
        if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
            return tool_error(format!(
                "`query` must be 1-{MAX_QUERY_CHARS} characters long."
            ));
        }
        let limit = args.limit.unwrap_or(DEFAULT_SEARCH_LIMIT);
        if !(1..=MAX_SEARCH_LIMIT).contains(&limit) {
            return tool_error(format!(
                "`limit` must be 1-{MAX_SEARCH_LIMIT}, got {limit}."
            ));
        }
        let results = match self.toolbox.github.search_repos(query, limit).await {
            Ok(results) => results,
            Err(e) => return tool_error(e),
        };
        let content = serde_json::to_string_pretty(&results).unwrap_or_default();
        let artifact = self
            .toolbox
            .store(ArtifactKind::SearchResults, None, content);
        tool_ok(
            format_search(&artifact, &results),
            json!({
                "artifact_id": artifact.id,
                "checksum": artifact.checksum,
                "bytes": artifact.bytes,
                "query": results.query,
                "total_count": results.total_count,
                "items": results.items,
            }),
        )
    }

    #[tool(
        name = "summarize",
        description = "Pipeline step 2, process the data: turn the search results stored under \
                       `source_id` into a Markdown report (an LLM-written overview, stats, and \
                       the list of sources). The report is stored as a new artifact; pass its \
                       `artifact_id` to `save_to_file`."
    )]
    async fn summarize(
        &self,
        Parameters(args): Parameters<SummarizeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(source) = self.toolbox.artifact(&args.source_id) else {
            return tool_error(format!(
                "Unknown artifact {:?}. Call `search` first and pass the `artifact_id` it returns.",
                args.source_id
            ));
        };
        if source.kind != ArtifactKind::SearchResults {
            return tool_error(format!(
                "`summarize` processes search results, but {} is a {} produced by `{}`.",
                source.id,
                source.kind.name(),
                source.produced_by
            ));
        }
        let results: SearchResults = match serde_json::from_str(&source.content) {
            Ok(results) => results,
            Err(e) => return tool_error(format!("{} is not readable search results: {e}", source.id)),
        };
        if results.items.is_empty() {
            return tool_error(format!(
                "{} has no repositories in it, so there is nothing to summarize. Search again \
                 with a broader query.",
                source.id
            ));
        }
        let focus = args
            .focus
            .as_deref()
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(|f| f.chars().take(MAX_FOCUS_CHARS).collect::<String>());
        let language = args
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| l.chars().take(MAX_LANGUAGE_CHARS).collect::<String>())
            .unwrap_or_else(|| "English".to_string());

        // The model gets the artifact's exact bytes, not a paraphrase of them.
        let messages = [
            json!({ "role": "system", "content": summarize_prompt(&language, focus.as_deref()) }),
            json!({ "role": "user", "content": source.content }),
        ];
        let overview = match self.toolbox.llm.chat(&messages, &[]).await.and_then(|m| {
            message_text(&m).ok_or_else(|| "the model returned an empty answer".to_string())
        }) {
            Ok(text) => text,
            Err(e) => return tool_error(format!("The LLM could not write the overview: {e}")),
        };

        let stats = stats(&results);
        let report = render_report(
            &source,
            &results,
            &stats,
            &overview,
            focus.as_deref(),
            unix_now(),
        );
        let artifact = self
            .toolbox
            .store(ArtifactKind::Summary, Some(source.id.as_str()), report);
        let text = format!(
            "Summarized {} ({}, {} repositories) into artifact {} ({}, {} bytes). \
             Next step: `save_to_file` with source_id \"{}\".\n\n{}",
            source.id,
            source.checksum,
            results.items.len(),
            artifact.id,
            artifact.checksum,
            artifact.bytes,
            artifact.id,
            artifact.content
        );
        tool_ok(
            text,
            json!({
                "artifact_id": artifact.id,
                "checksum": artifact.checksum,
                "bytes": artifact.bytes,
                "content": artifact.content,
                "source": {
                    "artifact_id": source.id,
                    "checksum": source.checksum,
                    "items": results.items.len(),
                },
                "stats": serde_json::to_value(&stats).unwrap_or_default(),
            }),
        )
    }

    #[tool(
        name = "save_to_file",
        description = "Pipeline step 3, save the result: write the artifact stored under \
                       `source_id` (usually the report from `summarize`) to a file in the \
                       server's output directory, then read the file back and confirm it holds \
                       exactly the artifact's bytes. Returns the path, size, checksum, and the \
                       artifact lineage (search -> summary -> file)."
    )]
    async fn save_to_file(
        &self,
        Parameters(args): Parameters<SaveToFileArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(source) = self.toolbox.artifact(&args.source_id) else {
            return tool_error(format!(
                "Unknown artifact {:?}. Pass the `artifact_id` returned by `summarize` (or `search`).",
                args.source_id
            ));
        };
        let filename = args
            .filename
            .as_deref()
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| {
                let extension = match source.kind {
                    ArtifactKind::SearchResults => "json",
                    ArtifactKind::Summary => "md",
                };
                format!("{}.{extension}", source.id)
            });
        if let Err(e) = validate_filename(&filename) {
            return tool_error(e);
        }
        let dir = &self.toolbox.output_dir;
        let path = dir.join(&filename);
        if !path.exists() && count_files(dir) >= MAX_OUTPUT_FILES {
            return tool_error(format!(
                "The output directory already holds {MAX_OUTPUT_FILES} files; overwrite an \
                 existing file name instead."
            ));
        }
        if let Err(e) = write_file(&path, &source.content) {
            return tool_error(format!("Could not write {}: {e}", path.display()));
        }

        // Read it back: the file has to hold exactly the artifact's bytes.
        let on_disk = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => return tool_error(format!("Wrote {} but could not read it back: {e}", path.display())),
        };
        let disk_checksum = checksum(&on_disk);
        if disk_checksum != source.checksum {
            return tool_error(format!(
                "{} reads back as {disk_checksum} instead of {}.",
                path.display(),
                source.checksum
            ));
        }

        let lineage = self.toolbox.lineage(&source.id);
        let chain = lineage
            .iter()
            .filter_map(|a| a["artifact_id"].as_str())
            .collect::<Vec<_>>()
            .join(" → ");
        tool_ok(
            format!(
                "Saved {} to {} ({} bytes, {}), verified by reading the file back. \
                 Lineage: {chain} → {filename}.",
                source.id,
                path.display(),
                on_disk.len(),
                disk_checksum
            ),
            json!({
                "filename": filename,
                "path": path.display().to_string(),
                "bytes": on_disk.len(),
                "checksum": disk_checksum,
                "verified_on_disk": true,
                "source": { "artifact_id": source.id, "checksum": source.checksum },
                "lineage": lineage,
            }),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PipelineServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("pipeline", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Three tools meant to be chained: `search` gets GitHub repositories, \
                 `summarize` turns them into a report, `save_to_file` saves the report. Each \
                 step returns an `artifact_id`; pass it to the next step as `source_id`.",
            )
    }
}

fn mcp_service(toolbox: Arc<Toolbox>) -> StreamableHttpService<PipelineServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(PipelineServer::new(toolbox.clone())),
        Default::default(),
        // Meant to be reached on the VDS by IP (see lesson 17), so any Host.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

// ---------------------------------------------------------------------------
// The LLM: used by `summarize` and by the agent.
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
// Running the chain: over MCP, as a fixed pipeline or driven by the agent.
// ---------------------------------------------------------------------------

#[derive(Serialize, Debug)]
struct ToolInfo {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

impl From<&Tool> for ToolInfo {
    fn from(tool: &Tool) -> Self {
        ToolInfo {
            name: tool.name.to_string(),
            description: tool.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*tool.input_schema).clone()),
        }
    }
}

#[derive(Serialize, Debug)]
struct McpInfo {
    url: String,
    server: Option<String>,
    tools: Vec<ToolInfo>,
}

/// One `tools/call` of the chain.
#[derive(Serialize, Debug)]
struct ToolCallTrace {
    step: usize,
    name: String,
    arguments: Value,
    is_error: bool,
    result_text: String,
    structured: Option<Value>,
    ms: u128,
}

/// One verification of a handoff between two steps.
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
    /// "pipeline" (fixed order, code wires the steps) or "agent" (the LLM does).
    mode: String,
    mcp: Option<McpInfo>,
    calls: Vec<ToolCallTrace>,
    llm_rounds: usize,
    answer: Option<String>,
    checks: Vec<Check>,
    ok: bool,
    saved_file: Option<String>,
    error: Option<String>,
}

async fn connect(mcp_url: &str) -> Result<RunningService<RoleClient, ()>, String> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(mcp_url.to_string()),
    );
    ().serve(transport)
        .await
        .map_err(|e| format!("MCP handshake with {mcp_url} failed: {e}"))
}

async fn discover(
    client: &RunningService<RoleClient, ()>,
    mcp_url: &str,
) -> Result<(Vec<Tool>, McpInfo), String> {
    let tools = client
        .list_all_tools()
        .await
        .map_err(|e| format!("tools/list failed: {e}"))?;
    let info = McpInfo {
        url: mcp_url.to_string(),
        server: client.peer_info().and_then(|info| {
            let info = serde_json::to_value(&*info).ok()?;
            let server = &info["serverInfo"];
            Some(format!(
                "{} v{}",
                server["name"].as_str()?,
                server["version"].as_str()?
            ))
        }),
        tools: tools.iter().map(ToolInfo::from).collect(),
    };
    Ok((tools, info))
}

fn json_object(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    }
}

/// Drops `null` fields, so an unset option is simply not sent.
fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(mut map) => {
            map.retain(|_, v| !v.is_null());
            Value::Object(map)
        }
        other => other,
    }
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

async fn call_mcp_tool(
    client: &RunningService<RoleClient, ()>,
    step: usize,
    name: &str,
    arguments: Value,
) -> ToolCallTrace {
    let started = Instant::now();
    let outcome = match &arguments {
        Value::Object(map) => client
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(map.clone()))
            .await
            .map_err(|e| format!("tools/call failed: {e}")),
        _ => Err(format!("Arguments aren't a JSON object: {arguments}")),
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
        "step {step}: tools/call {name} {arguments} -> {} in {} ms",
        if is_error { "error" } else { "ok" },
        started.elapsed().as_millis()
    );
    ToolCallTrace {
        step,
        name: name.to_string(),
        arguments,
        is_error,
        result_text,
        structured,
        ms: started.elapsed().as_millis(),
    }
}

#[derive(Deserialize, Debug, Default)]
struct PipelineRequest {
    query: String,
    limit: Option<u32>,
    focus: Option<String>,
    language: Option<String>,
    filename: Option<String>,
}

/// One pipeline stage: the tool to call, and how its arguments are built
/// from the request and the previous stage's `structuredContent`.
struct Stage {
    tool: &'static str,
    arguments: fn(&PipelineRequest, &Value) -> Value,
}

const PIPELINE: [Stage; 3] = [
    Stage {
        tool: "search",
        arguments: search_arguments,
    },
    Stage {
        tool: "summarize",
        arguments: summarize_arguments,
    },
    Stage {
        tool: "save_to_file",
        arguments: save_arguments,
    },
];

fn search_arguments(req: &PipelineRequest, _previous: &Value) -> Value {
    json!({ "query": req.query, "limit": req.limit })
}

fn summarize_arguments(req: &PipelineRequest, previous: &Value) -> Value {
    let source_id = previous["artifact_id"].clone();
    json!({ "source_id": source_id, "focus": req.focus, "language": req.language })
}

fn save_arguments(req: &PipelineRequest, previous: &Value) -> Value {
    let source_id = previous["artifact_id"].clone();
    json!({ "source_id": source_id, "filename": req.filename })
}

/// The fixed chain: no LLM decides anything here. Each step's output is the
/// next step's input, and the first failure stops the chain.
async fn pipeline_steps(
    client: &RunningService<RoleClient, ()>,
    mcp_url: &str,
    req: &PipelineRequest,
    run: &mut Run,
) -> Result<Option<String>, String> {
    let (_, info) = discover(client, mcp_url).await?;
    run.mcp = Some(info);
    let mut previous = Value::Null;
    for (i, stage) in PIPELINE.iter().enumerate() {
        let arguments = without_nulls((stage.arguments)(req, &previous));
        let trace = call_mcp_tool(client, i + 1, stage.tool, arguments).await;
        let failure = trace.is_error.then(|| trace.result_text.clone());
        previous = trace.structured.clone().unwrap_or(Value::Null);
        run.calls.push(trace);
        if let Some(e) = failure {
            return Err(format!(
                "Step {} `{}` failed, so the chain stopped there: {e}",
                i + 1,
                stage.tool
            ));
        }
    }
    Ok(None)
}

#[derive(Debug)]
struct Agent {
    llm: Arc<Llm>,
    mcp_url: String,
}

fn system_prompt() -> String {
    "You are an agent that runs a three-step pipeline with MCP tools: `search` gets the data \
     (GitHub repositories), `summarize` processes it into a report, and `save_to_file` saves the \
     result. For a request to research, summarize or save something, call `search`, then \
     `summarize` with `source_id` set to the `artifact_id` that `search` returned, then \
     `save_to_file` with `source_id` set to the `artifact_id` that `summarize` returned. Call \
     them one at a time, in that order, and pass ids exactly as returned; never retype or \
     rewrite the data yourself. Pass `language` to `summarize` matching the user's language, and \
     a `filename` only if the user asked for one. If a tool fails, stop and explain the error \
     plainly. Finish with 2-4 sentences: what was found and where it was saved. Reply in the \
     same language the user writes in."
        .to_string()
}

/// An MCP tool, re-described as an OpenAI-style function (see lesson 17).
fn openai_tool(tool: &Tool) -> Value {
    let mut parameters = (*tool.input_schema).clone();
    parameters.remove("$schema");
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description.as_deref().unwrap_or_default(),
            "parameters": parameters,
        }
    })
}

impl Agent {
    /// The same chain, but the LLM decides each call and wires the ids.
    async fn converse(
        &self,
        client: &RunningService<RoleClient, ()>,
        request: &str,
        run: &mut Run,
    ) -> Result<Option<String>, String> {
        let (tools, info) = discover(client, &self.mcp_url).await?;
        run.mcp = Some(info);
        let llm_tools: Vec<Value> = tools.iter().map(openai_tool).collect();

        let mut messages = vec![
            json!({ "role": "system", "content": system_prompt() }),
            json!({ "role": "user", "content": request }),
        ];
        for round in 1..=MAX_TOOL_ROUNDS {
            run.llm_rounds = round;
            let message = self.llm.chat(&messages, &llm_tools).await?;
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
                let arguments: Value = serde_json::from_str(raw_arguments)
                    .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
                let trace = call_mcp_tool(client, run.calls.len() + 1, name, arguments).await;
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": trace.result_text,
                }));
                run.calls.push(trace);
            }
        }
        Err(format!(
            "Gave up after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer."
        ))
    }
}

// ---------------------------------------------------------------------------
// Verifying the handoffs: did each step get exactly what the previous one made?
// ---------------------------------------------------------------------------

/// `owner/repo` (lowercased) for every github.com repository link in `text`.
fn github_repo_links(text: &str) -> BTreeSet<String> {
    const PREFIX: &str = "https://github.com/";
    let mut links = BTreeSet::new();
    for (start, _) in text.match_indices(PREFIX) {
        let path: String = text[start + PREFIX.len()..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
            .collect();
        let mut parts = path.split('/').filter(|p| !p.is_empty());
        if let (Some(owner), Some(repo)) = (parts.next(), parts.next()) {
            links.insert(format!("{owner}/{}", repo.trim_end_matches('.')).to_lowercase());
        }
    }
    links
}

/// Links in the summary to repositories the search never returned, and
/// search results the summary dropped.
fn grounding(summary: &str, items: &[Repo]) -> (Vec<String>, Vec<String>) {
    let known: BTreeSet<String> = items.iter().map(|r| r.full_name.to_lowercase()).collect();
    let linked = github_repo_links(summary);
    let ungrounded = linked.iter().filter(|l| !known.contains(*l)).cloned().collect();
    let missing = items
        .iter()
        .filter(|r| !linked.contains(&r.full_name.to_lowercase()))
        .map(|r| r.full_name.clone())
        .collect();
    (ungrounded, missing)
}

/// The latest successful `tool` call that produced `artifact_id`.
fn produced<'a>(calls: &'a [ToolCallTrace], tool: &str, artifact_id: &str) -> Option<&'a Value> {
    calls
        .iter()
        .rev()
        .filter(|c| c.name == tool && !c.is_error)
        .filter_map(|c| c.structured.as_ref())
        .find(|s| s["artifact_id"].as_str() == Some(artifact_id))
}

/// Walks the chain back from the last saved file (file -> summary -> search)
/// and checks every handoff, independently of what the tools claimed.
fn chain_checks(calls: &[ToolCallTrace], output_dir: &Path) -> Vec<Check> {
    let Some(saved) = calls
        .iter()
        .rev()
        .filter(|c| c.name == "save_to_file" && !c.is_error)
        .find_map(|c| c.structured.as_ref())
    else {
        return vec![Check::new(
            "Chain completed",
            false,
            "No successful `save_to_file` call, so there is no result to verify.".to_string(),
        )];
    };
    let summary_id = saved["source"]["artifact_id"].as_str().unwrap_or_default();
    let Some(summary) = produced(calls, "summarize", summary_id) else {
        return vec![Check::new(
            "summarize → save_to_file",
            false,
            format!("`save_to_file` saved {summary_id:?}, which no `summarize` call in this run produced."),
        )];
    };
    let search_id = summary["source"]["artifact_id"].as_str().unwrap_or_default();
    let Some(search) = produced(calls, "search", search_id) else {
        return vec![Check::new(
            "search → summarize",
            false,
            format!("`summarize` read {search_id:?}, which no `search` call in this run produced."),
        )];
    };
    let items: Vec<Repo> = serde_json::from_value(search["items"].clone()).unwrap_or_default();
    let summary_text = summary["content"].as_str().unwrap_or_default();
    let summary_checksum = summary["checksum"].as_str().unwrap_or_default();
    let mut checks = Vec::new();

    let read = &summary["source"];
    let same_input = read["checksum"] == search["checksum"]
        && read["items"].as_u64() == Some(items.len() as u64);
    checks.push(Check::new(
        "search → summarize",
        same_input,
        if same_input {
            format!(
                "`summarize` read {search_id} ({}, {} repositories): exactly what `search` returned.",
                search["checksum"].as_str().unwrap_or_default(),
                items.len()
            )
        } else {
            format!(
                "`summarize` read {} with {} items, but `search` returned {} with {} items.",
                read["checksum"],
                read["items"],
                search["checksum"],
                items.len()
            )
        },
    ));

    let (ungrounded, missing) = grounding(summary_text, &items);
    let grounded = ungrounded.is_empty() && missing.is_empty();
    let mut detail = Vec::new();
    if !missing.is_empty() {
        detail.push(format!("dropped from the summary: {}", missing.join(", ")));
    }
    if !ungrounded.is_empty() {
        detail.push(format!("linked but never returned by search: {}", ungrounded.join(", ")));
    }
    checks.push(Check::new(
        "Summary is grounded in the search results",
        grounded,
        if grounded {
            format!(
                "All {} repositories from {search_id} reached {summary_id}, and it links to no \
                 repository the search didn't return.",
                items.len()
            )
        } else {
            format!("{}.", detail.join("; "))
        },
    ));

    let same_output =
        saved["source"]["checksum"] == summary["checksum"] && saved["bytes"] == summary["bytes"];
    checks.push(Check::new(
        "summarize → save_to_file",
        same_output,
        if same_output {
            format!(
                "`save_to_file` wrote {summary_id} ({summary_checksum}, {} bytes): exactly what \
                 `summarize` produced.",
                summary["bytes"]
            )
        } else {
            format!(
                "`save_to_file` wrote {} ({} bytes), but `summarize` produced {} ({} bytes).",
                saved["source"]["checksum"], saved["bytes"], summary["checksum"], summary["bytes"]
            )
        },
    ));

    let filename = saved["filename"].as_str().unwrap_or_default();
    let on_disk = validate_filename(filename)
        .and_then(|_| std::fs::read_to_string(output_dir.join(filename)).map_err(|e| e.to_string()));
    checks.push(match on_disk {
        Ok(text) => {
            let found = checksum(&text);
            let same = found == summary_checksum;
            Check::new(
                "File on disk",
                same,
                if same {
                    format!(
                        "Re-read {filename} independently: {} bytes, {found}, identical to {summary_id}.",
                        text.len()
                    )
                } else {
                    format!("{filename} holds {found}, not {summary_checksum}.")
                },
            )
        }
        Err(e) => Check::new("File on disk", false, format!("Could not re-read {filename:?}: {e}")),
    });
    checks
}

fn finish(
    run: &mut Run,
    outcome: Result<Result<Option<String>, String>, tokio::time::error::Elapsed>,
    output_dir: &Path,
) {
    match outcome {
        Ok(Ok(answer)) => run.answer = answer,
        Ok(Err(e)) => run.error = Some(e),
        Err(_) => run.error = Some(format!("Timed out after {}s", RUN_TIMEOUT.as_secs())),
    }
    run.checks = chain_checks(&run.calls, output_dir);
    run.saved_file = run
        .calls
        .iter()
        .rev()
        .filter(|c| c.name == "save_to_file" && !c.is_error)
        .find_map(|c| c.structured.as_ref()?["filename"].as_str().map(str::to_string));
    run.ok = run.error.is_none() && run.checks.iter().all(|c| c.ok);
    if let Some(e) = &run.error {
        eprintln!("{} run failed: {e}", run.mode);
    }
}

async fn run_pipeline(app: &AppState, req: PipelineRequest) -> Run {
    let mut run = Run {
        mode: "pipeline".to_string(),
        ..Default::default()
    };
    let mcp_url = &app.agent.mcp_url;
    let outcome = tokio::time::timeout(RUN_TIMEOUT, async {
        let client = connect(mcp_url).await?;
        let result = pipeline_steps(&client, mcp_url, &req, &mut run).await;
        let _ = client.cancel().await;
        result
    })
    .await;
    finish(&mut run, outcome, &app.toolbox.output_dir);
    run
}

async fn run_agent(app: &AppState, request: &str) -> Run {
    let mut run = Run {
        mode: "agent".to_string(),
        ..Default::default()
    };
    let agent = &app.agent;
    let outcome = tokio::time::timeout(RUN_TIMEOUT, async {
        let client = connect(&agent.mcp_url).await?;
        let result = agent.converse(&client, request, &mut run).await;
        let _ = client.cancel().await;
        result
    })
    .await;
    finish(&mut run, outcome, &app.toolbox.output_dir);
    run
}

// ---------------------------------------------------------------------------
// HTTP API and page.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Settings {
    model: String,
    mcp_url: String,
    output_dir: String,
}

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    toolbox: Arc<Toolbox>,
    settings: Arc<Settings>,
}

async fn pipeline(State(app): State<AppState>, Json(req): Json<PipelineRequest>) -> Json<Run> {
    if req.query.trim().is_empty() {
        return Json(Run {
            mode: "pipeline".to_string(),
            error: Some("Enter a search query first.".to_string()),
            ..Default::default()
        });
    }
    Json(run_pipeline(&app, req).await)
}

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

async fn ask(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Run> {
    let question = req.question.trim();
    if question.is_empty() {
        return Json(Run {
            mode: "agent".to_string(),
            error: Some("Ask something first.".to_string()),
            ..Default::default()
        });
    }
    Json(run_agent(&app, question).await)
}

#[derive(Serialize)]
struct SavedFile {
    name: String,
    bytes: u64,
    modified: u64,
}

async fn list_files(State(app): State<AppState>) -> Json<Vec<SavedFile>> {
    let mut files: Vec<SavedFile> = std::fs::read_dir(&app.toolbox.output_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let name = entry.file_name().into_string().ok()?;
                    validate_filename(&name).ok()?;
                    let meta = entry.metadata().ok()?;
                    let modified = meta
                        .modified()
                        .ok()?
                        .duration_since(UNIX_EPOCH)
                        .ok()?
                        .as_secs();
                    Some(SavedFile {
                        name,
                        bytes: meta.len(),
                        modified,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.name.cmp(&b.name)));
    Json(files)
}

async fn read_saved_file(
    State(app): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<Value> {
    let content = validate_filename(&name).and_then(|_| {
        std::fs::read_to_string(app.toolbox.output_dir.join(&name)).map_err(|e| e.to_string())
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
    let mcp = mcp_service(state.toolbox.clone());
    Router::new()
        .route("/", get(index))
        .route("/api/config", get(config))
        .route("/api/pipeline", post(pipeline))
        .route("/api/ask", post(ask))
        .route("/api/files", get(list_files))
        .route("/api/files/{name}", get(read_saved_file))
        .with_state(state)
        .nest_service("/mcp", mcp)
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
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
    let mcp_url =
        env_non_empty("MCP_SERVER_URL").unwrap_or_else(|| format!("http://127.0.0.1:{port}/mcp"));
    // Relative to the working directory: the lesson folder under `cargo run`,
    // `~/apps/lesson-19` on the VDS (the systemd unit's WorkingDirectory).
    let output_dir = env_non_empty("OUTPUT_DIR").unwrap_or_else(|| "output".to_string());

    let llm = Arc::new(Llm {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .expect("failed to build HTTP client"),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model: model.clone(),
    });
    let github = GitHub {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client"),
        api_base: GITHUB_API.to_string(),
        token: env_non_empty("GITHUB_TOKEN"),
    };
    let toolbox = Arc::new(Toolbox::new(github, llm.clone(), PathBuf::from(&output_dir)));
    let state = AppState {
        agent: Arc::new(Agent {
            llm,
            mcp_url: mcp_url.clone(),
        }),
        toolbox,
        settings: Arc::new(Settings {
            model,
            mcp_url,
            output_dir,
        }),
    };

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port} (MCP at /mcp)");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson19-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn github_at(api_base: &str) -> GitHub {
        GitHub {
            client: reqwest::Client::new(),
            api_base: api_base.to_string(),
            token: None,
        }
    }

    fn llm_at(base: &str) -> Arc<Llm> {
        Arc::new(Llm {
            http: reqwest::Client::new(),
            endpoint: format!("{base}/chat/completions"),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
        })
    }

    fn search_fixture() -> Value {
        json!({
            "total_count": 42,
            "incomplete_results": false,
            "items": [
                {
                    "full_name": "octo/alpha",
                    "html_url": "https://github.com/octo/alpha",
                    "description": "The fastest MCP server framework",
                    "stargazers_count": 900,
                    "language": "Rust",
                    "topics": ["mcp", "rust"],
                    "updated_at": "2026-09-20T10:00:00Z"
                },
                {
                    "full_name": "octo/beta",
                    "html_url": "https://github.com/octo/beta",
                    "description": null,
                    "stargazers_count": 120,
                    "language": null,
                    "updated_at": "2026-09-01T10:00:00Z"
                }
            ]
        })
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Stands in for api.github.com's repository search.
    fn fake_github(body: Value) -> Router {
        Router::new().route(
            "/search/repositories",
            get(move || {
                let body = body.clone();
                async move { Json(body) }
            }),
        )
    }

    const OVERVIEW: &str = "OVERVIEW: octo/alpha leads with 900 stars.";

    /// Answers every chat request with the overview and records the requests.
    fn fake_summary_llm(requests: Arc<Mutex<Vec<Value>>>) -> Router {
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body);
                    Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": OVERVIEW } }]
                    }))
                }
            }),
        )
    }

    fn broken_llm() -> Router {
        Router::new().route(
            "/chat/completions",
            post(|| async {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": "model overloaded" } })),
                )
            }),
        )
    }

    /// The real app, `/mcp` included, on a random port.
    async fn spawn_app(github_base: &str, llm_base: &str, output_dir: PathBuf) -> AppState {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mcp_url = format!("http://{addr}/mcp");
        let llm = llm_at(llm_base);
        let state = AppState {
            agent: Arc::new(Agent {
                llm: llm.clone(),
                mcp_url: mcp_url.clone(),
            }),
            toolbox: Arc::new(Toolbox::new(github_at(github_base), llm, output_dir.clone())),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                mcp_url,
                output_dir: output_dir.display().to_string(),
            }),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        state
    }

    async fn duplex_client(server: PipelineServer) -> RunningService<RoleClient, ()> {
        let (server_io, client_io) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            let running = server.serve(server_io).await.expect("server handshake");
            let _ = running.waiting().await;
        });
        ().serve(client_io).await.expect("client handshake")
    }

    async fn call(
        client: &RunningService<RoleClient, ()>,
        name: &'static str,
        arguments: Value,
    ) -> CallToolResult {
        client
            .call_tool(CallToolRequestParams::new(name).with_arguments(json_object(arguments)))
            .await
            .unwrap()
    }

    fn failed_checks(run: &Run) -> Vec<String> {
        run.checks
            .iter()
            .filter(|c| !c.ok)
            .map(|c| format!("{}: {}", c.name, c.detail))
            .collect()
    }

    #[tokio::test]
    async fn tools_are_registered_with_described_parameters() {
        let toolbox = Arc::new(Toolbox::new(
            github_at("http://127.0.0.1:9"),
            llm_at("http://127.0.0.1:9"),
            temp_dir("tools"),
        ));
        let client = duplex_client(PipelineServer::new(toolbox)).await;
        let tools = client.list_all_tools().await.unwrap();
        client.cancel().await.unwrap();

        let mut names: Vec<&str> = tools.iter().map(|t| &*t.name).collect();
        names.sort_unstable();
        assert_eq!(names, ["save_to_file", "search", "summarize"]);
        for tool in &tools {
            assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
            let schema = Value::Object((*tool.input_schema).clone());
            for (property, spec) in schema["properties"].as_object().unwrap() {
                assert!(
                    spec["description"].as_str().is_some_and(|d| !d.is_empty()),
                    "{}.{property} has no description",
                    tool.name
                );
            }
            let required: Vec<&str> = schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let expected = if tool.name == "search" { "query" } else { "source_id" };
            assert_eq!(required, [expected], "{}", tool.name);
        }
    }

    #[tokio::test]
    async fn pipeline_chains_the_three_tools_and_the_data_arrives_intact() {
        let github = spawn(fake_github(search_fixture())).await;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm = spawn(fake_summary_llm(requests.clone())).await;
        let output_dir = temp_dir("pipeline");
        let state = spawn_app(&github, &llm, output_dir.clone()).await;

        let run = run_pipeline(
            &state,
            PipelineRequest {
                query: "mcp server language:rust".to_string(),
                limit: Some(2),
                filename: Some("report.md".to_string()),
                ..Default::default()
            },
        )
        .await;

        assert_eq!(run.error, None);
        let names: Vec<&str> = run.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["search", "summarize", "save_to_file"]);
        assert!(run.calls.iter().all(|c| !c.is_error));

        // Each step's input is the previous step's output.
        let search = run.calls[0].structured.as_ref().unwrap();
        let summary = run.calls[1].structured.as_ref().unwrap();
        let saved = run.calls[2].structured.as_ref().unwrap();
        assert_eq!(run.calls[1].arguments["source_id"], search["artifact_id"]);
        assert_eq!(run.calls[2].arguments["source_id"], summary["artifact_id"]);
        assert_eq!(saved["checksum"], summary["checksum"]);

        // The processing step's LLM got the search results byte for byte.
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let sent: SearchResults =
            serde_json::from_str(requests[0]["messages"][1]["content"].as_str().unwrap()).unwrap();
        let returned: Vec<Repo> = serde_json::from_value(search["items"].clone()).unwrap();
        assert_eq!(sent.items, returned);
        assert_eq!(sent.items[1].description, "");

        // The file holds the processed result, not the raw data.
        let file = std::fs::read_to_string(output_dir.join("report.md")).unwrap();
        assert_eq!(file, summary["content"].as_str().unwrap());
        assert!(file.contains(OVERVIEW));
        assert!(file.contains("[octo/alpha](https://github.com/octo/alpha) ★ 900 · Rust"));
        assert!(file.contains("- Languages: Rust 1, not specified 1"));

        assert_eq!(run.checks.len(), 4);
        assert!(run.ok, "failed checks: {:?}", failed_checks(&run));
        assert_eq!(run.saved_file.as_deref(), Some("report.md"));
    }

    #[tokio::test]
    async fn pipeline_stops_at_the_first_failed_step() {
        let github = spawn(fake_github(search_fixture())).await;
        let llm = spawn(broken_llm()).await;
        let output_dir = temp_dir("broken");
        let state = spawn_app(&github, &llm, output_dir.clone()).await;

        let run = run_pipeline(
            &state,
            PipelineRequest {
                query: "anything".to_string(),
                ..Default::default()
            },
        )
        .await;

        assert_eq!(run.calls.len(), 2, "save_to_file must not run");
        assert!(!run.calls[0].is_error);
        assert!(run.calls[1].is_error);
        assert!(run.calls[1].result_text.contains("model overloaded"));
        assert!(run.error.as_deref().unwrap().contains("Step 2 `summarize` failed"));
        assert!(!run.ok);
        assert!(!run.checks[0].ok);
        assert_eq!(count_files(&output_dir), 0);
    }

    #[tokio::test]
    async fn tool_arguments_are_validated() {
        let output_dir = temp_dir("validate");
        let toolbox = Arc::new(Toolbox::new(
            github_at("http://127.0.0.1:9"),
            llm_at("http://127.0.0.1:9"),
            output_dir.clone(),
        ));
        let summary = toolbox.store(ArtifactKind::Summary, None, "# Report\n".to_string());
        let client = duplex_client(PipelineServer::new(toolbox)).await;

        let empty = call(&client, "search", json!({ "query": "  " })).await;
        assert_eq!(empty.is_error, Some(true));
        let too_many = call(&client, "search", json!({ "query": "x", "limit": 50 })).await;
        assert_eq!(too_many.is_error, Some(true));

        let unknown = call(&client, "summarize", json!({ "source_id": "search-99" })).await;
        assert_eq!(unknown.is_error, Some(true));
        assert!(tool_result_text(&unknown).contains("Unknown artifact"));
        let wrong_kind = call(&client, "summarize", json!({ "source_id": summary.id })).await;
        assert_eq!(wrong_kind.is_error, Some(true));
        assert!(tool_result_text(&wrong_kind).contains("processes search results"));

        for bad in ["../escape.md", ".hidden.md", "report.exe", "a/b.md"] {
            let result = call(
                &client,
                "save_to_file",
                json!({ "source_id": summary.id, "filename": bad }),
            )
            .await;
            assert_eq!(result.is_error, Some(true), "{bad} was accepted");
        }
        assert_eq!(count_files(&output_dir), 0);

        let saved = call(&client, "save_to_file", json!({ "source_id": summary.id })).await;
        client.cancel().await.unwrap();
        assert_ne!(saved.is_error, Some(true), "{}", tool_result_text(&saved));
        let structured = saved.structured_content.unwrap();
        assert_eq!(structured["filename"], format!("{}.md", summary.id));
        assert_eq!(structured["verified_on_disk"], true);
        assert_eq!(
            std::fs::read_to_string(output_dir.join(format!("{}.md", summary.id))).unwrap(),
            "# Report\n"
        );
    }

    /// Plays the model in agent mode: search -> summarize -> save_to_file ->
    /// answer, reading each id from the previous tool result it was sent.
    /// Requests without tools are `summarize`'s own LLM call.
    fn fake_agent_llm() -> Router {
        Router::new().route(
            "/chat/completions",
            post(|Json(body): Json<Value>| async move {
                if body.get("tools").is_none() {
                    return Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": OVERVIEW } }]
                    }));
                }
                let messages = body["messages"].as_array().unwrap();
                let last_tool = messages
                    .iter()
                    .rev()
                    .find(|m| m["role"] == "tool")
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or_default();
                // The id the previous tool told us to pass on.
                let next_id = last_tool
                    .split("source_id \"")
                    .nth(1)
                    .and_then(|rest| rest.split('"').next())
                    .unwrap_or_default();
                let tool_results = messages.iter().filter(|m| m["role"] == "tool").count();
                let (name, arguments) = match tool_results {
                    0 => ("search", json!({ "query": "mcp server", "limit": 2 })),
                    1 => ("summarize", json!({ "source_id": next_id, "language": "English" })),
                    2 => ("save_to_file", json!({ "source_id": next_id, "filename": "agent.md" })),
                    _ => {
                        return Json(json!({ "choices": [{ "message": {
                            "role": "assistant",
                            "content": "Found octo/alpha and octo/beta; saved to agent.md."
                        } }] }));
                    }
                };
                Json(json!({ "choices": [{ "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": format!("call_{tool_results}"),
                        "type": "function",
                        "function": { "name": name, "arguments": arguments.to_string() }
                    }]
                } }] }))
            }),
        )
    }

    #[tokio::test]
    async fn agent_chains_the_tools_by_itself() {
        let github = spawn(fake_github(search_fixture())).await;
        let llm = spawn(fake_agent_llm()).await;
        let output_dir = temp_dir("agent");
        let state = spawn_app(&github, &llm, output_dir.clone()).await;

        let run = run_agent(&state, "Find Rust MCP servers, summarize them, save to agent.md").await;

        assert_eq!(run.error, None);
        let names: Vec<&str> = run.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["search", "summarize", "save_to_file"]);
        assert_eq!(run.llm_rounds, 4);
        assert!(run.answer.as_deref().unwrap().contains("agent.md"));
        assert!(run.ok, "failed checks: {:?}", failed_checks(&run));
        assert!(
            std::fs::read_to_string(output_dir.join("agent.md"))
                .unwrap()
                .contains(OVERVIEW)
        );
    }

    #[test]
    fn grounding_flags_invented_links_and_dropped_results() {
        let items: Vec<Repo> = serde_json::from_value::<GhSearch>(search_fixture())
            .unwrap()
            .items
            .into_iter()
            .map(Repo::from)
            .collect();
        let good = "See [octo/alpha](https://github.com/octo/alpha) and \
                    https://github.com/Octo/Beta/issues.";
        assert_eq!(grounding(good, &items), (Vec::<String>::new(), Vec::<String>::new()));

        let bad = "Only [octo/alpha](https://github.com/octo/alpha), plus \
                   https://github.com/someone/invented.";
        assert_eq!(
            grounding(bad, &items),
            (vec!["someone/invented".to_string()], vec!["octo/beta".to_string()])
        );
    }

    #[test]
    fn checks_catch_a_file_that_differs_from_the_summary() {
        let output_dir = temp_dir("tamper");
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(output_dir.join("r.md"), "tampered").unwrap();
        let content = "[octo/alpha](https://github.com/octo/alpha)";
        let trace = |step, name: &str, structured: Value| ToolCallTrace {
            step,
            name: name.to_string(),
            arguments: json!({}),
            is_error: false,
            result_text: String::new(),
            structured: Some(structured),
            ms: 0,
        };
        let calls = [
            trace(1, "search", json!({
                "artifact_id": "search-1",
                "checksum": "fnv1a64:1",
                "items": [{
                    "full_name": "octo/alpha", "url": "https://github.com/octo/alpha",
                    "description": "", "stars": 1, "language": null, "topics": [], "updated_at": ""
                }],
            })),
            trace(2, "summarize", json!({
                "artifact_id": "summary-2",
                "checksum": checksum(content),
                "bytes": content.len(),
                "content": content,
                "source": { "artifact_id": "search-1", "checksum": "fnv1a64:1", "items": 1 },
            })),
            trace(3, "save_to_file", json!({
                "filename": "r.md",
                "bytes": content.len(),
                "source": { "artifact_id": "summary-2", "checksum": checksum(content) },
            })),
        ];
        let checks = chain_checks(&calls, &output_dir);
        let passed: Vec<bool> = checks.iter().map(|c| c.ok).collect();
        assert_eq!(passed, [true, true, true, false], "{checks:?}");
    }

    #[test]
    fn filenames_stay_inside_the_output_directory() {
        for good in ["report.md", "rust-mcp_2026.json", "notes.txt"] {
            assert!(validate_filename(good).is_ok(), "{good}");
        }
        for bad in ["", "..", "../x.md", "/etc/passwd", "a/b.md", ".env", "x.sh", "no-extension"] {
            assert!(validate_filename(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn checksum_is_stable_and_sensitive() {
        assert_eq!(checksum(""), "fnv1a64:cbf29ce484222325");
        assert_ne!(checksum("report"), checksum("report "));
    }

    #[test]
    fn lineage_follows_derived_from() {
        let mut artifacts = Artifacts::default();
        let search = artifacts.add(ArtifactKind::SearchResults, None, "[]".to_string());
        let summary = artifacts.add(ArtifactKind::Summary, Some(search.id.as_str()), "# r".to_string());
        let ids: Vec<Value> = artifacts
            .lineage(&summary.id)
            .into_iter()
            .map(|a| a["artifact_id"].clone())
            .collect();
        assert_eq!(ids, [json!("search-1"), json!("summary-2")]);
    }
}

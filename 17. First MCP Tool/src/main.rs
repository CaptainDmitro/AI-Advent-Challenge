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
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const GITHUB_API: &str = "https://api.github.com";
const TOOL_NAME: &str = "list_recent_commits";
const DEFAULT_LIMIT: u32 = 10;
const MAX_LIMIT: u32 = 30;
const MAX_MESSAGE_CHARS: usize = 200;
/// Upper bound on LLM <-> tool round trips for one question, so a model that
/// keeps asking for tools can't loop forever.
const MAX_TOOL_ROUNDS: usize = 5;
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// The wrapped API: GitHub's public REST endpoint for a repository's commits.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GitHub {
    client: reqwest::Client,
    api_base: String,
    token: Option<String>,
}

/// One commit, trimmed down to what's worth handing to an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CommitSummary {
    sha: String,
    date: String,
    author: String,
    message: String,
    url: String,
}

#[derive(Deserialize)]
struct GhCommit {
    sha: String,
    html_url: String,
    commit: GhCommitDetail,
    /// The linked GitHub account; `null` when the commit email isn't tied to one.
    author: Option<GhUser>,
}

#[derive(Deserialize)]
struct GhCommitDetail {
    message: String,
    author: Option<GhGitAuthor>,
}

#[derive(Deserialize)]
struct GhGitAuthor {
    name: Option<String>,
    date: Option<String>,
}

#[derive(Deserialize)]
struct GhUser {
    login: String,
}

#[derive(Deserialize)]
struct GhError {
    message: String,
}

impl From<GhCommit> for CommitSummary {
    fn from(c: GhCommit) -> Self {
        let git_author = c.commit.author.as_ref();
        let first_line = c.commit.message.lines().next().unwrap_or_default();
        let mut message: String = first_line.chars().take(MAX_MESSAGE_CHARS).collect();
        if first_line.chars().count() > MAX_MESSAGE_CHARS {
            message.push('…');
        }
        CommitSummary {
            sha: c.sha.chars().take(7).collect(),
            date: git_author.and_then(|a| a.date.clone()).unwrap_or_default(),
            author: c
                .author
                .map(|u| u.login)
                .or_else(|| git_author.and_then(|a| a.name.clone()))
                .unwrap_or_else(|| "unknown".to_string()),
            message,
            url: c.html_url,
        }
    }
}

/// Validated, normalized input for one `GET /repos/{owner}/{repo}/commits`.
#[derive(Debug, PartialEq)]
struct CommitQuery {
    owner: String,
    repo: String,
    limit: u32,
    path: Option<String>,
    branch: Option<String>,
    since: Option<String>,
}

impl CommitQuery {
    fn repository(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }
}

impl GitHub {
    async fn list_commits(&self, q: &CommitQuery) -> Result<Vec<CommitSummary>, String> {
        let url = format!(
            "{}/repos/{}/{}/commits",
            self.api_base.trim_end_matches('/'),
            q.owner,
            q.repo
        );
        let mut params = vec![("per_page", q.limit.to_string())];
        if let Some(path) = &q.path {
            params.push(("path", path.clone()));
        }
        if let Some(branch) = &q.branch {
            params.push(("sha", branch.clone()));
        }
        if let Some(since) = &q.since {
            params.push(("since", since.clone()));
        }

        let mut request = self
            .client
            .get(&url)
            .query(&params)
            .header(ACCEPT, "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            // GitHub rejects requests without a User-Agent.
            .header(USER_AGENT, "ai-advent-first-mcp-tool");
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
                404 => format!(
                    "Repository {} was not found on GitHub (it may be private or misspelled).",
                    q.repository()
                ),
                409 => format!("Repository {} is empty: it has no commits yet.", q.repository()),
                403 | 429 if rate_limited => "GitHub API rate limit exceeded for this server. \
                     Try again later, or set GITHUB_TOKEN to raise the limit."
                    .to_string(),
                _ => format!("GitHub API error ({status}): {detail}"),
            });
        }

        let commits: Vec<GhCommit> =
            serde_json::from_str(&body).map_err(|e| format!("Unexpected GitHub response: {e}"))?;
        Ok(commits.into_iter().map(CommitSummary::from).collect())
    }
}

// ---------------------------------------------------------------------------
// The MCP server: one tool, registered with rmcp's macros.
// ---------------------------------------------------------------------------

/// Input of the `list_recent_commits` tool. rmcp derives the tool's JSON
/// Schema from this struct: each doc comment becomes that property's
/// `description`, and non-`Option` fields become `required`. That schema is
/// exactly what the LLM reads when it decides how to fill in the arguments.
#[derive(Debug, Deserialize, JsonSchema)]
struct ListCommitsArgs {
    /// Repository owner: a GitHub user or organization, e.g. "rust-lang".
    owner: String,
    /// Repository name, e.g. "rust".
    repo: String,
    /// How many commits to return, newest first. 1-30, default 10.
    limit: Option<u32>,
    /// Only return commits that touch this file or directory, e.g. "src/main.rs".
    path: Option<String>,
    /// Branch, tag or commit SHA to list from. Defaults to the repository's default branch.
    branch: Option<String>,
    /// Only return commits made on or after this date: "YYYY-MM-DD" or a full ISO 8601 timestamp.
    since: Option<String>,
}

fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 100
        && name != "."
        && name != ".."
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Models sometimes send `""` for an optional argument they meant to leave out.
fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn normalize_since(since: &str) -> Result<String, String> {
    let bytes = since.as_bytes();
    let is_date = bytes.len() >= 10
        && bytes[..10]
            .iter()
            .enumerate()
            .all(|(i, b)| if i == 4 || i == 7 { *b == b'-' } else { b.is_ascii_digit() });
    if !is_date {
        return Err(format!(
            "`since` must be a date like 2026-09-01 or an ISO 8601 timestamp, got {since:?}."
        ));
    }
    Ok(if since.len() == 10 {
        format!("{since}T00:00:00Z")
    } else {
        since.to_string()
    })
}

impl ListCommitsArgs {
    fn validate(self) -> Result<CommitQuery, String> {
        let owner = self.owner.trim().to_string();
        let repo = self.repo.trim().to_string();
        if !is_valid_name(&owner) || !is_valid_name(&repo) {
            return Err(format!(
                "`owner` and `repo` must be plain GitHub names (letters, digits, '-', '_', '.'), \
                 got {owner:?} / {repo:?}."
            ));
        }
        let since = non_empty(self.since)
            .map(|s| normalize_since(&s))
            .transpose()?;
        Ok(CommitQuery {
            owner,
            repo,
            limit: self.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT),
            path: non_empty(self.path),
            branch: non_empty(self.branch),
            since,
        })
    }
}

/// The text the LLM actually reads: one compact line per commit.
fn format_commits(q: &CommitQuery, commits: &[CommitSummary]) -> String {
    let mut scope = q.repository();
    if let Some(branch) = &q.branch {
        scope.push_str(&format!(" @ {branch}"));
    }
    if let Some(path) = &q.path {
        scope.push_str(&format!(", path {path}"));
    }
    if let Some(since) = &q.since {
        scope.push_str(&format!(", since {since}"));
    }
    if commits.is_empty() {
        return format!("No commits found in {scope}.");
    }
    let mut out = format!(
        "{} most recent commit(s) in {scope}, newest first (sha | date | author | message):\n",
        commits.len()
    );
    for c in commits {
        out.push_str(&format!(
            "- {} | {} | {} | {}\n",
            c.sha, c.date, c.author, c.message
        ));
    }
    out
}

#[derive(Debug, Clone)]
struct CommitsServer {
    github: GitHub,
    tool_router: ToolRouter<Self>,
}

/// `#[tool_router]` collects every `#[tool]` method below into
/// `Self::tool_router()`: registration is just annotating a method.
#[tool_router(router = tool_router)]
impl CommitsServer {
    fn new(github: GitHub) -> Self {
        Self {
            github,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "list_recent_commits",
        description = "List the most recent commits of a public GitHub repository, newest first. \
                       Returns each commit's short SHA, date, author and the first line of its \
                       message. Can be narrowed to a branch, a file or directory path, and a \
                       start date."
    )]
    async fn list_recent_commits(
        &self,
        Parameters(args): Parameters<ListCommitsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Anything that goes wrong *inside* the tool (bad argument, unknown
        // repo, rate limit) is a tool-level error: `is_error: true` with a
        // readable message the model can relay, not a JSON-RPC protocol error.
        let query = match args.validate() {
            Ok(query) => query,
            Err(e) => return Ok(CallToolResult::error(vec![ContentBlock::text(e)])),
        };
        match self.github.list_commits(&query).await {
            Ok(commits) => {
                let mut result = CallToolResult::success(vec![ContentBlock::text(
                    format_commits(&query, &commits),
                )]);
                result.structured_content = Some(json!({
                    "repository": query.repository(),
                    "count": commits.len(),
                    "commits": commits,
                }));
                Ok(result)
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(e)])),
        }
    }
}

/// `#[tool_handler]` wires `tools/list` and `tools/call` to the router above.
#[tool_handler(router = self.tool_router)]
impl ServerHandler for CommitsServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("github-commits", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only access to the public GitHub REST API: \
                 list the recent commits of any public repository.",
            )
    }
}

/// The MCP endpoint, as a tower service that axum mounts at `/mcp`.
fn mcp_service(github: GitHub) -> StreamableHttpService<CommitsServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(CommitsServer::new(github.clone())),
        Default::default(),
        // rmcp only accepts `Host: localhost` by default (DNS-rebinding
        // protection for servers on a developer's machine). This one is meant
        // to be reached on the VDS by IP, and it only exposes public,
        // read-only data, so any Host is fine.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

// ---------------------------------------------------------------------------
// The agent: discovers the MCP tools, lets the LLM call them, uses the result.
// ---------------------------------------------------------------------------

struct Agent {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    mcp_url: String,
}

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

/// One `tools/call` the agent made on the model's behalf.
#[derive(Serialize, Debug)]
struct ToolCallTrace {
    round: usize,
    name: String,
    arguments: Value,
    is_error: bool,
    result_text: String,
    structured: Option<Value>,
    ms: u128,
}

/// Everything one question produced, so the UI can show the whole chain.
#[derive(Serialize, Debug, Default)]
struct AgentRun {
    mcp: Option<McpInfo>,
    tool_calls: Vec<ToolCallTrace>,
    llm_rounds: usize,
    answer: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    /// Kept as raw JSON: it's appended to the conversation verbatim, so
    /// `tool_calls` (and any provider extras such as DeepSeek's
    /// `reasoning_content`) go back to the model exactly as they came.
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

fn system_prompt(today: &str) -> String {
    format!(
        "You are an assistant that answers questions about GitHub repositories. \
         You have tools from an MCP server: use them to fetch real data instead of guessing, \
         and never invent commits. Today's date is {today} (UTC); use it to turn relative \
         ranges like \"this week\" into a `since` date. If a tool returns an error, explain \
         it to the user plainly. Refer to commits by their short SHA. \
         Reply in the same language the user writes in."
    )
}

/// An MCP tool, re-described as an OpenAI-style function the LLM can call.
/// The MCP `inputSchema` already is a JSON Schema object, so it becomes
/// `parameters` as-is, minus the `$schema` marker some providers reject.
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
    round: usize,
    name: &str,
    raw_arguments: &str,
) -> ToolCallTrace {
    let started = Instant::now();
    let arguments: Value = serde_json::from_str(raw_arguments)
        .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
    let outcome = match &arguments {
        Value::Object(map) => client
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(map.clone()))
            .await
            .map_err(|e| format!("tools/call failed: {e}")),
        _ => Err(format!(
            "The model sent arguments that aren't a JSON object: {raw_arguments}"
        )),
    };
    let (is_error, result_text, structured) = match outcome {
        Ok(result) => (
            result.is_error.unwrap_or(false),
            tool_result_text(&result),
            result.structured_content,
        ),
        Err(e) => (true, e, None),
    };
    ToolCallTrace {
        round,
        name: name.to_string(),
        arguments,
        is_error,
        result_text,
        structured,
        ms: started.elapsed().as_millis(),
    }
}

impl Agent {
    async fn run(&self, question: &str) -> AgentRun {
        let mut run = AgentRun::default();
        let outcome = tokio::time::timeout(RUN_TIMEOUT, self.run_inner(question, &mut run)).await;
        match outcome {
            Ok(Ok(answer)) => run.answer = Some(answer),
            Ok(Err(e)) => run.error = Some(e),
            Err(_) => run.error = Some(format!("Timed out after {}s", RUN_TIMEOUT.as_secs())),
        }
        run
    }

    async fn run_inner(&self, question: &str, run: &mut AgentRun) -> Result<String, String> {
        let transport = StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(self.mcp_url.clone()),
        );
        let client = ()
            .serve(transport)
            .await
            .map_err(|e| format!("MCP handshake with {} failed: {e}", self.mcp_url))?;
        let result = self.converse(&client, question, run).await;
        let _ = client.cancel().await;
        result
    }

    async fn converse(
        &self,
        client: &RunningService<RoleClient, ()>,
        question: &str,
        run: &mut AgentRun,
    ) -> Result<String, String> {
        // 1. Discover what the MCP server offers.
        let tools = client
            .list_all_tools()
            .await
            .map_err(|e| format!("tools/list failed: {e}"))?;
        run.mcp = Some(McpInfo {
            url: self.mcp_url.clone(),
            server: client
                .peer_info()
                .and_then(|info| info.server_info.as_ref())
                .map(|s| format!("{} v{}", s.name, s.version)),
            tools: tools.iter().map(ToolInfo::from).collect(),
        });
        let llm_tools: Vec<Value> = tools.iter().map(openai_tool).collect();

        let mut messages = vec![
            json!({ "role": "system", "content": system_prompt(&today()) }),
            json!({ "role": "user", "content": question }),
        ];

        for round in 1..=MAX_TOOL_ROUNDS {
            run.llm_rounds = round;
            // 2. Let the model decide: answer now, or call a tool first.
            let message = self.call_llm(&messages, &llm_tools).await?;
            let tool_calls = message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if tool_calls.is_empty() {
                let answer = message
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                return if answer.is_empty() {
                    Err("The model returned an empty answer.".to_string())
                } else {
                    Ok(answer)
                };
            }

            // 3. Run each requested call over MCP and hand the result back.
            messages.push(message);
            for call in tool_calls {
                let id = call["id"].as_str().unwrap_or_default().to_string();
                let name = call["function"]["name"].as_str().unwrap_or_default();
                let raw_arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
                let trace = call_mcp_tool(client, round, name, raw_arguments).await;
                println!(
                    "tools/call {name} {raw_arguments} -> {} in {} ms",
                    if trace.is_error { "error" } else { "ok" },
                    trace.ms
                );
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": trace.result_text,
                }));
                run.tool_calls.push(trace);
            }
        }
        Err(format!(
            "Gave up after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer."
        ))
    }

    async fn call_llm(&self, messages: &[Value], tools: &[Value]) -> Result<Value, String> {
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

/// Today's UTC date as `YYYY-MM-DD`, without pulling in a date crate.
fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0);
    civil_from_days(days as i64)
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

// ---------------------------------------------------------------------------
// HTTP: the web UI, its JSON API, and the MCP endpoint, all on one port.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

async fn ask(State(agent): State<Arc<Agent>>, Json(req): Json<AskRequest>) -> Json<AgentRun> {
    let question = req.question.trim();
    if question.is_empty() {
        return Json(AgentRun {
            error: Some("Ask a question first.".to_string()),
            ..Default::default()
        });
    }
    let run = agent.run(question).await;
    match &run.error {
        Some(e) => eprintln!("Question failed: {e}"),
        None => println!(
            "Answered with {} tool call(s) in {} LLM round(s)",
            run.tool_calls.len(),
            run.llm_rounds
        ),
    }
    Json(run)
}

#[derive(Serialize)]
struct ConfigResponse {
    mcp_url: String,
    model: String,
    tool: &'static str,
}

async fn config(State(agent): State<Arc<Agent>>) -> Json<ConfigResponse> {
    Json(ConfigResponse {
        mcp_url: agent.mcp_url.clone(),
        model: agent.model.clone(),
        tool: TOOL_NAME,
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(agent: Arc<Agent>, github: GitHub) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/config", get(config))
        .route("/api/ask", post(ask))
        .with_state(agent)
        .nest_service("/mcp", mcp_service(github))
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
    // By default the agent connects to this same process's `/mcp` endpoint,
    // over real HTTP, like any other MCP client would.
    let mcp_url =
        env_non_empty("MCP_SERVER_URL").unwrap_or_else(|| format!("http://127.0.0.1:{port}/mcp"));

    let github = GitHub {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client"),
        api_base: GITHUB_API.to_string(),
        token: env_non_empty("GITHUB_TOKEN"),
    };
    let agent = Arc::new(Agent {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .expect("failed to build HTTP client"),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model,
        mcp_url,
    });
    println!(
        "MCP server: /mcp (GitHub token: {}); agent connects to {}",
        if github.token.is_some() { "set" } else { "not set" },
        agent.mcp_url
    );

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(agent, github)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::RawQuery, http::StatusCode};
    use std::sync::Mutex;

    fn obj(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    fn github_at(api_base: &str) -> GitHub {
        GitHub {
            client: reqwest::Client::new(),
            api_base: api_base.to_string(),
            token: None,
        }
    }

    fn commits_fixture() -> Value {
        json!([
            {
                "sha": "abc1234def5678900000000000000000000000000",
                "html_url": "https://github.com/octo/demo/commit/abc1234def",
                "commit": {
                    "message": "Add MCP tool\n\nA longer body that should not reach the model.",
                    "author": { "name": "Octo Cat", "date": "2026-09-25T10:00:00Z" }
                },
                "author": { "login": "octocat" }
            },
            {
                "sha": "9876543fedcba00000000000000000000000000000",
                "html_url": "https://github.com/octo/demo/commit/9876543fed",
                "commit": {
                    "message": "Initial commit",
                    "author": { "name": "Someone", "date": "2026-09-20T08:00:00Z" }
                },
                "author": null
            }
        ])
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Stands in for api.github.com: answers the commits endpoint with a fixed
    /// status and body, and records the query string it was called with.
    fn fake_github(status: StatusCode, body: Value, seen: Arc<Mutex<Option<String>>>) -> Router {
        Router::new().route(
            "/repos/{owner}/{repo}/commits",
            get(move |RawQuery(query): RawQuery| {
                let seen = seen.clone();
                let body = body.clone();
                async move {
                    *seen.lock().unwrap() = Some(query.unwrap_or_default());
                    (status, Json(body))
                }
            }),
        )
    }

    /// A client connected to `CommitsServer` over an in-memory pipe.
    async fn duplex_client(github: GitHub) -> RunningService<RoleClient, ()> {
        let (server_io, client_io) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            let running = CommitsServer::new(github)
                .serve(server_io)
                .await
                .expect("server handshake");
            let _ = running.waiting().await;
        });
        ().serve(client_io).await.expect("client handshake")
    }

    #[tokio::test]
    async fn tool_is_registered_with_described_parameters() {
        let client = duplex_client(github_at("http://127.0.0.1:9")).await;
        let tools = client.list_all_tools().await.unwrap();
        client.cancel().await.unwrap();

        assert_eq!(tools.len(), 1);
        let tool = &tools[0];
        assert_eq!(tool.name, TOOL_NAME);
        assert!(tool.description.as_deref().unwrap().contains("commits"));

        let schema = Value::Object((*tool.input_schema).clone());
        let mut properties: Vec<&str> = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        properties.sort_unstable();
        assert_eq!(properties, ["branch", "limit", "owner", "path", "repo", "since"]);

        let mut required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        required.sort_unstable();
        assert_eq!(required, ["owner", "repo"]);

        for property in properties {
            let description = schema["properties"][property]["description"].as_str();
            assert!(description.is_some_and(|d| !d.is_empty()), "{property} has no description");
        }
    }

    #[tokio::test]
    async fn call_tool_returns_commits_from_github() {
        let seen = Arc::new(Mutex::new(None));
        let base = spawn(fake_github(StatusCode::OK, commits_fixture(), seen.clone())).await;
        let client = duplex_client(github_at(&base)).await;

        let result = client
            .call_tool(CallToolRequestParams::new(TOOL_NAME).with_arguments(obj(json!({
                "owner": "octo",
                "repo": "demo",
                "limit": 2,
                "since": "2026-09-01",
                "path": "",
            }))))
            .await
            .unwrap();
        client.cancel().await.unwrap();

        assert_ne!(result.is_error, Some(true));
        let text = tool_result_text(&result);
        assert!(text.contains("- abc1234 | 2026-09-25T10:00:00Z | octocat | Add MCP tool"), "{text}");
        assert!(text.contains("- 9876543 | 2026-09-20T08:00:00Z | Someone | Initial commit"), "{text}");
        assert!(!text.contains("longer body"), "{text}");

        let structured = result.structured_content.unwrap();
        assert_eq!(structured["repository"], "octo/demo");
        assert_eq!(structured["count"], 2);
        assert_eq!(structured["commits"][0]["sha"], "abc1234");

        let query = seen.lock().unwrap().clone().unwrap();
        assert!(query.contains("per_page=2"), "{query}");
        assert!(query.contains("since=2026-09-01T00"), "{query}");
        assert!(!query.contains("path="), "an empty path must be dropped: {query}");
    }

    #[tokio::test]
    async fn missing_repository_is_a_tool_error() {
        let seen = Arc::new(Mutex::new(None));
        let body = json!({ "message": "Not Found" });
        let base = spawn(fake_github(StatusCode::NOT_FOUND, body, seen)).await;
        let client = duplex_client(github_at(&base)).await;

        let result = client
            .call_tool(
                CallToolRequestParams::new(TOOL_NAME)
                    .with_arguments(obj(json!({ "owner": "octo", "repo": "nope" }))),
            )
            .await
            .unwrap();
        client.cancel().await.unwrap();

        assert_eq!(result.is_error, Some(true));
        assert!(tool_result_text(&result).contains("octo/nope was not found"));
    }

    #[tokio::test]
    async fn invalid_owner_is_rejected_before_calling_github() {
        let seen = Arc::new(Mutex::new(None));
        let base = spawn(fake_github(StatusCode::OK, commits_fixture(), seen.clone())).await;
        let client = duplex_client(github_at(&base)).await;

        let result = client
            .call_tool(
                CallToolRequestParams::new(TOOL_NAME)
                    .with_arguments(obj(json!({ "owner": "../admin", "repo": "demo" }))),
            )
            .await
            .unwrap();
        client.cancel().await.unwrap();

        assert_eq!(result.is_error, Some(true));
        assert!(seen.lock().unwrap().is_none(), "GitHub must not be called");
    }

    /// Stands in for the LLM: first asks for `list_recent_commits`, then, once
    /// the tool result is in the conversation, answers from it.
    fn fake_llm(requests: Arc<Mutex<Vec<Value>>>) -> Router {
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body.clone());
                    let tool_result = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|m| m["role"] == "tool")
                        .and_then(|m| m["content"].as_str())
                        .map(str::to_string);
                    let message = match tool_result {
                        None => json!({
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "list_recent_commits",
                                    "arguments": "{\"owner\":\"octo\",\"repo\":\"demo\",\"limit\":2}"
                                }
                            }]
                        }),
                        Some(result) => {
                            // Build the answer out of the tool result itself.
                            let latest = result.lines().find(|l| l.starts_with("- ")).unwrap();
                            let fields: Vec<&str> = latest[2..].split(" | ").collect();
                            json!({
                                "role": "assistant",
                                "content": format!("The latest commit is {}: {}", fields[0], fields[3]),
                            })
                        }
                    };
                    Json(json!({ "choices": [{ "message": message }] }))
                }
            }),
        )
    }

    #[tokio::test]
    async fn agent_calls_the_mcp_tool_and_uses_the_result() {
        let seen = Arc::new(Mutex::new(None));
        let github_base = spawn(fake_github(StatusCode::OK, commits_fixture(), seen)).await;
        let llm_requests = Arc::new(Mutex::new(Vec::new()));
        let llm_base = spawn(fake_llm(llm_requests.clone())).await;

        // The real app, `/mcp` included, on a random port; the agent reaches
        // the tool over Streamable HTTP exactly as it does in production.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let agent = Arc::new(Agent {
            http: reqwest::Client::new(),
            endpoint: format!("{llm_base}/chat/completions"),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
            mcp_url: format!("http://{addr}/mcp"),
        });
        let router = app(agent.clone(), github_at(&github_base));
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let run = agent.run("What changed in octo/demo?").await;

        assert!(run.error.is_none(), "{:?}", run.error);
        let mcp = run.mcp.as_ref().unwrap();
        assert_eq!(mcp.server.as_deref(), Some(&*format!("github-commits v{}", env!("CARGO_PKG_VERSION"))));
        assert_eq!(mcp.tools.len(), 1);
        assert_eq!(run.llm_rounds, 2);
        assert_eq!(run.tool_calls.len(), 1);
        let call = &run.tool_calls[0];
        assert_eq!(call.name, TOOL_NAME);
        assert_eq!(call.arguments["repo"], "demo");
        assert!(!call.is_error, "{}", call.result_text);
        assert!(call.result_text.contains("abc1234"));
        assert_eq!(run.answer.as_deref(), Some("The latest commit is abc1234: Add MCP tool"));

        let requests = llm_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let function = &requests[0]["tools"][0]["function"];
        assert_eq!(function["name"], TOOL_NAME);
        assert_eq!(function["parameters"]["required"].as_array().unwrap().len(), 2);
        assert!(function["parameters"].get("$schema").is_none());
        let messages = requests[1]["messages"].as_array().unwrap();
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_1");
    }

    #[test]
    fn validate_normalizes_arguments() {
        let args = ListCommitsArgs {
            owner: " rust-lang ".to_string(),
            repo: "rust".to_string(),
            limit: Some(500),
            path: Some("  ".to_string()),
            branch: Some("master".to_string()),
            since: Some("2026-09-01".to_string()),
        };
        assert_eq!(
            args.validate().unwrap(),
            CommitQuery {
                owner: "rust-lang".to_string(),
                repo: "rust".to_string(),
                limit: MAX_LIMIT,
                path: None,
                branch: Some("master".to_string()),
                since: Some("2026-09-01T00:00:00Z".to_string()),
            }
        );

        let bad_since = ListCommitsArgs {
            owner: "a".to_string(),
            repo: "b".to_string(),
            limit: None,
            path: None,
            branch: None,
            since: Some("last week".to_string()),
        };
        assert!(bad_since.validate().unwrap_err().contains("since"));
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), "1970-01-01");
        assert_eq!(civil_from_days(11_017), "2000-03-01");
        assert_eq!(civil_from_days(20_722), "2026-09-26");
    }
}

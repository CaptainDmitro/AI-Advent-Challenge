mod diff;
mod git;
mod github;
mod index;
mod mcp;
mod rerank;
mod review;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
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
use tokio::sync::Mutex;

use diff::FileDiff;
use git::{DEFAULT_REPO_URL, Repo, RepoState, default_repo_dir};
use github::{GitHub, Posted};
use index::{Index, env_non_empty, load_corpus, local};
use mcp::project_service;
use rerank::Heuristic;
use review::{Candidate, Comment, Dropped, Finding, Fragment, Query, Target};

const MAX_TOOL_ROUNDS: usize = 6;
/// A tool result is clipped to this before it goes to the model.
const MAX_RESULT_CHARS: usize = 12_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const RUN_TIMEOUT: Duration = Duration::from_secs(600);

const USAGE: &str = "Usage:
  ai-code-review                     the web page on $PORT (default 3000)
  ai-code-review review [options]    review once, print the Markdown review

Options:
  --pr N            review pull request N (metadata from the GitHub API)
  --base REV        review REV...HEAD instead (default master)
  --head REV        the head to review with --base (default HEAD)
  --repo-dir DIR    the git checkout (default: $REPO_DIR, or .. inside a clone)
  --post            post the review as a PR comment, or update the earlier one
  --out FILE        also write the Markdown to FILE";

static NEXT_WORKTREE: AtomicU64 = AtomicU64::new(0);

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("\n… (clipped)");
    out
}

/// `https://github.com/owner/name.git` → `owner/name`.
fn repo_name(url: &str) -> String {
    let trimmed = url.trim_end_matches('/').trim_end_matches(".git");
    let parts: Vec<&str> = trimmed.rsplit(['/', ':']).take(2).collect();
    match parts.as_slice() {
        [name, owner] if !owner.is_empty() => format!("{owner}/{name}"),
        _ => trimmed.to_string(),
    }
}

// ---------------------------------------------------------------------------
// The MCP client side: the reviewer's connection to the `project` server.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct ToolCallTrace {
    step: usize,
    /// Which part of the diff the call was made for.
    batch: usize,
    round: usize,
    tool: String,
    arguments: Value,
    is_error: bool,
    result: String,
    ms: u128,
}

impl ToolCallTrace {
    /// `read_file(path: src/x.rs)`, for the PR comment.
    fn label(&self) -> String {
        let args = match &self.arguments {
            Value::Object(map) => map
                .iter()
                .map(|(k, v)| {
                    let v = v.as_str().map_or_else(|| v.to_string(), str::to_string);
                    format!("{k}: {v}")
                })
                .collect::<Vec<_>>()
                .join(", "),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        format!("{}({args})", self.tool)
    }
}

struct Project {
    client: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

/// `initialize` + `tools/list` over Streamable HTTP, exactly as with a remote
/// server.
async fn connect(url: &str) -> Result<Project, String> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.to_string()),
    );
    let client = ()
        .serve(transport)
        .await
        .map_err(|e| format!("MCP handshake with {url} failed: {e}"))?;
    match client.list_all_tools().await {
        Ok(tools) => Ok(Project { client, tools }),
        Err(e) => {
            let _ = client.cancel().await;
            Err(format!("tools/list failed: {e}"))
        }
    }
}

async fn connect_with_timeout(url: &str) -> Result<Project, String> {
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

impl Project {
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

    async fn call(&self, step: usize, batch: usize, round: usize, name: &str, arguments: Value) -> ToolCallTrace {
        let started = Instant::now();
        let outcome = if !self.tools.iter().any(|t| &*t.name == name) {
            Err(format!(
                "The project server has no tool `{name}` (it has: {}).",
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
        let (is_error, result) = match outcome {
            Ok(r) => (r.is_error.unwrap_or(false), tool_result_text(&r)),
            Err(e) => (true, e),
        };
        println!(
            "mcp {step}: {name}({arguments}) -> {} in {} ms",
            if is_error { "error" } else { "ok" },
            started.elapsed().as_millis()
        );
        ToolCallTrace {
            step,
            batch,
            round,
            tool: name.to_string(),
            arguments,
            is_error,
            result: clip(&result, MAX_RESULT_CHARS),
            ms: started.elapsed().as_millis(),
        }
    }

    async fn close(self) {
        let _ = self.client.cancel().await;
    }
}

/// The `project` MCP server over one checkout, on a private port for the
/// length of one review: the reviewer reaches the PR's head over real MCP,
/// the same way it would reach a remote server.
async fn spawn_project_server(repo: Arc<Repo>) -> Result<(String, tokio::task::JoinHandle<()>), String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Can't start the MCP server: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Can't start the MCP server: {e}"))?;
    let router = Router::new().nest_service("/mcp/project", project_service(repo));
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Ok((format!("http://{addr}/mcp/project"), server))
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
        let mut body = json!({ "model": self.model, "messages": messages, "temperature": 0.1 });
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
// The review pipeline.
// ---------------------------------------------------------------------------

/// A pull request by number, or two revisions of the checkout.
#[derive(Deserialize, Default, Debug, Clone)]
struct ReviewRequest {
    pr: Option<u64>,
    base: Option<String>,
    head: Option<String>,
}

#[derive(Serialize, Default, Debug)]
struct Stats {
    files: usize,
    /// Files whose patch went to the model.
    reviewed: usize,
    additions: usize,
    deletions: usize,
}

#[derive(Serialize, Debug)]
struct IndexInfo {
    documents: usize,
    chunks: usize,
    doc_chunks: usize,
    code_chunks: usize,
    embedder: String,
    built_ms: u128,
}

#[derive(Serialize, Default, Debug)]
struct ReviewReply {
    target: Option<Target>,
    /// `checkout` when the head was already checked out, `worktree` when a
    /// temporary worktree at the head was used.
    checkout: Option<String>,
    files: Vec<FileDiff>,
    stats: Stats,
    summary: Option<String>,
    findings: Vec<Finding>,
    dropped: Vec<Dropped>,
    queries: Vec<Query>,
    fragments: Vec<Fragment>,
    candidates: Vec<Candidate>,
    calls: Vec<ToolCallTrace>,
    tools: Vec<String>,
    mcp_error: Option<String>,
    batches: usize,
    not_reviewed: Vec<String>,
    llm_rounds: usize,
    index: Option<IndexInfo>,
    /// The PR comment.
    markdown: Option<String>,
    posted: Option<Posted>,
    post_error: Option<String>,
    error: Option<String>,
    model: String,
    ms: u128,
}

#[derive(Clone)]
struct Settings {
    model: String,
    repo_url: String,
    repo_name: String,
    language: String,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<Llm>,
    /// The main checkout: the clone on the VDS, the PR checkout in Actions.
    repo: Arc<Repo>,
    github: Arc<GitHub>,
    settings: Arc<Settings>,
    /// One review at a time: they share the checkout's refs and worktrees.
    lock: Arc<Mutex<()>>,
}

/// Reviews a PR or two revisions and renders the comment; with `post`,
/// also posts it to the PR (a failed review is posted too, so the PR shows
/// why there is no review).
async fn review(app: &AppState, req: ReviewRequest, post: bool) -> ReviewReply {
    let started = Instant::now();
    let mut reply = ReviewReply {
        model: app.settings.model.clone(),
        ..Default::default()
    };
    {
        let _one_at_a_time = app.lock.lock().await;
        match tokio::time::timeout(RUN_TIMEOUT, run_review(app, &req, &mut reply)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => reply.error = Some(e),
            Err(_) => reply.error = Some(format!("Timed out after {}s.", RUN_TIMEOUT.as_secs())),
        }
    }
    let markdown = match &reply.error {
        Some(error) => review::failure_markdown(error, reply.target.as_ref()),
        None => {
            let fallback = Target::default();
            let calls: Vec<String> = reply.calls.iter().map(ToolCallTrace::label).collect();
            review::markdown(&Comment {
                repo: &app.settings.repo_name,
                model: &app.settings.model,
                target: reply.target.as_ref().unwrap_or(&fallback),
                files: &reply.files,
                summary: reply.summary.as_deref().unwrap_or_default(),
                findings: &reply.findings,
                fragments: &reply.fragments,
                calls: &calls,
                not_reviewed: &reply.not_reviewed,
                seconds: started.elapsed().as_secs_f64(),
            })
        }
    };
    if post {
        match req.pr {
            Some(pr) => match app.github.upsert_comment(pr, &markdown).await {
                Ok(posted) => {
                    println!("review comment {} on PR #{pr}: {}", posted.action, posted.url);
                    reply.posted = Some(posted);
                }
                Err(e) => reply.post_error = Some(e),
            },
            None => {
                reply.post_error = Some("Posting needs a pull request number.".to_string());
            }
        }
    }
    reply.markdown = Some(markdown);
    reply.ms = started.elapsed().as_millis();
    if let Some(e) = &reply.error {
        eprintln!("review failed: {e}");
    }
    reply
}

async fn run_review(app: &AppState, req: &ReviewRequest, reply: &mut ReviewReply) -> Result<(), String> {
    match app.repo.state() {
        RepoState::Ready => {}
        RepoState::Cloning => {
            return Err("The repository is still being cloned; try again in a minute.".to_string());
        }
        RepoState::Unavailable { reason } => return Err(reason),
    }
    let target = resolve_target(app, req).await?;
    reply.target = Some(target.clone());

    // 1. The diff: what the head changed since it left the base.
    let files = diff::parse(&app.repo.diff(&target.base_sha, &target.head_sha).await?);
    reply.stats = Stats {
        files: files.len(),
        reviewed: files.iter().filter(|f| f.skipped.is_none()).count(),
        additions: files.iter().map(|f| f.additions).sum(),
        deletions: files.iter().map(|f| f.deletions).sum(),
    };
    reply.files = files.clone();
    if reply.stats.reviewed == 0 {
        reply.summary = Some(if files.is_empty() {
            "No changes between these commits.".to_string()
        } else {
            "Nothing to review: every changed file was skipped.".to_string()
        });
        return Ok(());
    }

    // 2. The files at the head: the checkout itself, or a worktree.
    let checkout = prepare_checkout(&app.repo, &target.head_sha).await?;
    reply.checkout = Some(if checkout.worktree { "worktree" } else { "checkout" }.to_string());
    let result = review_at(app, &checkout.dir, &target, &files, reply).await;
    if checkout.worktree {
        app.repo.remove_worktree(&checkout.dir).await;
    }
    result
}

/// The commits to compare. A PR comes from the GitHub API; its commits are
/// fetched into the checkout when they aren't there yet (on the VDS).
async fn resolve_target(app: &AppState, req: &ReviewRequest) -> Result<Target, String> {
    if let Some(number) = req.pr {
        let pr = app.github.pull(number).await?;
        if !(app.repo.has_commit(&pr.base_sha).await && app.repo.has_commit(&pr.head_sha).await) {
            let base = git::valid_ref(&pr.base_ref)?;
            println!("fetching PR #{number} and {base}");
            app.repo
                .fetch(&[
                    format!("+refs/pull/{number}/head:refs/review/pr-{number}"),
                    format!("+refs/heads/{base}:refs/remotes/origin/{base}"),
                ])
                .await?;
            for sha in [&pr.base_sha, &pr.head_sha] {
                if !app.repo.has_commit(sha).await {
                    return Err(format!(
                        "Commit {sha} of PR #{number} isn't in the checkout even after fetching it."
                    ));
                }
            }
        }
        return Ok(Target {
            pr: Some(number),
            title: pr.title,
            description: pr.body,
            author: Some(pr.author),
            url: Some(pr.url),
            base_ref: pr.base_ref,
            head_ref: pr.head_ref,
            base_sha: pr.base_sha,
            head_sha: pr.head_sha,
        });
    }
    let pick = |value: &Option<String>, default: &'static str| {
        value
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_string()
    };
    let base = pick(&req.base, "master");
    let head = pick(&req.head, "HEAD");
    let base_sha = app.repo.rev_parse(&base).await?;
    let head_sha = app.repo.rev_parse(&head).await?;
    Ok(Target {
        pr: None,
        title: format!("{base}...{head}"),
        description: String::new(),
        author: None,
        url: None,
        base_ref: base,
        head_ref: head,
        base_sha,
        head_sha,
    })
}

struct Checkout {
    dir: PathBuf,
    worktree: bool,
}

/// The head's files: the checkout itself when it is at the head with no
/// edits (the PR checkout in GitHub Actions), otherwise a temporary
/// detached worktree at the head (the clone on the VDS, which stays on its
/// own branch).
async fn prepare_checkout(repo: &Repo, head_sha: &str) -> Result<Checkout, String> {
    let current = repo.rev_parse("HEAD").await?;
    let dirty = !repo
        .git(&["status", "--porcelain=v1", "--untracked-files=no"])
        .await?
        .trim()
        .is_empty();
    if current == head_sha && !dirty {
        return Ok(Checkout {
            dir: repo.dir.clone(),
            worktree: false,
        });
    }
    let dir = std::env::temp_dir().join(format!(
        "ai-code-review-{}-{}",
        std::process::id(),
        NEXT_WORKTREE.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    repo.add_worktree(&dir, head_sha).await?;
    Ok(Checkout {
        dir,
        worktree: true,
    })
}

/// Everything after the diff, over the head's files in `dir`: the index,
/// the retrieval, the MCP server, the model, the checks.
async fn review_at(
    app: &AppState,
    dir: &Path,
    target: &Target,
    files: &[FileDiff],
    reply: &mut ReviewReply,
) -> Result<(), String> {
    // 3. RAG: the whole repository at the head, docs and code.
    let started = Instant::now();
    let index = Index::build(load_corpus(dir)?, local()).await?;
    let heuristic = Heuristic::new(&index.chunks);
    let doc_chunks = index.chunks.iter().filter(|c| c.kind == "doc").count();
    reply.index = Some(IndexInfo {
        documents: index.documents,
        chunks: index.chunks.len(),
        doc_chunks,
        code_chunks: index.chunks.len() - doc_chunks,
        embedder: index.embedder.id(),
        built_ms: started.elapsed().as_millis(),
    });
    reply.queries = review::queries(files);
    let retrieval = review::retrieve(&index, &heuristic, files, &reply.queries).await?;
    reply.fragments = retrieval.fragments;
    reply.candidates = retrieval.candidates;
    println!(
        "indexed {} files ({} chunks) in {} ms; {} queries -> {} fragments",
        index.documents,
        index.chunks.len(),
        started.elapsed().as_millis(),
        reply.queries.len(),
        reply.fragments.len()
    );

    // 4. MCP: the project server over the head, for the model's tool calls.
    let head_repo = Arc::new(Repo::new(dir.to_path_buf(), app.repo.url.clone()));
    let (project, server) = match spawn_project_server(head_repo).await {
        Ok((url, server)) => match connect_with_timeout(&url).await {
            Ok(project) => (Some(project), Some(server)),
            Err(e) => {
                server.abort();
                reply.mcp_error = Some(e);
                (None, None)
            }
        },
        Err(e) => {
            reply.mcp_error = Some(e);
            (None, None)
        }
    };
    if let Some(project) = &project {
        reply.tools = project.tool_names();
    }
    let tools = project.as_ref().map(Project::llm_tools).unwrap_or_default();
    let system = review::system_prompt(
        &app.settings.repo_name,
        &app.settings.language,
        &reply.fragments,
        &reply.tools,
    );

    // 5. The model, one conversation per batch of files.
    let plan = review::batches(files);
    reply.batches = plan.batches.len();
    reply.not_reviewed = plan.not_reviewed.clone();
    let mut summaries = Vec::new();
    let mut raw = Vec::new();
    let mut result = Ok(());
    for (i, patches) in plan.batches.iter().enumerate() {
        let user = review::user_prompt(target, files, patches, (i + 1, plan.batches.len()));
        match review_batch(app, project.as_ref(), &tools, &system, &user, i + 1, reply).await {
            Ok((summary, findings)) => {
                if !summary.is_empty() {
                    summaries.push(summary);
                }
                raw.extend(findings);
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }
    if let Some(project) = project {
        project.close().await;
    }
    if let Some(server) = server {
        server.abort();
    }
    result?;

    // 6. Only findings that point into the diff survive.
    let (findings, dropped) = review::validate(&raw, files, &reply.fragments);
    for fragment in reply.fragments.iter_mut() {
        fragment.cited = findings.iter().any(|f| f.sources.contains(&fragment.n));
    }
    reply.findings = findings;
    reply.dropped = dropped;
    reply.summary = Some(summaries.join(" "));
    Ok(())
}

/// One review conversation: the model may call the MCP tools for a few
/// rounds, then answers with the JSON review. A reply that isn't JSON gets
/// one more chance.
async fn review_batch(
    app: &AppState,
    project: Option<&Project>,
    tools: &[Value],
    system: &str,
    user: &str,
    batch: usize,
    reply: &mut ReviewReply,
) -> Result<(String, Vec<Value>), String> {
    let mut messages = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": user }),
    ];
    for round in 1..=MAX_TOOL_ROUNDS {
        reply.llm_rounds += 1;
        let last = round == MAX_TOOL_ROUNDS;
        if last {
            messages.push(json!({
                "role": "user",
                "content": "No more tool calls: give the final JSON review now."
            }));
        }
        let offered: &[Value] = if last { &[] } else { tools };
        let message = app.llm.chat(&messages, offered).await?;
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            let text = message_text(&message)
                .ok_or_else(|| "The model returned an empty review.".to_string())?;
            return match review::extract_json(&text) {
                Ok(value) => Ok(review::split_output(&value)),
                Err(e) => {
                    messages.push(message);
                    messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "That was not valid JSON ({e}). Reply again with only the JSON \
                             object described in the instructions."
                        ),
                    }));
                    reply.llm_rounds += 1;
                    let retry = app.llm.chat(&messages, &[]).await?;
                    let text = message_text(&retry).unwrap_or_default();
                    review::extract_json(&text)
                        .map(|value| review::split_output(&value))
                        .map_err(|e| format!("The model's review is not valid JSON: {e}"))
                }
            };
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
            let trace = match project {
                Some(project) => project.call(step, batch, round, name, arguments).await,
                None => ToolCallTrace {
                    step,
                    batch,
                    round,
                    tool: name.to_string(),
                    arguments,
                    is_error: true,
                    result: "The project MCP server is unreachable.".to_string(),
                    ms: 0,
                },
            };
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
        "Gave up after {MAX_TOOL_ROUNDS} rounds without a final review."
    ))
}

// ---------------------------------------------------------------------------
// HTTP.
// ---------------------------------------------------------------------------

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// The page never posts: a preview of the review the Action posts.
async fn review_api(State(app): State<AppState>, Json(req): Json<ReviewRequest>) -> Json<ReviewReply> {
    Json(review(&app, req, false).await)
}

async fn pulls_api(State(app): State<AppState>) -> Json<Value> {
    Json(match app.github.open_pulls().await {
        Ok(pulls) => json!({ "pulls": pulls, "error": null }),
        Err(e) => json!({ "pulls": [], "error": e }),
    })
}

async fn status_api(State(app): State<AppState>) -> Json<Value> {
    let branch = app.repo.branch().await;
    Json(json!({
        "repo": {
            "name": app.settings.repo_name,
            "url": app.settings.repo_url,
            "dir": app.repo.dir.display().to_string(),
            "state": app.repo.state(),
        },
        "branch": branch.as_ref().ok(),
        "branch_error": branch.as_ref().err(),
        "github": {
            "api": app.github.api(),
            "repo": app.github.repo,
            "token": app.github.has_token(),
        },
        "model": app.settings.model,
        "language": app.settings.language,
        "threshold": review::THRESHOLD,
        "pinned": review::PINNED,
    }))
}

fn app(state: AppState) -> Router {
    let repo = state.repo.clone();
    Router::new()
        .route("/", get(index_page))
        .route("/api/status", get(status_api))
        .route("/api/pulls", get(pulls_api))
        .route("/api/review", post(review_api))
        .with_state(state)
        .nest_service("/mcp/project", project_service(repo))
}

fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build HTTP client")
}

// ---------------------------------------------------------------------------
// The command line: what the GitHub Action runs.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
struct Cli {
    pr: Option<u64>,
    base: Option<String>,
    head: Option<String>,
    repo_dir: Option<PathBuf>,
    post: bool,
    out: Option<PathBuf>,
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value."))
        };
        match arg.as_str() {
            "--pr" => {
                let number = value("--pr")?;
                cli.pr = Some(
                    number
                        .trim_start_matches('#')
                        .parse()
                        .map_err(|_| format!("--pr needs a pull request number, not `{number}`."))?,
                );
            }
            "--base" => cli.base = Some(value("--base")?),
            "--head" => cli.head = Some(value("--head")?),
            "--repo-dir" => cli.repo_dir = Some(PathBuf::from(value("--repo-dir")?)),
            "--out" => cli.out = Some(PathBuf::from(value("--out")?)),
            "--post" => cli.post = true,
            other => return Err(format!("Unknown argument `{other}`.")),
        }
    }
    if cli.post && cli.pr.is_none() {
        return Err("--post needs --pr.".to_string());
    }
    if cli.pr.is_some() && (cli.base.is_some() || cli.head.is_some()) {
        return Err("Use either --pr or --base/--head, not both.".to_string());
    }
    Ok(cli)
}

fn append(path: &str, text: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{text}")
}

/// `ai-code-review review ...`: one review, printed; written to `--out`
/// and the Actions job summary; posted with `--post`. The exit code is 1
/// only when the review or the posting failed, never because of findings.
async fn run_cli(state: AppState, cli: Cli) {
    state.repo.ensure(false).await;
    let request = ReviewRequest {
        pr: cli.pr,
        base: cli.base.clone(),
        head: cli.head.clone(),
    };
    let reply = review(&state, request, cli.post).await;
    let markdown = reply.markdown.clone().unwrap_or_default();
    println!("\n{markdown}");
    if let Some(path) = &cli.out
        && let Err(e) = std::fs::write(path, &markdown)
    {
        eprintln!("Can't write {}: {e}", path.display());
    }
    if let Some(summary) = env_non_empty("GITHUB_STEP_SUMMARY")
        && let Err(e) = append(&summary, &markdown)
    {
        eprintln!("Can't write the job summary: {e}");
    }
    eprintln!(
        "files: {} ({} reviewed), fragments: {}, MCP calls: {}, findings: {} ({} dropped), {} ms",
        reply.stats.files,
        reply.stats.reviewed,
        reply.fragments.len(),
        reply.calls.len(),
        reply.findings.len(),
        reply.dropped.len(),
        reply.ms
    );
    if let Some(e) = &reply.post_error {
        eprintln!("Posting the review failed: {e}");
    }
    if reply.error.is_some() || reply.post_error.is_some() {
        std::process::exit(1);
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match args.first().map(String::as_str) {
        Some("review") => match parse_cli(&args[1..]) {
            Ok(cli) => Some(cli),
            Err(e) => {
                eprintln!("Error: {e}\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            return;
        }
        _ => None,
    };
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let base_url =
        env_non_empty("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.deepseek.com".to_string());
    let model = env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key = env_non_empty("OPENAI_API_KEY").unwrap_or_else(|| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    let language = env_non_empty("REVIEW_LANGUAGE").unwrap_or_else(|| "English".to_string());
    // Under `cargo run` the repo this lesson sits in; `pr` in the Action;
    // `~/apps/lesson-32/repo` on the VDS, cloned on first start.
    let repo_dir = cli
        .as_ref()
        .and_then(|c| c.repo_dir.clone())
        .or_else(|| env_non_empty("REPO_DIR").map(PathBuf::from))
        .unwrap_or_else(default_repo_dir);
    let repo_url = env_non_empty("REPO_URL").unwrap_or_else(|| DEFAULT_REPO_URL.to_string());
    let clone = cli.is_none() && env_non_empty("REPO_CLONE").is_none_or(|v| v.trim() != "off");
    // GitHub Actions sets GITHUB_REPOSITORY and GITHUB_API_URL.
    let github_repo = env_non_empty("GITHUB_REPOSITORY").unwrap_or_else(|| repo_name(&repo_url));
    let github_api = env_non_empty("GITHUB_API_URL").unwrap_or_else(|| github::DEFAULT_API.to_string());

    let repo = Arc::new(Repo::new(repo_dir, repo_url.clone()));
    let state = AppState {
        llm: Arc::new(Llm {
            http: http_client(Duration::from_secs(180)),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model: model.clone(),
        }),
        repo: repo.clone(),
        github: Arc::new(GitHub::new(
            http_client(Duration::from_secs(30)),
            &github_api,
            env_non_empty("GITHUB_TOKEN"),
            github_repo.clone(),
        )),
        settings: Arc::new(Settings {
            model,
            repo_name: github_repo,
            repo_url,
            language,
        }),
        lock: Arc::new(Mutex::new(())),
    };
    println!(
        "repository {} in {}",
        state.settings.repo_name,
        repo.dir.display()
    );

    if let Some(cli) = cli {
        run_cli(state, cli).await;
        return;
    }

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    // The clone can take a while; the page works meanwhile and says so.
    tokio::spawn(async move {
        repo.ensure(clone).await;
    });
    println!("Listening on http://localhost:{port} (reviews at /api/review, MCP at /mcp/project)");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::Path as UrlPath;
    use axum::http::HeaderMap;
    use axum::routing::patch;
    use std::sync::Mutex as StdMutex;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson32-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn git_out(dir: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "init.defaultBranch=master",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("git must be installed to run these tests");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git(dir: &Path, args: &[&str]) {
        git_out(dir, args);
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    const MAIN_RS: &str = "mod retry;

use retry::{MAX_RETRIES, retry_delay};

/// Calls `f` until it succeeds, waiting longer after every failure.
pub fn call_with_retry<T>(mut f: impl FnMut() -> Result<T, String>) -> Result<T, String> {
    let mut last = String::new();
    for attempt in 0..MAX_RETRIES {
        match f() {
            Ok(value) => return Ok(value),
            Err(e) => last = e,
        }
        std::thread::sleep(retry_delay(attempt));
    }
    Err(last)
}

fn main() {
    let answer = call_with_retry(|| Ok::<_, String>(42)).unwrap();
    println!(\"{answer}\");
}
";

    const RETRY_RS: &str = "use std::time::Duration;

/// The most attempts `call_with_retry` makes.
pub const MAX_RETRIES: u32 = 5;

/// How long to wait after failed attempt number `attempt` (0-based):
/// 200 ms, then doubling.
pub fn retry_delay(attempt: u32) -> Duration {
    Duration::from_millis(200 * 2u64.pow(attempt))
}
";

    const JITTER_RS: &str = "use std::time::Duration;

/// A second backoff, with jitter.
pub fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(100 * u64::from(attempt + 1))
}
";

    /// A small project: master with the docs and one lesson, and a feature
    /// branch with one more commit that changes the retry loop (line 8)
    /// and adds a second backoff. The feature branch is checked out.
    fn temp_repo(name: &str) -> PathBuf {
        let dir = temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "README.md", "# Demo\n\nA monorepo of course lessons.\n");
        write(
            &dir,
            "docs/conventions.md",
            "# Conventions\n\n## Code\n\n- Errors are strings for the UI: a handler returns \
             an error as a display string and never panics, so no `unwrap()` on anything that \
             comes from the outside.\n- Retries go through `retry_delay` in `src/retry.rs`; \
             don't write a second backoff.\n- Tests need no network and no keys.\n",
        );
        write(
            &dir,
            "docs/architecture.md",
            "# Architecture\n\n## Layout\n\nEvery lesson folder is an independent Cargo crate \
             with its own README and source files.\n",
        );
        write(
            &dir,
            "07. Retry Lesson/README.md",
            "# 07. Retry Lesson\n\n## What this is\n\n`call_with_retry` calls a fallible \
             function up to `MAX_RETRIES` attempts, sleeping `retry_delay(attempt)` between \
             them: exponential backoff from 200 ms.\n",
        );
        write(&dir, "07. Retry Lesson/src/retry.rs", RETRY_RS);
        write(&dir, "07. Retry Lesson/src/main.rs", MAIN_RS);
        git(&dir, &["init", "-q"]);
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "Initial commit"]);
        git(&dir, &["checkout", "-q", "-b", "feature/faster-retries"]);
        write(
            &dir,
            "07. Retry Lesson/src/main.rs",
            &MAIN_RS.replace("0..MAX_RETRIES", "0..=MAX_RETRIES"),
        );
        write(&dir, "07. Retry Lesson/src/jitter.rs", JITTER_RS);
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "Retry once more, add jitter"]);
        dir
    }

    const REVIEW_JSON: &str = r#"{"summary": "Retries one more time than documented and adds a second backoff.",
 "findings": [
  {"category": "bug", "severity": "high", "file": "07. Retry Lesson/src/main.rs", "line": 8,
   "title": "Off-by-one: MAX_RETRIES + 1 attempts", "details": "0..=MAX_RETRIES runs six times.",
   "suggestion": "Use `0..MAX_RETRIES`.", "sources": [1, 99]},
  {"category": "architecture", "severity": "medium", "file": "07. Retry Lesson/src/jitter.rs",
   "line": 999, "title": "A second backoff", "details": "retry_delay already exists.", "sources": [1]},
  {"category": "recommendation", "severity": "low", "file": "src/elsewhere.rs", "line": 1,
   "title": "Not in this PR"},
  {"category": "bug", "file": "07. Retry Lesson/src/main.rs", "line": 8, "title": ""}
 ]}"#;

    /// A scripted OpenAI-compatible LLM: the first request with tools gets a
    /// `read_file` call, everything after it gets `review`.
    fn fake_llm(review: &'static str, requests: Arc<StdMutex<Vec<Value>>>) -> Router {
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body.clone());
                    let answered = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|m| m["role"] == "assistant")
                        .count();
                    if answered == 0 && body.get("tools").is_some() {
                        let arguments = json!({ "path": "07. Retry Lesson/src/retry.rs" });
                        return Json(json!({ "choices": [{ "message": {
                            "role": "assistant", "content": null,
                            "tool_calls": [{ "id": "call_0", "type": "function", "function": {
                                "name": "read_file", "arguments": arguments.to_string()
                            } }]
                        } }] }));
                    }
                    Json(json!({ "choices": [{ "message": {
                        "role": "assistant", "content": review
                    } }] }))
                }
            }),
        )
    }

    #[derive(Default)]
    struct FakeGitHub {
        comments: Vec<(u64, String)>,
        auth: Vec<String>,
    }

    fn comment_json(id: u64, body: &str) -> Value {
        json!({ "id": id, "body": body, "html_url": format!("https://github.test/c/{id}") })
    }

    /// The parts of the GitHub API the reviewer uses, with comments kept in
    /// memory.
    fn fake_github(pull: Value, state: Arc<StdMutex<FakeGitHub>>) -> Router {
        let pull = Arc::new(pull);
        let one = pull.clone();
        let (list, add, edit) = (state.clone(), state.clone(), state);
        Router::new()
            .route(
                "/repos/{owner}/{name}/pulls/{number}",
                get(move || {
                    let pull = one.clone();
                    async move { Json((*pull).clone()) }
                }),
            )
            .route(
                "/repos/{owner}/{name}/pulls",
                get(move || {
                    let pull = pull.clone();
                    async move { Json(json!([(*pull).clone()])) }
                }),
            )
            .route(
                "/repos/{owner}/{name}/issues/{number}/comments",
                get(move || {
                    let state = list.clone();
                    async move {
                        let comments: Vec<Value> = state
                            .lock()
                            .unwrap()
                            .comments
                            .iter()
                            .map(|(id, body)| comment_json(*id, body))
                            .collect();
                        Json(json!(comments))
                    }
                })
                .post(move |headers: HeaderMap, Json(body): Json<Value>| {
                    let state = add.clone();
                    async move {
                        let mut state = state.lock().unwrap();
                        let auth = headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or_default()
                            .to_string();
                        state.auth.push(auth);
                        let id = 100 + state.comments.len() as u64;
                        let text = body["body"].as_str().unwrap_or_default().to_string();
                        state.comments.push((id, text.clone()));
                        Json(comment_json(id, &text))
                    }
                }),
            )
            .route(
                "/repos/{owner}/{name}/issues/comments/{id}",
                patch(
                    move |UrlPath((_, _, id)): UrlPath<(String, String, u64)>,
                          Json(body): Json<Value>| {
                        let state = edit.clone();
                        async move {
                            let mut state = state.lock().unwrap();
                            let text = body["body"].as_str().unwrap_or_default().to_string();
                            if let Some(c) = state.comments.iter_mut().find(|(cid, _)| *cid == id) {
                                c.1 = text.clone();
                            }
                            Json(comment_json(id, &text))
                        }
                    },
                ),
            )
    }

    fn pull_json(base_sha: &str, head_sha: &str) -> Value {
        json!({
            "number": 7, "title": "Faster retries", "body": "Retry once more, and add jitter.",
            "html_url": "https://github.test/octo/demo/pull/7", "user": { "login": "dev" },
            "draft": false,
            "base": { "ref": "master", "sha": base_sha, "repo": { "full_name": "octo/demo" } },
            "head": { "ref": "feature/faster-retries", "sha": head_sha, "repo": { "full_name": "octo/demo" } }
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

    /// The real app on a random port, over `repo_dir` (never cloned).
    async fn spawn_app(llm_base: &str, github_base: &str, repo_dir: PathBuf) -> (AppState, String) {
        let repo = Arc::new(Repo::new(repo_dir, DEFAULT_REPO_URL.to_string()));
        repo.ensure(false).await;
        let state = AppState {
            llm: Arc::new(Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{llm_base}/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            }),
            repo,
            github: Arc::new(GitHub::new(
                reqwest::Client::new(),
                github_base,
                Some("test-token".to_string()),
                "octo/demo".to_string(),
            )),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                repo_url: DEFAULT_REPO_URL.to_string(),
                repo_name: "octo/demo".to_string(),
                language: "English".to_string(),
            }),
            lock: Arc::new(Mutex::new(())),
        };
        let base = spawn(app(state.clone())).await;
        (state, base)
    }

    async fn post_review(base: &str, body: Value) -> Value {
        reqwest::Client::new()
            .post(format!("{base}/api/review"))
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    #[test]
    fn the_command_line_is_parsed() {
        let args = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(
            parse_cli(&args("--repo-dir pr --pr 12 --post")).unwrap(),
            Cli {
                pr: Some(12),
                repo_dir: Some(PathBuf::from("pr")),
                post: true,
                ..Default::default()
            }
        );
        assert_eq!(parse_cli(&args("--pr #3")).unwrap().pr, Some(3));
        let local = parse_cli(&args("--base origin/master --out review.md")).unwrap();
        assert_eq!(local.base.as_deref(), Some("origin/master"));
        assert_eq!(local.out, Some(PathBuf::from("review.md")));
        for bad in ["--post", "--pr", "--pr x", "--pr 1 --base master", "--force"] {
            assert!(parse_cli(&args(bad)).is_err(), "{bad} was accepted");
        }
        assert_eq!(
            repo_name("https://github.com/CaptainDmitro/AI-Advent-Challenge.git"),
            "CaptainDmitro/AI-Advent-Challenge"
        );
    }

    #[tokio::test]
    async fn a_pr_is_reviewed_with_rag_and_mcp_and_the_comment_is_posted_once() {
        let dir = temp_repo("pr");
        let base_sha = git_out(&dir, &["rev-parse", "master"]);
        let head_sha = git_out(&dir, &["rev-parse", "feature/faster-retries"]);
        // The checkout stays on master, as the clone on the VDS would: the
        // review has to use a worktree at the PR's head.
        git(&dir, &["checkout", "-q", "master"]);
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(REVIEW_JSON, requests.clone())).await;
        let comments = Arc::new(StdMutex::new(FakeGitHub::default()));
        let github = spawn(fake_github(pull_json(&base_sha, &head_sha), comments.clone())).await;
        let (state, _) = spawn_app(&llm, &github, dir.clone()).await;

        let pr = ReviewRequest {
            pr: Some(7),
            ..Default::default()
        };
        let reply = review(&state, pr.clone(), true).await;
        assert!(reply.error.is_none(), "{:?}", reply.error);
        assert_eq!(reply.checkout.as_deref(), Some("worktree"));
        let target = reply.target.as_ref().unwrap();
        assert_eq!((target.base_sha.as_str(), target.head_sha.as_str()), (base_sha.as_str(), head_sha.as_str()));
        assert_eq!(target.title, "Faster retries");

        // The diff.
        let paths: Vec<&str> = reply.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["07. Retry Lesson/src/jitter.rs", "07. Retry Lesson/src/main.rs"]);
        assert_eq!((reply.stats.additions, reply.stats.deletions), (7, 1));
        assert!(reply.queries.iter().any(|q| q.text.contains("MAX_RETRIES")), "{:?}", reply.queries);

        // RAG: the rules first, then docs and code from the rest of the repo.
        let first = &reply.fragments[0];
        assert_eq!((first.n, first.source.as_str(), first.pinned), (1, "docs/conventions.md", true));
        assert!(
            reply.fragments.iter().any(|f| f.source == "07. Retry Lesson/src/retry.rs"
                && f.kind == "code"
                && f.text.contains("fn retry_delay")),
            "{:#?}",
            reply.candidates
        );
        assert!(
            reply.fragments.iter().any(|f| f.source == "07. Retry Lesson/README.md" && f.kind == "doc"),
            "{:#?}",
            reply.candidates
        );
        // Nothing the diff already shows.
        assert!(!reply.fragments.iter().any(|f| f.source.ends_with("src/main.rs")
            && f.start_line <= 8
            && f.end_line >= 8));
        let index = reply.index.as_ref().unwrap();
        assert_eq!(index.documents, 7, "README, 2 docs, the lesson README and 3 sources");

        // MCP: the model read a file from the PR's head.
        assert_eq!(reply.tools.len(), 6);
        assert_eq!(reply.calls.len(), 1);
        let call = &reply.calls[0];
        assert_eq!(call.tool, "read_file");
        assert!(!call.is_error && call.result.contains("pub fn retry_delay"), "{}", call.result);

        // What the model was given.
        let seen = requests.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        let system = seen[0]["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("[1] docs/conventions.md › Conventions › Code"), "{system}");
        assert!(system.contains("fn retry_delay"), "{system}");
        let user = seen[0]["messages"][1]["content"].as_str().unwrap();
        assert!(user.contains("Pull request #7: \"Faster retries\" by dev"), "{user}");
        assert!(user.contains("    8 +     for attempt in 0..=MAX_RETRIES {"), "{user}");
        assert!(user.contains("      -     for attempt in 0..MAX_RETRIES {"), "{user}");
        assert_eq!(seen[0]["tools"].as_array().unwrap().len(), 6);
        let tool_message = seen[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap();
        assert!(tool_message["content"].as_str().unwrap().contains("retry_delay"));

        // The findings, checked.
        assert_eq!(reply.findings.len(), 2, "{:#?}", reply.findings);
        let bug = &reply.findings[0];
        assert_eq!(bug.category, review::Category::Bug);
        assert_eq!((bug.file.as_str(), bug.line), ("07. Retry Lesson/src/main.rs", Some(8)));
        assert_eq!(bug.sources, [1]);
        let arch = &reply.findings[1];
        assert_eq!((arch.line, arch.line_outside_diff), (None, Some(999)));
        assert_eq!(reply.dropped.len(), 2, "{:#?}", reply.dropped);
        assert!(reply.fragments[0].cited);

        // The comment.
        let md = reply.markdown.as_deref().unwrap();
        assert!(md.starts_with(review::MARKER), "{md}");
        assert!(md.contains("Off-by-one: MAX_RETRIES + 1 attempts"));
        assert!(md.contains(&format!(
            "https://github.com/octo/demo/blob/{head_sha}/07.%20Retry%20Lesson/src/main.rs#L8"
        )), "{md}");
        assert!(md.contains("_No recommendations._"));
        assert!(md.contains("`read_file(path: 07. Retry Lesson/src/retry.rs)`"), "{md}");
        let posted = reply.posted.as_ref().unwrap();
        assert_eq!(posted.action, "created");

        // The worktree is gone again.
        let worktrees = git_out(&dir, &["worktree", "list", "--porcelain"]);
        assert_eq!(worktrees.matches("worktree ").count(), 1, "{worktrees}");

        // A new push: the same comment is edited, not a second one added.
        let again = review(&state, pr, true).await;
        assert!(again.error.is_none(), "{:?}", again.error);
        let edited = again.posted.as_ref().unwrap();
        assert_eq!((edited.action.as_str(), edited.id), ("updated", posted.id));
        let github_state = comments.lock().unwrap();
        assert_eq!(github_state.comments.len(), 1);
        assert!(github_state.comments[0].1.contains("Off-by-one"));
        assert_eq!(github_state.auth, ["Bearer test-token"]);
        drop(github_state);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn the_page_reviews_two_revisions_without_posting() {
        let dir = temp_repo("page");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(REVIEW_JSON, requests.clone())).await;
        let comments = Arc::new(StdMutex::new(FakeGitHub::default()));
        let head_sha = git_out(&dir, &["rev-parse", "HEAD"]);
        let github = spawn(fake_github(pull_json("0", &head_sha), comments.clone())).await;
        let (_, base) = spawn_app(&llm, &github, dir.clone()).await;

        let reply = post_review(&base, json!({ "base": "master", "head": "HEAD" })).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["checkout"], "checkout", "the head is checked out: no worktree");
        assert!(reply["target"]["pr"].is_null());
        assert_eq!(reply["target"]["title"], "master...HEAD");
        assert_eq!(reply["findings"].as_array().unwrap().len(), 2);
        assert_eq!(reply["calls"][0]["tool"], "read_file");
        assert!(reply["posted"].is_null());
        assert!(comments.lock().unwrap().comments.is_empty(), "the page never posts");
        assert!(reply["markdown"].as_str().unwrap().starts_with(review::MARKER));

        // No changes: no model.
        let before = requests.lock().unwrap().len();
        let same = post_review(&base, json!({ "base": "HEAD", "head": "HEAD" })).await;
        assert!(same["error"].is_null(), "{same}");
        assert_eq!(same["summary"], "No changes between these commits.");
        assert_eq!(requests.lock().unwrap().len(), before);

        // A bad revision is an error string, not an HTTP error.
        let bad = post_review(&base, json!({ "base": "no-such-branch" })).await;
        assert!(bad["error"].as_str().unwrap().contains("no-such-branch"), "{bad}");
        let option = post_review(&base, json!({ "base": "--output=/tmp/x" })).await;
        assert!(option["error"].as_str().unwrap().contains("not a valid revision"), "{option}");

        let pulls: Value = reqwest::get(format!("{base}/api/pulls")).await.unwrap().json().await.unwrap();
        assert_eq!(pulls["pulls"][0]["number"], 7);
        let status: Value = reqwest::get(format!("{base}/api/status")).await.unwrap().json().await.unwrap();
        assert_eq!(status["repo"]["state"]["state"], "ready");
        assert_eq!(status["branch"]["branch"], "feature/faster-retries");
        assert_eq!(status["github"]["repo"], "octo/demo");
        let page = reqwest::get(&base).await.unwrap().text().await.unwrap();
        assert!(page.contains("/api/review"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_failed_review_is_posted_as_a_failure_and_cleans_up() {
        let dir = temp_repo("fail");
        let base_sha = git_out(&dir, &["rev-parse", "master"]);
        let head_sha = git_out(&dir, &["rev-parse", "HEAD"]);
        git(&dir, &["checkout", "-q", "master"]);
        let comments = Arc::new(StdMutex::new(FakeGitHub::default()));
        let github = spawn(fake_github(pull_json(&base_sha, &head_sha), comments.clone())).await;
        // Nothing listens on port 9: every LLM request fails.
        let (state, _) = spawn_app("http://127.0.0.1:9", &github, dir.clone()).await;

        let reply = review(&state, ReviewRequest { pr: Some(7), ..Default::default() }, true).await;
        let error = reply.error.as_deref().unwrap();
        assert!(error.contains("LLM request failed"), "{error}");
        assert!(!reply.fragments.is_empty(), "retrieval ran before the model");
        assert_eq!(reply.posted.as_ref().unwrap().action, "created");
        let posted = comments.lock().unwrap().comments[0].1.clone();
        assert!(posted.starts_with(review::MARKER) && posted.contains("The review failed"), "{posted}");
        let worktrees = git_out(&dir, &["worktree", "list", "--porcelain"]);
        assert_eq!(worktrees.matches("worktree ").count(), 1, "{worktrees}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

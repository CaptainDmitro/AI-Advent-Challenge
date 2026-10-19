mod diff;
mod mcp;
mod workspace;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::header,
    response::{Html, IntoResponse},
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

use diff::{FileDiff, patch, summary, tree_diff};
use mcp::files_service;
use workspace::Workspace;

/// A goal like "update the docs from the code" takes a search, a few reads
/// and a few edits, so more rounds than the question-answering lessons.
const MAX_TOOL_ROUNDS: usize = 16;
/// A tool result is clipped to this before it goes to the model.
const MAX_RESULT_CHARS: usize = 12_000;
const MAX_TASK_CHARS: usize = 4_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const RUN_TIMEOUT: Duration = Duration::from_secs(300);
/// The tools that change files.
const WRITE_TOOLS: [&str; 2] = ["write_file", "edit_file"];

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
// The MCP client side: the assistant's connection to the `files` server.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct ToolCallTrace {
    step: usize,
    /// 0 for the calls the assistant makes itself before the model runs.
    round: usize,
    tool: String,
    arguments: Value,
    /// Made by the assistant (`list_files` at the start), not chosen by the
    /// model.
    auto: bool,
    is_error: bool,
    result: String,
    structured: Option<Value>,
    ms: u128,
}

struct FilesClient {
    client: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

/// `initialize` + `tools/list` over Streamable HTTP, exactly as with a remote
/// server.
async fn connect(url: &str) -> Result<FilesClient, String> {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.to_string()),
    );
    let client = ()
        .serve(transport)
        .await
        .map_err(|e| format!("MCP handshake with {url} failed: {e}"))?;
    match client.list_all_tools().await {
        Ok(tools) => Ok(FilesClient { client, tools }),
        Err(e) => {
            let _ = client.cancel().await;
            Err(format!("tools/list failed: {e}"))
        }
    }
}

async fn connect_with_timeout(url: &str) -> Result<FilesClient, String> {
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

impl FilesClient {
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
                "The files server has no tool `{name}` (it has: {}).",
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
            "mcp {step}: {name} -> {} in {} ms",
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

/// OpenAI sends a tool call's arguments as a JSON string; some compatible
/// servers send the object itself. Malformed JSON is passed on as a string,
/// so the call fails with an error the model can read.
fn tool_arguments(raw: &Value) -> Value {
    match raw {
        Value::String(s) if s.trim().is_empty() => json!({}),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Value::Object(_) => raw.clone(),
        _ => json!({}),
    }
}

fn message_text(message: &Value) -> Option<String> {
    let text = message.get("content")?.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

// ---------------------------------------------------------------------------
// Running a task.
// ---------------------------------------------------------------------------

#[derive(Serialize, Default, Debug)]
struct Reply {
    message: String,
    /// The changes were shown and then rolled back.
    dry_run: bool,
    answer: Option<String>,
    error: Option<String>,
    calls: Vec<ToolCallTrace>,
    /// The tools the MCP server listed.
    tools: Vec<String>,
    /// What this task changed: the project before it vs. after it, computed
    /// from the files themselves, not from what the model says.
    changes: Vec<FileDiff>,
    /// The same as one unified diff.
    diff: String,
    /// One line: `2 files changed: ...`.
    summary: String,
    changed: bool,
    /// A dry run's changes were rolled back.
    reverted: bool,
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
    ws: Arc<Workspace>,
    settings: Arc<Settings>,
    /// One task at a time, so "what this task changed" is exactly that.
    run_lock: Arc<tokio::sync::Mutex<()>>,
}

fn system_prompt(files: &str, tools: &[String], dry_run: bool) -> String {
    let tools = if tools.is_empty() {
        "The files MCP server is not reachable right now, so there are no tools: say that the \
         project couldn't be opened and stop."
            .to_string()
    } else {
        format!(
            "The `files` MCP server: {}. list_files, read_file, search_files and get_diff \
             read; write_file and edit_file change files on disk and answer with the diff.",
            tools.join(", ")
        )
    };
    let mode = if dry_run {
        "\n\nThis is a DRY RUN: make the changes exactly as you would for real. The user is \
         shown their diff and then they are rolled back."
    } else {
        ""
    };
    format!(
        "You are a project assistant that works with the files of a software project. The user \
         gives you a goal (find, check, update, generate); you decide which files to open and do \
         the work through the tools.\n\n\
         PROJECT FILES (live, from list_files, fetched before you were asked):\n{files}\n\n\
         TOOLS: {tools}\n\n\
         Rules:\n\
         - Work from the files, never from memory or guesses: search_files to find where \
         something is, read_file before you state what a file says or change it. Open every \
         file the goal depends on (code, docs, rules), not just the first match.\n\
         - Finding usages: search, then list every place as `path:line` with what it does \
         there; say how many places and files there are.\n\
         - Checking rules or invariants: find the rules (e.g. a rules file), check each one \
         against the files, and report each as passed or violated with `path:line` evidence.\n\
         - Changing files: only when the goal asks you to create, update, fix or generate \
         something. Use edit_file for a targeted change (old_text copied exactly from \
         read_file, without the line numbers) and write_file for a new file. Keep each file's \
         language, format and style; follow existing examples (e.g. an existing ADR or \
         changelog entry). Update documentation from what the code actually does.\n\
         - Say that you changed a file only after the tool succeeded. If a tool fails, read \
         the error, fix the call and retry.\n\
         - End with a short summary: what you found or did, and the list of files you \
         changed (or \"no files changed\").\n\
         - Answer in the language of the user's message, in Markdown, concisely.{mode}"
    )
}

async fn ask(app: &AppState, req: AskRequest) -> Reply {
    let started = Instant::now();
    let message = req.message.trim().to_string();
    let mut reply = Reply {
        message: message.clone(),
        dry_run: req.dry_run,
        mcp_url: app.settings.mcp_url.clone(),
        ..Default::default()
    };
    if message.is_empty() {
        reply.error = Some("Describe the task.".to_string());
    } else if message.chars().count() > MAX_TASK_CHARS {
        reply.error = Some(format!(
            "The task is longer than {MAX_TASK_CHARS} characters."
        ));
    } else {
        let _turn = app.run_lock.lock().await;
        let before = app.ws.snapshot();
        match tokio::time::timeout(RUN_TIMEOUT, run(app, &message, req.dry_run, &mut reply)).await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => reply.error = Some(e),
            Err(_) => reply.error = Some(format!("Timed out after {}s.", RUN_TIMEOUT.as_secs())),
        }
        // What changed is read from the disk, even when the run failed
        // halfway: the edits it made are real.
        match (before, app.ws.snapshot()) {
            (Ok(before), Ok(after)) => {
                reply.changes = tree_diff(&before, &after);
                reply.diff = patch(&reply.changes);
                reply.summary = summary(&reply.changes);
                reply.changed = !reply.changes.is_empty();
                if req.dry_run && reply.changed {
                    match app.ws.restore(&before) {
                        Ok(_) => reply.reverted = true,
                        Err(e) => {
                            reply.error = Some(format!("The dry run couldn't be rolled back: {e}"))
                        }
                    }
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                if reply.error.is_none() {
                    reply.error = Some(format!("Can't read the project: {e}"));
                }
            }
        }
    }
    reply.ms = started.elapsed().as_millis();
    if let Some(e) = &reply.error {
        eprintln!("task failed: {e}");
    }
    reply
}

async fn run(app: &AppState, task: &str, dry_run: bool, reply: &mut Reply) -> Result<(), String> {
    let files = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(files) => Some(files),
        Err(e) => {
            reply.mcp_error = Some(e);
            None
        }
    };
    let result = work(app, files.as_ref(), task, dry_run, reply).await;
    if let Some(files) = files {
        files.close().await;
    }
    result
}

async fn work(
    app: &AppState,
    files: Option<&FilesClient>,
    task: &str,
    dry_run: bool,
    reply: &mut Reply,
) -> Result<(), String> {
    // The assistant opens the project itself: the file list goes into the
    // prompt, so the model starts from what is there.
    let tree = match files {
        Some(files) => {
            reply.tools = files.tool_names();
            let trace = files.call(1, 0, "list_files", json!({}), true).await;
            let text = if trace.is_error {
                format!("list_files failed: {}", trace.result)
            } else {
                trace.result.clone()
            };
            reply.calls.push(trace);
            text
        }
        None => format!(
            "Unknown: the files MCP server is unreachable ({}).",
            reply.mcp_error.as_deref().unwrap_or_default()
        ),
    };

    let tools = files.map(FilesClient::llm_tools).unwrap_or_default();
    let mut messages = vec![
        json!({ "role": "system", "content": system_prompt(&tree, &reply.tools, dry_run) }),
        json!({ "role": "user", "content": task }),
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
            reply.answer = Some(text);
            return Ok(());
        }
        messages.push(message);
        for call in tool_calls {
            let id = call["id"].as_str().unwrap_or_default().to_string();
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let arguments = tool_arguments(&call["function"]["arguments"]);
            let step = reply.calls.len() + 1;
            let trace = match files {
                Some(files) => files.call(step, round, name, arguments, false).await,
                None => ToolCallTrace {
                    step,
                    round,
                    tool: name.to_string(),
                    arguments,
                    auto: false,
                    is_error: true,
                    result: "The files MCP server is unreachable.".to_string(),
                    structured: None,
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
    dry_run: bool,
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Reply> {
    Json(ask(&app, req).await)
}

/// The changes since the baseline.
fn baseline_diff(app: &AppState) -> Result<Vec<FileDiff>, String> {
    Ok(tree_diff(app.ws.baseline(), &app.ws.snapshot()?))
}

/// The file tree for the page, with what changed since the baseline.
async fn files_api(State(app): State<AppState>) -> Json<Value> {
    let files = app.ws.list(None);
    let changes = baseline_diff(&app);
    let changes: Vec<Value> = changes
        .as_ref()
        .map(|c| {
            c.iter()
                .map(|d| json!({ "path": d.path, "status": d.status }))
                .collect()
        })
        .unwrap_or_default();
    match files {
        Ok(files) => Json(json!({ "files": files, "changes": changes, "error": null })),
        Err(e) => Json(json!({ "files": [], "changes": changes, "error": e })),
    }
}

#[derive(Deserialize)]
struct FileQuery {
    path: String,
}

async fn file_api(State(app): State<AppState>, Query(query): Query<FileQuery>) -> Json<Value> {
    match app.ws.read(&query.path) {
        Ok((path, content)) => Json(json!({ "path": path, "content": content, "error": null })),
        Err(e) => Json(json!({ "path": query.path, "content": null, "error": e })),
    }
}

async fn diff_api(State(app): State<AppState>) -> Json<Value> {
    match baseline_diff(&app) {
        Ok(diffs) => Json(json!({
            "summary": summary(&diffs),
            "patch": patch(&diffs),
            "files": diffs,
            "error": null,
        })),
        Err(e) => Json(json!({ "summary": "", "patch": "", "files": [], "error": e })),
    }
}

/// The same as a file to save and `git apply`.
async fn patch_api(State(app): State<AppState>) -> impl IntoResponse {
    let text = match baseline_diff(&app) {
        Ok(diffs) => patch(&diffs),
        Err(e) => format!("# error: {e}\n"),
    };
    (
        [
            (header::CONTENT_TYPE, "text/x-diff; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "inline; filename=\"changes.patch\"",
            ),
        ],
        text,
    )
}

async fn status_api(State(app): State<AppState>) -> Json<Value> {
    let tools = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(files) => {
            let names = files.tool_names();
            files.close().await;
            Ok(names)
        }
        Err(e) => Err(e),
    };
    let count = app.ws.list(None).map(|f| f.len()).unwrap_or(0);
    Json(json!({
        "workspace": {
            "root": app.ws.root().display().to_string(),
            "demo": app.ws.is_demo(),
            "files": count,
            "baseline_files": app.ws.baseline().len(),
        },
        "mcp": {
            "url": app.settings.mcp_url,
            "tools": tools.as_ref().ok(),
            "error": tools.as_ref().err(),
        },
        "model": app.settings.model,
    }))
}

/// Puts the demo project back, so the demo can be run again.
async fn reset_api(State(app): State<AppState>) -> Json<Value> {
    let _turn = app.run_lock.lock().await;
    match app.ws.reset() {
        Ok(touched) => Json(json!({ "ok": true, "restored": touched, "error": null })),
        Err(e) => Json(json!({ "ok": false, "restored": [], "error": e })),
    }
}

fn app(state: AppState) -> Router {
    let ws = state.ws.clone();
    Router::new()
        .route("/", get(index_page))
        .route("/api/status", get(status_api))
        .route("/api/files", get(files_api))
        .route("/api/file", get(file_api))
        .route("/api/diff", get(diff_api))
        .route("/api/diff.patch", get(patch_api))
        .route("/api/ask", post(ask_api))
        .route("/api/reset", post(reset_api))
        .with_state(state)
        .nest_service("/mcp/files", files_service(ws))
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
    // PROJECT_DIR: any existing folder. Otherwise the demo project in
    // `workspace/` (relative to the working directory: `~/apps/lesson-34/` on
    // the VDS), created from sample/ on the first start.
    let ws = match env_non_empty("PROJECT_DIR") {
        Some(dir) => Workspace::open_dir(PathBuf::from(dir)),
        None => Workspace::open_demo(PathBuf::from(
            env_non_empty("WORKSPACE_DIR").unwrap_or_else(|| "workspace".to_string()),
        )),
    };
    let ws = Arc::new(ws.unwrap_or_else(|e| {
        eprintln!("Error: can't open the project: {e}");
        std::process::exit(1);
    }));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    let local_port = listener.local_addr().unwrap().port();
    let mcp_url = env_non_empty("MCP_URL")
        .unwrap_or_else(|| format!("http://127.0.0.1:{local_port}/mcp/files"));

    println!(
        "Project in {} ({}). Listening on http://localhost:{port} (MCP server at /mcp/files, \
         tasks at /api/ask)",
        ws.root().display(),
        if ws.is_demo() { "demo" } else { "PROJECT_DIR" }
    );
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
        ws,
        settings: Arc::new(Settings { model, mcp_url }),
        run_lock: Arc::new(tokio::sync::Mutex::new(())),
    };
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    /// A fresh folder for a demo project.
    fn temp_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lesson34-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// A scripted OpenAI-compatible LLM: the n-th request of a conversation
    /// (n = assistant messages so far) gets the n-th round of tool calls, and
    /// the final `answer` after the last one.
    fn fake_llm(
        rounds: Vec<Vec<(&'static str, Value)>>,
        answer: &'static str,
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

    /// The real app on a random port, over a fresh demo project in `root`.
    async fn spawn_app(llm_base: &str, root: PathBuf) -> (AppState, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = AppState {
            llm: Arc::new(Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{llm_base}/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            }),
            ws: Arc::new(Workspace::open_demo(root).unwrap()),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                mcp_url: format!("{base}/mcp/files"),
            }),
            run_lock: Arc::new(tokio::sync::Mutex::new(())),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (state, base)
    }

    async fn post_ask(base: &str, message: &str, dry_run: bool) -> Value {
        reqwest::Client::new()
            .post(format!("{base}/api/ask"))
            .json(&json!({ "message": message, "dry_run": dry_run }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn get_json(url: String) -> Value {
        reqwest::get(url).await.unwrap().json().await.unwrap()
    }

    fn tools_of(reply: &Value) -> Vec<String> {
        reply["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["tool"].as_str().unwrap().to_string())
            .collect()
    }

    fn system_of(request: &Value) -> String {
        request["messages"][0]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    #[test]
    fn tool_arguments_come_as_a_string_or_an_object() {
        assert_eq!(tool_arguments(&json!("{\"path\": \"a.md\"}")), json!({ "path": "a.md" }));
        assert_eq!(tool_arguments(&json!({ "path": "a.md" })), json!({ "path": "a.md" }));
        assert_eq!(tool_arguments(&json!("")), json!({}));
        assert_eq!(tool_arguments(&Value::Null), json!({}));
        assert_eq!(tool_arguments(&json!("{oops")), json!("{oops"));
    }

    const DOCS_OLD: &str = "Body: `{\"title\": \"...\", \"body\": \"...\"}`. Returns the note, or `404`.";
    const DOCS_NEW: &str = "Body: `{\"title\": \"...\", \"body\": \"...\"}`. Returns the note, or `404`.\n\n\
        ### DELETE /notes/{note_id}\n\n\
        Deletes one of the user's own notes. Auth: required. Returns `204`.";

    #[tokio::test]
    async fn files_mcp_server_reads_searches_and_writes() {
        let root = temp_root("mcp");
        let (state, _) = spawn_app("http://127.0.0.1:9", root.clone()).await;
        let files = connect(&state.settings.mcp_url).await.unwrap();

        let mut names = files.tool_names();
        names.sort();
        assert_eq!(
            names,
            ["edit_file", "get_diff", "list_files", "read_file", "search_files", "write_file"]
        );
        for tool in &files.tools {
            assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
        }
        assert_eq!(files.llm_tools().len(), 6);

        // list_files
        let all = files.call(1, 0, "list_files", json!({}), true).await;
        assert!(!all.is_error, "{}", all.result);
        assert_eq!(all.structured.as_ref().unwrap()["total"], workspace::SAMPLE.len());
        assert!(all.result.contains("- app/routes/notes.py ("));
        let routes = files.call(2, 1, "list_files", json!({ "dir": "app/routes" }), false).await;
        assert_eq!(routes.structured.unwrap()["total"], 2);
        assert!(files.call(3, 1, "list_files", json!({ "dir": "../" }), false).await.is_error);

        // read_file
        let head = files
            .call(4, 1, "read_file", json!({ "path": "./app/routes/notes.py", "start_line": 1, "end_line": 3 }), false)
            .await;
        assert!(!head.is_error, "{}", head.result);
        let h = head.structured.unwrap();
        assert_eq!(h["path"], "app/routes/notes.py");
        assert_eq!((h["start_line"].as_u64(), h["end_line"].as_u64()), (Some(1), Some(3)));
        assert_eq!(h["truncated"], true);
        assert!(head.result.contains("   1 | \"\"\"Notes: list, read, create, update, delete and share.\"\"\""));
        assert!(files.call(5, 1, "read_file", json!({ "path": "app/main.py", "start_line": 999 }), false).await.is_error);
        assert!(files.call(6, 1, "read_file", json!({ "path": "nope.py" }), false).await.is_error);

        // search_files: every place `require_auth` is used.
        let usages = files.call(7, 1, "search_files", json!({ "pattern": "require_auth" }), false).await;
        let u = usages.structured.unwrap();
        assert_eq!(u["total"], 13);
        assert_eq!(u["files"].as_array().unwrap().len(), 6);
        let in_app = files
            .call(8, 1, "search_files", json!({ "pattern": "REQUIRE_AUTH", "path": "app" }), false)
            .await;
        let a = in_app.structured.unwrap();
        assert_eq!((a["total"].as_u64(), a["files"].as_array().unwrap().len()), (Some(10), 3));
        assert!(in_app.result.contains("app/routes/notes.py:30: def list_notes"), "{}", in_app.result);
        let routes = files
            .call(9, 1, "search_files", json!({ "pattern": "@router\\.(get|post|put|delete)", "regex": true, "extension": ".py" }), false)
            .await;
        assert_eq!(routes.structured.unwrap()["total"], 9);
        let none = files.call(10, 1, "search_files", json!({ "pattern": "legacy_hash", "path": "docs" }), false).await;
        assert!(!none.is_error && none.result.starts_with("No matches"));
        assert!(files.call(11, 1, "search_files", json!({ "pattern": "(", "regex": true }), false).await.is_error);
        assert!(files.call(12, 1, "search_files", json!({ "pattern": " " }), false).await.is_error);

        // edit_file: exactly one occurrence, or replace_all.
        let ambiguous = files
            .call(13, 1, "edit_file", json!({ "path": "docs/api.md", "old_text": "Auth: required.", "new_text": "Auth: yes." }), false)
            .await;
        assert!(ambiguous.is_error && ambiguous.result.contains("5 times"), "{}", ambiguous.result);
        let missing = files
            .call(14, 1, "edit_file", json!({ "path": "docs/api.md", "old_text": "  12 | nope", "new_text": "x" }), false)
            .await;
        assert!(missing.is_error && missing.result.contains("not found"));
        let edit = files
            .call(15, 1, "edit_file", json!({ "path": "docs/api.md", "old_text": DOCS_OLD, "new_text": DOCS_NEW }), false)
            .await;
        assert!(!edit.is_error, "{}", edit.result);
        let e = edit.structured.unwrap();
        assert_eq!((e["status"].as_str(), e["added"].as_u64(), e["removed"].as_u64()), (Some("modified"), Some(4), Some(0)));
        assert!(edit.result.contains("+### DELETE /notes/{note_id}"), "{}", edit.result);
        assert!(std::fs::read_to_string(root.join("docs/api.md")).unwrap().contains("### DELETE /notes/{note_id}"));

        // write_file: a new file, saved at once.
        let adr = files
            .call(16, 1, "write_file", json!({ "path": "docs/adr/0002-token-expiry.md", "content": "# ADR 0002: Token expiry\n" }), false)
            .await;
        assert!(!adr.is_error, "{}", adr.result);
        assert_eq!(adr.structured.unwrap()["status"], "added");
        assert!(adr.result.starts_with("Created `docs/adr/0002-token-expiry.md`"));
        assert!(root.join("docs/adr/0002-token-expiry.md").exists());
        for bad in ["../evil.md", ".env", "app/run.exe", "/tmp/x.md"] {
            let r = files.call(17, 1, "write_file", json!({ "path": bad, "content": "x" }), false).await;
            assert!(r.is_error, "{bad}: {}", r.result);
        }

        // get_diff: both changes against the sample.
        let diff = files.call(18, 1, "get_diff", json!({}), false).await;
        let d = diff.structured.unwrap();
        let changed: Vec<&str> = d["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
        assert_eq!(changed, ["docs/adr/0002-token-expiry.md", "docs/api.md"]);
        assert!(d["diff"].as_str().unwrap().contains("--- /dev/null\n+++ b/docs/adr/0002-token-expiry.md"));
        let only_api = files.call(19, 1, "get_diff", json!({ "path": "docs/api.md" }), false).await;
        assert_eq!(only_api.structured.unwrap()["files"].as_array().unwrap().len(), 1);

        let unknown = files.call(20, 1, "delete_file", json!({}), false).await;
        assert!(unknown.is_error && unknown.result.contains("no tool `delete_file`"));
        files.close().await;
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Scenario 1: "find every place this API is used" reads and changes
    /// nothing.
    #[tokio::test]
    async fn finding_usages_reads_and_changes_nothing() {
        let root = temp_root("usages");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![vec![("search_files", json!({ "pattern": "require_auth", "path": "app" }))]],
            "`require_auth` is used in 10 places in 3 files; `delete_note` (app/routes/notes.py:58) has no auth.",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, root.clone()).await;

        let reply = post_ask(&base, "Найди все места, где используется require_auth", false).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(tools_of(&reply), ["list_files", "search_files"]);
        assert_eq!(reply["calls"][0]["auto"], true);
        assert_eq!(reply["calls"][1]["auto"], false);
        assert_eq!(reply["calls"][1]["structured"]["total"], 10);
        assert_eq!(reply["changed"], false);
        assert!(reply["changes"].as_array().unwrap().is_empty());
        assert_eq!(reply["summary"], "No changes.");
        assert_eq!(reply["llm_rounds"], 2);

        let requests = requests.lock().unwrap();
        let system = system_of(&requests[0]);
        assert!(system.contains("PROJECT FILES") && system.contains("- app/routes/notes.py ("), "{system}");
        assert!(!system.contains("DRY RUN"));
        assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 6);
        // The search result went back to the model.
        let tool_message = requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap()
            .clone();
        assert!(tool_message["content"].as_str().unwrap().contains("app/routes/users.py:45: def me"));
        drop(requests);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Scenario 2: the docs are updated from the code, saved to disk, shown
    /// as a diff, and the demo can start over.
    #[tokio::test]
    async fn docs_are_updated_from_the_code_and_saved() {
        let root = temp_root("docs");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![
                vec![
                    ("search_files", json!({ "pattern": "@router\\.", "regex": true, "path": "app/routes" })),
                    ("read_file", json!({ "path": "docs/api.md" })),
                ],
                // A first try with the line number copied along: the error
                // goes back to the model, which corrects the call.
                vec![("edit_file", json!({ "path": "docs/api.md", "old_text": "  50 | ### PUT /notes/{note_id}", "new_text": "x" }))],
                vec![("edit_file", json!({ "path": "docs/api.md", "old_text": DOCS_OLD, "new_text": DOCS_NEW }))],
            ],
            "Added `DELETE /notes/{note_id}` to docs/api.md. Files changed: docs/api.md.",
            requests.clone(),
        ))
        .await;
        let (state, base) = spawn_app(&llm, root.clone()).await;

        let reply = post_ask(&base, "Обнови docs/api.md по коду", false).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(
            tools_of(&reply),
            ["list_files", "search_files", "read_file", "edit_file", "edit_file"]
        );
        assert_eq!(reply["calls"][3]["is_error"], true);
        assert_eq!(reply["calls"][4]["is_error"], false);
        assert_eq!(reply["changed"], true);
        assert_eq!(reply["reverted"], false);
        let changes = reply["changes"].as_array().unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0]["path"], "docs/api.md");
        assert_eq!(changes[0]["status"], "modified");
        let diff = reply["diff"].as_str().unwrap();
        assert!(diff.starts_with("--- a/docs/api.md\n+++ b/docs/api.md\n"), "{diff}");
        assert!(diff.contains("+### DELETE /notes/{note_id}") && !diff.contains("\n-"), "{diff}");
        assert!(reply["summary"].as_str().unwrap().starts_with("1 file changed: docs/api.md (modified"));

        // Saved: on disk, in the page's views and in the patch.
        let on_disk = std::fs::read_to_string(root.join("docs/api.md")).unwrap();
        assert!(on_disk.contains("Deletes one of the user's own notes."));
        let file = get_json(format!("{base}/api/file?path=docs/api.md")).await;
        assert_eq!(file["content"].as_str().unwrap(), on_disk);
        let tree = get_json(format!("{base}/api/files")).await;
        assert_eq!(tree["changes"], json!([{ "path": "docs/api.md", "status": "modified" }]));
        let all = get_json(format!("{base}/api/diff")).await;
        assert_eq!(all["patch"], reply["diff"]);
        let patch_file = reqwest::get(format!("{base}/api/diff.patch")).await.unwrap();
        assert!(patch_file.headers()[header::CONTENT_TYPE].to_str().unwrap().starts_with("text/x-diff"));
        assert_eq!(patch_file.text().await.unwrap(), diff);

        // Each tool result went back to the model, the failed one as an error.
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        let tool_messages: Vec<String> = requests[3]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["content"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(tool_messages.len(), 4);
        assert!(tool_messages[1].contains("`docs/api.md`, lines 1-"));
        assert!(tool_messages[2].starts_with("Error: old_text was not found"));
        assert!(tool_messages[3].contains("Updated `docs/api.md` (+4 -0 lines)"));
        drop(requests);

        // Reset puts the sample back for the next run.
        let reset: Value = reqwest::Client::new()
            .post(format!("{base}/api/reset"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(reset["ok"], true);
        assert_eq!(reset["restored"], json!(["docs/api.md"]));
        assert_eq!(state.ws.snapshot().unwrap(), workspace::sample());
        assert_eq!(get_json(format!("{base}/api/diff")).await["files"], json!([]));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Scenario 3: a new file (an ADR in the style of the existing one) in a
    /// dry run: the diff is returned and the project is left as it was.
    #[tokio::test]
    async fn a_dry_run_returns_the_diff_and_rolls_it_back() {
        let root = temp_root("dry-run");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![
                vec![("read_file", json!({ "path": "docs/adr/0001-use-fastapi.md" }))],
                vec![(
                    "write_file",
                    json!({ "path": "docs/adr/0002-token-expiry.md", "content": "# ADR 0002: Expiring tokens\n\n- Status: proposed\n" }),
                )],
            ],
            "Wrote docs/adr/0002-token-expiry.md following ADR 0001.",
            requests.clone(),
        ))
        .await;
        let (state, base) = spawn_app(&llm, root.clone()).await;

        let reply = post_ask(&base, "Напиши ADR про истекающие токены", true).await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["dry_run"], true);
        assert_eq!(tools_of(&reply), ["list_files", "read_file", "write_file"]);
        assert_eq!(reply["changed"], true);
        assert_eq!(reply["reverted"], true);
        assert_eq!(reply["changes"][0]["status"], "added");
        assert!(reply["diff"].as_str().unwrap().contains("+++ b/docs/adr/0002-token-expiry.md\n@@ -0,0 +1,3 @@"));
        assert!(!root.join("docs/adr/0002-token-expiry.md").exists());
        assert_eq!(state.ws.snapshot().unwrap(), workspace::sample());
        assert!(system_of(&requests.lock().unwrap()[0]).contains("DRY RUN"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn status_files_and_validation() {
        let root = temp_root("status");
        let (_, base) = spawn_app("http://127.0.0.1:9", root.clone()).await;
        let status = get_json(format!("{base}/api/status")).await;
        assert_eq!(status["mcp"]["tools"].as_array().unwrap().len(), 6);
        assert_eq!(status["workspace"]["demo"], true);
        assert_eq!(status["workspace"]["files"], workspace::SAMPLE.len());

        let tree = get_json(format!("{base}/api/files")).await;
        assert!(tree["error"].is_null(), "{tree}");
        assert_eq!(tree["files"].as_array().unwrap().len(), workspace::SAMPLE.len());
        assert_eq!(tree["changes"], json!([]));
        let config = get_json(format!("{base}/api/file?path=app/config.py")).await;
        assert!(config["content"].as_str().unwrap().contains("SECRET_KEY"));
        let escape = get_json(format!("{base}/api/file?path=../Cargo.toml")).await;
        assert!(escape["content"].is_null() && escape["error"].is_string());

        let empty = post_ask(&base, "   ", false).await;
        assert_eq!(empty["error"], "Describe the task.");
        // The LLM is down: the error is a string in the reply, nothing changed.
        let down = post_ask(&base, "Check the rules", false).await;
        assert!(down["error"].as_str().unwrap().contains("LLM request failed"), "{down}");
        assert_eq!(tools_of(&down), ["list_files"]);
        assert_eq!(down["changed"], false);

        let page = reqwest::get(&base).await.unwrap().text().await.unwrap();
        assert!(page.contains("/api/ask") && page.contains("/api/diff"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

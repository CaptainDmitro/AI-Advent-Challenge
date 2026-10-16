mod git;
mod index;
mod mcp;
mod rerank;

use std::path::PathBuf;
use std::sync::Arc;
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

use git::{BranchInfo, DEFAULT_REPO_URL, Repo, RepoState, default_repo_dir};
use index::{Document, Hit, Index, builtin_documents, env_non_empty, load_repo, local};
use mcp::project_service;
use rerank::Heuristic;

/// Candidates from the vector search, before reranking.
const TOP_K: usize = 20;
/// At most this many fragments go into the prompt.
const TOP_N: usize = 5;
/// The reranker score a fragment needs (lessons 23-25).
const THRESHOLD: f32 = 0.30;
const MAX_TOOL_ROUNDS: usize = 8;
/// A tool result is clipped to this before it goes to the model.
const MAX_RESULT_CHARS: usize = 12_000;
const MAX_QUESTION_CHARS: usize = 2_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const RUN_TIMEOUT: Duration = Duration::from_secs(180);

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
// The documentation index, kept in step with the checkout.
// ---------------------------------------------------------------------------

/// What the current index was built from.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Origin {
    /// The docs built into the binary: there is no checkout (yet).
    Builtin,
    /// The checkout, at HEAD + working-tree signature.
    Checkout { signature: String },
}

#[derive(Clone)]
struct Snapshot {
    index: Arc<Index>,
    heuristic: Arc<Heuristic>,
    origin: Origin,
    built_ms: u128,
}

#[derive(Serialize, Clone, Debug)]
struct IndexInfo {
    origin: Origin,
    documents: usize,
    chunks: usize,
    sources: Vec<String>,
    embedder: String,
    built_ms: u128,
    /// Why the last rebuild from the checkout failed, if it did.
    error: Option<String>,
}

impl Snapshot {
    fn info(&self, error: Option<String>) -> IndexInfo {
        IndexInfo {
            origin: self.origin.clone(),
            documents: self.index.documents,
            chunks: self.index.chunks.len(),
            sources: self.index.titles.iter().map(|(s, _)| s.clone()).collect(),
            embedder: self.index.embedder.id(),
            built_ms: self.built_ms,
            error,
        }
    }
}

async fn build(docs: Vec<Document>, origin: Origin) -> Result<Snapshot, String> {
    let started = Instant::now();
    let index = Index::build(docs, local()).await?;
    let heuristic = Heuristic::new(&index.chunks);
    let built_ms = started.elapsed().as_millis();
    println!(
        "indexed {} documents, {} chunks from {:?} in {built_ms} ms",
        index.documents,
        index.chunks.len(),
        origin
    );
    Ok(Snapshot {
        index: Arc::new(index),
        heuristic: Arc::new(heuristic),
        origin,
        built_ms,
    })
}

struct CorpusState {
    snapshot: Snapshot,
    error: Option<String>,
}

struct Corpus {
    state: Mutex<CorpusState>,
}

impl Corpus {
    async fn builtin() -> Result<Corpus, String> {
        let snapshot = build(builtin_documents(), Origin::Builtin).await?;
        Ok(Corpus {
            state: Mutex::new(CorpusState {
                snapshot,
                error: None,
            }),
        })
    }

    /// The index for the checkout as it is right now. When the checkout's
    /// signature (HEAD + working tree) differs from the one the index was
    /// built at, it is rebuilt first: a `git checkout`, a pull or an edited
    /// doc is picked up by the next question. A failed rebuild keeps the
    /// last good index and reports why.
    async fn current(&self, repo: &Repo, force: bool) -> (Snapshot, Option<String>) {
        let mut guard = self.state.lock().await;
        if repo.state() == RepoState::Ready {
            match repo.signature().await {
                Ok(signature) => {
                    let fresh = matches!(&guard.snapshot.origin,
                        Origin::Checkout { signature: built } if *built == signature);
                    if force || !fresh {
                        let rebuilt = match load_repo(&repo.dir) {
                            Ok(docs) => build(docs, Origin::Checkout { signature }).await,
                            Err(e) => Err(e),
                        };
                        match rebuilt {
                            Ok(snapshot) => {
                                guard.snapshot = snapshot;
                                guard.error = None;
                            }
                            Err(e) => guard.error = Some(e),
                        }
                    }
                }
                Err(e) => guard.error = Some(e),
            }
        }
        (guard.snapshot.clone(), guard.error.clone())
    }
}

// ---------------------------------------------------------------------------
// The MCP client side: the assistant's connection to the `project` server.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct ToolCallTrace {
    step: usize,
    /// 0 for the call the assistant makes itself before the model runs.
    round: usize,
    tool: String,
    arguments: Value,
    /// Made by the assistant (`git_branch` on every question), not chosen by
    /// the model.
    auto: bool,
    is_error: bool,
    result: String,
    structured: Option<Value>,
    ms: u128,
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
// The /help command.
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Command {
    /// `/help` alone (`None`), `/help <question>`, or plain text.
    Help(Option<String>),
    Unknown(String),
}

fn parse_command(message: &str) -> Command {
    let text = message.trim();
    if let Some(rest) = text.strip_prefix("/help")
        && (rest.is_empty() || rest.starts_with(char::is_whitespace))
    {
        let question = rest.trim();
        return Command::Help((!question.is_empty()).then(|| question.to_string()));
    }
    if text.starts_with('/') {
        let name = text.split_whitespace().next().unwrap_or(text);
        return Command::Unknown(name.to_string());
    }
    Command::Help((!text.is_empty()).then(|| text.to_string()))
}

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
    /// `help` or `unknown`.
    command: String,
    question: Option<String>,
    /// What the documentation was searched with.
    search_query: Option<String>,
    answer: Option<String>,
    error: Option<String>,
    /// From the `project` MCP server's `git_branch`.
    branch: Option<BranchInfo>,
    sources: Vec<Source>,
    candidates: Vec<Candidate>,
    calls: Vec<ToolCallTrace>,
    /// The tools the MCP server listed.
    tools: Vec<String>,
    mcp_url: String,
    mcp_error: Option<String>,
    llm_rounds: usize,
    index: Option<IndexInfo>,
    ms: u128,
}

#[derive(Clone)]
struct Settings {
    model: String,
    repo_url: String,
    repo_name: String,
    mcp_url: String,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<Llm>,
    repo: Arc<Repo>,
    corpus: Arc<Corpus>,
    settings: Arc<Settings>,
}

const TRANSLATE_PROMPT: &str = "You turn a developer's question about a software repository \
    into a short English search query for its documentation (README, AGENTS.md, DEPLOYMENT.md, \
    docs/ and lesson READMEs). Keep file names, commands, numbers and identifiers exactly as \
    written. Reply with the query only, one line, no quotes.";

/// The documentation is in English and the index is lexical, so a question
/// in Cyrillic is searched together with an English rewrite of it.
fn needs_translation(question: &str) -> bool {
    question
        .chars()
        .any(|c| matches!(c, '\u{0400}'..='\u{04FF}'))
}

fn system_prompt(repo_name: &str, branch_text: &str, fragments: &[Source], tools: &[String]) -> String {
    let mut docs = String::new();
    if fragments.is_empty() {
        docs.push_str(
            "(No documentation fragment passed the relevance threshold for this question. \
             Use the tools, or say the documentation doesn't cover it.)\n",
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
        "The project MCP server is not reachable right now, so there are no tools: answer from \
         the documentation only."
            .to_string()
    } else {
        format!(
            "The `project` MCP server gives read-only access to the git checkout: {}. Call them \
             when the question is about the current state of the code (which files exist, \
             what is in a file, recent commits, what changed on this branch, uncommitted \
             edits) or when the fragments don't cover it. Don't call a tool whose answer is \
             already above.",
            tools.join(", ")
        )
    };
    format!(
        "You are the developer assistant of the repository {repo_name}. You answer /help \
         questions about the project: its structure, conventions, lessons, how to build, test \
         and deploy it.\n\n\
         PROJECT STATE (live, from the `project` MCP server's git_branch tool):\n{branch_text}\n\n\
         DOCUMENTATION fragments retrieved for this question from README.md, AGENTS.md, \
         DEPLOYMENT.md, docs/ and the lesson READMEs:\n{docs}\n\
         TOOLS: {tools}\n\n\
         Rules:\n\
         - Base every fact on the fragments or on tool results. Cite a fragment as [n] right \
         after the fact it supports (one number per bracket). Name the tool when a fact comes \
         from one, e.g. \"(list_files)\".\n\
         - If neither the documentation nor the tools answer the question, say so plainly. \
         Never invent files, commands, settings or ports.\n\
         - Copy paths, commands, ports and names exactly as the sources write them.\n\
         - Answer in the language of the question, in Markdown, concisely: the answer first, \
         then the details. Mention the current branch when the answer depends on it."
    )
}

async fn help(app: &AppState, message: &str) -> Reply {
    let started = Instant::now();
    let mut reply = Reply {
        message: message.trim().to_string(),
        mcp_url: app.settings.mcp_url.clone(),
        ..Default::default()
    };
    match parse_command(message) {
        Command::Unknown(name) => {
            reply.command = "unknown".to_string();
            reply.error = Some(format!(
                "Unknown command `{name}`. Use `/help` to see what I can do, or \
                 `/help <question>` to ask about the project."
            ));
        }
        Command::Help(question) => {
            reply.command = "help".to_string();
            if question
                .as_ref()
                .is_some_and(|q| q.chars().count() > MAX_QUESTION_CHARS)
            {
                reply.error = Some(format!(
                    "The question is longer than {MAX_QUESTION_CHARS} characters."
                ));
            } else {
                reply.question = question.clone();
                match tokio::time::timeout(RUN_TIMEOUT, run_help(app, question, &mut reply)).await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => reply.error = Some(e),
                    Err(_) => {
                        reply.error = Some(format!("Timed out after {}s.", RUN_TIMEOUT.as_secs()))
                    }
                }
            }
        }
    }
    reply.ms = started.elapsed().as_millis();
    if let Some(e) = &reply.error {
        eprintln!("/help failed: {e}");
    }
    reply
}

async fn run_help(app: &AppState, question: Option<String>, reply: &mut Reply) -> Result<(), String> {
    let (snapshot, index_error) = app.corpus.current(&app.repo, false).await;
    reply.index = Some(snapshot.info(index_error));

    // MCP: connect to the project server, and ask it for the branch first,
    // every time, so the answer always knows where the project stands.
    let project = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(project) => Some(project),
        Err(e) => {
            reply.mcp_error = Some(e);
            None
        }
    };
    let branch_text = match &project {
        Some(project) => {
            reply.tools = project.tool_names();
            let trace = project.call(1, 0, "git_branch", json!({}), true).await;
            let text = if trace.is_error {
                format!("git_branch failed: {}", trace.result)
            } else {
                reply.branch = trace
                    .structured
                    .clone()
                    .and_then(|v| serde_json::from_value(v).ok());
                trace.result.clone()
            };
            reply.calls.push(trace);
            text
        }
        None => format!(
            "Unknown: the project MCP server is unreachable ({}).",
            reply.mcp_error.as_deref().unwrap_or_default()
        ),
    };

    let result = match question {
        None => {
            let text = overview(app, reply);
            reply.answer = Some(text);
            Ok(())
        }
        Some(question) => answer(app, &snapshot, project.as_ref(), &question, &branch_text, reply).await,
    };
    if let Some(project) = project {
        project.close().await;
    }
    result
}

/// `/help` with no question: what the assistant is and can do. No LLM call.
fn overview(app: &AppState, reply: &Reply) -> String {
    let branch = match &reply.branch {
        Some(b) if b.detached => format!("detached HEAD at `{}`", b.head),
        Some(b) => format!("`{}` (HEAD `{}`)", b.branch, b.head),
        None => format!(
            "unknown ({})",
            reply
                .calls
                .first()
                .map(|c| c.result.as_str())
                .or(reply.mcp_error.as_deref())
                .unwrap_or("no answer")
        ),
    };
    let (documents, chunks, origin) = reply.index.as_ref().map_or((0, 0, String::new()), |i| {
        (
            i.documents,
            i.chunks,
            match &i.origin {
                Origin::Builtin => "the docs built into this binary (no checkout yet)".to_string(),
                Origin::Checkout { .. } => format!("the checkout in `{}`", app.repo.dir.display()),
            },
        )
    });
    let tools = if reply.tools.is_empty() {
        "none (the MCP server is unreachable)".to_string()
    } else {
        reply
            .tools
            .iter()
            .map(|t| format!("`{t}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "**Developer assistant for `{repo}`.** Current branch: {branch}, from the `project` MCP \
         server.\n\n\
         **Commands**\n\
         - `/help`: this overview.\n\
         - `/help <question>`: ask about the project (structure, conventions, lessons, build \
         and deploy). Plain text without a command works the same way.\n\n\
         **How an answer is made**\n\
         1. `git_branch` on the `project` MCP server, every time.\n\
         2. A search of the documentation: {documents} documents, {chunks} chunks from \
         {origin}, reranked, fragments above {THRESHOLD:.2}.\n\
         3. The model ({model}) answers from those fragments with [n] citations and can call \
         the MCP tools: {tools}.\n\n\
         **Try**\n\
         - `/help How is the repository structured?`\n\
         - `/help How do I add a new lesson?`\n\
         - `/help Which port and service name will lesson 31 get on the VDS?`\n\
         - `/help What changed on this branch compared to master?`\n\
         - `/help Какие уроки используют MCP?`",
        repo = app.settings.repo_name,
        model = app.settings.model,
    )
}

async fn answer(
    app: &AppState,
    snapshot: &Snapshot,
    project: Option<&Project>,
    question: &str,
    branch_text: &str,
    reply: &mut Reply,
) -> Result<(), String> {
    let search_query = if needs_translation(question) {
        let messages = [
            json!({ "role": "system", "content": TRANSLATE_PROMPT }),
            json!({ "role": "user", "content": question }),
        ];
        match app.llm.chat(&messages, &[]).await.ok().and_then(|m| message_text(&m)) {
            Some(english) => format!("{question} {english}"),
            None => question.to_string(),
        }
    } else {
        question.to_string()
    };
    reply.search_query = Some(search_query.clone());

    // RAG: vector search, heuristic rerank, threshold (lessons 21-25).
    let hits = snapshot.index.search(&search_query, TOP_K).await?;
    let terms = snapshot.heuristic.terms(&search_query);
    let mut scored: Vec<(f32, Hit)> = hits
        .into_iter()
        .map(|h| (snapshot.heuristic.score(&terms, &h).score, h))
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

    let tools = project.map(Project::llm_tools).unwrap_or_default();
    let mut messages = vec![
        json!({
            "role": "system",
            "content": system_prompt(&app.settings.repo_name, branch_text, &reply.sources, &reply.tools),
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
            let trace = match project {
                Some(project) => project.call(step, round, name, arguments, false).await,
                None => ToolCallTrace {
                    step,
                    round,
                    tool: name.to_string(),
                    arguments,
                    auto: false,
                    is_error: true,
                    result: "The project MCP server is unreachable.".to_string(),
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
struct HelpRequest {
    message: String,
}

async fn help_api(State(app): State<AppState>, Json(req): Json<HelpRequest>) -> Json<Reply> {
    Json(help(&app, &req.message).await)
}

async fn status_api(State(app): State<AppState>) -> Json<Value> {
    let (snapshot, index_error) = app.corpus.current(&app.repo, false).await;
    let branch = app.repo.branch().await;
    let tools = match connect_with_timeout(&app.settings.mcp_url).await {
        Ok(project) => {
            let names = project.tool_names();
            project.close().await;
            Ok(names)
        }
        Err(e) => Err(e),
    };
    Json(json!({
        "repo": {
            "name": app.settings.repo_name,
            "url": app.settings.repo_url,
            "dir": app.repo.dir.display().to_string(),
            "state": app.repo.state(),
        },
        "branch": branch.as_ref().ok(),
        "branch_error": branch.as_ref().err(),
        "index": snapshot.info(index_error),
        "mcp": {
            "url": app.settings.mcp_url,
            "tools": tools.as_ref().ok(),
            "error": tools.as_ref().err(),
        },
        "model": app.settings.model,
        "threshold": THRESHOLD,
    }))
}

async fn reindex_api(State(app): State<AppState>) -> Json<IndexInfo> {
    let (snapshot, error) = app.corpus.current(&app.repo, true).await;
    Json(snapshot.info(error))
}

fn app(state: AppState) -> Router {
    let repo = state.repo.clone();
    Router::new()
        .route("/", get(index_page))
        .route("/api/status", get(status_api))
        .route("/api/help", post(help_api))
        .route("/api/reindex", post(reindex_api))
        .with_state(state)
        .nest_service("/mcp/project", project_service(repo))
}

fn print_reply(reply: &Reply) {
    if let Some(b) = &reply.branch {
        println!("branch: {} (HEAD {})", b.branch, b.head);
    }
    println!();
    match (&reply.answer, &reply.error) {
        (Some(answer), _) => println!("{answer}"),
        (None, Some(error)) => println!("Error: {error}"),
        (None, None) => println!("(no answer)"),
    }
    if !reply.sources.is_empty() {
        println!("\nSources:");
        for s in &reply.sources {
            println!(
                "  [{}] {} › {} (score {:.2}{})",
                s.n,
                s.source,
                s.section,
                s.score,
                if s.cited { ", cited" } else { "" }
            );
        }
    }
    if !reply.calls.is_empty() {
        println!("\nMCP calls:");
        for c in &reply.calls {
            println!(
                "  {}. {}({}){}{}",
                c.step,
                c.tool,
                c.arguments,
                if c.auto { " [auto]" } else { "" },
                if c.is_error { " -> error" } else { "" }
            );
        }
    }
}

fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build HTTP client")
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = args.first().is_some_and(|a| a == "help");
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let base_url =
        env_non_empty("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.deepseek.com".to_string());
    let model = env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key = env_non_empty("OPENAI_API_KEY").unwrap_or_else(|| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    // Relative to the working directory: under `cargo run` the repo this
    // lesson sits in; `~/apps/lesson-31/repo` on the VDS, cloned on first start.
    let repo_dir = env_non_empty("REPO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(default_repo_dir);
    let repo_url = env_non_empty("REPO_URL").unwrap_or_else(|| DEFAULT_REPO_URL.to_string());
    let clone = env_non_empty("REPO_CLONE").is_none_or(|v| v.trim() != "off");

    let bind = if cli {
        "127.0.0.1:0".to_string()
    } else {
        format!("0.0.0.0:{port}")
    };
    let listener = tokio::net::TcpListener::bind(bind.as_str()).await.unwrap();
    let local_port = listener.local_addr().unwrap().port();
    let mcp_url = env_non_empty("MCP_URL")
        .unwrap_or_else(|| format!("http://127.0.0.1:{local_port}/mcp/project"));

    let repo = Arc::new(Repo::new(repo_dir, repo_url.clone()));
    let corpus = Arc::new(Corpus::builtin().await.unwrap_or_else(|e| {
        eprintln!("Error: can't index the built-in docs: {e}");
        std::process::exit(1);
    }));
    let state = AppState {
        llm: Arc::new(Llm {
            http: http_client(Duration::from_secs(120)),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model: model.clone(),
        }),
        repo: repo.clone(),
        corpus: corpus.clone(),
        settings: Arc::new(Settings {
            model,
            repo_name: repo_name(&repo_url),
            repo_url,
            mcp_url,
        }),
    };
    println!(
        "repository {} in {}",
        state.settings.repo_name,
        repo.dir.display()
    );

    if cli {
        // `cargo run -- help [question]`: the same /help, with the MCP server
        // on a private port for the length of one question.
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        repo.ensure(clone).await;
        let message = format!("/help {}", args[1..].join(" "));
        print_reply(&help(&state, &message).await);
        return;
    }

    // The clone can take a while; the page and the built-in docs work
    // meanwhile, and the index switches to the checkout once it's there.
    tokio::spawn(async move {
        repo.ensure(clone).await;
        let (_, error) = corpus.current(&repo, false).await;
        if let Some(e) = error {
            eprintln!("indexing the checkout failed: {e}");
        }
    });
    println!(
        "Listening on http://localhost:{port} (MCP server at /mcp/project, /help at /api/help)"
    );
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson31-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn git(dir: &Path, args: &[&str]) {
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
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    const ARCHITECTURE: &str = "# Architecture\n\n## Repository layout\n\nWhere does each \
        lesson live? Each lesson folder lives at the repository root, named `NN. Title`. \
        Every lesson folder is an independent Cargo crate with `src/main.rs` and \
        `static/index.html`; there is no Cargo workspace.\n\n## Deploy\n\nA deployed lesson \
        listens on port 4000 plus its number, as the systemd user service \
        ai-advent-lesson-NN.\n";

    /// A small project: master with the docs and one lesson, and a feature
    /// branch checked out with one more commit.
    fn temp_repo(name: &str) -> PathBuf {
        let dir = temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, "README.md", "# Demo Project\n\nA monorepo of course lessons.\n");
        write(
            &dir,
            "AGENTS.md",
            "# Agent instructions\n\n## Adding a new lesson\n\nCreate the next numbered \
             folder and add a row to the root README table of contents.\n",
        );
        write(&dir, "docs/architecture.md", ARCHITECTURE);
        write(
            &dir,
            "07. Context Persistence/README.md",
            "# 07. Context Persistence\n\nThe agent saves its conversation history to a JSON file.\n",
        );
        write(&dir, "07. Context Persistence/src/main.rs", "fn main() {}\n");
        git(&dir, &["init", "-q"]);
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "Initial commit"]);
        git(&dir, &["checkout", "-q", "-b", "feature/help-command"]);
        write(
            &dir,
            "docs/help.md",
            "# Help command\n\nThe /help command answers questions about the project.\n",
        );
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "Add the help command docs"]);
        dir
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

    /// The real app on a random port, over `repo_dir` (never cloned).
    async fn spawn_app(llm_base: &str, repo_dir: PathBuf) -> (AppState, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
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
            corpus: Arc::new(Corpus::builtin().await.unwrap()),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                repo_url: DEFAULT_REPO_URL.to_string(),
                repo_name: repo_name(DEFAULT_REPO_URL),
                mcp_url: format!("{base}/mcp/project"),
            }),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (state, base)
    }

    async fn ask(base: &str, message: &str) -> Value {
        reqwest::Client::new()
            .post(format!("{base}/api/help"))
            .json(&json!({ "message": message }))
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

    #[test]
    fn commands_are_parsed() {
        assert_eq!(parse_command("/help"), Command::Help(None));
        assert_eq!(parse_command("  /help   "), Command::Help(None));
        assert_eq!(
            parse_command("/help where are the docs?"),
            Command::Help(Some("where are the docs?".to_string()))
        );
        assert_eq!(
            parse_command("where are the docs?"),
            Command::Help(Some("where are the docs?".to_string()))
        );
        assert_eq!(parse_command("/helpme"), Command::Unknown("/helpme".to_string()));
        assert_eq!(parse_command("/deploy now"), Command::Unknown("/deploy".to_string()));
        assert_eq!(
            repo_name("https://github.com/CaptainDmitro/AI-Advent-Challenge.git"),
            "CaptainDmitro/AI-Advent-Challenge"
        );
        assert_eq!(repo_name("git@github.com:octo/demo.git"), "octo/demo");
        assert!(needs_translation("Как устроен репозиторий?"));
        assert!(!needs_translation("How is the repo laid out?"));
    }

    #[tokio::test]
    async fn project_mcp_server_reads_the_checkout() {
        let dir = temp_repo("mcp");
        let (state, _) = spawn_app("http://127.0.0.1:9", dir.clone()).await;
        let project = connect(&state.settings.mcp_url).await.unwrap();

        let mut names = project.tool_names();
        names.sort();
        assert_eq!(
            names,
            ["git_branch", "git_diff", "git_log", "git_status", "list_files", "read_file"]
        );
        for tool in &project.tools {
            assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
        }
        assert_eq!(project.llm_tools().len(), 6);

        let branch = project.call(1, 0, "git_branch", json!({}), true).await;
        assert!(!branch.is_error, "{}", branch.result);
        let info: BranchInfo = serde_json::from_value(branch.structured.unwrap()).unwrap();
        assert_eq!(info.branch, "feature/help-command");
        assert!(!info.detached);
        assert_eq!(info.branches, ["feature/help-command", "master"]);
        assert!(branch.result.contains("Current branch: feature/help-command"));

        let docs = project.call(2, 1, "list_files", json!({ "path": "docs" }), false).await;
        assert!(!docs.is_error, "{}", docs.result);
        assert_eq!(docs.structured.as_ref().unwrap()["total"], 2);
        assert!(docs.result.contains("docs/architecture.md") && docs.result.contains("docs/help.md"));
        let all = project.call(3, 1, "list_files", json!({}), false).await;
        assert!(all.result.contains("07. Context Persistence/ (2)"), "{}", all.result);
        let readmes = project
            .call(4, 1, "list_files", json!({ "contains": "readme" }), false)
            .await;
        assert_eq!(readmes.structured.as_ref().unwrap()["total"], 2);

        let file = project
            .call(5, 1, "read_file", json!({ "path": "docs/architecture.md" }), false)
            .await;
        assert!(!file.is_error && file.result.contains("Repository layout"), "{}", file.result);
        for bad in ["../outside.md", ".git/config", "/etc/passwd", "docs", "nope.md"] {
            let r = project.call(6, 1, "read_file", json!({ "path": bad }), false).await;
            assert!(r.is_error, "{bad} was read: {}", r.result);
        }

        let log = project.call(7, 1, "git_log", json!({ "limit": 5 }), false).await;
        assert_eq!(log.structured.as_ref().unwrap()["commits"].as_array().unwrap().len(), 2);
        assert!(log.result.contains("Add the help command docs") && log.result.contains("Initial commit"));
        let only_docs = project
            .call(8, 1, "git_log", json!({ "path": "docs/help.md" }), false)
            .await;
        assert_eq!(only_docs.structured.as_ref().unwrap()["commits"].as_array().unwrap().len(), 1);
        assert!(project.call(9, 1, "git_log", json!({ "limit": 99 }), false).await.is_error);

        let branch_diff = project.call(10, 1, "git_diff", json!({ "base": "master" }), false).await;
        assert!(branch_diff.result.contains("docs/help.md"), "{}", branch_diff.result);
        assert!(!branch_diff.result.contains("architecture.md"));
        for bad in ["--output=/tmp/x", "master..HEAD"] {
            let r = project.call(11, 1, "git_diff", json!({ "base": bad }), false).await;
            assert!(r.is_error, "{bad} was accepted");
        }

        let clean = project.call(12, 1, "git_status", json!({}), false).await;
        assert!(clean.result.contains("clean"), "{}", clean.result);
        write(&dir, "README.md", "# Demo Project\n\nA monorepo of course lessons.\nUncommitted line\n");
        let dirty = project.call(13, 1, "git_status", json!({}), false).await;
        assert!(dirty.result.contains("M README.md"), "{}", dirty.result);
        let edits = project.call(14, 1, "git_diff", json!({}), false).await;
        assert!(edits.result.contains("+Uncommitted line"), "{}", edits.result);

        let unknown = project.call(15, 1, "rm_rf", json!({}), false).await;
        assert!(unknown.is_error && unknown.result.contains("no tool `rm_rf`"));
        project.close().await;
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn help_alone_shows_the_commands_and_the_branch_without_the_llm() {
        let dir = temp_repo("overview");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(vec![], "unused", "unused", requests.clone())).await;
        let (_, base) = spawn_app(&llm, dir.clone()).await;

        let reply = ask(&base, "/help").await;
        assert!(reply["error"].is_null(), "{reply}");
        let answer = reply["answer"].as_str().unwrap();
        assert!(answer.contains("/help <question>"), "{answer}");
        assert!(answer.contains("`feature/help-command`"), "{answer}");
        assert!(answer.contains("5 documents"), "{answer}");
        assert_eq!(reply["branch"]["branch"], "feature/help-command");
        assert_eq!(reply["calls"][0]["tool"], "git_branch");
        assert_eq!(reply["calls"][0]["auto"], true);
        assert_eq!(reply["tools"].as_array().unwrap().len(), 6);
        assert_eq!(reply["index"]["origin"]["kind"], "checkout");
        assert!(requests.lock().unwrap().is_empty(), "the overview needs no model");

        let unknown = ask(&base, "/deploy now").await;
        assert!(unknown["error"].as_str().unwrap().contains("Unknown command `/deploy`"));
        assert!(unknown["calls"].as_array().unwrap().is_empty());
        assert!(requests.lock().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn help_answers_from_the_docs_and_the_mcp_tools() {
        let dir = temp_repo("answer");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![vec![("list_files", json!({ "path": "docs" }))]],
            "Each lesson is its own folder at the root, an independent Cargo crate [1]. \
             The docs are architecture.md and help.md (list_files).",
            "unused",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, dir.clone()).await;

        let reply = ask(&base, "/help Where does each lesson live, and is there a Cargo workspace?").await;
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["branch"]["branch"], "feature/help-command");

        // The branch came from MCP before the model ran; then the model's own call.
        let calls = reply["calls"].as_array().unwrap();
        let tools: Vec<&str> = calls.iter().map(|c| c["tool"].as_str().unwrap()).collect();
        assert_eq!(tools, ["git_branch", "list_files"]);
        assert_eq!(calls[0]["auto"], true);
        assert_eq!(calls[1]["round"], 1);
        assert!(calls.iter().all(|c| c["is_error"] == false), "{calls:?}");

        // RAG: the architecture doc is retrieved, given to the model and cited.
        assert_eq!(reply["candidates"][0]["source"], "docs/architecture.md", "{reply}");
        let sources = reply["sources"].as_array().unwrap();
        assert_eq!(sources[0]["source"], "docs/architecture.md", "{reply}");
        assert_eq!(sources[0]["cited"], true);
        assert_eq!(reply["llm_rounds"], 2);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let system = system_of(&requests[0]);
        assert!(system.contains("Current branch: feature/help-command"), "{system}");
        assert!(system.contains("[1] docs/architecture.md"), "{system}");
        assert!(system.contains("there is no Cargo workspace"), "{system}");
        assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 6);
        // The tool's result went back to the model.
        let tool_message = requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap();
        assert!(tool_message["content"].as_str().unwrap().contains("docs/help.md"));
        drop(requests);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn the_index_follows_the_checkout() {
        let dir = temp_repo("follow");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(vec![], "ok", "unused", requests)).await;
        let (_, base) = spawn_app(&llm, dir.clone()).await;

        let first = ask(&base, "/help what is the zanzibar flag?").await;
        assert_eq!(first["index"]["documents"], 5, "{}", first["index"]);
        assert!(
            !first["sources"].as_array().unwrap().iter().any(|s| s["source"] == "docs/zanzibar.md")
        );

        // A new doc, not even committed: picked up by the next question.
        write(
            &dir,
            "docs/zanzibar.md",
            "# Zanzibar flag\n\nThe zanzibar flag switches the assistant to quux mode.\n",
        );
        let second = ask(&base, "/help what is the zanzibar flag?").await;
        assert_eq!(second["index"]["documents"], 6);
        assert_ne!(second["index"]["origin"], first["index"]["origin"]);
        assert_eq!(second["sources"][0]["source"], "docs/zanzibar.md", "{second}");

        // Another branch: its docs, its name.
        std::fs::remove_file(dir.join("docs/zanzibar.md")).unwrap();
        git(&dir, &["checkout", "-q", "master"]);
        let third = ask(&base, "/help what is the zanzibar flag?").await;
        assert_eq!(third["branch"]["branch"], "master");
        assert_eq!(third["index"]["documents"], 4, "docs/help.md is only on the feature branch");
        assert!(third["sources"].as_array().unwrap().is_empty(), "{third}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_russian_question_is_searched_in_english_too() {
        let dir = temp_repo("russian");
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![],
            "Каждый урок — отдельная папка [1].",
            "where does each lesson folder live cargo workspace",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, dir.clone()).await;

        let reply = ask(&base, "/help Где лежат уроки и есть ли Cargo workspace?").await;
        assert!(reply["error"].is_null(), "{reply}");
        let query = reply["search_query"].as_str().unwrap();
        assert!(query.starts_with("Где лежат уроки") && query.contains("lesson folder live"));
        assert_eq!(reply["sources"][0]["source"], "docs/architecture.md", "{reply}");
        let requests = requests.lock().unwrap();
        assert!(requests[0].get("tools").is_none(), "the first call is the translation");
        assert_eq!(requests.len(), 2);
        drop(requests);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn without_a_checkout_it_says_so_and_uses_the_built_in_docs() {
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(
            vec![vec![("delete_everything", json!({}))]],
            "Lessons deploy to port 4000 + NN [1].",
            "unused",
            requests.clone(),
        ))
        .await;
        let (_, base) = spawn_app(&llm, temp_dir("no-checkout")).await;

        let reply = ask(&base, "/help Which port does a deployed lesson get?").await;
        assert!(reply["error"].is_null(), "{reply}");
        assert!(reply["branch"].is_null());
        assert_eq!(reply["index"]["origin"]["kind"], "builtin");
        assert_eq!(reply["index"]["documents"], index::BUILTIN.len());
        let calls = reply["calls"].as_array().unwrap();
        assert_eq!(calls[0]["tool"], "git_branch");
        assert_eq!(calls[0]["is_error"], true);
        assert_eq!(calls[1]["tool"], "delete_everything");
        assert!(calls[1]["result"].as_str().unwrap().contains("no tool"));
        assert!(!reply["candidates"].as_array().unwrap().is_empty(), "{reply}");

        let requests = requests.lock().unwrap();
        assert!(system_of(&requests[0]).contains("git_branch failed"));
        let tool_message = requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "tool")
            .unwrap();
        assert!(tool_message["content"].as_str().unwrap().starts_with("Error:"));
    }

    #[tokio::test]
    async fn status_reports_the_repo_the_index_and_the_mcp_server() {
        let dir = temp_repo("status");
        let (_, base) = spawn_app("http://127.0.0.1:9", dir.clone()).await;
        let status: Value = reqwest::get(format!("{base}/api/status"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["repo"]["state"]["state"], "ready");
        assert_eq!(status["branch"]["branch"], "feature/help-command");
        assert_eq!(status["mcp"]["tools"].as_array().unwrap().len(), 6);
        assert_eq!(status["index"]["documents"], 5);
        let page = reqwest::get(&base).await.unwrap().text().await.unwrap();
        assert!(page.contains("/api/help"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

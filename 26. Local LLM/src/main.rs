use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen2.5:3b";
const SYSTEM_PROMPT: &str = "You are a helpful assistant running locally on the user's machine. \
                             Answer clearly, in the language of the question.";

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

impl Message {
    fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

// Ollama's native /api/chat, not the OpenAI-compatible /v1: only the native
// endpoint reports model load time and generation time separately.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Options>,
}

#[derive(Serialize)]
struct Options {
    temperature: f32,
}

// Durations are in nanoseconds.
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Message,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    total_duration: u64,
}

#[derive(Deserialize)]
struct OllamaError {
    error: String,
}

#[derive(Deserialize)]
struct VersionResponse {
    version: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

#[derive(Serialize, Debug, PartialEq)]
struct Stats {
    prompt_tokens: u64,
    completion_tokens: u64,
    tokens_per_sec: f64,
    load_secs: f64,
    total_secs: f64,
}

impl Stats {
    fn from_response(r: &OllamaChatResponse) -> Self {
        let eval_secs = secs(r.eval_duration);
        Self {
            prompt_tokens: r.prompt_eval_count,
            completion_tokens: r.eval_count,
            tokens_per_sec: if eval_secs > 0.0 {
                r.eval_count as f64 / eval_secs
            } else {
                0.0
            },
            load_secs: secs(r.load_duration),
            total_secs: secs(r.total_duration),
        }
    }
}

fn secs(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}

/// Accepts the same shapes Ollama's own `OLLAMA_HOST` does: with or without a scheme.
fn normalize_host(host: &str) -> String {
    let host = host.trim().trim_end_matches('/');
    if host.is_empty() {
        DEFAULT_HOST.to_string()
    } else if host.contains("://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

/// The chat: one conversation, held in memory, answered by a model served by
/// Ollama on this machine. No API key — nothing leaves `OLLAMA_HOST`.
struct AppState {
    client: reqwest::Client,
    host: String,
    default_model: String,
    history: Mutex<Vec<Message>>,
}

impl AppState {
    fn new(host: &str, default_model: &str) -> Self {
        // Generous: on CPU a long answer from a cold model can take minutes.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            host: normalize_host(host),
            default_model: default_model.to_string(),
            history: Mutex::new(vec![Message::new("system", SYSTEM_PROMPT)]),
        }
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> Result<String, String> {
        let response = request
            .send()
            .await
            .map_err(|e| format!("Can't reach Ollama at {}: {e}. Is `ollama serve` running?", self.host))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<OllamaError>(&body)
                .map(|e| e.error)
                .unwrap_or(body);
            return Err(format!("Ollama error ({status}): {detail}"));
        }
        Ok(body)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let body = self
            .send(self.client.get(format!("{}{path}", self.host)))
            .await?;
        serde_json::from_str(&body).map_err(|e| format!("Failed to parse {path}: {e}"))
    }

    async fn status(&self) -> StatusResponse {
        let mut status = StatusResponse {
            host: self.host.clone(),
            default_model: self.default_model.clone(),
            version: None,
            models: Vec::new(),
            error: None,
        };
        let result = async {
            let version = self.get_json::<VersionResponse>("/api/version").await?;
            let tags = self.get_json::<TagsResponse>("/api/tags").await?;
            Ok::<_, String>((version.version, tags.models))
        }
        .await;
        match result {
            Ok((version, models)) => {
                status.version = Some(version);
                status.models = models.into_iter().map(|m| m.name).collect();
            }
            Err(e) => status.error = Some(e),
        }
        status
    }

    async fn respond(
        &self,
        user_message: &str,
        model: &str,
        temperature: Option<f32>,
    ) -> Result<(String, Stats), String> {
        let messages = {
            let mut history = self.history.lock().unwrap();
            history.push(Message::new("user", user_message));
            history.clone()
        };

        let request = OllamaChatRequest {
            model,
            messages: &messages,
            stream: false,
            options: temperature.map(|t| Options {
                temperature: t.clamp(0.0, 2.0),
            }),
        };
        let result = async {
            let body = self
                .send(
                    self.client
                        .post(format!("{}/api/chat", self.host))
                        .json(&request),
                )
                .await?;
            serde_json::from_str::<OllamaChatResponse>(&body)
                .map_err(|e| format!("Failed to parse response: {e}"))
        }
        .await;

        let mut history = self.history.lock().unwrap();
        match result {
            Ok(response) => {
                let stats = Stats::from_response(&response);
                let reply = response.message.content;
                history.push(Message::new("assistant", &reply));
                Ok((reply, stats))
            }
            Err(e) => {
                // A failed turn leaves no trace, so a retry isn't sent twice.
                history.pop();
                Err(e)
            }
        }
    }

    fn reset(&self) {
        let mut history = self.history.lock().unwrap();
        history.clear();
        history.push(Message::new("system", SYSTEM_PROMPT));
    }
}

#[derive(Serialize)]
struct StatusResponse {
    host: String,
    default_model: String,
    version: Option<String>,
    models: Vec<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct ChatResponse {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<Stats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    Json(state.status().await)
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let model = req
        .model
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| state.default_model.clone());
    let message = req.message.trim();
    if message.is_empty() {
        return Json(ChatResponse {
            model,
            reply: None,
            stats: None,
            error: Some("Empty message.".to_string()),
        });
    }
    let response = match state.respond(message, &model, req.temperature).await {
        Ok((reply, stats)) => ChatResponse {
            model,
            reply: Some(reply),
            stats: Some(stats),
            error: None,
        },
        Err(e) => ChatResponse {
            model,
            reply: None,
            stats: None,
            error: Some(e),
        },
    };
    Json(response)
}

async fn reset(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    state.reset();
    Json(serde_json::json!({ "ok": true }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState::new(
        &env_or("OLLAMA_HOST", DEFAULT_HOST),
        &env_or("OLLAMA_MODEL", DEFAULT_MODEL),
    ));
    println!(
        "Ollama at {}, default model {}",
        state.host, state.default_model
    );

    let app = Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route("/api/chat", post(chat))
        .route("/api/reset", post(reset))
        .with_state(state);

    let port = env_or("PORT", "3000");
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    struct Recorded {
        path: String,
        body: String,
    }

    /// A one-shot HTTP server standing in for Ollama: answers each incoming
    /// connection with the next canned response and records what it was sent.
    fn fake_ollama(responses: Vec<(u16, String)>) -> (String, thread::JoinHandle<Vec<Recorded>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut request_body = vec![0; length];
                reader.read_exact(&mut request_body).unwrap();
                seen.push(Recorded {
                    path: request_line
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .to_string(),
                    body: String::from_utf8(request_body).unwrap(),
                });
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
            seen
        });
        (host, handle)
    }

    /// A host nothing listens on: bind a port, then let it go.
    fn closed_host() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    }

    fn ollama_reply(content: &str) -> String {
        serde_json::json!({
            "model": "qwen2.5:3b",
            "message": { "role": "assistant", "content": content },
            "done": true,
            "prompt_eval_count": 20,
            "eval_count": 3,
            "eval_duration": 500_000_000u64,
            "load_duration": 250_000_000u64,
            "total_duration": 1_000_000_000u64
        })
        .to_string()
    }

    fn ask(message: &str, model: Option<&str>, temperature: Option<f32>) -> Json<ChatRequest> {
        Json(ChatRequest {
            message: message.to_string(),
            model: model.map(str::to_string),
            temperature,
        })
    }

    #[tokio::test]
    async fn status_lists_the_installed_models() {
        let (host, server) = fake_ollama(vec![
            (200, r#"{"version":"0.40.3"}"#.to_string()),
            (
                200,
                r#"{"models":[{"name":"qwen2.5:3b"},{"name":"llama3.2:latest"}]}"#.to_string(),
            ),
        ]);
        let state = Arc::new(AppState::new(&host, "qwen2.5:3b"));
        let Json(got) = status(State(state)).await;
        assert_eq!(got.version.as_deref(), Some("0.40.3"));
        assert_eq!(got.models, ["qwen2.5:3b", "llama3.2:latest"]);
        assert_eq!(got.default_model, "qwen2.5:3b");
        assert!(got.error.is_none());

        let paths: Vec<String> = server.join().unwrap().into_iter().map(|r| r.path).collect();
        assert_eq!(paths, ["/api/version", "/api/tags"]);
    }

    #[tokio::test]
    async fn status_says_when_ollama_is_not_running() {
        let state = Arc::new(AppState::new(&closed_host(), "qwen2.5:3b"));
        let Json(got) = status(State(state)).await;
        assert!(got.version.is_none());
        assert!(got.models.is_empty());
        let error = got.error.unwrap();
        assert!(error.contains("Can't reach Ollama"), "{error}");
    }

    #[tokio::test]
    async fn chat_keeps_the_history_and_returns_the_counters() {
        let (host, server) = fake_ollama(vec![
            (200, ollama_reply("Париж")),
            (200, ollama_reply("Около 2,1 млн человек.")),
        ]);
        let state = Arc::new(AppState::new(&host, "qwen2.5:3b"));

        let Json(first) = chat(
            State(state.clone()),
            ask("Столица Франции?", None, Some(0.0)),
        )
        .await;
        assert_eq!(first.reply.as_deref(), Some("Париж"));
        assert_eq!(first.model, "qwen2.5:3b");
        assert_eq!(
            first.stats,
            Some(Stats {
                prompt_tokens: 20,
                completion_tokens: 3,
                tokens_per_sec: 6.0,
                load_secs: 0.25,
                total_secs: 1.0,
            })
        );

        let Json(second) = chat(
            State(state.clone()),
            ask("  А сколько там жителей?  ", Some("llama3.2"), Some(5.0)),
        )
        .await;
        assert_eq!(second.reply.as_deref(), Some("Около 2,1 млн человек."));
        assert_eq!(second.model, "llama3.2");

        let seen = server.join().unwrap();
        assert!(seen.iter().all(|r| r.path == "/api/chat"));
        let first_body: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
        assert_eq!(first_body["model"], "qwen2.5:3b");
        assert_eq!(first_body["stream"], false);
        assert_eq!(first_body["options"]["temperature"], 0.0);
        assert_eq!(first_body["messages"].as_array().unwrap().len(), 2);

        // The second request carries the whole dialogue, and the temperature
        // is clamped to Ollama's range.
        let second_body: serde_json::Value = serde_json::from_str(&seen[1].body).unwrap();
        assert_eq!(second_body["model"], "llama3.2");
        assert_eq!(second_body["options"]["temperature"], 2.0);
        let roles: Vec<&str> = second_body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(second_body["messages"][2]["content"], "Париж");
        assert_eq!(second_body["messages"][3]["content"], "А сколько там жителей?");
    }

    #[tokio::test]
    async fn a_failed_turn_is_reported_and_leaves_no_trace() {
        let (host, server) = fake_ollama(vec![(
            404,
            r#"{"error":"model \"nope\" not found, try pulling it first"}"#.to_string(),
        )]);
        let state = Arc::new(AppState::new(&host, "qwen2.5:3b"));

        let Json(response) = chat(State(state.clone()), ask("hi", Some("nope"), None)).await;
        let error = response.error.unwrap();
        assert!(error.contains("404") && error.contains("not found"), "{error}");
        assert!(response.reply.is_none());
        server.join().unwrap();

        // An empty message never reaches Ollama (nothing is listening anyway).
        let Json(response) = chat(State(state.clone()), ask("   ", None, None)).await;
        assert_eq!(response.error.as_deref(), Some("Empty message."));

        let history = state.history.lock().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }

    #[tokio::test]
    async fn reset_keeps_only_the_system_prompt() {
        let state = Arc::new(AppState::new(&closed_host(), "qwen2.5:3b"));
        state
            .history
            .lock()
            .unwrap()
            .extend([Message::new("user", "hi"), Message::new("assistant", "hello")]);
        reset(State(state.clone())).await;
        let history = state.history.lock().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, SYSTEM_PROMPT);
    }

    #[test]
    fn normalizes_ollama_host_like_ollama_does() {
        assert_eq!(normalize_host(""), DEFAULT_HOST);
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(normalize_host("http://gpu-box:11434/"), "http://gpu-box:11434");
    }
}

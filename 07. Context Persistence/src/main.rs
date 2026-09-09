use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer clearly and concisely.";

/// The only two models this agent is allowed to call - both live behind the
/// same endpoint/key, so switching between them is just a different `model`
/// value on the same request, not a different provider.
const MODEL_FLASH: &str = "deepseek-v4-flash";
const MODEL_PRO: &str = "deepseek-v4-pro";

/// Per-request overrides a caller may supply on top of the running
/// conversation. Every field is optional; omitted ones are left out of the
/// API request entirely rather than sent as an explicit null/default.
struct ChatOptions {
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
}

/// The agent: owns the conversation history, knows how to turn a user
/// message into an LLM call and back into a reply, and persists the history
/// to `history_path` after every completed turn so a restarted process picks
/// up exactly where it left off.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    default_model: String,
    history_path: String,
    history: Mutex<Vec<Message>>,
}

impl Agent {
    fn new(
        client: reqwest::Client,
        endpoint: String,
        api_key: String,
        default_model: String,
        history_path: String,
    ) -> Self {
        let history = Self::load_history(&history_path);
        Self {
            client,
            endpoint,
            api_key,
            default_model,
            history_path,
            history: Mutex::new(history),
        }
    }

    fn seed_history() -> Vec<Message> {
        vec![Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }]
    }

    /// Loads a previously persisted conversation from disk. Any problem
    /// reading or parsing the file (missing, empty, corrupted) is treated
    /// the same way: fall back to a fresh conversation rather than fail
    /// startup over history that was never load-bearing to begin with.
    fn load_history(path: &str) -> Vec<Message> {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Self::seed_history(),
        };
        match serde_json::from_str::<Vec<Message>>(&contents) {
            Ok(history) if !history.is_empty() => history,
            Ok(_) => Self::seed_history(),
            Err(e) => {
                eprintln!("Warning: failed to parse {path} ({e}); starting a fresh conversation.");
                Self::seed_history()
            }
        }
    }

    /// Writes the full current history to disk. Best-effort: a write failure
    /// is logged but never allowed to break the in-memory conversation the
    /// user is actively having.
    fn persist(&self) {
        let history = self.history.lock().unwrap().clone();
        match serde_json::to_string_pretty(&history) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.history_path, json) {
                    eprintln!("Warning: failed to write {}: {e}", self.history_path);
                }
            }
            Err(e) => eprintln!("Warning: failed to serialize history: {e}"),
        }
    }

    /// Only `MODEL_FLASH`/`MODEL_PRO` may ever be sent upstream - an
    /// unrecognized or missing request falls back to this agent's default
    /// rather than forwarding arbitrary client input to the API.
    fn resolve_model(&self, requested: Option<&str>) -> String {
        match requested {
            Some(MODEL_FLASH) => MODEL_FLASH.to_string(),
            Some(MODEL_PRO) => MODEL_PRO.to_string(),
            _ => self.default_model.clone(),
        }
    }

    async fn respond(&self, user_message: &str, options: ChatOptions) -> String {
        let messages = {
            let mut history = self.history.lock().unwrap();
            history.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
            history.clone()
        };

        let stop = options
            .stop
            .map(|s| {
                s.into_iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .take(4)
                    .collect::<Vec<_>>()
            })
            .filter(|s| !s.is_empty());

        let request = ChatCompletionRequest {
            model: self.resolve_model(options.model.as_deref()),
            messages,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            stop,
        };

        let response = match self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                self.history.lock().unwrap().pop();
                return format!("Request failed: {e}");
            }
        };

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            self.history.lock().unwrap().pop();
            let detail = serde_json::from_str::<ErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return format!("API error ({status}): {detail}");
        }

        let reply = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(parsed) => parsed
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_else(|| "No response returned from the model.".to_string()),
            Err(e) => {
                self.history.lock().unwrap().pop();
                return format!("Failed to parse response: {e}");
            }
        };

        self.history.lock().unwrap().push(Message {
            role: "assistant".to_string(),
            content: reply.clone(),
        });
        self.persist();

        reply
    }

    fn reset(&self) {
        {
            let mut history = self.history.lock().unwrap();
            *history = Self::seed_history();
        }
        self.persist();
    }

    fn visible_history(&self) -> Vec<Message> {
        self.history
            .lock()
            .unwrap()
            .iter()
            .filter(|m| m.role != "system")
            .cloned()
            .collect()
    }
}

#[cfg(test)]
impl Agent {
    fn history_snapshot(&self) -> Vec<Message> {
        self.history.lock().unwrap().clone()
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ChatResponse {
    reply: String,
}

#[derive(Serialize)]
struct HistoryResponse {
    messages: Vec<Message>,
}

async fn chat(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let reply = agent
        .respond(
            &req.message,
            ChatOptions {
                model: req.model,
                temperature: req.temperature,
                max_tokens: req.max_tokens,
                stop: req.stop,
            },
        )
        .await;
    Json(ChatResponse { reply })
}

async fn reset(State(agent): State<Arc<Agent>>) -> Json<ChatResponse> {
    agent.reset();
    Json(ChatResponse {
        reply: "Conversation reset.".to_string(),
    })
}

/// Lets the page rehydrate the visible chat bubbles on load, so a browser
/// reload after the server restarts shows the same conversation rather than
/// an empty chat window sitting on top of an agent that still remembers.
async fn history(State(agent): State<Arc<Agent>>) -> Json<HistoryResponse> {
    Json(HistoryResponse {
        messages: agent.visible_history(),
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    // Falls back to MODEL_FLASH if OPENAI_MODEL is unset or isn't one of the
    // two models this agent is allowed to call.
    let default_model = match std::env::var("OPENAI_MODEL") {
        Ok(m) if m == MODEL_FLASH || m == MODEL_PRO => m,
        _ => MODEL_FLASH.to_string(),
    };
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    let history_path = std::env::var("HISTORY_FILE").unwrap_or_else(|_| "history.json".to_string());

    let agent = Agent::new(
        reqwest::Client::new(),
        format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        default_model,
        history_path.clone(),
    );
    let restored_turns = agent.history.lock().unwrap().len().saturating_sub(1);
    if restored_turns > 0 {
        println!("Restored {restored_turns} message(s) from {history_path}");
    } else {
        println!("No previous history at {history_path}; starting a fresh conversation.");
    }
    let agent = Arc::new(agent);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/reset", post(reset))
        .route("/api/history", get(history))
        .with_state(agent);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_history_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "ai-advent-lesson-07-test-{}-{}.json",
                std::process::id(),
                n
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn test_agent_with_path(history_path: String) -> Agent {
        Agent::new(
            reqwest::Client::new(),
            "http://localhost:0/chat/completions".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
            history_path,
        )
    }

    fn test_agent() -> Agent {
        test_agent_with_path(temp_history_path())
    }

    #[test]
    fn new_seeds_only_the_system_prompt_when_no_history_file_exists() {
        let agent = test_agent();
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }

    #[test]
    fn new_loads_previously_persisted_history() {
        let path = temp_history_path();
        let seeded = vec![
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "hello".to_string(),
            },
        ];
        std::fs::write(&path, serde_json::to_string(&seeded).unwrap()).unwrap();

        let agent = test_agent_with_path(path.clone());
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].content, "hi");
        assert_eq!(history[2].content, "hello");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_falls_back_to_seed_on_invalid_json() {
        let path = temp_history_path();
        std::fs::write(&path, "not json").unwrap();

        let agent = test_agent_with_path(path.clone());
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reset_restores_only_the_system_prompt_and_persists_it() {
        let path = temp_history_path();
        let agent = test_agent_with_path(path.clone());
        agent.history.lock().unwrap().push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.reset();

        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");

        let on_disk: Vec<Message> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk.len(), 1);
        assert_eq!(on_disk[0].role, "system");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn visible_history_omits_the_system_prompt() {
        let agent = test_agent();
        agent.history.lock().unwrap().push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        let visible = agent.visible_history();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].role, "user");
    }

    #[test]
    fn resolve_model_honors_an_allowed_request() {
        let agent = test_agent();
        assert_eq!(agent.resolve_model(Some(MODEL_PRO)), MODEL_PRO);
        assert_eq!(agent.resolve_model(Some(MODEL_FLASH)), MODEL_FLASH);
    }

    #[test]
    fn resolve_model_falls_back_to_the_default_otherwise() {
        let agent = test_agent();
        assert_eq!(agent.resolve_model(None), "test-model");
        assert_eq!(agent.resolve_model(Some("gpt-4o")), "test-model");
    }
}

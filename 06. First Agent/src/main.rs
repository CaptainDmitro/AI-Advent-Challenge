use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
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

/// The agent: a standalone entity that owns the conversation history and knows
/// how to turn a user message into an LLM call and back into a reply. Callers
/// only ever see `respond`/`reset` - never the HTTP request/response shape.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    history: Mutex<Vec<Message>>,
}

impl Agent {
    fn new(client: reqwest::Client, endpoint: String, api_key: String, model: String) -> Self {
        Self {
            client,
            endpoint,
            api_key,
            model,
            history: Mutex::new(vec![Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            }]),
        }
    }

    async fn respond(&self, user_message: &str) -> String {
        let messages = {
            let mut history = self.history.lock().unwrap();
            history.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
            history.clone()
        };

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
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

        reply
    }

    fn reset(&self) {
        let mut history = self.history.lock().unwrap();
        history.clear();
        history.push(Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        });
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
}

#[derive(Serialize)]
struct ChatResponse {
    reply: String,
}

async fn chat(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let reply = agent.respond(&req.message).await;
    Json(ChatResponse { reply })
}

async fn reset(State(agent): State<Arc<Agent>>) -> Json<ChatResponse> {
    agent.reset();
    Json(ChatResponse {
        reply: "Conversation reset.".to_string(),
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "deepseek-v4-pro".to_string());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });

    let agent = Arc::new(Agent::new(
        reqwest::Client::new(),
        format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model,
    ));

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/reset", post(reset))
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

    fn test_agent() -> Agent {
        Agent::new(
            reqwest::Client::new(),
            "http://localhost:0/chat/completions".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
        )
    }

    #[test]
    fn new_seeds_only_the_system_prompt() {
        let agent = test_agent();
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }

    #[test]
    fn reset_restores_only_the_system_prompt() {
        let agent = test_agent();
        agent.history.lock().unwrap().push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.reset();
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }
}

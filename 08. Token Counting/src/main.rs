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
    usage: Option<Usage>,
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

/// Token accounting straight from the API's own `usage` field on a chat
/// completion response - the authoritative number for the turn that just
/// happened, as opposed to our own pre-call estimate.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// What a single chat turn (or a reset/history fetch) reports about token
/// usage: our own heuristic estimate computed before the call, the actual
/// usage the provider returned (when it did), and a running total against a
/// configurable reference context limit.
#[derive(Serialize)]
struct TokenReport {
    /// This request's user message, counted locally before the call.
    estimated_request_tokens: u32,
    /// The whole history before this turn was added, counted locally.
    estimated_history_tokens_before: u32,
    /// This turn's reply, counted locally (0 when the turn failed).
    estimated_response_tokens: u32,
    /// The provider's own accounting for this turn, when it returns one.
    actual: Option<Usage>,
    /// Tokens now sitting in history: `actual.total_tokens` when the
    /// provider gave us one, otherwise our own estimate of the full,
    /// updated history.
    history_tokens_after: u32,
    /// A configurable reference limit (see `CONTEXT_LIMIT_TOKENS`) - not a
    /// guarantee of the real backend's context window, just something to
    /// watch a running total against.
    context_limit: u32,
    percent_of_limit: f32,
    /// Only present when `PRICE_PER_1M_INPUT_TOKENS`/`PRICE_PER_1M_OUTPUT_TOKENS`
    /// are configured; uses actual usage when available, the estimate otherwise.
    estimated_cost_usd: Option<f64>,
}

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer clearly and concisely.";

/// The only two models this agent is allowed to call - both live behind the
/// same endpoint/key, so switching between them is just a different `model`
/// value on the same request, not a different provider.
const MODEL_FLASH: &str = "deepseek-v4-flash";
const MODEL_PRO: &str = "deepseek-v4-pro";

/// Roughly typical chat-format overhead per message (role/name delimiters)
/// on top of the content itself, matching the widely-cited ~4-tokens-per-message
/// rule of thumb for OpenAI-style chat formatting. Approximate, like the rest
/// of this heuristic.
const MESSAGE_OVERHEAD_TOKENS: u32 = 4;

/// A dependency-free, deliberately approximate token estimate: runs of
/// letters/digits are charged at ~4 characters per token (typical BPE
/// subword granularity), and each punctuation/symbol character is charged
/// as its own token. This is not any particular provider's real tokenizer -
/// DeepSeek's isn't public - so treat it as a ballpark, not ground truth.
/// The API's own `usage` field (see `Usage`) is the authoritative number.
fn estimate_tokens(text: &str) -> u32 {
    fn flush(tokens: &mut u32, run_len: u32) {
        if run_len > 0 {
            *tokens = tokens.saturating_add(run_len.div_ceil(4).max(1));
        }
    }

    let mut tokens: u32 = 0;
    let mut run_len: u32 = 0;

    for ch in text.chars() {
        if ch.is_whitespace() {
            flush(&mut tokens, run_len);
            run_len = 0;
        } else if ch.is_alphanumeric() {
            run_len += 1;
        } else {
            flush(&mut tokens, run_len);
            run_len = 0;
            tokens = tokens.saturating_add(1);
        }
    }
    flush(&mut tokens, run_len);

    tokens
}

fn message_tokens(message: &Message) -> u32 {
    estimate_tokens(&message.content) + MESSAGE_OVERHEAD_TOKENS
}

fn history_tokens(history: &[Message]) -> u32 {
    history.iter().map(message_tokens).sum()
}

/// Per-request overrides a caller may supply on top of the running
/// conversation. Every field is optional; omitted ones are left out of the
/// API request entirely rather than sent as an explicit null/default.
struct ChatOptions {
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
}

struct RespondResult {
    reply: String,
    tokens: TokenReport,
}

/// The agent: owns the conversation history, knows how to turn a user
/// message into an LLM call and back into a reply, persists the history to
/// `history_path` after every completed turn, and reports token usage
/// (estimated and, when the provider supplies it, actual) for every turn.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    default_model: String,
    history_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    history: Mutex<Vec<Message>>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    fn new(
        client: reqwest::Client,
        endpoint: String,
        api_key: String,
        default_model: String,
        history_path: String,
        context_limit: u32,
        price_per_1m_input: Option<f64>,
        price_per_1m_output: Option<f64>,
    ) -> Self {
        let history = Self::load_history(&history_path);
        Self {
            client,
            endpoint,
            api_key,
            default_model,
            history_path,
            context_limit,
            price_per_1m_input,
            price_per_1m_output,
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

    fn history_tokens_now(&self) -> u32 {
        history_tokens(&self.history.lock().unwrap())
    }

    /// Cost in USD for the given token counts, using whichever
    /// price-per-1M-token rates are configured. `None` if either rate is
    /// unset, rather than guessing at a provider's real pricing.
    fn estimate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> Option<f64> {
        let input_rate = self.price_per_1m_input?;
        let output_rate = self.price_per_1m_output?;
        Some(
            (prompt_tokens as f64 / 1_000_000.0) * input_rate
                + (completion_tokens as f64 / 1_000_000.0) * output_rate,
        )
    }

    fn snapshot(&self, before: u32) -> TokenReport {
        let after = self.history_tokens_now();
        TokenReport {
            estimated_request_tokens: 0,
            estimated_history_tokens_before: before,
            estimated_response_tokens: 0,
            actual: None,
            history_tokens_after: after,
            context_limit: self.context_limit,
            percent_of_limit: (after as f32 / self.context_limit as f32) * 100.0,
            estimated_cost_usd: None,
        }
    }

    async fn respond(&self, user_message: &str, options: ChatOptions) -> RespondResult {
        let (messages, estimated_request_tokens, estimated_history_before) = {
            let mut history = self.history.lock().unwrap();
            let history_before = history_tokens(&history);
            history.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
            let request_tokens = message_tokens(history.last().unwrap());
            (history.clone(), request_tokens, history_before)
        };

        let make_report = |actual: Option<Usage>,
                            estimated_response_tokens: u32,
                            history_after_estimate: u32|
         -> TokenReport {
            let history_tokens_after = actual
                .as_ref()
                .map(|u| u.total_tokens)
                .unwrap_or(history_after_estimate);
            let percent_of_limit =
                (history_tokens_after as f32 / self.context_limit as f32) * 100.0;
            let estimated_cost_usd = match &actual {
                Some(u) => self.estimate_cost(u.prompt_tokens, u.completion_tokens),
                None => self.estimate_cost(estimated_request_tokens, estimated_response_tokens),
            };
            TokenReport {
                estimated_request_tokens,
                estimated_history_tokens_before: estimated_history_before,
                estimated_response_tokens,
                actual,
                history_tokens_after,
                context_limit: self.context_limit,
                percent_of_limit,
                estimated_cost_usd,
            }
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
                return RespondResult {
                    reply: format!("Request failed: {e}"),
                    tokens: make_report(None, 0, estimated_history_before),
                };
            }
        };

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            self.history.lock().unwrap().pop();
            let detail = serde_json::from_str::<ErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return RespondResult {
                reply: format!("API error ({status}): {detail}"),
                tokens: make_report(None, 0, estimated_history_before),
            };
        }

        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(p) => p,
            Err(e) => {
                self.history.lock().unwrap().pop();
                return RespondResult {
                    reply: format!("Failed to parse response: {e}"),
                    tokens: make_report(None, 0, estimated_history_before),
                };
            }
        };

        let reply = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_else(|| "No response returned from the model.".to_string());
        let estimated_response_tokens = estimate_tokens(&reply) + MESSAGE_OVERHEAD_TOKENS;

        let history_after_estimate = {
            let mut history = self.history.lock().unwrap();
            history.push(Message {
                role: "assistant".to_string(),
                content: reply.clone(),
            });
            history_tokens(&history)
        };
        self.persist();

        RespondResult {
            tokens: make_report(parsed.usage, estimated_response_tokens, history_after_estimate),
            reply,
        }
    }

    fn reset(&self) -> TokenReport {
        let before = self.history_tokens_now();
        {
            let mut history = self.history.lock().unwrap();
            *history = Self::seed_history();
        }
        self.persist();
        self.snapshot(before)
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
    tokens: TokenReport,
}

#[derive(Serialize)]
struct HistoryResponse {
    messages: Vec<Message>,
    tokens: TokenReport,
}

async fn chat(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let result = agent
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
    Json(ChatResponse {
        reply: result.reply,
        tokens: result.tokens,
    })
}

async fn reset(State(agent): State<Arc<Agent>>) -> Json<ChatResponse> {
    let tokens = agent.reset();
    Json(ChatResponse {
        reply: "Conversation reset.".to_string(),
        tokens,
    })
}

/// Lets the page rehydrate the visible chat bubbles and the token stats bar
/// on load, so a browser reload after the server restarts shows the same
/// conversation and running totals.
async fn history(State(agent): State<Arc<Agent>>) -> Json<HistoryResponse> {
    let tokens = agent.snapshot(agent.history_tokens_now());
    Json(HistoryResponse {
        messages: agent.visible_history(),
        tokens,
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
    let history_path =
        std::env::var("HISTORY_FILE").unwrap_or_else(|_| "history.json".to_string());
    // A reference limit for the "% of limit" indicator, not a guarantee of
    // the real backend's context window - see the README for why.
    let context_limit: u32 = std::env::var("CONTEXT_LIMIT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64_000);
    let price_per_1m_input = std::env::var("PRICE_PER_1M_INPUT_TOKENS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0);
    let price_per_1m_output = std::env::var("PRICE_PER_1M_OUTPUT_TOKENS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0);

    let agent = Agent::new(
        reqwest::Client::new(),
        format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        default_model,
        history_path.clone(),
        context_limit,
        price_per_1m_input,
        price_per_1m_output,
    );
    let restored_turns = agent.history.lock().unwrap().len().saturating_sub(1);
    if restored_turns > 0 {
        println!(
            "Restored {restored_turns} message(s) from {history_path} (~{} tokens)",
            agent.history_tokens_now()
        );
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
                "ai-advent-lesson-08-test-{}-{}.json",
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
            1000,
            Some(2.0),
            Some(4.0),
        )
    }

    fn test_agent() -> Agent {
        test_agent_with_path(temp_history_path())
    }

    #[test]
    fn estimate_tokens_is_zero_for_empty_text() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_grows_with_length() {
        let short = estimate_tokens("hello");
        let long = estimate_tokens(&"hello world ".repeat(50));
        assert!(long > short * 10);
    }

    #[test]
    fn estimate_tokens_counts_punctuation_separately() {
        let base = estimate_tokens("hello");
        let with_punct = estimate_tokens("hello!");
        assert_eq!(with_punct, base + 1);
    }

    #[test]
    fn history_tokens_includes_the_seeded_system_prompt() {
        let agent = test_agent();
        assert!(agent.history_tokens_now() > 0);
    }

    #[test]
    fn reset_reports_before_and_after_and_persists_the_drop() {
        let path = temp_history_path();
        let agent = test_agent_with_path(path.clone());
        agent.history.lock().unwrap().push(Message {
            role: "user".to_string(),
            content: "a fairly long message to push the token count up a bit".to_string(),
        });
        let before_reset = agent.history_tokens_now();

        let report = agent.reset();
        assert_eq!(report.estimated_history_tokens_before, before_reset);
        assert!(report.history_tokens_after < before_reset);
        assert_eq!(agent.history_tokens_now(), report.history_tokens_after);

        let on_disk: Vec<Message> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk.len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn estimate_cost_is_none_without_configured_prices() {
        let agent = Agent::new(
            reqwest::Client::new(),
            "http://localhost:0/chat/completions".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
            temp_history_path(),
            1000,
            None,
            None,
        );
        assert_eq!(agent.estimate_cost(1000, 1000), None);
    }

    #[test]
    fn estimate_cost_scales_with_configured_rates() {
        let agent = test_agent(); // $2/1M input, $4/1M output
        let cost = agent.estimate_cost(1_000_000, 1_000_000).unwrap();
        assert!((cost - 6.0).abs() < 1e-9);
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

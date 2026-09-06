use std::time::Instant;

use axum::{Json, extract::State, response::Html, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
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

#[derive(Deserialize, Clone, Copy)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

struct ModelConfig {
    endpoint: String,
    api_key: String,
    model: String,
    input_cost_per_1m: Option<f64>,
    output_cost_per_1m: Option<f64>,
}

impl ModelConfig {
    fn from_env(
        base_url_var: &str,
        base_url_default: Option<&str>,
        api_key_var: &str,
        api_key_default: Option<&str>,
        model_var: &str,
        model_default: &str,
        input_cost_per_1m: Option<f64>,
        output_cost_per_1m: Option<f64>,
    ) -> Self {
        let base_url = std::env::var(base_url_var).unwrap_or_else(|_| {
            base_url_default
                .unwrap_or_else(|| {
                    eprintln!("Error: {base_url_var} environment variable is not set.");
                    std::process::exit(1);
                })
                .to_string()
        });
        let api_key = std::env::var(api_key_var).unwrap_or_else(|_| {
            api_key_default
                .unwrap_or_else(|| {
                    eprintln!("Error: {api_key_var} environment variable is not set.");
                    std::process::exit(1);
                })
                .to_string()
        });
        let model = std::env::var(model_var).unwrap_or_else(|_| model_default.to_string());

        ModelConfig {
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model,
            input_cost_per_1m,
            output_cost_per_1m,
        }
    }
}

// Flat per-1M-token rates, approximating DeepSeek's published off-peak, cache-miss
// pricing (https://api-docs.deepseek.com/quick_start/pricing) as a single representative
// number — actual DeepSeek billing varies up to 6x with cache hits and peak hours, which
// this comparison tool doesn't attempt to model.
const MEDIUM_INPUT_COST_PER_1M: f64 = 0.22;
const MEDIUM_OUTPUT_COST_PER_1M: f64 = 0.66;
const STRONG_INPUT_COST_PER_1M: f64 = 0.66;
const STRONG_OUTPUT_COST_PER_1M: f64 = 1.98;

struct AppState {
    client: reqwest::Client,
    weak: ModelConfig,
    medium: ModelConfig,
    strong: ModelConfig,
}

#[derive(Serialize)]
struct RunResponse {
    model: String,
    answer: String,
    elapsed_ms: u128,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    cost_usd: Option<f64>,
}

async fn run_model(client: &reqwest::Client, config: &ModelConfig, prompt: &str) -> RunResponse {
    let request = ChatRequest {
        model: config.model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let started = Instant::now();
    let response = client
        .post(&config.endpoint)
        .bearer_auth(&config.api_key)
        .json(&request)
        .send()
        .await;
    let elapsed_ms = started.elapsed().as_millis();

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            return RunResponse {
                model: config.model.clone(),
                answer: format!("Request failed: {e}"),
                elapsed_ms,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                cost_usd: None,
            };
        }
    };

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if !status.is_success() {
        let detail = serde_json::from_str::<ErrorResponse>(&body)
            .map(|e| e.error.message)
            .unwrap_or(body);
        return RunResponse {
            model: config.model.clone(),
            answer: format!("API error ({status}): {detail}"),
            elapsed_ms,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cost_usd: None,
        };
    }

    let parsed: ChatResponse = match serde_json::from_str(&body) {
        Ok(p) => p,
        Err(e) => {
            return RunResponse {
                model: config.model.clone(),
                answer: format!("Failed to parse response: {e}"),
                elapsed_ms,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                cost_usd: None,
            };
        }
    };

    let answer = parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_else(|| "No response returned from the model.".to_string());

    let cost_usd = parsed.usage.and_then(|u| {
        match (config.input_cost_per_1m, config.output_cost_per_1m) {
            (Some(input_rate), Some(output_rate)) => Some(
                (u.prompt_tokens as f64 / 1_000_000.0) * input_rate
                    + (u.completion_tokens as f64 / 1_000_000.0) * output_rate,
            ),
            _ => None,
        }
    });

    RunResponse {
        model: config.model.clone(),
        answer,
        elapsed_ms,
        prompt_tokens: parsed.usage.map(|u| u.prompt_tokens),
        completion_tokens: parsed.usage.map(|u| u.completion_tokens),
        total_tokens: parsed.usage.map(|u| u.total_tokens),
        cost_usd,
    }
}

#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

async fn weak(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<RunResponse> {
    Json(run_model(&state.client, &state.weak, &req.prompt).await)
}

async fn medium(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<RunResponse> {
    Json(run_model(&state.client, &state.medium, &req.prompt).await)
}

async fn strong(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<RunResponse> {
    Json(run_model(&state.client, &state.strong, &req.prompt).await)
}

#[derive(Deserialize)]
struct TierResult {
    model: String,
    answer: String,
    elapsed_ms: u128,
    total_tokens: Option<u32>,
    cost_usd: Option<f64>,
}

impl TierResult {
    fn describe(&self, label: &str) -> String {
        let tokens = self
            .total_tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let cost = self
            .cost_usd
            .map(|c| format!("${c:.6}"))
            .unwrap_or_else(|| "n/a (free/local)".to_string());
        format!(
            "{label} — {} ({} ms, {} tokens, cost {}):\n{}",
            self.model, self.elapsed_ms, tokens, cost, self.answer
        )
    }
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    prompt: String,
    weak: TierResult,
    medium: TierResult,
    strong: TierResult,
}

async fn analyze(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<AnalyzeRequest>,
) -> Json<RunResponse> {
    let analysis_prompt = format!(
        "The same task was sent to three LLMs of increasing capability. Compare their \
         answers: note differences in quality/correctness, and weigh that against the \
         speed and cost each one took. Give a short verdict on which was the best \
         trade-off. Respond in the same language as the original task.\n\n\
         Task: {}\n\n{}\n\n{}\n\n{}",
        req.prompt,
        req.weak.describe("Weak model"),
        req.medium.describe("Medium model"),
        req.strong.describe("Strong model"),
    );

    Json(run_model(&state.client, &state.strong, &analysis_prompt).await)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let weak_config = ModelConfig::from_env(
        "WEAK_BASE_URL",
        Some("http://localhost:11434/v1"),
        "WEAK_API_KEY",
        Some("ollama"),
        "WEAK_MODEL",
        "qwen2.5:0.5b",
        None,
        None,
    );
    let medium_config = ModelConfig::from_env(
        "MEDIUM_BASE_URL",
        None,
        "MEDIUM_API_KEY",
        None,
        "MEDIUM_MODEL",
        "deepseek-v4-flash",
        Some(MEDIUM_INPUT_COST_PER_1M),
        Some(MEDIUM_OUTPUT_COST_PER_1M),
    );
    let strong_config = ModelConfig::from_env(
        "STRONG_BASE_URL",
        None,
        "STRONG_API_KEY",
        None,
        "STRONG_MODEL",
        "deepseek-v4-pro",
        Some(STRONG_INPUT_COST_PER_1M),
        Some(STRONG_OUTPUT_COST_PER_1M),
    );

    let state = std::sync::Arc::new(AppState {
        client: reqwest::Client::new(),
        weak: weak_config,
        medium: medium_config,
        strong: strong_config,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/weak", post(weak))
        .route("/api/medium", post(medium))
        .route("/api/strong", post(strong))
        .route("/api/analyze", post(analyze))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

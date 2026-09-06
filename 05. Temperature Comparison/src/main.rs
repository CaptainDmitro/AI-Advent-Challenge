use std::sync::Arc;

use axum::{Json, extract::State, response::Html, routing::{get, post}, Router};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
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

struct AppState {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
}

// Failures are turned into a display string rather than an error response, so one
// box/run failing never breaks the rest of the page.
async fn call_llm(state: &AppState, prompt: &str, temperature: f32) -> String {
    let request = ChatRequest {
        model: state.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        temperature,
    };

    let response = match state
        .client
        .post(&state.endpoint)
        .bearer_auth(&state.api_key)
        .json(&request)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("Request failed: {e}"),
    };

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if !status.is_success() {
        let detail = serde_json::from_str::<ErrorResponse>(&body)
            .map(|e| e.error.message)
            .unwrap_or(body);
        return format!("API error ({status}): {detail}");
    }

    match serde_json::from_str::<ChatResponse>(&body) {
        Ok(parsed) => parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_else(|| "No response returned from the model.".to_string()),
        Err(e) => format!("Failed to parse response: {e}"),
    }
}

#[derive(Deserialize)]
struct RunRequest {
    prompt: String,
    temperature: f32,
}

#[derive(Serialize)]
struct RunResponse {
    answer: String,
}

async fn run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RunRequest>,
) -> Json<RunResponse> {
    let answer = call_llm(&state, &req.prompt, req.temperature).await;
    Json(RunResponse { answer })
}

#[derive(Deserialize)]
struct TemperatureGroup {
    temperature: f32,
    answer: String,
}

#[derive(Deserialize)]
struct AnalyzeRequest {
    prompt: String,
    groups: Vec<TemperatureGroup>,
}

async fn analyze(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzeRequest>,
) -> Json<RunResponse> {
    let mut sections = String::new();
    for group in &req.groups {
        sections.push_str(&format!(
            "\nTemperature {}: {}\n",
            group.temperature, group.answer
        ));
    }

    let analysis_prompt = format!(
        "The same task was sent to one model at each of several temperature settings. \
         Compare the settings on accuracy and creativity, and conclude which kind of \
         task each temperature setting suits best. (Diversity across repeated runs at \
         the same temperature isn't captured here — only one sample per setting — so \
         don't claim to observe it, just note that higher temperatures are expected to \
         vary more across repeated runs.) Respond in the same language as the original \
         task.\n\nTask: {}\n{}",
        req.prompt, sections
    );

    // Judging benefits from a focused, low-temperature pass regardless of what was tested.
    let answer = call_llm(&state, &analysis_prompt, 0.0).await;
    Json(RunResponse { answer })
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

    let state = Arc::new(AppState {
        client: reqwest::Client::new(),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/run", post(run))
        .route("/api/analyze", post(analyze))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

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

// Any failure (network, HTTP status, parsing) is turned into a display string instead
// of an error response, so one box's failure never breaks the rest of the page.
async fn call_llm(state: &AppState, messages: Vec<Message>) -> String {
    let request = ChatRequest {
        model: state.model.clone(),
        messages,
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

fn user_message(content: &str) -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: content.to_string(),
    }]
}

#[derive(Deserialize)]
struct PromptRequest {
    prompt: String,
}

#[derive(Serialize)]
struct AnswerResponse {
    answer: String,
}

async fn direct(State(state): State<Arc<AppState>>, Json(req): Json<PromptRequest>) -> Json<AnswerResponse> {
    let answer = call_llm(&state, user_message(&req.prompt)).await;
    Json(AnswerResponse { answer })
}

async fn step_by_step(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<AnswerResponse> {
    let prompt = format!("{}\n\nРешай пошагово.", req.prompt);
    let answer = call_llm(&state, user_message(&prompt)).await;
    Json(AnswerResponse { answer })
}

#[derive(Serialize)]
struct PlannedResponse {
    generated_prompt: String,
    answer: String,
}

async fn planned(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<PlannedResponse> {
    let planning_request = format!(
        "Составь подробный промпт, который поможет решить следующую задачу. \
         В ответе выведи только сам промпт, без пояснений.\n\nЗадача: {}",
        req.prompt
    );
    let generated_prompt = call_llm(&state, user_message(&planning_request)).await;
    let answer = call_llm(&state, user_message(&generated_prompt)).await;
    Json(PlannedResponse {
        generated_prompt,
        answer,
    })
}

#[derive(Serialize)]
struct ExpertsResponse {
    analyst: String,
    engineer: String,
    critic: String,
}

async fn experts(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Json<ExpertsResponse> {
    let ask = |role: &str| {
        let prompt = format!(
            "Ты выступаешь в роли {role}. Реши следующую задачу со своей точки зрения:\n\n{}",
            req.prompt
        );
        call_llm(&state, user_message(&prompt))
    };

    let (analyst, engineer, critic) = tokio::join!(
        ask("аналитика"),
        ask("инженера"),
        ask("критика"),
    );

    Json(ExpertsResponse {
        analyst,
        engineer,
        critic,
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
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
        .route("/api/direct", post(direct))
        .route("/api/step-by-step", post(step_by_step))
        .route("/api/planned", post(planned))
        .route("/api/experts", post(experts))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

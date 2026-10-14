use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Semaphore, SemaphorePermit};

const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen2.5:3b";
const RATE_WINDOW: Duration = Duration::from_secs(60);
/// Chat-template tokens Qwen adds around every message (`<|im_start|>role\n … <|im_end|>\n`).
const TOKENS_PER_MESSAGE: usize = 4;
/// Successful latencies kept for avg/p50/p95.
const LATENCY_WINDOW: usize = 200;

/// Every limit the service enforces. All of them come from the environment.
#[derive(Serialize, Clone, Debug)]
struct Limits {
    /// Per API key, sliding 60 s window. Counted before anything else, so a
    /// flood of junk requests is cut off as cheaply as real ones.
    requests_per_minute: usize,
    /// Prompt (estimated) + answer (`max_tokens`) must fit. Also sent to
    /// Ollama as `num_ctx`, so Ollama never silently cuts the start of a prompt.
    max_context_tokens: usize,
    /// Cap on `max_tokens` (Ollama's `num_predict`). Larger requests are clamped.
    max_output_tokens: usize,
    /// Requests generating at the same time on the Mac.
    max_concurrent: usize,
    /// Requests allowed to wait for a slot. One more gets 503.
    max_queue: usize,
    /// Request body size; a bigger body is refused before it is parsed.
    max_body_bytes: usize,
}

impl Limits {
    fn from_env() -> Self {
        Self {
            requests_per_minute: env_num("RATE_LIMIT_PER_MINUTE", 20),
            max_context_tokens: env_num("MAX_CONTEXT_TOKENS", 4096),
            max_output_tokens: env_num("MAX_OUTPUT_TOKENS", 512),
            max_concurrent: env_num("MAX_CONCURRENT", 2),
            max_queue: env_num("MAX_QUEUE", 4),
            max_body_bytes: env_num("MAX_BODY_BYTES", 256 * 1024),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

/// Rough token count without a tokenizer. Qwen's tokenizer is byte-level BPE:
/// English runs at ≈4 bytes per token, Russian at more (every Cyrillic letter is
/// 2 bytes), so bytes / 4 errs on the side of refusing, never of overflowing.
fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| m.content.len().div_ceil(4) + TOKENS_PER_MESSAGE)
        .sum()
}

// ---------------------------------------------------------------- errors

/// An OpenAI-style error: a real HTTP status (clients and load balancers act on
/// 401/429/503) plus `{"error": {"message", "type", "code"}}` the page shows as is.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    message: String,
    retry_after: Option<u64>,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            code,
            message: message.into(),
            retry_after: None,
        }
    }

    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", code, message)
    }

    fn unauthorized(message: &str) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", "invalid_api_key", message)
    }

    fn rate_limited(limit: usize, wait: Duration) -> Self {
        let secs = (wait.as_secs_f64().ceil() as u64).max(1);
        Self {
            retry_after: Some(secs),
            ..Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "rate_limit_exceeded",
                format!("Rate limit reached: {limit} requests per minute per key. Retry in {secs} s."),
            )
        }
    }

    fn busy(limits: &Limits) -> Self {
        Self {
            retry_after: Some(2),
            ..Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "server_busy",
                format!(
                    "Busy: {} request(s) generating and {} waiting, the queue is full. Retry shortly.",
                    limits.max_concurrent, limits.max_queue
                ),
            )
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        if let Some(secs) = self.retry_after {
            headers.insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        if self.status == StatusCode::UNAUTHORIZED {
            headers.insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        let body = json!({ "error": { "message": self.message, "type": self.kind, "code": self.code } });
        (self.status, headers, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------- API keys

/// Keys from `API_KEYS` (comma-separated) if set. Otherwise from `file`, one per
/// line, created with one fresh key on first start. The file sits next to the
/// binary on the VDS, and a redeploy only replaces the binary and `.env`, so the
/// key survives redeploys.
fn load_keys(from_env: &str, file: &Path) -> std::io::Result<Vec<String>> {
    let parse = |s: &str| -> Vec<String> {
        s.split([',', '\n'])
            .map(str::trim)
            .filter(|k| !k.is_empty() && !k.starts_with('#'))
            .map(str::to_string)
            .collect()
    };
    let keys = parse(from_env);
    if !keys.is_empty() {
        return Ok(keys);
    }
    if file.exists() {
        let keys = parse(&std::fs::read_to_string(file)?);
        if !keys.is_empty() {
            return Ok(keys);
        }
    }
    let key = generate_key()?;
    std::fs::write(file, format!("{key}\n"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600))?;
    }
    println!("Generated a new API key in {}", file.display());
    Ok(vec![key])
}

fn generate_key() -> std::io::Result<String> {
    let mut bytes = [0u8; 24];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("sk-local-{hex}"))
}

/// Compares every byte, so the time taken doesn't reveal how much of a guess was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// What a key is shown as in stats and logs.
fn key_label(key: &str) -> String {
    let head: String = key.chars().take(12).collect();
    format!("{head}…")
}

// ---------------------------------------------------------------- rate limit

/// Sliding-window limiter: the timestamps of each key's requests in the last minute.
struct RateLimiter {
    per_window: usize,
    window: Duration,
    hits: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl RateLimiter {
    fn new(per_window: usize, window: Duration) -> Self {
        Self {
            per_window: per_window.max(1),
            window,
            hits: Mutex::new(HashMap::new()),
        }
    }

    /// Records a request. `Ok(remaining)` in this window, or `Err(wait)` until a slot frees.
    fn check(&self, key: &str, now: Instant) -> Result<usize, Duration> {
        let mut hits = self.hits.lock().unwrap();
        let queue = hits.entry(key.to_string()).or_default();
        self.expire(queue, now);
        if let Some(&oldest) = queue.front()
            && queue.len() >= self.per_window
        {
            return Err(self.window.saturating_sub(now.duration_since(oldest)));
        }
        queue.push_back(now);
        Ok(self.per_window - queue.len())
    }

    /// Requests left in the window, without recording one.
    fn remaining(&self, key: &str, now: Instant) -> usize {
        let mut hits = self.hits.lock().unwrap();
        let queue = hits.entry(key.to_string()).or_default();
        self.expire(queue, now);
        self.per_window.saturating_sub(queue.len())
    }

    fn expire(&self, queue: &mut VecDeque<Instant>, now: Instant) {
        while let Some(&t) = queue.front() {
            if now.duration_since(t) >= self.window {
                queue.pop_front();
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------- concurrency

/// At most `max_concurrent` requests reach Ollama at once; up to `max_queue`
/// more wait in FIFO order; anything beyond that is refused right away instead
/// of piling up until every client times out.
struct Gate {
    slots: Semaphore,
    max_queue: usize,
    running: AtomicUsize,
    queued: AtomicUsize,
}

/// A slot on the model. Dropping it (answer sent, error, or client gone) frees it.
struct Slot<'a> {
    _permit: SemaphorePermit<'a>,
    running: &'a AtomicUsize,
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.running.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A place in the queue, given back even if the client disconnects while waiting.
struct InQueue<'a>(&'a AtomicUsize);

impl Drop for InQueue<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Gate {
    fn new(max_concurrent: usize, max_queue: usize) -> Self {
        Self {
            slots: Semaphore::new(max_concurrent.max(1)),
            max_queue,
            running: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
        }
    }

    /// A free slot now, or one after waiting in the queue, or `None` when the queue is full.
    async fn enter(&self) -> Option<Slot<'_>> {
        let permit = match self.slots.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                if self.queued.fetch_add(1, Ordering::SeqCst) >= self.max_queue {
                    self.queued.fetch_sub(1, Ordering::SeqCst);
                    return None;
                }
                let _in_queue = InQueue(&self.queued);
                self.slots.acquire().await.ok()?
            }
        };
        self.running.fetch_add(1, Ordering::SeqCst);
        Some(Slot {
            _permit: permit,
            running: &self.running,
        })
    }
}

// ---------------------------------------------------------------- metrics

#[derive(Default)]
struct Metrics {
    by_status: BTreeMap<u16, u64>,
    latencies_ms: VecDeque<u64>,
    completion_tokens: u64,
}

#[derive(Serialize, Debug, PartialEq)]
struct Latency {
    count: usize,
    avg: u64,
    p50: u64,
    p95: u64,
    max: u64,
}

impl Metrics {
    fn latency(&self) -> Latency {
        let mut sorted: Vec<u64> = self.latencies_ms.iter().copied().collect();
        sorted.sort_unstable();
        let pick = |q: f64| -> u64 {
            if sorted.is_empty() {
                0
            } else {
                sorted[((sorted.len() - 1) as f64 * q).round() as usize]
            }
        };
        Latency {
            count: sorted.len(),
            avg: if sorted.is_empty() {
                0
            } else {
                sorted.iter().sum::<u64>() / sorted.len() as u64
            },
            p50: pick(0.5),
            p95: pick(0.95),
            max: sorted.last().copied().unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------- Ollama

// Ollama's native /api/chat: unlike /v1 it takes num_ctx and reports the load
// and generation times.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    num_ctx: usize,
    num_predict: usize,
}

// Durations are in nanoseconds.
#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Message,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    load_duration: u64,
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

// ---------------------------------------------------------------- the service

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<Message>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

/// The private AI service: a gateway in front of an Ollama nobody else can
/// reach. It owns the keys, the limits, the queue and the counters; the model
/// itself stays on the Mac.
struct AppState {
    client: reqwest::Client,
    ollama: String,
    default_model: String,
    keys: Vec<String>,
    limits: Limits,
    rate: RateLimiter,
    gate: Gate,
    metrics: Mutex<Metrics>,
    next_id: AtomicU64,
    started: Instant,
}

impl AppState {
    fn new(ollama: &str, default_model: &str, keys: Vec<String>, limits: Limits) -> Self {
        // Generous: a queued request waits for the ones ahead of it, and a cold
        // model on a busy Mac can take a while.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            ollama: normalize_host(ollama),
            default_model: default_model.to_string(),
            keys,
            rate: RateLimiter::new(limits.requests_per_minute, RATE_WINDOW),
            gate: Gate::new(limits.max_concurrent, limits.max_queue),
            limits,
            metrics: Mutex::new(Metrics::default()),
            next_id: AtomicU64::new(1),
            started: Instant::now(),
        }
    }

    fn authenticate(&self, headers: &HeaderMap) -> Result<String, ApiError> {
        let presented = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::trim)
            .unwrap_or("");
        if presented.is_empty() {
            return Err(ApiError::unauthorized(
                "Missing API key. Send `Authorization: Bearer <key>`.",
            ));
        }
        if self
            .keys
            .iter()
            .any(|k| constant_time_eq(k.as_bytes(), presented.as_bytes()))
        {
            Ok(presented.to_string())
        } else {
            Err(ApiError::unauthorized("Invalid API key."))
        }
    }

    /// The effective model and `max_tokens`, and the prompt estimate — or why the request can't run.
    fn validate(&self, req: &ChatCompletionRequest) -> Result<(String, usize, usize), ApiError> {
        if req.stream {
            return Err(ApiError::bad_request(
                "stream_unsupported",
                "Streaming isn't supported by this service: send \"stream\": false.",
            ));
        }
        if req.messages.is_empty() {
            return Err(ApiError::bad_request("empty_messages", "`messages` is empty."));
        }
        if let Some(m) = req
            .messages
            .iter()
            .find(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
        {
            return Err(ApiError::bad_request(
                "invalid_role",
                format!("Unknown role `{}`: use system, user or assistant.", m.role),
            ));
        }
        if req.messages.iter().all(|m| m.content.trim().is_empty()) {
            return Err(ApiError::bad_request("empty_messages", "Every message is empty."));
        }
        let requested = req.max_tokens.unwrap_or(self.limits.max_output_tokens);
        if requested == 0 {
            return Err(ApiError::bad_request(
                "invalid_max_tokens",
                "`max_tokens` must be at least 1.",
            ));
        }
        let max_tokens = requested.min(self.limits.max_output_tokens);
        let prompt = estimate_tokens(&req.messages);
        if prompt + max_tokens > self.limits.max_context_tokens {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "context_length_exceeded",
                format!(
                    "Too long: ≈{prompt} prompt tokens + {max_tokens} for the answer = {} tokens, \
                     over this service's context limit of {}. Shorten the conversation or lower max_tokens.",
                    prompt + max_tokens,
                    self.limits.max_context_tokens
                ),
            ));
        }
        let model = req
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .unwrap_or(self.default_model.as_str())
            .to_string();
        Ok((model, max_tokens, prompt))
    }

    async fn ollama_get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, ApiError> {
        let response = self
            .client
            .get(format!("{}{path}", self.ollama))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| self.connect_error(e))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Self::upstream_status(status.as_u16(), body));
        }
        serde_json::from_str(&body).map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                "bad_upstream_response",
                format!("Failed to parse Ollama's {path}: {e}"),
            )
        })
    }

    fn connect_error(&self, e: reqwest::Error) -> ApiError {
        if e.is_timeout() {
            ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                "upstream_error",
                "ollama_timeout",
                format!("Ollama at {} didn't answer in time: {e}", self.ollama),
            )
        } else {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                "ollama_unreachable",
                format!(
                    "Can't reach Ollama at {}: {e}. Is `ollama serve` running \
                     (or, on the VDS, is `./tunnel.sh` up on the Mac)?",
                    self.ollama
                ),
            )
        }
    }

    fn upstream_status(status: u16, body: String) -> ApiError {
        let detail = serde_json::from_str::<OllamaError>(&body)
            .map(|e| e.error)
            .unwrap_or(body);
        let (http, code) = if status == 404 {
            (StatusCode::NOT_FOUND, "model_not_found")
        } else {
            (StatusCode::BAD_GATEWAY, "ollama_error")
        };
        ApiError::new(http, "upstream_error", code, format!("Ollama error ({status}): {detail}"))
    }

    async fn call_ollama(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f32>,
        max_tokens: usize,
    ) -> Result<OllamaChatResponse, ApiError> {
        let request = OllamaChatRequest {
            model,
            messages,
            stream: false,
            options: OllamaOptions {
                temperature: temperature.map(|t| t.clamp(0.0, 2.0)),
                num_ctx: self.limits.max_context_tokens,
                num_predict: max_tokens,
            },
        };
        let response = self
            .client
            .post(format!("{}/api/chat", self.ollama))
            .json(&request)
            .send()
            .await
            .map_err(|e| self.connect_error(e))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Self::upstream_status(status.as_u16(), body));
        }
        serde_json::from_str(&body).map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                "bad_upstream_response",
                format!("Failed to parse Ollama's answer: {e}"),
            )
        })
    }

    /// Auth → rate limit → validation → queue → Ollama. Every refusal happens
    /// before the model is touched, and the cheapest checks come first.
    async fn complete(
        &self,
        headers: &HeaderMap,
        body: Result<Json<ChatCompletionRequest>, JsonRejection>,
        started: Instant,
    ) -> Result<Response, ApiError> {
        let key = self.authenticate(headers)?;
        let remaining = self
            .rate
            .check(&key, Instant::now())
            .map_err(|wait| ApiError::rate_limited(self.limits.requests_per_minute, wait))?;
        let Json(req) = body.map_err(|rejection| {
            ApiError::new(
                rejection.status(),
                "invalid_request_error",
                "invalid_body",
                rejection.body_text(),
            )
        })?;
        let (model, max_tokens, prompt_estimate) = self.validate(&req)?;

        let _slot = self
            .gate
            .enter()
            .await
            .ok_or_else(|| ApiError::busy(&self.limits))?;
        let queue_ms = started.elapsed().as_millis() as u64;
        let upstream_started = Instant::now();
        let reply = self
            .call_ollama(&model, &req.messages, req.temperature, max_tokens)
            .await?;
        let upstream_ms = upstream_started.elapsed().as_millis() as u64;
        self.metrics.lock().unwrap().completion_tokens += reply.eval_count;

        let eval_secs = reply.eval_duration as f64 / 1e9;
        let tokens_per_sec = if eval_secs > 0.0 {
            reply.eval_count as f64 / eval_secs
        } else {
            0.0
        };
        let finish_reason = match reply.done_reason.as_deref() {
            Some("length") => "length",
            _ => "stop",
        };
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let body = json!({
            "id": format!("chatcmpl-{id}"),
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": reply.message.content },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": reply.prompt_eval_count,
                "completion_tokens": reply.eval_count,
                "total_tokens": reply.prompt_eval_count + reply.eval_count,
            },
            // Not in OpenAI's schema: what the gateway did with this request.
            "service": {
                "estimated_prompt_tokens": prompt_estimate,
                "max_tokens": max_tokens,
                "num_ctx": self.limits.max_context_tokens,
                "queue_ms": queue_ms,
                "upstream_ms": upstream_ms,
                "load_ms": reply.load_duration / 1_000_000,
                "tokens_per_sec": tokens_per_sec,
                "rate_limit_remaining": remaining,
            },
        });
        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-limit-requests", HeaderValue::from(self.limits.requests_per_minute));
        headers.insert("x-ratelimit-remaining-requests", HeaderValue::from(remaining));
        Ok((headers, Json(body)).into_response())
    }

    fn record(&self, status: StatusCode, elapsed: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics.by_status.entry(status.as_u16()).or_default() += 1;
        if status.is_success() {
            if metrics.latencies_ms.len() == LATENCY_WINDOW {
                metrics.latencies_ms.pop_front();
            }
            metrics.latencies_ms.push_back(elapsed.as_millis() as u64);
        }
    }
}

// ---------------------------------------------------------------- handlers

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let started = Instant::now();
    let response = match state.complete(&headers, body, started).await {
        Ok(response) => response,
        Err(e) => e.into_response(),
    };
    state.record(response.status(), started.elapsed());
    response
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    state.authenticate(&headers)?;
    let tags = state.ollama_get::<TagsResponse>("/api/tags").await?;
    let data: Vec<_> = tags
        .models
        .into_iter()
        .map(|m| json!({ "id": m.name, "object": "model", "owned_by": "ollama" }))
        .collect();
    Ok(Json(json!({ "object": "list", "data": data, "default": state.default_model })))
}

/// Live state of the service for the calling key: limits, queue, outcomes, latency.
async fn stats(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Result<Json<serde_json::Value>, ApiError> {
    let key = state.authenticate(&headers)?;
    let metrics = state.metrics.lock().unwrap();
    Ok(Json(json!({
        "limits": state.limits,
        "running": state.gate.running.load(Ordering::SeqCst),
        "queued": state.gate.queued.load(Ordering::SeqCst),
        "requests": metrics.by_status.values().sum::<u64>(),
        "by_status": metrics.by_status,
        "latency_ms": metrics.latency(),
        "completion_tokens": metrics.completion_tokens,
        "uptime_secs": state.started.elapsed().as_secs(),
        "you": {
            "key": key_label(&key),
            "remaining_this_minute": state.rate.remaining(&key, Instant::now()),
        },
    })))
}

/// Public: is the service up, and can it reach the model? Says nothing about keys or usage.
async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let ollama = match state.ollama_get::<VersionResponse>("/api/version").await {
        Ok(v) => json!({ "reachable": true, "version": v.version }),
        Err(e) => json!({ "reachable": false, "error": e.message }),
    };
    Json(json!({
        "status": "ok",
        "default_model": state.default_model,
        "ollama": ollama,
    }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: Arc<AppState>) -> Router {
    let body_limit = state.limits.max_body_bytes;
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/stats", get(stats))
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_num(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
        .max(1)
}

#[tokio::main]
async fn main() {
    let keys_file = env_or("API_KEYS_FILE", "api_keys.txt");
    let keys = load_keys(&env_or("API_KEYS", ""), Path::new(&keys_file))
        .unwrap_or_else(|e| panic!("can't load or create the API key file {keys_file}: {e}"));
    let limits = Limits::from_env();
    let state = Arc::new(AppState::new(
        &env_or("OLLAMA_HOST", DEFAULT_HOST),
        &env_or("OLLAMA_MODEL", DEFAULT_MODEL),
        keys,
        limits,
    ));
    println!(
        "Ollama at {}, default model {}, {} API key(s){}",
        state.ollama,
        state.default_model,
        state.keys.len(),
        if std::env::var("API_KEYS").is_ok_and(|v| !v.trim().is_empty()) {
            " from API_KEYS".to_string()
        } else {
            format!(" from {keys_file}")
        }
    );
    println!("Limits: {:?}", state.limits);

    let port = env_or("PORT", "3000");
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const KEY: &str = "sk-local-test-one";
    const OTHER_KEY: &str = "sk-local-test-two";

    fn test_limits() -> Limits {
        Limits {
            requests_per_minute: 100,
            max_context_tokens: 4096,
            max_output_tokens: 512,
            max_concurrent: 2,
            max_queue: 4,
            max_body_bytes: 64 * 1024,
        }
    }

    /// A stand-in for Ollama: echoes the last message back after `delay_ms`,
    /// records every request, and tracks how many it served at the same time.
    #[derive(Default)]
    struct Fake {
        delay_ms: u64,
        fail: Option<(u16, &'static str)>,
        requests: Mutex<Vec<Value>>,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
    }

    async fn fake_chat(State(fake): State<Arc<Fake>>, Json(body): Json<Value>) -> Response {
        let now = fake.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        fake.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(fake.delay_ms)).await;
        fake.in_flight.fetch_sub(1, Ordering::SeqCst);
        let last = body["messages"]
            .as_array()
            .and_then(|m| m.last())
            .and_then(|m| m["content"].as_str())
            .unwrap_or("")
            .to_string();
        fake.requests.lock().unwrap().push(body);
        if let Some((status, error)) = fake.fail {
            return (StatusCode::from_u16(status).unwrap(), Json(json!({ "error": error }))).into_response();
        }
        Json(json!({
            "model": "qwen2.5:3b",
            "message": { "role": "assistant", "content": format!("echo: {last}") },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 3,
            "eval_duration": 500_000_000u64,
            "load_duration": 250_000_000u64,
            "total_duration": 1_000_000_000u64
        }))
        .into_response()
    }

    async fn serve(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    async fn fake_ollama(delay_ms: u64, fail: Option<(u16, &'static str)>) -> (String, Arc<Fake>) {
        let fake = Arc::new(Fake {
            delay_ms,
            fail,
            ..Fake::default()
        });
        let router = Router::new()
            .route("/api/chat", post(fake_chat))
            .route("/api/version", get(|| async { Json(json!({ "version": "0.12.0" })) }))
            .route(
                "/api/tags",
                get(|| async { Json(json!({ "models": [{ "name": "qwen2.5:3b" }, { "name": "llama3.2:latest" }] })) }),
            )
            .with_state(fake.clone());
        (serve(router).await, fake)
    }

    /// A host nothing listens on: bind a port, then let it go.
    fn closed_host() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    }

    async fn service(ollama: &str, limits: Limits) -> (String, Arc<AppState>) {
        let state = Arc::new(AppState::new(
            ollama,
            "qwen2.5:3b",
            vec![KEY.to_string(), OTHER_KEY.to_string()],
            limits,
        ));
        (serve(app(state.clone())).await, state)
    }

    async fn post_chat(base: &str, key: Option<&str>, body: Value) -> (u16, reqwest::header::HeaderMap, Value) {
        let mut request = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&body);
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.unwrap();
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        (status, headers, response.json().await.unwrap_or(Value::Null))
    }

    async fn get_json(base: &str, path: &str, key: Option<&str>) -> (u16, Value) {
        let mut request = reqwest::Client::new().get(format!("{base}{path}"));
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap_or(Value::Null))
    }

    fn hi() -> Value {
        json!({ "messages": [{ "role": "user", "content": "hi" }] })
    }

    #[tokio::test]
    async fn everything_but_the_page_and_health_needs_a_key() {
        let (ollama, fake) = fake_ollama(0, None).await;
        let (base, _) = service(&ollama, test_limits()).await;

        let (status, headers, body) = post_chat(&base, None, hi()).await;
        assert_eq!(status, 401);
        assert_eq!(headers["www-authenticate"], "Bearer");
        assert_eq!(body["error"]["code"], "invalid_api_key");
        let (status, _, body) = post_chat(&base, Some("sk-local-wrong"), hi()).await;
        assert_eq!(status, 401);
        assert_eq!(body["error"]["message"], "Invalid API key.");
        assert_eq!(get_json(&base, "/api/stats", None).await.0, 401);
        assert_eq!(get_json(&base, "/v1/models", Some("nope")).await.0, 401);
        assert!(fake.requests.lock().unwrap().is_empty(), "a refused request never reaches Ollama");

        let (status, health) = get_json(&base, "/health", None).await;
        assert_eq!(status, 200);
        assert_eq!(health["ollama"]["reachable"], true);
        assert_eq!(health["ollama"]["version"], "0.12.0");
        let page = reqwest::get(format!("{base}/")).await.unwrap();
        assert_eq!(page.status().as_u16(), 200);

        let (status, models) = get_json(&base, "/v1/models", Some(OTHER_KEY)).await;
        assert_eq!(status, 200);
        assert_eq!(models["data"][1]["id"], "llama3.2:latest");
        assert_eq!(models["default"], "qwen2.5:3b");
    }

    #[tokio::test]
    async fn chat_completion_is_openai_shaped_and_carries_the_limits_to_ollama() {
        let (ollama, fake) = fake_ollama(0, None).await;
        let (base, _) = service(&ollama, test_limits()).await;

        let request = json!({
            "messages": [
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "Столица Франции?" },
                { "role": "assistant", "content": "Париж." },
                { "role": "user", "content": "А Италии?" }
            ],
            "temperature": 0.5,
            "max_tokens": 9999
        });
        let (status, headers, body) = post_chat(&base, Some(KEY), request).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "qwen2.5:3b");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "echo: А Италии?");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 12);
        assert_eq!(body["usage"]["completion_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 15);
        assert_eq!(body["service"]["max_tokens"], 512, "max_tokens is capped");
        assert_eq!(body["service"]["tokens_per_sec"], 6.0);
        assert_eq!(body["service"]["load_ms"], 250);
        assert_eq!(body["service"]["rate_limit_remaining"], 99);
        assert_eq!(headers["x-ratelimit-limit-requests"], "100");
        assert_eq!(headers["x-ratelimit-remaining-requests"], "99");

        let sent = fake.requests.lock().unwrap()[0].clone();
        assert_eq!(sent["stream"], false);
        assert_eq!(sent["model"], "qwen2.5:3b");
        assert_eq!(sent["options"]["num_ctx"], 4096);
        assert_eq!(sent["options"]["num_predict"], 512);
        assert_eq!(sent["options"]["temperature"], 0.5);
        assert_eq!(sent["messages"].as_array().unwrap().len(), 4);

        // Another model, and no temperature: Ollama's default is left alone.
        let (status, _, body) = post_chat(
            &base,
            Some(KEY),
            json!({ "model": " llama3.2 ", "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 16 }),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(body["model"], "llama3.2");
        let sent = fake.requests.lock().unwrap()[1].clone();
        assert_eq!(sent["options"]["num_predict"], 16);
        assert!(sent["options"].get("temperature").is_none());

        let (status, stats) = get_json(&base, "/api/stats", Some(KEY)).await;
        assert_eq!(status, 200);
        assert_eq!(stats["requests"], 2);
        assert_eq!(stats["by_status"]["200"], 2);
        assert_eq!(stats["completion_tokens"], 6);
        assert_eq!(stats["latency_ms"]["count"], 2);
        assert_eq!(stats["you"]["remaining_this_minute"], 98);
        assert_eq!(stats["limits"]["max_context_tokens"], 4096);
    }

    #[tokio::test]
    async fn bad_requests_are_refused_before_ollama() {
        let (ollama, fake) = fake_ollama(0, None).await;
        let (base, state) = service(&ollama, test_limits()).await;

        let cases = [
            (json!({ "messages": [] }), 400, "empty_messages"),
            (json!({ "messages": [{ "role": "user", "content": "  " }] }), 400, "empty_messages"),
            (json!({ "messages": [{ "role": "tool", "content": "x" }] }), 400, "invalid_role"),
            (json!({ "messages": [{ "role": "user", "content": "x" }], "stream": true }), 400, "stream_unsupported"),
            (json!({ "messages": [{ "role": "user", "content": "x" }], "max_tokens": 0 }), 400, "invalid_max_tokens"),
            (json!({ "messages": "nope" }), 422, "invalid_body"),
        ];
        for (body, want_status, want_code) in cases {
            let (status, _, got) = post_chat(&base, Some(KEY), body.clone()).await;
            assert_eq!(status, want_status, "{body} → {got}");
            assert_eq!(got["error"]["code"], want_code, "{body}");
        }

        // Not JSON at all.
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(KEY)
            .header("content-type", "application/json")
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);

        // ≈5000 estimated tokens: over the 4096 context, refused with the numbers.
        let long = "word ".repeat(4000);
        let (status, _, got) = post_chat(
            &base,
            Some(KEY),
            json!({ "messages": [{ "role": "user", "content": long }] }),
        )
        .await;
        assert_eq!(status, 413);
        assert_eq!(got["error"]["code"], "context_length_exceeded");
        let message = got["error"]["message"].as_str().unwrap();
        assert!(message.contains("5004") && message.contains("4096"), "{message}");

        // Fits alone, but not with the answer it asks room for.
        let medium = "word ".repeat(3000); // 3750 + 4 tokens
        let (status, _, _) = post_chat(
            &base,
            Some(KEY),
            json!({ "messages": [{ "role": "user", "content": medium }], "max_tokens": 512 }),
        )
        .await;
        assert_eq!(status, 413);
        let (status, _, _) = post_chat(
            &base,
            Some(KEY),
            json!({ "messages": [{ "role": "user", "content": medium }], "max_tokens": 100 }),
        )
        .await;
        assert_eq!(status, 200);

        // A body over the size limit is refused before it is even parsed.
        let huge = "x".repeat(70 * 1024);
        let (status, _, got) = post_chat(
            &base,
            Some(KEY),
            json!({ "messages": [{ "role": "user", "content": huge }] }),
        )
        .await;
        assert_eq!(status, 413);
        assert_eq!(got["error"]["code"], "invalid_body");

        assert_eq!(fake.requests.lock().unwrap().len(), 1, "only the one valid request reached Ollama");
        let metrics = state.metrics.lock().unwrap();
        assert_eq!(metrics.by_status[&400], 6);
        assert_eq!(metrics.by_status[&413], 3);
        assert_eq!(metrics.by_status[&200], 1);
    }

    #[tokio::test]
    async fn rate_limit_is_per_key_and_says_when_to_retry() {
        let (ollama, fake) = fake_ollama(0, None).await;
        let limits = Limits {
            requests_per_minute: 3,
            ..test_limits()
        };
        let (base, _) = service(&ollama, limits).await;

        // Junk counts too: the limit is checked before the body.
        let junk = json!({ "messages": [] });
        for _ in 0..2 {
            assert_eq!(post_chat(&base, Some(KEY), junk.clone()).await.0, 400);
        }
        let (status, headers, _) = post_chat(&base, Some(KEY), hi()).await;
        assert_eq!(status, 200);
        assert_eq!(headers["x-ratelimit-remaining-requests"], "0");

        let (status, headers, body) = post_chat(&base, Some(KEY), hi()).await;
        assert_eq!(status, 429);
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
        let retry: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
        assert!((1..=60).contains(&retry), "{retry}");

        // Another key has its own budget.
        assert_eq!(post_chat(&base, Some(OTHER_KEY), hi()).await.0, 200);
        assert_eq!(fake.requests.lock().unwrap().len(), 2);

        let (_, stats) = get_json(&base, "/api/stats", Some(KEY)).await;
        assert_eq!(stats["you"]["remaining_this_minute"], 0);
        assert_eq!(stats["by_status"]["429"], 1);
    }

    #[test]
    fn rate_window_slides() {
        let limiter = RateLimiter::new(2, Duration::from_secs(60));
        let t0 = Instant::now();
        assert_eq!(limiter.check("a", t0), Ok(1));
        assert_eq!(limiter.check("a", t0 + Duration::from_secs(10)), Ok(0));
        assert_eq!(
            limiter.check("a", t0 + Duration::from_secs(20)),
            Err(Duration::from_secs(40))
        );
        assert_eq!(limiter.remaining("b", t0), 2);
        // The first hit is a minute old: one slot again.
        assert_eq!(limiter.check("a", t0 + Duration::from_secs(60)), Ok(0));
        assert_eq!(limiter.remaining("a", t0 + Duration::from_secs(71)), 1);
    }

    async fn parallel(base: &str, n: usize) -> Vec<(u16, Value)> {
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let base = base.to_string();
                tokio::spawn(async move {
                    let body = json!({ "messages": [{ "role": "user", "content": format!("request {i}") }] });
                    let (status, headers, body) = post_chat(&base, Some(KEY), body).await;
                    if status == 503 {
                        assert_eq!(headers["retry-after"], "2");
                    }
                    (status, body)
                })
            })
            .collect();
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }
        results
    }

    fn count(results: &[(u16, Value)], status: u16) -> usize {
        results.iter().filter(|(s, _)| *s == status).count()
    }

    #[tokio::test]
    async fn parallel_requests_share_the_slots_and_wait_in_the_queue() {
        let (ollama, fake) = fake_ollama(200, None).await;
        let (base, state) = service(&ollama, test_limits()).await;

        // 2 slots + 4 in the queue: six at once all get answered, never more than two at a time.
        let results = parallel(&base, 6).await;
        assert_eq!(count(&results, 200), 6, "{results:?}");
        assert_eq!(fake.max_in_flight.load(Ordering::SeqCst), 2);
        // Each one got its own answer, not a neighbour's.
        for (_, body) in &results {
            let reply = body["choices"][0]["message"]["content"].as_str().unwrap();
            assert!(reply.starts_with("echo: request "), "{reply}");
        }
        assert!(results.iter().any(|(_, b)| b["service"]["queue_ms"].as_u64().unwrap() >= 150));
        assert_eq!(state.gate.running.load(Ordering::SeqCst), 0);
        assert_eq!(state.gate.queued.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_full_queue_answers_503_at_once() {
        let (ollama, fake) = fake_ollama(400, None).await;
        let limits = Limits {
            max_concurrent: 1,
            max_queue: 2,
            ..test_limits()
        };
        let (base, state) = service(&ollama, limits).await;

        // 1 generating + 2 waiting; the other two are turned away.
        let results = parallel(&base, 5).await;
        assert_eq!(count(&results, 200), 3, "{results:?}");
        assert_eq!(count(&results, 503), 2, "{results:?}");
        let busy = results.iter().find(|(s, _)| *s == 503).unwrap();
        assert_eq!(busy.1["error"]["code"], "server_busy");
        assert_eq!(fake.max_in_flight.load(Ordering::SeqCst), 1);
        assert_eq!(fake.requests.lock().unwrap().len(), 3);
        assert_eq!(state.gate.queued.load(Ordering::SeqCst), 0);

        // And the service recovers: the next one goes through.
        assert_eq!(post_chat(&base, Some(KEY), hi()).await.0, 200);
    }

    #[tokio::test]
    async fn ollama_problems_come_back_as_readable_errors() {
        let (base, _) = service(&closed_host(), test_limits()).await;
        let (status, _, body) = post_chat(&base, Some(KEY), hi()).await;
        assert_eq!(status, 502);
        assert_eq!(body["error"]["code"], "ollama_unreachable");
        assert!(body["error"]["message"].as_str().unwrap().contains("Can't reach Ollama"));
        let (status, health) = get_json(&base, "/health", None).await;
        assert_eq!(status, 200, "the service itself is up");
        assert_eq!(health["ollama"]["reachable"], false);

        let (ollama, _) = fake_ollama(0, Some((404, "model \"nope\" not found, try pulling it first"))).await;
        let (base, state) = service(&ollama, test_limits()).await;
        let (status, _, body) = post_chat(
            &base,
            Some(KEY),
            json!({ "model": "nope", "messages": [{ "role": "user", "content": "hi" }] }),
        )
        .await;
        assert_eq!(status, 404);
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(body["error"]["message"].as_str().unwrap().contains("try pulling it first"));
        // The slot was given back.
        assert_eq!(state.gate.running.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn keys_come_from_the_env_or_a_generated_file() {
        let dir = std::env::temp_dir().join(format!("private-llm-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("api_keys.txt");
        let _ = std::fs::remove_file(&file);

        assert_eq!(load_keys(" a , b ,", &file).unwrap(), ["a", "b"]);
        assert!(!file.exists(), "API_KEYS set: no file is written");

        let generated = load_keys("", &file).unwrap();
        assert_eq!(generated.len(), 1);
        assert!(generated[0].starts_with("sk-local-") && generated[0].len() == 9 + 48);
        // The second start reads the same key back.
        assert_eq!(load_keys("", &file).unwrap(), generated);

        std::fs::write(&file, "# comment\nk1\n\nk2\n").unwrap();
        assert_eq!(load_keys("", &file).unwrap(), ["k1", "k2"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn small_helpers() {
        let messages = vec![
            Message { role: "user".into(), content: "abcd".into() },
            Message { role: "user".into(), content: "Привет".into() }, // 12 bytes
        ];
        assert_eq!(estimate_tokens(&messages), (1 + 4) + (3 + 4));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert_eq!(key_label("sk-local-0123456789"), "sk-local-012…");
        assert_eq!(normalize_host(""), DEFAULT_HOST);
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(normalize_host("http://gpu-box:11434/"), "http://gpu-box:11434");

        let mut metrics = Metrics::default();
        metrics.latencies_ms.extend([100, 200, 300, 400, 1000]);
        assert_eq!(
            metrics.latency(),
            Latency { count: 5, avg: 400, p50: 300, p95: 1000, max: 1000 }
        );
    }
}

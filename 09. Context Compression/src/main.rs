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

/// What a single chat turn (or a reset/history fetch) reports: token usage
/// as in lesson 08, plus the numbers that make context compression legible -
/// what would have been sent without it, what was actually sent, and how
/// much that saved.
#[derive(Serialize)]
struct TokenReport {
    /// This request's user message, counted locally before the call.
    estimated_request_tokens: u32,
    /// The whole raw history before this turn was added, counted locally.
    estimated_history_tokens_before: u32,
    /// This turn's reply, counted locally (0 when the turn failed).
    estimated_response_tokens: u32,
    /// The provider's own accounting for this turn, when it returns one.
    /// `prompt_tokens` here reflects whatever was actually sent - the full
    /// history, or the compacted system+summary+last-N context.
    actual: Option<Usage>,
    /// The complete, uncompressed conversation so far - every message ever
    /// exchanged, never trimmed. This is the persisted record, and what
    /// lesson 08's agent would have sent every turn.
    raw_history_tokens_after: u32,
    /// What this turn's request would have cost with compaction switched
    /// off: the full raw history including the new user message.
    uncompacted_context_tokens: u32,
    /// What was actually sent to the model this turn: the full history when
    /// compaction is off, or system + summary + last N messages when on.
    sent_context_tokens: u32,
    /// `uncompacted_context_tokens - sent_context_tokens`. Zero when
    /// compaction is off (nothing to save - the two are identical).
    tokens_saved_this_turn: u32,
    /// Whether this turn's request used the compacted context or the full
    /// history.
    compaction_enabled: bool,
    /// Whether a background summarization (folding older messages into the
    /// summary) ran as part of this turn.
    compaction_triggered: bool,
    /// Tokens currently held in the running summary text (0 if none yet).
    summary_tokens: u32,
    /// How many raw messages are currently folded into the summary.
    summarized_message_count: usize,
    /// A configurable reference limit (see `CONTEXT_LIMIT_TOKENS`) - not a
    /// guarantee of the real backend's context window, just something to
    /// watch what's actually being sent against.
    context_limit: u32,
    /// `sent_context_tokens` (or the API's real `prompt_tokens`, when
    /// available) against `context_limit` - this is what determines
    /// whether the *next* call risks getting rejected, which is why it
    /// tracks the sent context rather than the ever-growing raw history.
    percent_of_limit: f32,
    /// Only present when `PRICE_PER_1M_INPUT_TOKENS`/`PRICE_PER_1M_OUTPUT_TOKENS`
    /// are configured; uses actual usage when available, the estimate otherwise.
    estimated_cost_usd: Option<f64>,
}

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer clearly and concisely.";

/// The only two models this agent is allowed to call for chat turns - both
/// live behind the same endpoint/key, so switching between them is just a
/// different `model` value on the same request, not a different provider.
const MODEL_FLASH: &str = "deepseek-v4-flash";
const MODEL_PRO: &str = "deepseek-v4-pro";

/// Roughly typical chat-format overhead per message (role/name delimiters)
/// on top of the content itself, matching the widely-cited ~4-tokens-per-message
/// rule of thumb for OpenAI-style chat formatting. Approximate, like the rest
/// of this heuristic.
const MESSAGE_OVERHEAD_TOKENS: u32 = 4;

/// Instructs the (cheap, same-provider) model used for the background
/// summarization call. Kept separate from `SYSTEM_PROMPT` since this call
/// never talks to the user - its only output is the updated summary text.
const SUMMARY_SYSTEM_PROMPT: &str = "You condense conversation history for reuse as context in a later request. You will be given an optional PRIOR SUMMARY and a NEW block of turns. Produce a single, concise, updated summary that folds the new block into the prior one: preserve facts, names, decisions, numbers, and anything a later turn might need to refer back to; drop small talk and filler. Output only the updated summary text, with no preamble, headings, or commentary.";

/// The synthetic system-role message the summary gets wrapped in when it's
/// spliced into a compacted request, so the model sees it as background
/// context rather than something either party said.
const SUMMARY_MESSAGE_PREFIX: &str = "Summary of earlier conversation:\n";

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

/// Persisted separately from `history.json`: the running summary text and
/// how many of the raw (non-system) messages are already folded into it.
/// Keeping this apart from history means the full, uncompressed transcript
/// is never lost or rewritten - the summary is purely an additional, more
/// compact view onto the same data.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct SummaryState {
    summary: Option<String>,
    summarized_through: usize,
}

/// Decides whether a compaction pass is due, and if so, which slice of the
/// not-yet-summarized messages to fold in. Pulled out as a free function
/// (no I/O, no locking) so the threshold/window math can be unit tested
/// without a network call.
///
/// `non_system_len` is the number of non-system messages in history.
/// Returns `(start, count)`: fold `count` messages starting at offset
/// `start` within the non-system slice (i.e. history indices
/// `1+start .. 1+start+count`, skipping the system prompt at index 0).
fn plan_compaction(
    non_system_len: usize,
    keep_last_n: usize,
    summarized_through: usize,
    summarize_every: usize,
    force: bool,
) -> Option<(usize, usize)> {
    let foldable = non_system_len.saturating_sub(keep_last_n);
    let unfolded = foldable.saturating_sub(summarized_through);
    if unfolded == 0 {
        return None;
    }
    if !force && unfolded < summarize_every {
        return None;
    }
    Some((summarized_through, unfolded))
}

/// Builds the message list actually sent to the model for one turn.
///
/// - `use_compaction: false` - the full raw history, unchanged (lesson 08's
///   behavior).
/// - `use_compaction: true` - the system prompt, then (if a summary exists)
///   one synthetic system message carrying it, then only the last
///   `keep_last_n` raw messages. Everything older than that window is
///   represented solely by the summary, never sent verbatim again.
fn build_context(
    history: &[Message],
    summary: &Option<String>,
    use_compaction: bool,
    keep_last_n: usize,
) -> Vec<Message> {
    if !use_compaction || history.is_empty() {
        return history.to_vec();
    }

    let mut context = Vec::with_capacity(keep_last_n + 2);
    context.push(history[0].clone());
    if let Some(summary_text) = summary {
        context.push(Message {
            role: "system".to_string(),
            content: format!("{SUMMARY_MESSAGE_PREFIX}{summary_text}"),
        });
    }

    let non_system_len = history.len() - 1;
    let take = keep_last_n.min(non_system_len);
    let tail_start = history.len() - take;
    context.extend_from_slice(&history[tail_start..]);
    context
}

fn build_summary_prompt(prior: &Option<String>, block: &[Message]) -> String {
    let mut prompt = String::new();
    if let Some(prior_summary) = prior {
        prompt.push_str("PRIOR SUMMARY:\n");
        prompt.push_str(prior_summary);
        prompt.push_str("\n\n");
    }
    prompt.push_str("NEW TURNS TO FOLD IN:\n");
    for message in block {
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt
}

/// Per-request overrides a caller may supply on top of the running
/// conversation. Every field is optional; omitted ones are left out of the
/// API request entirely rather than sent as an explicit null/default.
struct ChatOptions {
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
    /// `None` defers to the agent's configured default - see
    /// `COMPACTION_ENABLED`. An explicit value lets the UI A/B a single
    /// conversation with compaction on vs. off.
    use_compaction: Option<bool>,
}

struct RespondResult {
    reply: String,
    tokens: TokenReport,
}

/// Bundles `Agent::new`'s configuration so the constructor doesn't grow an
/// ever-longer flat argument list every time a lesson adds a knob.
struct AgentConfig {
    endpoint: String,
    api_key: String,
    default_model: String,
    history_path: String,
    summary_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    keep_last_n: usize,
    summarize_every: usize,
    compaction_default: bool,
}

/// The agent: owns the conversation history, knows how to turn a user
/// message into an LLM call and back into a reply, persists the history to
/// `history_path` after every completed turn, and - new in this lesson -
/// maintains a running summary of everything older than the last
/// `keep_last_n` messages, persisted separately to `summary_path`, so a
/// compacted request (system + summary + last N) can stand in for the full
/// history without ever discarding the full history itself.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    default_model: String,
    history_path: String,
    summary_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    keep_last_n: usize,
    summarize_every: usize,
    compaction_default: bool,
    history: Mutex<Vec<Message>>,
    summary_state: Mutex<SummaryState>,
}

impl Agent {
    fn new(client: reqwest::Client, config: AgentConfig) -> Self {
        let history = Self::load_history(&config.history_path);
        let summary_state = Self::load_summary_state(&config.summary_path);
        Self {
            client,
            endpoint: config.endpoint,
            api_key: config.api_key,
            default_model: config.default_model,
            history_path: config.history_path,
            summary_path: config.summary_path,
            context_limit: config.context_limit,
            price_per_1m_input: config.price_per_1m_input,
            price_per_1m_output: config.price_per_1m_output,
            keep_last_n: config.keep_last_n,
            summarize_every: config.summarize_every,
            compaction_default: config.compaction_default,
            history: Mutex::new(history),
            summary_state: Mutex::new(summary_state),
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

    /// Same fall-back-on-any-problem policy as `load_history`: a missing or
    /// corrupted summary file just means "no summary yet", never a startup
    /// failure.
    fn load_summary_state(path: &str) -> SummaryState {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return SummaryState::default(),
        };
        match serde_json::from_str::<SummaryState>(&contents) {
            Ok(state) => state,
            Err(e) => {
                eprintln!(
                    "Warning: failed to parse {path} ({e}); starting with no summary."
                );
                SummaryState::default()
            }
        }
    }

    /// Writes the full current history to disk. Best-effort: a write failure
    /// is logged but never allowed to break the in-memory conversation the
    /// user is actively having.
    fn persist_history(&self) {
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

    fn persist_summary(&self) {
        let state = self.summary_state.lock().unwrap().clone();
        match serde_json::to_string_pretty(&state) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.summary_path, json) {
                    eprintln!("Warning: failed to write {}: {e}", self.summary_path);
                }
            }
            Err(e) => eprintln!("Warning: failed to serialize summary state: {e}"),
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

    fn summary_snapshot(&self) -> SummaryState {
        self.summary_state.lock().unwrap().clone()
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

    /// A descriptive snapshot for endpoints that aren't mid-turn (reset,
    /// history-on-load): what a request would look like *right now* under
    /// this agent's default compaction setting, with no real call made.
    fn snapshot(&self) -> TokenReport {
        let history = self.history.lock().unwrap().clone();
        let summary_state = self.summary_snapshot();
        let raw_tokens = history_tokens(&history);
        let sent = build_context(
            &history,
            &summary_state.summary,
            self.compaction_default,
            self.keep_last_n,
        );
        let sent_context_tokens = history_tokens(&sent);
        let summary_tokens = summary_state
            .summary
            .as_deref()
            .map(estimate_tokens)
            .unwrap_or(0);

        TokenReport {
            estimated_request_tokens: 0,
            estimated_history_tokens_before: raw_tokens,
            estimated_response_tokens: 0,
            actual: None,
            raw_history_tokens_after: raw_tokens,
            uncompacted_context_tokens: raw_tokens,
            sent_context_tokens,
            tokens_saved_this_turn: raw_tokens.saturating_sub(sent_context_tokens),
            compaction_enabled: self.compaction_default,
            compaction_triggered: false,
            summary_tokens,
            summarized_message_count: summary_state.summarized_through,
            context_limit: self.context_limit,
            percent_of_limit: (sent_context_tokens as f32 / self.context_limit as f32) * 100.0,
            estimated_cost_usd: None,
        }
    }

    async fn respond(&self, user_message: &str, options: ChatOptions) -> RespondResult {
        let use_compaction = options.use_compaction.unwrap_or(self.compaction_default);

        let (full_history, estimated_request_tokens, estimated_history_before) = {
            let mut history = self.history.lock().unwrap();
            let history_before = history_tokens(&history);
            history.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
            let request_tokens = message_tokens(history.last().unwrap());
            (history.clone(), request_tokens, history_before)
        };

        let summary_before = self.summary_snapshot();
        let sent_messages = build_context(
            &full_history,
            &summary_before.summary,
            use_compaction,
            self.keep_last_n,
        );
        let uncompacted_context_tokens = history_tokens(&full_history);
        let sent_context_tokens = history_tokens(&sent_messages);
        let summary_tokens_before = summary_before
            .summary
            .as_deref()
            .map(estimate_tokens)
            .unwrap_or(0);

        let make_report = |actual: Option<Usage>,
                            estimated_response_tokens: u32,
                            raw_history_tokens_after: u32,
                            compaction_triggered: bool,
                            summarized_message_count: usize|
         -> TokenReport {
            let effective_sent = actual
                .as_ref()
                .map(|u| u.prompt_tokens)
                .unwrap_or(sent_context_tokens);
            let percent_of_limit = (effective_sent as f32 / self.context_limit as f32) * 100.0;
            let estimated_cost_usd = match &actual {
                Some(u) => self.estimate_cost(u.prompt_tokens, u.completion_tokens),
                None => self.estimate_cost(sent_context_tokens, estimated_response_tokens),
            };
            TokenReport {
                estimated_request_tokens,
                estimated_history_tokens_before: estimated_history_before,
                estimated_response_tokens,
                actual,
                raw_history_tokens_after,
                uncompacted_context_tokens,
                sent_context_tokens,
                tokens_saved_this_turn: uncompacted_context_tokens.saturating_sub(sent_context_tokens),
                compaction_enabled: use_compaction,
                compaction_triggered,
                summary_tokens: summary_tokens_before,
                summarized_message_count,
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
            messages: sent_messages,
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
                    tokens: make_report(
                        None,
                        0,
                        estimated_history_before,
                        false,
                        summary_before.summarized_through,
                    ),
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
                tokens: make_report(
                    None,
                    0,
                    estimated_history_before,
                    false,
                    summary_before.summarized_through,
                ),
            };
        }

        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(p) => p,
            Err(e) => {
                self.history.lock().unwrap().pop();
                return RespondResult {
                    reply: format!("Failed to parse response: {e}"),
                    tokens: make_report(
                        None,
                        0,
                        estimated_history_before,
                        false,
                        summary_before.summarized_through,
                    ),
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

        let raw_history_tokens_after = {
            let mut history = self.history.lock().unwrap();
            history.push(Message {
                role: "assistant".to_string(),
                content: reply.clone(),
            });
            history_tokens(&history)
        };
        self.persist_history();

        let compaction_triggered = self.run_compaction(false).await;
        let summarized_message_count = self.summary_snapshot().summarized_through;

        RespondResult {
            tokens: make_report(
                parsed.usage,
                estimated_response_tokens,
                raw_history_tokens_after,
                compaction_triggered,
                summarized_message_count,
            ),
            reply,
        }
    }

    /// Folds the oldest not-yet-summarized messages (everything older than
    /// the last `keep_last_n`) into the running summary via one extra LLM
    /// call, when `plan_compaction` says it's due (or unconditionally, when
    /// `force` is set - used by `/api/compact` for demoing without waiting
    /// for `summarize_every` more messages). Best-effort: any failure just
    /// means the summary stays as it was and gets retried on the next turn;
    /// it never breaks the primary chat flow.
    async fn run_compaction(&self, force: bool) -> bool {
        let (block, summarized_through_before, prior_summary) = {
            let history = self.history.lock().unwrap();
            let state = self.summary_state.lock().unwrap();
            let non_system_len = history.len() - 1;
            match plan_compaction(
                non_system_len,
                self.keep_last_n,
                state.summarized_through,
                self.summarize_every,
                force,
            ) {
                Some((start, count)) => {
                    let begin = 1 + start;
                    let end = begin + count;
                    (history[begin..end].to_vec(), start, state.summary.clone())
                }
                None => return false,
            }
        };

        let request = ChatCompletionRequest {
            model: self.default_model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: SUMMARY_SYSTEM_PROMPT.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: build_summary_prompt(&prior_summary, &block),
                },
            ],
            temperature: Some(0.2),
            max_tokens: None,
            stop: None,
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
                eprintln!("Warning: compaction request failed, will retry next turn: {e}");
                return false;
            }
        };

        if !response.status().is_success() {
            eprintln!(
                "Warning: compaction API error ({}), will retry next turn.",
                response.status()
            );
            return false;
        }

        let body = response.text().await.unwrap_or_default();
        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: failed to parse compaction response, will retry next turn: {e}");
                return false;
            }
        };
        let new_summary = match parsed.choices.into_iter().next() {
            Some(choice) => choice.message.content,
            None => return false,
        };

        {
            let mut state = self.summary_state.lock().unwrap();
            // Guards against a concurrent compaction call (e.g. an
            // overlapping `/api/compact`) having already advanced things
            // out from under this one.
            if state.summarized_through != summarized_through_before {
                return false;
            }
            state.summary = Some(new_summary);
            state.summarized_through = summarized_through_before + block.len();
        }
        self.persist_summary();
        true
    }

    fn reset(&self) -> TokenReport {
        {
            let mut history = self.history.lock().unwrap();
            *history = Self::seed_history();
        }
        {
            let mut state = self.summary_state.lock().unwrap();
            *state = SummaryState::default();
        }
        self.persist_history();
        self.persist_summary();
        self.snapshot()
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
    compaction: Option<bool>,
}

#[derive(Serialize)]
struct ChatResponse {
    reply: String,
    tokens: TokenReport,
}

#[derive(Serialize)]
struct HistoryResponse {
    messages: Vec<Message>,
    summary: Option<String>,
    tokens: TokenReport,
}

#[derive(Serialize)]
struct SummaryResponse {
    summary: Option<String>,
    summary_tokens: u32,
    summarized_message_count: usize,
    keep_last_n: usize,
    summarize_every: usize,
}

#[derive(Serialize)]
struct CompactResponse {
    triggered: bool,
    summary: Option<String>,
    summary_tokens: u32,
    summarized_message_count: usize,
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
                use_compaction: req.compaction,
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

/// Lets the page rehydrate the visible chat bubbles, the current summary,
/// and the token stats bar on load, so a browser reload after the server
/// restarts shows the same conversation and running totals.
async fn history(State(agent): State<Arc<Agent>>) -> Json<HistoryResponse> {
    let tokens = agent.snapshot();
    Json(HistoryResponse {
        messages: agent.visible_history(),
        summary: agent.summary_snapshot().summary,
        tokens,
    })
}

/// A dedicated read of just the compaction state, for a UI panel that wants
/// to show the running summary without also re-fetching the whole
/// transcript.
async fn summary(State(agent): State<Arc<Agent>>) -> Json<SummaryResponse> {
    let state = agent.summary_snapshot();
    let summary_tokens = state.summary.as_deref().map(estimate_tokens).unwrap_or(0);
    Json(SummaryResponse {
        summary: state.summary,
        summary_tokens,
        summarized_message_count: state.summarized_through,
        keep_last_n: agent.keep_last_n,
        summarize_every: agent.summarize_every,
    })
}

/// Forces an immediate compaction pass regardless of the `summarize_every`
/// threshold, so the effect can be demonstrated without first sending ten
/// more messages. A no-op (returns `triggered: false`) if there's nothing
/// past the last-N window left to fold.
async fn compact(State(agent): State<Arc<Agent>>) -> Json<CompactResponse> {
    let triggered = agent.run_compaction(true).await;
    let state = agent.summary_snapshot();
    let summary_tokens = state.summary.as_deref().map(estimate_tokens).unwrap_or(0);
    Json(CompactResponse {
        triggered,
        summary: state.summary,
        summary_tokens,
        summarized_message_count: state.summarized_through,
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
    let summary_path =
        std::env::var("SUMMARY_FILE").unwrap_or_else(|_| "summary.json".to_string());
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
    let keep_last_n: usize = std::env::var("KEEP_LAST_N_MESSAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(6);
    let summarize_every: usize = std::env::var("SUMMARIZE_EVERY_N_MESSAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(10);
    let compaction_default = std::env::var("COMPACTION_ENABLED")
        .ok()
        .map(|v| !v.eq_ignore_ascii_case("false") && v != "0")
        .unwrap_or(true);

    let agent = Agent::new(
        reqwest::Client::new(),
        AgentConfig {
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            default_model,
            history_path: history_path.clone(),
            summary_path: summary_path.clone(),
            context_limit,
            price_per_1m_input,
            price_per_1m_output,
            keep_last_n,
            summarize_every,
            compaction_default,
        },
    );
    let restored_turns = agent.history.lock().unwrap().len().saturating_sub(1);
    if restored_turns > 0 {
        println!(
            "Restored {restored_turns} message(s) from {history_path} (~{} tokens); compaction {}.",
            agent.history_tokens_now(),
            if compaction_default { "on by default" } else { "off by default" }
        );
    } else {
        println!(
            "No previous history at {history_path}; starting a fresh conversation. Compaction {}.",
            if compaction_default { "on by default" } else { "off by default" }
        );
    }
    let agent = Arc::new(agent);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/reset", post(reset))
        .route("/api/history", get(history))
        .route("/api/summary", get(summary))
        .route("/api/compact", post(compact))
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

    fn temp_path(label: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "ai-advent-lesson-09-{label}-{}-{}.json",
                std::process::id(),
                n
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn test_agent_with_paths(history_path: String, summary_path: String) -> Agent {
        Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                default_model: "test-model".to_string(),
                history_path,
                summary_path,
                context_limit: 1000,
                price_per_1m_input: Some(2.0),
                price_per_1m_output: Some(4.0),
                keep_last_n: 4,
                summarize_every: 10,
                compaction_default: true,
            },
        )
    }

    fn test_agent() -> Agent {
        test_agent_with_paths(temp_path("history"), temp_path("summary"))
    }

    fn sample_history(non_system_count: usize) -> Vec<Message> {
        let mut history = vec![Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }];
        for i in 0..non_system_count {
            history.push(Message {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("message {i}"),
            });
        }
        history
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
    fn plan_compaction_is_none_when_history_fits_within_the_keep_window() {
        assert_eq!(plan_compaction(5, 6, 0, 10, false), None);
    }

    #[test]
    fn plan_compaction_is_none_below_the_threshold() {
        assert_eq!(plan_compaction(15, 6, 0, 10, false), None); // 9 unfolded, < 10
    }

    #[test]
    fn plan_compaction_fires_once_the_threshold_is_reached() {
        assert_eq!(plan_compaction(16, 6, 0, 10, false), Some((0, 10)));
    }

    #[test]
    fn plan_compaction_accounts_for_what_is_already_summarized() {
        // 25 non-system messages, keep 6, already summarized 10 -> 9 unfolded, below threshold.
        assert_eq!(plan_compaction(25, 6, 10, 10, false), None);
        // One more message tips it to exactly 10 unfolded.
        assert_eq!(plan_compaction(26, 6, 10, 10, false), Some((10, 10)));
    }

    #[test]
    fn plan_compaction_force_ignores_the_threshold_but_not_an_empty_fold() {
        assert_eq!(plan_compaction(9, 6, 0, 10, true), Some((0, 3)));
        assert_eq!(plan_compaction(6, 6, 0, 10, true), None); // nothing past the window
    }

    #[test]
    fn build_context_without_compaction_returns_full_history_unchanged() {
        let history = sample_history(12);
        let ctx = build_context(&history, &Some("SUMMARY".to_string()), false, 4);
        assert_eq!(ctx.len(), history.len());
        assert_eq!(ctx[1].content, "message 0");
    }

    #[test]
    fn build_context_with_compaction_and_a_summary_keeps_only_the_tail() {
        let history = sample_history(12);
        let ctx = build_context(&history, &Some("SUMMARY TEXT".to_string()), true, 4);
        // system + summary-note + last 4 raw messages.
        assert_eq!(ctx.len(), 6);
        assert_eq!(ctx[0].role, "system");
        assert!(ctx[1].content.contains("SUMMARY TEXT"));
        assert_eq!(ctx[2].content, "message 8");
        assert_eq!(ctx[5].content, "message 11");
    }

    #[test]
    fn build_context_with_compaction_and_no_summary_skips_the_summary_note() {
        let history = sample_history(12);
        let ctx = build_context(&history, &None, true, 4);
        assert_eq!(ctx.len(), 5); // system + last 4, no summary note
        assert_eq!(ctx[1].content, "message 8");
    }

    #[test]
    fn build_context_never_duplicates_messages_when_history_is_shorter_than_the_window() {
        let history = sample_history(2);
        let ctx = build_context(&history, &None, true, 4);
        assert_eq!(ctx.len(), 3); // system + both messages, nothing to summarize yet
    }

    #[test]
    fn history_tokens_includes_the_seeded_system_prompt() {
        let agent = test_agent();
        assert!(agent.history_tokens_now() > 0);
    }

    #[test]
    fn reset_clears_history_and_summary_and_persists_both() {
        let history_path = temp_path("history");
        let summary_path = temp_path("summary");
        let agent = test_agent_with_paths(history_path.clone(), summary_path.clone());
        agent.history.lock().unwrap().push(Message {
            role: "user".to_string(),
            content: "a fairly long message to push the token count up a bit".to_string(),
        });
        agent.summary_state.lock().unwrap().summary = Some("stale summary".to_string());

        let report = agent.reset();
        assert_eq!(report.raw_history_tokens_after, agent.history_tokens_now());
        assert_eq!(agent.summary_snapshot().summary, None);

        let history_on_disk: Vec<Message> =
            serde_json::from_str(&std::fs::read_to_string(&history_path).unwrap()).unwrap();
        assert_eq!(history_on_disk.len(), 1);
        let summary_on_disk: SummaryState =
            serde_json::from_str(&std::fs::read_to_string(&summary_path).unwrap()).unwrap();
        assert_eq!(summary_on_disk.summary, None);

        let _ = std::fs::remove_file(&history_path);
        let _ = std::fs::remove_file(&summary_path);
    }

    #[test]
    fn estimate_cost_is_none_without_configured_prices() {
        let agent = Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                default_model: "test-model".to_string(),
                history_path: temp_path("history"),
                summary_path: temp_path("summary"),
                context_limit: 1000,
                price_per_1m_input: None,
                price_per_1m_output: None,
                keep_last_n: 4,
                summarize_every: 10,
                compaction_default: true,
            },
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
        assert_eq!(agent.summary_snapshot().summary, None);
    }

    #[test]
    fn new_loads_previously_persisted_history_and_summary() {
        let history_path = temp_path("history");
        let summary_path = temp_path("summary");
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
        std::fs::write(&history_path, serde_json::to_string(&seeded).unwrap()).unwrap();
        let seeded_summary = SummaryState {
            summary: Some("earlier context".to_string()),
            summarized_through: 4,
        };
        std::fs::write(
            &summary_path,
            serde_json::to_string(&seeded_summary).unwrap(),
        )
        .unwrap();

        let agent = test_agent_with_paths(history_path.clone(), summary_path.clone());
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 3);
        assert_eq!(history[1].content, "hi");
        assert_eq!(history[2].content, "hello");
        let summary = agent.summary_snapshot();
        assert_eq!(summary.summary, Some("earlier context".to_string()));
        assert_eq!(summary.summarized_through, 4);

        let _ = std::fs::remove_file(&history_path);
        let _ = std::fs::remove_file(&summary_path);
    }

    #[test]
    fn new_falls_back_to_seed_on_invalid_json() {
        let history_path = temp_path("history");
        std::fs::write(&history_path, "not json").unwrap();

        let agent = test_agent_with_paths(history_path.clone(), temp_path("summary"));
        let history = agent.history_snapshot();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");

        let _ = std::fs::remove_file(&history_path);
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

    #[test]
    fn snapshot_reports_zero_savings_when_nothing_is_summarized_yet() {
        let agent = test_agent();
        let report = agent.snapshot();
        assert_eq!(report.tokens_saved_this_turn, 0);
        assert_eq!(report.summarized_message_count, 0);
    }

    #[test]
    fn snapshot_reports_savings_once_a_summary_exists_and_history_exceeds_the_window() {
        let agent = test_agent(); // keep_last_n = 4
        {
            let mut history = agent.history.lock().unwrap();
            for i in 0..12 {
                history.push(Message {
                    role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                    content: format!("a reasonably long message body number {i}"),
                });
            }
        }
        agent.summary_state.lock().unwrap().summary = Some("condensed context".to_string());
        agent.summary_state.lock().unwrap().summarized_through = 8;

        let report = agent.snapshot();
        assert!(report.sent_context_tokens < report.uncompacted_context_tokens);
        assert!(report.tokens_saved_this_turn > 0);
    }
}

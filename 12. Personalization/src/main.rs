use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use serde::de::DeserializeOwned;
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

/// Token accounting straight from the API's own `usage` field - the
/// authoritative number for the turn that just happened, as opposed to our
/// own pre-call estimate.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer clearly and concisely.";

/// Roughly typical chat-format overhead per message (role/name delimiters)
/// on top of the content itself, matching the widely-cited ~4-tokens-per-message
/// rule of thumb for OpenAI-style chat formatting. Approximate, like the rest
/// of this heuristic - the API's own `usage` field is the authoritative number.
const MESSAGE_OVERHEAD_TOKENS: u32 = 4;

/// A dependency-free, deliberately approximate token estimate: runs of
/// letters/digits are charged at ~4 characters per token, and each
/// punctuation/symbol character is charged as its own token. Not any
/// particular provider's real tokenizer - just a ballpark.
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

// ---------------------------------------------------------------------
// Personalization: an explicit, user-authored profile that sits beside the
// three memory layers from lesson 11 rather than inside them.
//
// The distinction that matters here is *how* each thing gets written:
// long-term memory (see below) is inferred by an LLM from what the user
// says over time ("what does the agent happen to have picked up about
// me"); the profile is only ever written by an explicit API call ("what
// the user has deliberately told the agent to always do"). Style, tone,
// format and hard constraints belong in the profile precisely because
// they shouldn't be left to inference - a user who wants terse, jargon-
// free answers wants that guaranteed, not "usually".
// ---------------------------------------------------------------------

/// The personalization profile: preferences about *how* to answer, as
/// opposed to long-term memory's facts about *what* is known about the
/// user. Every field is optional/empty by default so a fresh agent has no
/// profile at all and behaves exactly like lesson 11's plain agent.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
struct UserProfile {
    #[serde(default)]
    style: Option<String>,
    #[serde(default)]
    tone: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    constraints: Vec<String>,
}

impl UserProfile {
    fn is_empty(&self) -> bool {
        self.style.is_none()
            && self.tone.is_none()
            && self.format.is_none()
            && self.language.is_none()
            && self.constraints.is_empty()
    }
}

// ---------------------------------------------------------------------
// The three memory layers (unchanged from lesson 11)
// ---------------------------------------------------------------------

/// Short-term memory: the live, ordered transcript of the *current
/// conversation*. This is the only layer with an unconditional write rule -
/// every user/assistant turn is appended here automatically, with no
/// judgment call about whether it's "worth" keeping. It is also the only
/// layer a plain `/api/reset` ever touches.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct ShortTermMemory {
    messages: Vec<Message>,
}

/// Working memory: a scratchpad scoped to exactly one task. Nothing written
/// here is meant to outlive the task - see `Agent::finish_task` for the one
/// explicit bridge (promotion) that lets a fact survive into long-term
/// memory before this layer is thrown away. `task` is `None` whenever no
/// task is active, which is also the gate that decides whether this layer
/// is extracted into or injected into a request at all.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct WorkingMemory {
    task: Option<String>,
    facts: BTreeMap<String, String>,
}

/// Long-term memory: durable, cross-task, cross-session knowledge about the
/// user - identity, standing preferences, decisions explicitly meant to
/// stick. Nothing here is ever cleared by `/api/reset` or
/// `/api/task/finish`; only an explicit `/api/memory/forget` call touches
/// it. Unlike the profile above, this is written by the LLM extractor, not
/// the user directly - it's what the agent *infers*, not what it's *told*.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct LongTermMemory {
    facts: BTreeMap<String, String>,
}

/// Which layer an explicit `/api/memory/remember` or `/api/memory/forget`
/// call addresses. Deliberately excludes any notion of "write short-term
/// facts" - short-term isn't a key-value store, it's the raw dialogue. The
/// profile has its own dedicated `/api/profile` endpoints instead of going
/// through this enum, since it isn't a bag of arbitrary facts.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MemoryLayer {
    ShortTerm,
    Working,
    LongTerm,
}

/// Renders every (key, value) pair as its own bullet line, sorted by key
/// since both fact maps are `BTreeMap`s.
fn format_facts_lines(facts: &BTreeMap<String, String>) -> String {
    let mut s = String::new();
    for (key, value) in facts {
        s.push_str(&format!("- {key}: {value}\n"));
    }
    s
}

/// `None` when the profile has nothing set, so no empty note gets spliced
/// into a request - a fresh agent with no profile behaves exactly like one
/// that was never given this feature at all.
fn format_profile_block(profile: &UserProfile) -> Option<String> {
    if profile.is_empty() {
        return None;
    }
    let mut s = String::from(
        "User profile - apply these preferences to every reply in this session, \
         even though the user won't restate them each turn:\n",
    );
    if let Some(style) = &profile.style {
        s.push_str(&format!("- Style: {style}\n"));
    }
    if let Some(tone) = &profile.tone {
        s.push_str(&format!("- Tone: {tone}\n"));
    }
    if let Some(format) = &profile.format {
        s.push_str(&format!("- Response format: {format}\n"));
    }
    if let Some(language) = &profile.language {
        s.push_str(&format!("- Language: {language}\n"));
    }
    if !profile.constraints.is_empty() {
        s.push_str("Hard constraints (never violate these):\n");
        for c in &profile.constraints {
            s.push_str(&format!("- {c}\n"));
        }
    }
    Some(s)
}

/// `None` when there's nothing captured yet, so no empty note gets spliced
/// into a request.
fn format_long_term_block(facts: &BTreeMap<String, String>) -> Option<String> {
    if facts.is_empty() {
        return None;
    }
    Some(format!(
        "What you durably know about this user:\n{}",
        format_facts_lines(facts)
    ))
}

/// `None` whenever no task is active - working memory has nothing to say
/// about a request that isn't part of any task.
fn format_working_block(task: &Option<String>, facts: &BTreeMap<String, String>) -> Option<String> {
    let task = task.as_ref()?;
    if facts.is_empty() {
        Some(format!("Current task: {task}\n"))
    } else {
        Some(format!(
            "Current task: {task}\nKnown details for this task:\n{}",
            format_facts_lines(facts)
        ))
    }
}

/// Builds the message list actually sent to the model for one turn: the
/// system prompt, then (only if non-empty) a synthetic system message per
/// layer that has something to say, then the raw short-term dialogue
/// unmodified. The order is deliberate: the profile (how to answer, set
/// once and rarely changed) comes first, then long-term facts (what's
/// known about the user, changes occasionally), then working memory
/// (changes once per task), then the live conversation (changes every
/// turn) - each block earlier in the list is more stable than the one
/// after it.
fn build_context(
    system_prompt: &str,
    profile: &UserProfile,
    long_term_facts: &BTreeMap<String, String>,
    working_task: &Option<String>,
    working_facts: &BTreeMap<String, String>,
    short_term_messages: &[Message],
) -> Vec<Message> {
    let mut context = Vec::with_capacity(short_term_messages.len() + 4);
    context.push(Message {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    });
    if let Some(block) = format_profile_block(profile) {
        context.push(Message {
            role: "system".to_string(),
            content: block,
        });
    }
    if let Some(block) = format_long_term_block(long_term_facts) {
        context.push(Message {
            role: "system".to_string(),
            content: block,
        });
    }
    if let Some(block) = format_working_block(working_task, working_facts) {
        context.push(Message {
            role: "system".to_string(),
            content: block,
        });
    }
    context.extend_from_slice(short_term_messages);
    context
}

// ---------------------------------------------------------------------
// Explicit, narrowly-scoped extraction prompts - one per destination.
// There is deliberately no extraction prompt for the profile: it is never
// written by the LLM, only by an explicit `/api/profile` call.
// ---------------------------------------------------------------------

/// Governs the working-memory extractor: runs only while a task is active,
/// and is deliberately blind to anything that isn't scoped to *this* task.
const WORKING_MEMORY_PROMPT: &str = "You maintain the working memory for ONE specific task an assistant is currently helping with. You will be given the TASK, the CURRENT WORKING FACTS as a JSON object, and the LATEST TURN (the user's message and the assistant's reply). Extract only facts needed to complete THIS task: parameters, constraints, choices, and progress specific to it. Do NOT include anything about the user as a person that would still matter after this task is finished (identity, standing preferences, recurring habits) - that belongs to a different memory system and must be left out here. Return the complete, updated facts as a single JSON object of string keys to string values, and nothing else - no prose, no markdown code fences, no commentary. Keep keys short and snake_case, and reuse an existing key when a new message updates the same fact. If nothing new or changed, return the CURRENT WORKING FACTS unchanged.";

/// Governs the long-term extractor: runs on every turn regardless of
/// whether a task is active, and is deliberately blind to task-scoped
/// detail. It is also blind to style/format/tone preferences now that
/// those have a dedicated home in the profile - it only extracts facts
/// about the user, not instructions about how to talk to them.
const LONG_TERM_MEMORY_PROMPT: &str = "You maintain a small, durable long-term memory about a user, meant to outlive any single task or conversation: who they are, standing preferences about their life or work, recurring constraints, and decisions they explicitly asked to be remembered permanently. You will be given the CURRENT LONG-TERM FACTS as a JSON object and the LATEST TURN. Only extract something if it would still be true and relevant in a completely different, unrelated task later - never include a fact that only matters to the task at hand. Do NOT extract preferences about communication style, tone, format, or language - those are handled by a separate personalization profile and must be left out here. Return the complete, updated facts as a single JSON object of string keys to string values, and nothing else - no prose, no markdown code fences, no commentary. Keep keys short and snake_case, and reuse an existing key when a new message updates the same fact. If nothing new or durable, return the CURRENT LONG-TERM FACTS unchanged.";

/// Governs the promotion step run once, explicitly, when a task finishes:
/// decides which (if any) working-memory facts are durable enough to
/// survive the task being thrown away.
const PROMOTION_PROMPT: &str = "A task has just finished. You will be given the TASK, its WORKING FACTS (about to be discarded) as a JSON object, and the CURRENT LONG-TERM FACTS as a JSON object. Decide which working facts, if any, describe something durable about the user that should survive beyond this task (identity, standing preference, recurring constraint) rather than being task-specific detail that should simply be forgotten. Return the complete, updated LONG-TERM facts as a single JSON object of string keys to string values, merging in only what's durable - and nothing else, no prose, no markdown code fences, no commentary. If nothing qualifies, return the CURRENT LONG-TERM FACTS unchanged.";

fn build_working_memory_prompt(
    task: &str,
    current: &BTreeMap<String, String>,
    user_message: &str,
    assistant_reply: &str,
) -> String {
    let current_json = serde_json::to_string(current).unwrap_or_else(|_| "{}".to_string());
    format!(
        "TASK:\n{task}\n\nCURRENT WORKING FACTS:\n{current_json}\n\nLATEST TURN:\nuser: {user_message}\nassistant: {assistant_reply}\n"
    )
}

fn build_long_term_prompt(current: &BTreeMap<String, String>, user_message: &str, assistant_reply: &str) -> String {
    let current_json = serde_json::to_string(current).unwrap_or_else(|_| "{}".to_string());
    format!(
        "CURRENT LONG-TERM FACTS:\n{current_json}\n\nLATEST TURN:\nuser: {user_message}\nassistant: {assistant_reply}\n"
    )
}

fn build_promotion_prompt(
    task: &str,
    working_facts: &BTreeMap<String, String>,
    long_term_facts: &BTreeMap<String, String>,
) -> String {
    let working_json = serde_json::to_string(working_facts).unwrap_or_else(|_| "{}".to_string());
    let long_term_json = serde_json::to_string(long_term_facts).unwrap_or_else(|_| "{}".to_string());
    format!(
        "TASK:\n{task}\n\nWORKING FACTS (about to be discarded):\n{working_json}\n\nCURRENT LONG-TERM FACTS:\n{long_term_json}\n"
    )
}

/// The extraction calls aren't asked to use JSON mode (this course's
/// backend isn't guaranteed to support it), so a reply is sometimes wrapped
/// in a ```json fence despite the prompt asking for none. Stripped here
/// rather than tightening the prompt further, since a model ignoring
/// instructions is exactly the kind of thing this has to tolerate anyway.
fn strip_json_fences(text: &str) -> String {
    let trimmed = text.trim();
    let trimmed = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    trimmed.strip_suffix("```").unwrap_or(trimmed).trim().to_string()
}

/// Parses an LLM-provided facts blob into a flat string map. Shared by the
/// working-memory extractor, the long-term extractor, and the promotion
/// step - all three ask for exactly the same shape of answer, just scoped
/// to a different question.
fn parse_facts_response(raw: &str) -> Result<BTreeMap<String, String>, serde_json::Error> {
    serde_json::from_str(&strip_json_fences(raw))
}

// ---------------------------------------------------------------------
// Persistence - each layer (including the profile) lives in its own file,
// loaded/saved independently, so the separation between them is visible on
// disk and a corrupt file for one layer can never take another down with
// it.
// ---------------------------------------------------------------------

fn load_or_default<T: Default + DeserializeOwned>(path: &str) -> T {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return T::default(),
    };
    match serde_json::from_str::<T>(&contents) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("Warning: failed to parse {path} ({e}); starting fresh for this layer.");
            T::default()
        }
    }
}

fn save_json<T: Serialize>(path: &str, value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("Warning: failed to write {path}: {e}");
            }
        }
        Err(e) => eprintln!("Warning: failed to serialize state for {path}: {e}"),
    }
}

/// What a chat turn or a memory inspection reports about the memory system:
/// how many tokens each layer contributed to the last request built (or
/// would contribute right now), plus how big each layer currently is.
#[derive(Serialize)]
struct MemoryReport {
    system_prompt_tokens: u32,
    profile_tokens: u32,
    long_term_tokens: u32,
    working_memory_tokens: u32,
    short_term_tokens: u32,
    sent_context_tokens: u32,
    actual: Option<Usage>,
    profile_is_set: bool,
    long_term_facts_count: usize,
    working_facts_count: usize,
    short_term_message_count: usize,
    active_task: Option<String>,
    context_limit: u32,
    percent_of_limit: f32,
    estimated_cost_usd: Option<f64>,
}

struct ChatOptions {
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

struct RespondResult {
    reply: String,
    memory: MemoryReport,
}

/// Bundles `Agent::new`'s configuration so the constructor doesn't grow an
/// ever-longer flat argument list.
struct AgentConfig {
    endpoint: String,
    api_key: String,
    model: String,
    profile_path: String,
    short_term_path: String,
    working_memory_path: String,
    long_term_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
}

/// The agent: owns the personalization profile and all three memory
/// layers, each behind its own mutex and persisted to its own file. No
/// method ever locks more than one layer's mutex across an `.await` point,
/// and every mutation is followed by a persist call scoped to exactly the
/// layer that changed.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    profile_path: String,
    short_term_path: String,
    working_memory_path: String,
    long_term_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    profile: Mutex<UserProfile>,
    short_term: Mutex<ShortTermMemory>,
    working: Mutex<WorkingMemory>,
    long_term: Mutex<LongTermMemory>,
}

impl Agent {
    fn new(client: reqwest::Client, config: AgentConfig) -> Self {
        let profile = load_or_default::<UserProfile>(&config.profile_path);
        let short_term = load_or_default::<ShortTermMemory>(&config.short_term_path);
        let working = load_or_default::<WorkingMemory>(&config.working_memory_path);
        let long_term = load_or_default::<LongTermMemory>(&config.long_term_path);
        Self {
            client,
            endpoint: config.endpoint,
            api_key: config.api_key,
            model: config.model,
            profile_path: config.profile_path,
            short_term_path: config.short_term_path,
            working_memory_path: config.working_memory_path,
            long_term_path: config.long_term_path,
            context_limit: config.context_limit,
            price_per_1m_input: config.price_per_1m_input,
            price_per_1m_output: config.price_per_1m_output,
            profile: Mutex::new(profile),
            short_term: Mutex::new(short_term),
            working: Mutex::new(working),
            long_term: Mutex::new(long_term),
        }
    }

    fn persist_profile(&self) {
        save_json(&self.profile_path, &*self.profile.lock().unwrap());
    }

    fn persist_short_term(&self) {
        save_json(&self.short_term_path, &*self.short_term.lock().unwrap());
    }

    fn persist_working(&self) {
        save_json(&self.working_memory_path, &*self.working.lock().unwrap());
    }

    fn persist_long_term(&self) {
        save_json(&self.long_term_path, &*self.long_term.lock().unwrap());
    }

    fn profile(&self) -> UserProfile {
        self.profile.lock().unwrap().clone()
    }

    /// Full replace, not a merge - the profile is small and explicit
    /// enough that "what you send is what gets stored" is easier to
    /// reason about than partial-update semantics.
    fn set_profile(&self, profile: UserProfile) {
        *self.profile.lock().unwrap() = profile;
        self.persist_profile();
    }

    fn reset_profile(&self) {
        self.set_profile(UserProfile::default());
    }

    fn short_term_messages(&self) -> Vec<Message> {
        self.short_term.lock().unwrap().messages.clone()
    }

    fn working_task(&self) -> Option<String> {
        self.working.lock().unwrap().task.clone()
    }

    fn working_facts(&self) -> BTreeMap<String, String> {
        self.working.lock().unwrap().facts.clone()
    }

    fn long_term_facts(&self) -> BTreeMap<String, String> {
        self.long_term.lock().unwrap().facts.clone()
    }

    fn estimate_cost(&self, prompt_tokens: u32, completion_tokens: u32) -> Option<f64> {
        let input_rate = self.price_per_1m_input?;
        let output_rate = self.price_per_1m_output?;
        Some(
            (prompt_tokens as f64 / 1_000_000.0) * input_rate
                + (completion_tokens as f64 / 1_000_000.0) * output_rate,
        )
    }

    /// A single point of contact with the chat-completions endpoint, shared
    /// by the main reply and all three background extraction calls.
    async fn call_chat(
        &self,
        messages: Vec<Message>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<(String, Option<Usage>), String> {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature,
            max_tokens,
            stop: None,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Request failed: {e}"))?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            let detail = serde_json::from_str::<ErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return Err(format!("API error ({status}): {detail}"));
        }

        let parsed = serde_json::from_str::<ChatCompletionResponse>(&body)
            .map_err(|e| format!("Failed to parse response: {e}"))?;
        let ChatCompletionResponse { choices, usage } = parsed;
        let content = choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_else(|| "No response returned from the model.".to_string());
        Ok((content, usage))
    }

    /// Builds a `MemoryReport` from whatever the profile and three memory
    /// layers hold right now, given the token count of whatever request
    /// this report is describing (a real one just sent, or a hypothetical
    /// one for inspection).
    fn compose_report(&self, actual: Option<Usage>, sent_context_tokens: u32) -> MemoryReport {
        let profile = self.profile();
        let long_term = self.long_term_facts();
        let working_task = self.working_task();
        let working_facts = self.working_facts();
        let short_term_len = self.short_term.lock().unwrap().messages.len();
        let short_term_tokens = history_tokens(&self.short_term.lock().unwrap().messages);

        let system_prompt_tokens = message_tokens(&Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        });
        let profile_tokens = format_profile_block(&profile)
            .map(|b| estimate_tokens(&b) + MESSAGE_OVERHEAD_TOKENS)
            .unwrap_or(0);
        let long_term_tokens = format_long_term_block(&long_term)
            .map(|b| estimate_tokens(&b) + MESSAGE_OVERHEAD_TOKENS)
            .unwrap_or(0);
        let working_memory_tokens = format_working_block(&working_task, &working_facts)
            .map(|b| estimate_tokens(&b) + MESSAGE_OVERHEAD_TOKENS)
            .unwrap_or(0);

        let effective_sent = actual.as_ref().map(|u| u.prompt_tokens).unwrap_or(sent_context_tokens);
        let percent_of_limit = (effective_sent as f32 / self.context_limit as f32) * 100.0;
        let estimated_cost_usd = actual
            .as_ref()
            .and_then(|u| self.estimate_cost(u.prompt_tokens, u.completion_tokens));

        MemoryReport {
            system_prompt_tokens,
            profile_tokens,
            long_term_tokens,
            working_memory_tokens,
            short_term_tokens,
            sent_context_tokens,
            actual,
            profile_is_set: !profile.is_empty(),
            long_term_facts_count: long_term.len(),
            working_facts_count: working_facts.len(),
            short_term_message_count: short_term_len,
            active_task: working_task,
            context_limit: self.context_limit,
            percent_of_limit,
            estimated_cost_usd,
        }
    }

    /// A descriptive snapshot for endpoints that aren't mid-turn (memory
    /// inspection, reset, task start/finish, profile changes): what a
    /// request would look like *right now*, with no real call made.
    fn snapshot(&self) -> MemoryReport {
        let profile = self.profile();
        let long_term = self.long_term_facts();
        let working_task = self.working_task();
        let working_facts = self.working_facts();
        let short_term = self.short_term_messages();
        let sent_messages = build_context(
            SYSTEM_PROMPT,
            &profile,
            &long_term,
            &working_task,
            &working_facts,
            &short_term,
        );
        let sent_context_tokens = history_tokens(&sent_messages);
        self.compose_report(None, sent_context_tokens)
    }

    async fn respond(&self, user_message: &str, options: ChatOptions) -> RespondResult {
        // Short-term's write rule is unconditional: the turn goes in
        // regardless of what happens next.
        {
            let mut st = self.short_term.lock().unwrap();
            st.messages.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
        }

        let profile = self.profile();
        let long_term = self.long_term_facts();
        let working_task = self.working_task();
        let working_facts = self.working_facts();
        let short_term = self.short_term_messages();

        let sent_messages = build_context(
            SYSTEM_PROMPT,
            &profile,
            &long_term,
            &working_task,
            &working_facts,
            &short_term,
        );
        let sent_context_tokens = history_tokens(&sent_messages);

        match self.call_chat(sent_messages, options.temperature, options.max_tokens).await {
            Ok((reply, usage)) => {
                {
                    let mut st = self.short_term.lock().unwrap();
                    st.messages.push(Message {
                        role: "assistant".to_string(),
                        content: reply.clone(),
                    });
                }
                self.persist_short_term();

                // Working memory is only ever written to while a task is
                // active - this is the explicit gate, not a judgment call
                // made by the extractor itself.
                if let Some(task) = working_task.as_deref() {
                    self.update_working_memory(task, user_message, &reply).await;
                }
                // Long-term extraction runs on every turn, task or no
                // task. The profile is never touched here - it only ever
                // changes via an explicit `/api/profile` call.
                self.update_long_term_memory(user_message, &reply).await;

                let memory = self.compose_report(usage, sent_context_tokens);
                RespondResult { reply, memory }
            }
            Err(err) => {
                // The turn never happened as far as the model is
                // concerned - don't let a failed call leave a half-turn in
                // short-term memory.
                self.short_term.lock().unwrap().messages.pop();
                let memory = self.compose_report(None, sent_context_tokens);
                RespondResult { reply: err, memory }
            }
        }
    }

    /// Strategy for working memory: runs only while a task is active
    /// (callers gate on that) and only ever touches `self.working.facts`.
    /// Best-effort: any failure - network, a non-200, unparseable JSON -
    /// just leaves the prior facts in place and gets retried on the next
    /// turn; it never blocks or breaks the chat reply it rode in on.
    async fn update_working_memory(&self, task: &str, user_message: &str, assistant_reply: &str) {
        let current = self.working_facts();
        let prompt = build_working_memory_prompt(task, &current, user_message, assistant_reply);
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: WORKING_MEMORY_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        match self.call_chat(messages, Some(0.0), None).await {
            Ok((raw, _)) => match parse_facts_response(&raw) {
                Ok(facts) => {
                    self.working.lock().unwrap().facts = facts;
                    self.persist_working();
                }
                Err(e) => eprintln!("Warning: working-memory response wasn't valid JSON, keeping prior facts: {e}"),
            },
            Err(e) => eprintln!("Warning: working-memory update failed, keeping prior facts: {e}"),
        }
    }

    /// Long-term's extractor: runs every turn, only ever touches
    /// `self.long_term.facts`. Same best-effort contract as working memory.
    async fn update_long_term_memory(&self, user_message: &str, assistant_reply: &str) {
        let current = self.long_term_facts();
        let prompt = build_long_term_prompt(&current, user_message, assistant_reply);
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: LONG_TERM_MEMORY_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        match self.call_chat(messages, Some(0.0), None).await {
            Ok((raw, _)) => match parse_facts_response(&raw) {
                Ok(facts) => {
                    self.long_term.lock().unwrap().facts = facts;
                    self.persist_long_term();
                }
                Err(e) => eprintln!("Warning: long-term response wasn't valid JSON, keeping prior facts: {e}"),
            },
            Err(e) => eprintln!("Warning: long-term update failed, keeping prior facts: {e}"),
        }
    }

    /// Runs once, explicitly, when a task finishes with promotion
    /// requested: asks the model which (if any) working facts are durable
    /// enough to merge into long-term before working memory is thrown
    /// away. A no-op when working memory is already empty - nothing to
    /// consider promoting.
    async fn run_promotion(&self, task: &str) {
        let working_facts = self.working_facts();
        if working_facts.is_empty() {
            return;
        }
        let current_long_term = self.long_term_facts();
        let prompt = build_promotion_prompt(task, &working_facts, &current_long_term);
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: PROMOTION_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        match self.call_chat(messages, Some(0.0), None).await {
            Ok((raw, _)) => match parse_facts_response(&raw) {
                Ok(facts) => {
                    self.long_term.lock().unwrap().facts = facts;
                    self.persist_long_term();
                }
                Err(e) => {
                    eprintln!("Warning: promotion response wasn't valid JSON; discarding working memory without promoting: {e}")
                }
            },
            Err(e) => eprintln!("Warning: promotion call failed; discarding working memory without promoting: {e}"),
        }
    }

    /// Starts a new task: sets `working.task` and clears any facts left
    /// over from before (there shouldn't be any, since `finish_task`
    /// always clears them, but this makes the guarantee explicit rather
    /// than assumed). Refuses to start a second task on top of an active
    /// one - the lifecycle boundary has to be crossed explicitly via
    /// `finish_task` first, not implicitly overwritten.
    fn start_task(&self, name: String) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        if w.task.is_some() {
            return Err("A task is already active; finish it before starting a new one.".to_string());
        }
        w.task = Some(name);
        w.facts = BTreeMap::new();
        drop(w);
        self.persist_working();
        Ok(())
    }

    fn clear_working(&self) {
        let mut w = self.working.lock().unwrap();
        w.task = None;
        w.facts = BTreeMap::new();
        drop(w);
        self.persist_working();
    }

    /// Ends the active task. With `promote: true` (the default from the
    /// API), runs the promotion step first so durable facts survive into
    /// long-term memory; with `promote: false`, working memory is simply
    /// discarded - an explicit "this was throwaway" signal.
    async fn finish_task(&self, promote: bool) -> Result<(), String> {
        let task = {
            let w = self.working.lock().unwrap();
            match &w.task {
                Some(t) => t.clone(),
                None => return Err("No task is currently active.".to_string()),
            }
        };
        if promote {
            self.run_promotion(&task).await;
        }
        self.clear_working();
        Ok(())
    }

    /// Explicit, human-directed writes that bypass the LLM extractors
    /// entirely - the clearest demonstration that "where does this go" is
    /// a deliberate choice, not something only an extraction prompt
    /// decides.
    fn remember(&self, layer: MemoryLayer, key: String, value: String) -> Result<(), String> {
        match layer {
            MemoryLayer::ShortTerm => Err(
                "Short-term memory is the raw dialogue, not a key-value store; there's nothing to \"remember\" into it directly.".to_string(),
            ),
            MemoryLayer::Working => {
                let mut w = self.working.lock().unwrap();
                if w.task.is_none() {
                    return Err("No task is active; start one before writing to working memory.".to_string());
                }
                w.facts.insert(key, value);
                drop(w);
                self.persist_working();
                Ok(())
            }
            MemoryLayer::LongTerm => {
                self.long_term.lock().unwrap().facts.insert(key, value);
                self.persist_long_term();
                Ok(())
            }
        }
    }

    /// Wipes exactly one layer and nothing else. Forgetting working memory
    /// also clears the active task, since working memory without a task is
    /// a meaningless state.
    fn forget(&self, layer: MemoryLayer) {
        match layer {
            MemoryLayer::ShortTerm => {
                self.short_term.lock().unwrap().messages.clear();
                self.persist_short_term();
            }
            MemoryLayer::Working => self.clear_working(),
            MemoryLayer::LongTerm => {
                self.long_term.lock().unwrap().facts = BTreeMap::new();
                self.persist_long_term();
            }
        }
    }

    /// Clears the current conversation only. The profile, working memory
    /// (the active task), and long-term memory are untouched - resetting a
    /// conversation does not make the agent forget who it's talking to,
    /// how it was told to talk, or what task it's in the middle of.
    fn reset_conversation(&self) {
        self.short_term.lock().unwrap().messages.clear();
        self.persist_short_term();
    }
}

// ---------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatResponse {
    reply: String,
    memory: MemoryReport,
}

#[derive(Serialize)]
struct MemoryInspection {
    profile: UserProfile,
    short_term: Vec<Message>,
    working_task: Option<String>,
    working_facts: BTreeMap<String, String>,
    long_term_facts: BTreeMap<String, String>,
    report: MemoryReport,
}

#[derive(Deserialize)]
struct StartTaskRequest {
    name: String,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct FinishTaskRequest {
    #[serde(default = "default_true")]
    promote: bool,
}

#[derive(Deserialize)]
struct RememberRequest {
    layer: MemoryLayer,
    key: String,
    value: String,
}

#[derive(Deserialize)]
struct ForgetRequest {
    layer: MemoryLayer,
}

async fn chat(State(agent): State<Arc<Agent>>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    let result = agent
        .respond(
            &req.message,
            ChatOptions {
                temperature: req.temperature,
                max_tokens: req.max_tokens,
            },
        )
        .await;
    Json(ChatResponse {
        reply: result.reply,
        memory: result.memory,
    })
}

async fn get_memory(State(agent): State<Arc<Agent>>) -> Json<MemoryInspection> {
    Json(MemoryInspection {
        profile: agent.profile(),
        short_term: agent.short_term_messages(),
        working_task: agent.working_task(),
        working_facts: agent.working_facts(),
        long_term_facts: agent.long_term_facts(),
        report: agent.snapshot(),
    })
}

async fn reset(State(agent): State<Arc<Agent>>) -> Json<MemoryReport> {
    agent.reset_conversation();
    Json(agent.snapshot())
}

async fn start_task(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<StartTaskRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent.start_task(req.name).map_err(|e| (StatusCode::CONFLICT, e))?;
    Ok(Json(agent.snapshot()))
}

async fn finish_task(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<FinishTaskRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent.finish_task(req.promote).await.map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.snapshot()))
}

async fn remember(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent
        .remember(req.layer, req.key, req.value)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.snapshot()))
}

async fn forget(State(agent): State<Arc<Agent>>, Json(req): Json<ForgetRequest>) -> Json<MemoryReport> {
    agent.forget(req.layer);
    Json(agent.snapshot())
}

async fn get_profile(State(agent): State<Arc<Agent>>) -> Json<UserProfile> {
    Json(agent.profile())
}

/// Full replace: the body is the complete new profile. Any field left out
/// of the request JSON defaults to empty/unset (see `UserProfile`'s
/// `#[serde(default)]` fields), so posting `{"style": "concise"}` clears
/// every other preference rather than merging - explicit over implicit.
async fn set_profile(State(agent): State<Arc<Agent>>, Json(profile): Json<UserProfile>) -> Json<UserProfile> {
    agent.set_profile(profile.clone());
    Json(profile)
}

async fn reset_profile(State(agent): State<Arc<Agent>>) -> Json<UserProfile> {
    agent.reset_profile();
    Json(agent.profile())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    let model = std::env::var("OPENAI_MODEL")
        .ok()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    let profile_path = std::env::var("PROFILE_FILE").unwrap_or_else(|_| "profile.json".to_string());
    let short_term_path = std::env::var("SHORT_TERM_FILE").unwrap_or_else(|_| "short_term.json".to_string());
    let working_memory_path = std::env::var("WORKING_MEMORY_FILE").unwrap_or_else(|_| "working_memory.json".to_string());
    let long_term_path = std::env::var("LONG_TERM_FILE").unwrap_or_else(|_| "long_term.json".to_string());
    // A reference limit for the "% of limit" indicator, not a guarantee of
    // the real backend's context window.
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
        AgentConfig {
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model,
            profile_path: profile_path.clone(),
            short_term_path: short_term_path.clone(),
            working_memory_path: working_memory_path.clone(),
            long_term_path: long_term_path.clone(),
            context_limit,
            price_per_1m_input,
            price_per_1m_output,
        },
    );

    let task_note = agent
        .working_task()
        .map(|t| format!(" for task \"{t}\""))
        .unwrap_or_default();
    let profile_note = if agent.profile().is_empty() {
        "no personalization profile set".to_string()
    } else {
        "personalization profile restored".to_string()
    };
    println!(
        "Restored {} short-term message(s), {} working fact(s){task_note}, {} long-term fact(s), {profile_note}.",
        agent.short_term_messages().len(),
        agent.working_facts().len(),
        agent.long_term_facts().len(),
    );

    let agent = Arc::new(agent);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/memory", get(get_memory))
        .route("/api/reset", post(reset))
        .route("/api/task/start", post(start_task))
        .route("/api/task/finish", post(finish_task))
        .route("/api/memory/remember", post(remember))
        .route("/api/memory/forget", post(forget))
        .route("/api/profile", get(get_profile).put(set_profile).delete(reset_profile))
        .with_state(agent);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
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
            .join(format!("ai-advent-lesson-12-{label}-{}-{}.json", std::process::id(), n))
            .to_string_lossy()
            .into_owned()
    }

    fn test_agent() -> Agent {
        Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
                profile_path: temp_path("profile"),
                short_term_path: temp_path("short"),
                working_memory_path: temp_path("working"),
                long_term_path: temp_path("long"),
                context_limit: 1000,
                price_per_1m_input: Some(2.0),
                price_per_1m_output: Some(4.0),
            },
        )
    }

    fn sample_profile() -> UserProfile {
        UserProfile {
            style: Some("concise, no fluff".to_string()),
            tone: Some("direct and professional".to_string()),
            format: Some("short bullet points".to_string()),
            language: Some("English".to_string()),
            constraints: vec!["Never use emojis".to_string()],
        }
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
    fn user_profile_default_is_empty() {
        assert!(UserProfile::default().is_empty());
    }

    #[test]
    fn user_profile_with_any_field_set_is_not_empty() {
        let mut p = UserProfile::default();
        p.style = Some("concise".to_string());
        assert!(!p.is_empty());

        let mut p2 = UserProfile::default();
        p2.constraints.push("no emojis".to_string());
        assert!(!p2.is_empty());
    }

    #[test]
    fn format_profile_block_is_none_when_empty() {
        assert_eq!(format_profile_block(&UserProfile::default()), None);
    }

    #[test]
    fn format_profile_block_includes_every_set_field() {
        let block = format_profile_block(&sample_profile()).unwrap();
        assert!(block.contains("concise, no fluff"));
        assert!(block.contains("direct and professional"));
        assert!(block.contains("short bullet points"));
        assert!(block.contains("English"));
        assert!(block.contains("Never use emojis"));
    }

    #[test]
    fn format_profile_block_omits_unset_fields() {
        let profile = UserProfile {
            style: Some("detailed".to_string()),
            ..Default::default()
        };
        let block = format_profile_block(&profile).unwrap();
        assert!(block.contains("detailed"));
        assert!(!block.contains("Tone:"));
        assert!(!block.contains("Response format:"));
        assert!(!block.contains("Language:"));
        assert!(!block.contains("Hard constraints"));
    }

    #[test]
    fn format_long_term_block_is_none_when_empty() {
        assert_eq!(format_long_term_block(&BTreeMap::new()), None);
    }

    #[test]
    fn format_long_term_block_lists_facts_sorted() {
        let mut facts = BTreeMap::new();
        facts.insert("z_last".to_string(), "z".to_string());
        facts.insert("a_first".to_string(), "a".to_string());
        let block = format_long_term_block(&facts).unwrap();
        let a_pos = block.find("a_first").unwrap();
        let z_pos = block.find("z_last").unwrap();
        assert!(a_pos < z_pos);
    }

    #[test]
    fn format_working_block_is_none_without_an_active_task() {
        assert_eq!(format_working_block(&None, &BTreeMap::new()), None);
    }

    #[test]
    fn format_working_block_includes_task_and_facts() {
        let mut facts = BTreeMap::new();
        facts.insert("budget".to_string(), "$200".to_string());
        let block = format_working_block(&Some("Plan a party".to_string()), &facts).unwrap();
        assert!(block.contains("Plan a party"));
        assert!(block.contains("budget: $200"));
    }

    #[test]
    fn build_context_with_no_memory_or_profile_is_just_system_and_dialogue() {
        let dialogue = vec![Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let ctx = build_context(
            SYSTEM_PROMPT,
            &UserProfile::default(),
            &BTreeMap::new(),
            &None,
            &BTreeMap::new(),
            &dialogue,
        );
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "system");
        assert_eq!(ctx[1].content, "hi");
    }

    #[test]
    fn build_context_includes_profile_long_term_and_working_blocks_in_order() {
        let mut long_term = BTreeMap::new();
        long_term.insert("diet".to_string(), "vegetarian".to_string());
        let mut working = BTreeMap::new();
        working.insert("budget".to_string(), "$200".to_string());
        let dialogue = vec![Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let ctx = build_context(
            SYSTEM_PROMPT,
            &sample_profile(),
            &long_term,
            &Some("Plan a party".to_string()),
            &working,
            &dialogue,
        );
        assert_eq!(ctx.len(), 5);
        assert_eq!(ctx[0].role, "system");
        assert!(ctx[1].content.contains("concise, no fluff"));
        assert!(ctx[2].content.contains("vegetarian"));
        assert!(ctx[3].content.contains("Plan a party"));
        assert_eq!(ctx[4].content, "hi");
    }

    #[test]
    fn strip_json_fences_removes_a_json_code_fence() {
        let raw = "```json\n{\"a\": \"b\"}\n```";
        assert_eq!(strip_json_fences(raw), "{\"a\": \"b\"}");
    }

    #[test]
    fn strip_json_fences_passes_through_plain_json() {
        let raw = "{\"a\": \"b\"}";
        assert_eq!(strip_json_fences(raw), "{\"a\": \"b\"}");
    }

    #[test]
    fn parse_facts_response_parses_valid_json() {
        let facts = parse_facts_response("{\"goal\": \"ship\"}").unwrap();
        assert_eq!(facts.get("goal"), Some(&"ship".to_string()));
    }

    #[test]
    fn parse_facts_response_rejects_non_object_json() {
        assert!(parse_facts_response("[1,2,3]").is_err());
    }

    #[test]
    fn build_working_memory_prompt_includes_task_facts_and_turn() {
        let mut facts = BTreeMap::new();
        facts.insert("budget".to_string(), "$200".to_string());
        let prompt = build_working_memory_prompt("Plan a party", &facts, "make it $300", "updated the budget");
        assert!(prompt.contains("Plan a party"));
        assert!(prompt.contains("$200"));
        assert!(prompt.contains("make it $300"));
        assert!(prompt.contains("updated the budget"));
    }

    #[test]
    fn build_long_term_prompt_includes_current_facts_and_turn() {
        let mut facts = BTreeMap::new();
        facts.insert("diet".to_string(), "vegetarian".to_string());
        let prompt = build_long_term_prompt(&facts, "I'm also allergic to nuts", "noted");
        assert!(prompt.contains("vegetarian"));
        assert!(prompt.contains("allergic to nuts"));
    }

    #[test]
    fn build_promotion_prompt_includes_task_and_both_fact_maps() {
        let mut working = BTreeMap::new();
        working.insert("budget".to_string(), "$200".to_string());
        let mut long_term = BTreeMap::new();
        long_term.insert("diet".to_string(), "vegetarian".to_string());
        let prompt = build_promotion_prompt("Plan a party", &working, &long_term);
        assert!(prompt.contains("Plan a party"));
        assert!(prompt.contains("$200"));
        assert!(prompt.contains("vegetarian"));
    }

    #[test]
    fn load_or_default_returns_default_when_file_missing() {
        let path = temp_path("missing");
        let value: ShortTermMemory = load_or_default(&path);
        assert!(value.messages.is_empty());
    }

    #[test]
    fn load_or_default_falls_back_on_invalid_json() {
        let path = temp_path("invalid");
        std::fs::write(&path, "not json").unwrap();
        let value: WorkingMemory = load_or_default(&path);
        assert!(value.task.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_json_then_load_or_default_round_trips() {
        let path = temp_path("roundtrip");
        let original = sample_profile();
        save_json(&path, &original);
        let loaded: UserProfile = load_or_default(&path);
        assert_eq!(loaded, original);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_starts_with_empty_profile_and_layers_when_no_files_exist() {
        let agent = test_agent();
        assert!(agent.profile().is_empty());
        assert!(agent.short_term_messages().is_empty());
        assert!(agent.working_task().is_none());
        assert!(agent.working_facts().is_empty());
        assert!(agent.long_term_facts().is_empty());
    }

    #[test]
    fn set_profile_replaces_and_persists() {
        let agent = test_agent();
        agent.set_profile(sample_profile());
        assert_eq!(agent.profile(), sample_profile());
    }

    #[test]
    fn set_profile_is_a_full_replace_not_a_merge() {
        let agent = test_agent();
        agent.set_profile(sample_profile());
        let partial = UserProfile {
            style: Some("detailed".to_string()),
            ..Default::default()
        };
        agent.set_profile(partial.clone());
        assert_eq!(agent.profile(), partial);
        assert_eq!(agent.profile().tone, None);
    }

    #[test]
    fn reset_profile_clears_back_to_default() {
        let agent = test_agent();
        agent.set_profile(sample_profile());
        agent.reset_profile();
        assert!(agent.profile().is_empty());
    }

    #[test]
    fn start_task_sets_active_task_and_clears_working_facts() {
        let agent = test_agent();
        agent.working.lock().unwrap().facts.insert("stale".to_string(), "value".to_string());
        agent.start_task("Plan a party".to_string()).unwrap();
        assert_eq!(agent.working_task(), Some("Plan a party".to_string()));
        assert!(agent.working_facts().is_empty());
    }

    #[test]
    fn start_task_rejects_when_a_task_is_already_active() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string()).unwrap();
        assert!(agent.start_task("Plan a meeting".to_string()).is_err());
    }

    #[tokio::test]
    async fn finish_task_without_promotion_clears_working_memory_and_leaves_long_term_untouched() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string()).unwrap();
        agent.working.lock().unwrap().facts.insert("budget".to_string(), "$200".to_string());
        agent.long_term.lock().unwrap().facts.insert("diet".to_string(), "vegetarian".to_string());

        agent.finish_task(false).await.unwrap();

        assert!(agent.working_task().is_none());
        assert!(agent.working_facts().is_empty());
        assert_eq!(agent.long_term_facts().get("diet"), Some(&"vegetarian".to_string()));
    }

    #[tokio::test]
    async fn finish_task_rejects_when_no_task_is_active() {
        let agent = test_agent();
        assert!(agent.finish_task(false).await.is_err());
    }

    #[test]
    fn remember_writes_directly_into_the_requested_layer() {
        let agent = test_agent();
        agent
            .remember(MemoryLayer::LongTerm, "diet".to_string(), "vegetarian".to_string())
            .unwrap();
        assert_eq!(agent.long_term_facts().get("diet"), Some(&"vegetarian".to_string()));
    }

    #[test]
    fn remember_rejects_short_term_since_its_not_a_key_value_store() {
        let agent = test_agent();
        assert!(agent
            .remember(MemoryLayer::ShortTerm, "k".to_string(), "v".to_string())
            .is_err());
    }

    #[test]
    fn remember_into_working_requires_an_active_task() {
        let agent = test_agent();
        assert!(agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .is_err());
        agent.start_task("Plan a party".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        assert_eq!(agent.working_facts().get("budget"), Some(&"$200".to_string()));
    }

    #[test]
    fn forget_short_term_clears_the_dialogue_only() {
        let agent = test_agent();
        agent.short_term.lock().unwrap().messages.push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.long_term.lock().unwrap().facts.insert("diet".to_string(), "vegetarian".to_string());
        agent.forget(MemoryLayer::ShortTerm);
        assert!(agent.short_term_messages().is_empty());
        assert_eq!(agent.long_term_facts().get("diet"), Some(&"vegetarian".to_string()));
    }

    #[test]
    fn forget_working_clears_facts_and_the_active_task() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        agent.forget(MemoryLayer::Working);
        assert!(agent.working_task().is_none());
        assert!(agent.working_facts().is_empty());
    }

    #[test]
    fn forget_long_term_clears_profile_facts_only() {
        let agent = test_agent();
        agent.long_term.lock().unwrap().facts.insert("diet".to_string(), "vegetarian".to_string());
        agent.short_term.lock().unwrap().messages.push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.forget(MemoryLayer::LongTerm);
        assert!(agent.long_term_facts().is_empty());
        assert!(!agent.short_term_messages().is_empty());
    }

    #[test]
    fn reset_conversation_clears_short_term_but_preserves_profile_working_and_long_term() {
        let agent = test_agent();
        agent.set_profile(sample_profile());
        agent.short_term.lock().unwrap().messages.push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.start_task("Plan a party".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        agent
            .remember(MemoryLayer::LongTerm, "diet".to_string(), "vegetarian".to_string())
            .unwrap();

        agent.reset_conversation();

        assert!(agent.short_term_messages().is_empty());
        assert_eq!(agent.profile(), sample_profile());
        assert_eq!(agent.working_task(), Some("Plan a party".to_string()));
        assert_eq!(agent.working_facts().get("budget"), Some(&"$200".to_string()));
        assert_eq!(agent.long_term_facts().get("diet"), Some(&"vegetarian".to_string()));
    }

    #[test]
    fn snapshot_reports_zero_profile_tokens_when_no_profile_is_set() {
        let agent = test_agent();
        let report = agent.snapshot();
        assert_eq!(report.profile_tokens, 0);
        assert!(!report.profile_is_set);
    }

    #[test]
    fn snapshot_reports_profile_tokens_once_a_profile_is_set() {
        let agent = test_agent();
        let before = agent.snapshot();
        agent.set_profile(sample_profile());
        let after = agent.snapshot();
        assert_eq!(before.profile_tokens, 0);
        assert!(after.profile_tokens > 0);
        assert!(after.profile_is_set);
    }

    #[test]
    fn snapshot_reports_zero_working_tokens_when_no_task_is_active() {
        let agent = test_agent();
        let report = agent.snapshot();
        assert_eq!(report.working_memory_tokens, 0);
        assert!(report.active_task.is_none());
    }

    #[test]
    fn snapshot_includes_working_tokens_only_while_a_task_is_active() {
        let agent = test_agent();
        let before = agent.snapshot();
        agent.start_task("Plan a party".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        let after = agent.snapshot();
        assert_eq!(before.working_memory_tokens, 0);
        assert!(after.working_memory_tokens > 0);
        assert_eq!(after.active_task, Some("Plan a party".to_string()));
    }

    #[test]
    fn estimate_cost_is_none_without_configured_prices() {
        let agent = Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
                profile_path: temp_path("profile"),
                short_term_path: temp_path("short"),
                working_memory_path: temp_path("working"),
                long_term_path: temp_path("long"),
                context_limit: 1000,
                price_per_1m_input: None,
                price_per_1m_output: None,
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
}

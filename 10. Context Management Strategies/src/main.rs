use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
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

/// Which of the three context-management strategies is currently governing
/// what gets sent to the model. Unlike lesson 09's compression, none of
/// these ever produce an LLM-written summary of the conversation itself:
///
/// - `SlidingWindow` - only the last `keep_last_n` raw messages are sent;
///   everything older is dropped outright, not folded into anything.
/// - `StickyFacts` - a small key-value memory (goals, constraints,
///   preferences, decisions - see `FACTS_SYSTEM_PROMPT`) is extracted after
///   every user turn and sent alongside the last `keep_last_n` messages.
/// - `Branching` - the active branch's *entire* raw history is sent,
///   unmodified; the mechanism that keeps things manageable is letting a
///   conversation fork at a checkpoint so two directions never pollute each
///   other's history, rather than trimming within either one.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Strategy {
    SlidingWindow,
    StickyFacts,
    Branching,
}

impl Strategy {
    /// Only used for parsing the `DEFAULT_STRATEGY` env var - request bodies
    /// deserialize `Strategy` directly via serde instead.
    fn from_env_str(s: &str) -> Option<Self> {
        match s {
            "sliding_window" => Some(Strategy::SlidingWindow),
            "sticky_facts" => Some(Strategy::StickyFacts),
            "branching" => Some(Strategy::Branching),
            _ => None,
        }
    }
}

/// What a single chat turn (or a reset/history fetch) reports.
#[derive(Serialize)]
struct TokenReport {
    /// This request's user message, counted locally before the call.
    estimated_request_tokens: u32,
    /// The active branch's raw history before this turn was added.
    estimated_history_tokens_before: u32,
    /// This turn's reply, counted locally (0 when the turn failed).
    estimated_response_tokens: u32,
    /// The provider's own accounting for this turn, when it returns one.
    actual: Option<Usage>,
    /// The complete, unbounded history of the active branch after this
    /// turn - never trimmed, regardless of which strategy is active.
    raw_history_tokens_after: u32,
    /// What this turn would have cost with no strategy applied at all: the
    /// full active-branch history including the new user message. This is
    /// the fair baseline all three strategies are compared against.
    uncompacted_context_tokens: u32,
    /// What was actually sent to the model this turn, under the active
    /// strategy.
    sent_context_tokens: u32,
    /// `uncompacted_context_tokens - sent_context_tokens`. Always zero for
    /// `Branching`, since it sends the full history by design.
    tokens_saved_this_turn: u32,
    /// Which strategy produced `sent_context_tokens` this turn.
    strategy: Strategy,
    /// How many key-value facts are currently held in sticky memory.
    facts_count: usize,
    /// Tokens the facts block would cost if spliced into a request (0 if
    /// there are no facts yet).
    facts_tokens: u32,
    /// The branch this turn was sent against.
    branch_id: String,
    branch_name: String,
    /// A configurable reference limit - not a guarantee of the real
    /// backend's context window, just something to watch what's actually
    /// being sent against.
    context_limit: u32,
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

/// Instructs the (cheap, same-provider) model used for the per-turn facts
/// extraction call. Deliberately narrow: this call's only output is a JSON
/// object, never prose, and it never talks to the user.
const FACTS_SYSTEM_PROMPT: &str = "You maintain a compact key-value memory of important facts from an ongoing conversation: goals, constraints, preferences, decisions, and agreements. You will be given the CURRENT FACTS as a JSON object and the LATEST TURN (the user's message and the assistant's reply to it). Return the complete, updated facts as a single JSON object of string keys to string values, and nothing else - no prose, no markdown code fences, no commentary. Keep keys short and snake_case, and reuse an existing key when a new message updates the same fact rather than creating a near-duplicate. Only include facts that would matter to a later turn; drop anything trivial. If nothing new or changed, return the CURRENT FACTS unchanged.";

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

/// One line of conversation. `messages[0]` is always the system prompt.
/// Branches are how this lesson implements "checkpoint, then fork two
/// directions": a fork is simply a new `Branch` whose `messages` is a copy
/// of another branch's messages up to some earlier point, after which the
/// two evolve completely independently - neither can see or pollute the
/// other's later turns.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Branch {
    id: String,
    name: String,
    messages: Vec<Message>,
    parent_id: Option<String>,
    /// How many *visible* (non-system) messages the parent branch had at
    /// the moment this branch was forked from it. `None` for the original
    /// branch, which was never forked from anything.
    forked_at: Option<usize>,
}

/// A saved position within a branch's history, so a fork can be created
/// from it later (possibly more than once, and possibly after the source
/// branch itself has kept going past this point).
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Checkpoint {
    id: String,
    label: String,
    branch_id: String,
    /// Number of visible (non-system) messages the source branch held when
    /// this checkpoint was taken. A fork replays exactly this many messages
    /// plus the system prompt - nothing the source branch said afterward.
    message_count: usize,
}

/// Everything persisted to `STATE_FILE` between restarts.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct PersistedState {
    branches: Vec<Branch>,
    active_branch_id: String,
    checkpoints: Vec<Checkpoint>,
    facts: BTreeMap<String, String>,
    strategy: Strategy,
    next_id: u64,
}

fn default_state(strategy: Strategy) -> PersistedState {
    PersistedState {
        branches: vec![Branch {
            id: "b0".to_string(),
            name: "Main".to_string(),
            messages: vec![Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            }],
            parent_id: None,
            forked_at: None,
        }],
        active_branch_id: "b0".to_string(),
        checkpoints: Vec::new(),
        facts: BTreeMap::new(),
        strategy,
        next_id: 1,
    }
}

/// Strategy 1 - Sliding Window: the system prompt, then only the last
/// `keep_last_n` raw messages. Everything older is simply gone from the
/// request; it is never folded into anything else.
fn sliding_window_context(messages: &[Message], keep_last_n: usize) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut context = Vec::with_capacity(keep_last_n + 1);
    context.push(messages[0].clone());
    let non_system_len = messages.len() - 1;
    let take = keep_last_n.min(non_system_len);
    let tail_start = messages.len() - take;
    context.extend_from_slice(&messages[tail_start..]);
    context
}

/// Renders the sticky facts as a single system-role block, or `None` when
/// there's nothing captured yet (so no empty note gets spliced in).
fn format_facts_block(facts: &BTreeMap<String, String>) -> Option<String> {
    if facts.is_empty() {
        return None;
    }
    let mut block = String::from("Known facts about this conversation:\n");
    for (key, value) in facts {
        block.push_str(&format!("- {key}: {value}\n"));
    }
    Some(block)
}

/// Strategy 2 - Sticky Facts: the sliding window, plus one synthetic
/// system message carrying the current key-value facts, inserted right
/// after the system prompt so the model sees it as background rather than
/// something either party said.
fn sticky_facts_context(
    messages: &[Message],
    keep_last_n: usize,
    facts: &BTreeMap<String, String>,
) -> Vec<Message> {
    let mut context = sliding_window_context(messages, keep_last_n);
    if let Some(block) = format_facts_block(facts) {
        let idx = 1.min(context.len());
        context.insert(
            idx,
            Message {
                role: "system".to_string(),
                content: block,
            },
        );
    }
    context
}

/// Builds the message list actually sent to the model for one turn, under
/// whichever strategy is active. `Branching` sends the active branch's
/// full history unchanged - its "management" comes from keeping each
/// branch's own history small by not merging directions, not from
/// trimming within one.
fn build_context(
    messages: &[Message],
    strategy: Strategy,
    keep_last_n: usize,
    facts: &BTreeMap<String, String>,
) -> Vec<Message> {
    match strategy {
        Strategy::Branching => messages.to_vec(),
        Strategy::SlidingWindow => sliding_window_context(messages, keep_last_n),
        Strategy::StickyFacts => sticky_facts_context(messages, keep_last_n, facts),
    }
}

fn build_facts_prompt(current: &BTreeMap<String, String>, user_message: &str, assistant_reply: &str) -> String {
    let current_json = serde_json::to_string(current).unwrap_or_else(|_| "{}".to_string());
    format!(
        "CURRENT FACTS:\n{current_json}\n\nLATEST TURN:\nuser: {user_message}\nassistant: {assistant_reply}\n"
    )
}

/// The facts-extraction call isn't asked to use JSON mode (this course's
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

/// Per-request overrides a caller may supply on top of the running
/// conversation. Every field is optional; omitted ones are left out of the
/// API request entirely rather than sent as an explicit null/default.
struct ChatOptions {
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
    /// `None` defers to the agent's currently configured strategy. An
    /// explicit value lets the UI switch strategies per turn without a
    /// separate round trip first.
    strategy: Option<Strategy>,
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
    state_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    keep_last_n: usize,
    default_strategy: Strategy,
}

/// The agent: owns every branch of the conversation (a fresh install has
/// exactly one, "Main"), the checkpoints that can be forked from, the
/// sticky facts memory, and which strategy currently governs what gets
/// sent. All four are persisted together to `state_path` after every
/// mutation.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    default_model: String,
    state_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    keep_last_n: usize,
    default_strategy: Strategy,
    branches: Mutex<Vec<Branch>>,
    active_branch_id: Mutex<String>,
    checkpoints: Mutex<Vec<Checkpoint>>,
    facts: Mutex<BTreeMap<String, String>>,
    strategy: Mutex<Strategy>,
    next_id: AtomicU64,
}

impl Agent {
    fn new(client: reqwest::Client, config: AgentConfig) -> Self {
        let state = Self::load_state(&config.state_path, config.default_strategy);
        Self {
            client,
            endpoint: config.endpoint,
            api_key: config.api_key,
            default_model: config.default_model,
            state_path: config.state_path,
            context_limit: config.context_limit,
            price_per_1m_input: config.price_per_1m_input,
            price_per_1m_output: config.price_per_1m_output,
            keep_last_n: config.keep_last_n,
            default_strategy: config.default_strategy,
            branches: Mutex::new(state.branches),
            active_branch_id: Mutex::new(state.active_branch_id),
            checkpoints: Mutex::new(state.checkpoints),
            facts: Mutex::new(state.facts),
            strategy: Mutex::new(state.strategy),
            next_id: AtomicU64::new(state.next_id),
        }
    }

    /// Loads previously persisted state from disk. Any problem reading or
    /// parsing the file (missing, empty, corrupted, or somehow left with no
    /// branches at all) falls back to a fresh single-branch conversation
    /// rather than failing startup over state that was never load-bearing
    /// to begin with.
    fn load_state(path: &str, default_strategy: Strategy) -> PersistedState {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return default_state(default_strategy),
        };
        match serde_json::from_str::<PersistedState>(&contents) {
            Ok(state) if !state.branches.is_empty() => state,
            Ok(_) => default_state(default_strategy),
            Err(e) => {
                eprintln!("Warning: failed to parse {path} ({e}); starting a fresh conversation.");
                default_state(default_strategy)
            }
        }
    }

    fn snapshot_persisted_state(&self) -> PersistedState {
        PersistedState {
            branches: self.branches.lock().unwrap().clone(),
            active_branch_id: self.active_branch_id.lock().unwrap().clone(),
            checkpoints: self.checkpoints.lock().unwrap().clone(),
            facts: self.facts.lock().unwrap().clone(),
            strategy: *self.strategy.lock().unwrap(),
            next_id: self.next_id.load(Ordering::Relaxed),
        }
    }

    /// Writes the full current state to disk. Best-effort: a write failure
    /// is logged but never allowed to break the in-memory conversation the
    /// user is actively having.
    fn persist_state(&self) {
        let state = self.snapshot_persisted_state();
        match serde_json::to_string_pretty(&state) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.state_path, json) {
                    eprintln!("Warning: failed to write {}: {e}", self.state_path);
                }
            }
            Err(e) => eprintln!("Warning: failed to serialize state: {e}"),
        }
    }

    fn allocate_id(&self, prefix: &str) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}{n}")
    }

    fn find_branch<'a>(branches: &'a [Branch], id: &str) -> Option<&'a Branch> {
        branches.iter().find(|b| b.id == id)
    }

    fn find_branch_mut<'a>(branches: &'a mut [Branch], id: &str) -> Option<&'a mut Branch> {
        branches.iter_mut().find(|b| b.id == id)
    }

    fn active_branch_id(&self) -> String {
        self.active_branch_id.lock().unwrap().clone()
    }

    fn active_branch_messages(&self) -> Vec<Message> {
        let id = self.active_branch_id();
        let branches = self.branches.lock().unwrap();
        Self::find_branch(&branches, &id)
            .map(|b| b.messages.clone())
            .unwrap_or_default()
    }

    fn strategy(&self) -> Strategy {
        *self.strategy.lock().unwrap()
    }

    fn set_strategy(&self, strategy: Strategy) {
        *self.strategy.lock().unwrap() = strategy;
        self.persist_state();
    }

    fn facts_snapshot(&self) -> BTreeMap<String, String> {
        self.facts.lock().unwrap().clone()
    }

    fn checkpoints_snapshot(&self) -> Vec<Checkpoint> {
        self.checkpoints.lock().unwrap().clone()
    }

    fn branches_summary(&self) -> Vec<BranchSummary> {
        let active_id = self.active_branch_id();
        self.branches
            .lock()
            .unwrap()
            .iter()
            .map(|b| BranchSummary {
                id: b.id.clone(),
                name: b.name.clone(),
                message_count: b.messages.len().saturating_sub(1),
                is_active: b.id == active_id,
                parent_id: b.parent_id.clone(),
                forked_at: b.forked_at,
            })
            .collect()
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

    fn pop_last_message(&self, branch_id: &str) {
        let mut branches = self.branches.lock().unwrap();
        if let Some(branch) = Self::find_branch_mut(&mut branches, branch_id) {
            branch.messages.pop();
        }
    }

    /// A descriptive snapshot for endpoints that aren't mid-turn (reset,
    /// history-on-load): what a request would look like *right now* under
    /// the active branch and strategy, with no real call made.
    fn snapshot(&self) -> TokenReport {
        let branch_id = self.active_branch_id();
        let messages = self.active_branch_messages();
        let strategy = self.strategy();
        let facts = self.facts_snapshot();
        let raw_tokens = history_tokens(&messages);
        let sent = build_context(&messages, strategy, self.keep_last_n, &facts);
        let sent_context_tokens = history_tokens(&sent);
        let facts_tokens = format_facts_block(&facts).map(|b| estimate_tokens(&b)).unwrap_or(0);
        let branch_name = {
            let branches = self.branches.lock().unwrap();
            Self::find_branch(&branches, &branch_id)
                .map(|b| b.name.clone())
                .unwrap_or_default()
        };

        TokenReport {
            estimated_request_tokens: 0,
            estimated_history_tokens_before: raw_tokens,
            estimated_response_tokens: 0,
            actual: None,
            raw_history_tokens_after: raw_tokens,
            uncompacted_context_tokens: raw_tokens,
            sent_context_tokens,
            tokens_saved_this_turn: raw_tokens.saturating_sub(sent_context_tokens),
            strategy,
            facts_count: facts.len(),
            facts_tokens,
            branch_id,
            branch_name,
            context_limit: self.context_limit,
            percent_of_limit: (sent_context_tokens as f32 / self.context_limit as f32) * 100.0,
            estimated_cost_usd: None,
        }
    }

    async fn respond(&self, user_message: &str, options: ChatOptions) -> RespondResult {
        let strategy = options.strategy.unwrap_or_else(|| self.strategy());
        let branch_id = self.active_branch_id();

        let (full_history, estimated_request_tokens, estimated_history_before) = {
            let mut branches = self.branches.lock().unwrap();
            let branch = Self::find_branch_mut(&mut branches, &branch_id).expect("active branch must exist");
            let history_before = history_tokens(&branch.messages);
            branch.messages.push(Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            });
            let request_tokens = message_tokens(branch.messages.last().unwrap());
            (branch.messages.clone(), request_tokens, history_before)
        };

        let facts_before = self.facts_snapshot();
        let sent_messages = build_context(&full_history, strategy, self.keep_last_n, &facts_before);
        let uncompacted_context_tokens = history_tokens(&full_history);
        let sent_context_tokens = history_tokens(&sent_messages);

        let branch_name = {
            let branches = self.branches.lock().unwrap();
            Self::find_branch(&branches, &branch_id)
                .map(|b| b.name.clone())
                .unwrap_or_default()
        };

        let make_report = |actual: Option<Usage>,
                            estimated_response_tokens: u32,
                            raw_history_tokens_after: u32,
                            facts_count: usize,
                            facts_tokens: u32|
         -> TokenReport {
            let effective_sent = actual.as_ref().map(|u| u.prompt_tokens).unwrap_or(sent_context_tokens);
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
                strategy,
                facts_count,
                facts_tokens,
                branch_id: branch_id.clone(),
                branch_name: branch_name.clone(),
                context_limit: self.context_limit,
                percent_of_limit,
                estimated_cost_usd,
            }
        };

        let facts_tokens_before = format_facts_block(&facts_before).map(|b| estimate_tokens(&b)).unwrap_or(0);

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
                self.pop_last_message(&branch_id);
                return RespondResult {
                    reply: format!("Request failed: {e}"),
                    tokens: make_report(None, 0, estimated_history_before, facts_before.len(), facts_tokens_before),
                };
            }
        };

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            self.pop_last_message(&branch_id);
            let detail = serde_json::from_str::<ErrorResponse>(&body)
                .map(|e| e.error.message)
                .unwrap_or(body);
            return RespondResult {
                reply: format!("API error ({status}): {detail}"),
                tokens: make_report(None, 0, estimated_history_before, facts_before.len(), facts_tokens_before),
            };
        }

        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(p) => p,
            Err(e) => {
                self.pop_last_message(&branch_id);
                return RespondResult {
                    reply: format!("Failed to parse response: {e}"),
                    tokens: make_report(None, 0, estimated_history_before, facts_before.len(), facts_tokens_before),
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
            let mut branches = self.branches.lock().unwrap();
            let branch = Self::find_branch_mut(&mut branches, &branch_id).expect("active branch must exist");
            branch.messages.push(Message {
                role: "assistant".to_string(),
                content: reply.clone(),
            });
            history_tokens(&branch.messages)
        };
        self.persist_state();

        self.update_facts(user_message, &reply).await;
        let facts_after = self.facts_snapshot();
        let facts_tokens_after = format_facts_block(&facts_after).map(|b| estimate_tokens(&b)).unwrap_or(0);

        RespondResult {
            tokens: make_report(
                parsed.usage,
                estimated_response_tokens,
                raw_history_tokens_after,
                facts_after.len(),
                facts_tokens_after,
            ),
            reply,
        }
    }

    /// Strategy 2's other half: after every turn (regardless of which
    /// strategy is currently active, so switching to Sticky Facts later
    /// sees an already-current memory), asks the same backend to fold the
    /// latest turn into the running key-value facts. Best-effort: any
    /// failure - network, a non-200, unparseable JSON - just leaves the
    /// prior facts in place and gets retried on the next turn; it never
    /// blocks or breaks the chat reply it rode in on.
    async fn update_facts(&self, user_message: &str, assistant_reply: &str) {
        let current = self.facts_snapshot();
        let prompt = build_facts_prompt(&current, user_message, assistant_reply);
        let request = ChatCompletionRequest {
            model: self.default_model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: FACTS_SYSTEM_PROMPT.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: prompt,
                },
            ],
            temperature: Some(0.0),
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
                eprintln!("Warning: facts update request failed, keeping prior facts: {e}");
                return;
            }
        };

        if !response.status().is_success() {
            eprintln!(
                "Warning: facts update API error ({}), keeping prior facts.",
                response.status()
            );
            return;
        }

        let body = response.text().await.unwrap_or_default();
        let parsed = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: failed to parse facts response, keeping prior facts: {e}");
                return;
            }
        };
        let raw = match parsed.choices.into_iter().next() {
            Some(choice) => choice.message.content,
            None => return,
        };
        let cleaned = strip_json_fences(&raw);
        match serde_json::from_str::<BTreeMap<String, String>>(&cleaned) {
            Ok(new_facts) => {
                *self.facts.lock().unwrap() = new_facts;
                self.persist_state();
            }
            Err(e) => {
                eprintln!("Warning: facts response wasn't valid JSON, keeping prior facts: {e}");
            }
        }
    }

    /// Strategy 3, step 1: records the active branch's current length so it
    /// can be forked from later - possibly more than once, and possibly
    /// after the branch itself has kept going past this point.
    fn create_checkpoint(&self, label: String) -> Checkpoint {
        let branch_id = self.active_branch_id();
        let message_count = {
            let branches = self.branches.lock().unwrap();
            Self::find_branch(&branches, &branch_id)
                .map(|b| b.messages.len().saturating_sub(1))
                .unwrap_or(0)
        };
        let checkpoint = Checkpoint {
            id: self.allocate_id("cp"),
            label,
            branch_id,
            message_count,
        };
        self.checkpoints.lock().unwrap().push(checkpoint.clone());
        self.persist_state();
        checkpoint
    }

    /// Strategy 3, step 2: creates a new branch that replays exactly the
    /// checkpoint's messages (system prompt + `message_count` visible
    /// messages) and nothing after it. The new branch does not become
    /// active automatically - call `switch_branch` for that - so creating
    /// two forks from the same checkpoint never has the second overwrite
    /// the first's view.
    fn fork_branch(&self, checkpoint_id: &str, name: String) -> Result<BranchSummary, String> {
        let (source_branch_id, message_count) = {
            let checkpoints = self.checkpoints.lock().unwrap();
            match checkpoints.iter().find(|c| c.id == checkpoint_id) {
                Some(cp) => (cp.branch_id.clone(), cp.message_count),
                None => return Err(format!("No checkpoint with id {checkpoint_id}")),
            }
        };
        let messages = {
            let branches = self.branches.lock().unwrap();
            match Self::find_branch(&branches, &source_branch_id) {
                Some(b) => {
                    let raw_len = (message_count + 1).min(b.messages.len());
                    b.messages[..raw_len].to_vec()
                }
                None => return Err(format!("Source branch {source_branch_id} no longer exists")),
            }
        };
        let new_branch = Branch {
            id: self.allocate_id("b"),
            name,
            messages,
            parent_id: Some(source_branch_id),
            forked_at: Some(message_count),
        };
        let summary = BranchSummary {
            id: new_branch.id.clone(),
            name: new_branch.name.clone(),
            message_count: new_branch.messages.len().saturating_sub(1),
            is_active: false,
            parent_id: new_branch.parent_id.clone(),
            forked_at: new_branch.forked_at,
        };
        self.branches.lock().unwrap().push(new_branch);
        self.persist_state();
        Ok(summary)
    }

    fn switch_branch(&self, branch_id: &str) -> Result<(), String> {
        let exists = self.branches.lock().unwrap().iter().any(|b| b.id == branch_id);
        if !exists {
            return Err(format!("No branch with id {branch_id}"));
        }
        *self.active_branch_id.lock().unwrap() = branch_id.to_string();
        self.persist_state();
        Ok(())
    }

    /// Clears everything back to a single fresh "Main" branch, no
    /// checkpoints, no facts, and the strategy this agent was configured
    /// with by default.
    fn reset(&self) -> TokenReport {
        let fresh = default_state(self.default_strategy);
        *self.branches.lock().unwrap() = fresh.branches;
        *self.active_branch_id.lock().unwrap() = fresh.active_branch_id;
        *self.checkpoints.lock().unwrap() = Vec::new();
        *self.facts.lock().unwrap() = BTreeMap::new();
        *self.strategy.lock().unwrap() = fresh.strategy;
        self.next_id.store(1, Ordering::Relaxed);
        self.persist_state();
        self.snapshot()
    }

    fn visible_history(&self) -> Vec<Message> {
        self.active_branch_messages()
            .into_iter()
            .filter(|m| m.role != "system")
            .collect()
    }
}

#[cfg(test)]
impl Agent {
    fn branch_messages_snapshot(&self, id: &str) -> Vec<Message> {
        let branches = self.branches.lock().unwrap();
        Self::find_branch(&branches, id).cloned().map(|b| b.messages).unwrap_or_default()
    }
}

/// What a UI lists per branch - never the raw `Branch`, so the (possibly
/// large) message list isn't re-sent just to render a switcher.
#[derive(Serialize)]
struct BranchSummary {
    id: String,
    name: String,
    message_count: usize,
    is_active: bool,
    parent_id: Option<String>,
    forked_at: Option<usize>,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
    strategy: Option<Strategy>,
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
    facts: BTreeMap<String, String>,
    branches: Vec<BranchSummary>,
    checkpoints: Vec<Checkpoint>,
    keep_last_n: usize,
}

#[derive(Deserialize)]
struct StrategyRequest {
    strategy: Strategy,
}

#[derive(Deserialize)]
struct CheckpointRequest {
    label: Option<String>,
}

#[derive(Deserialize)]
struct ForkRequest {
    checkpoint_id: String,
    name: String,
}

#[derive(Deserialize)]
struct SwitchRequest {
    branch_id: String,
}

async fn chat(State(agent): State<Arc<Agent>>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    let result = agent
        .respond(
            &req.message,
            ChatOptions {
                model: req.model,
                temperature: req.temperature,
                max_tokens: req.max_tokens,
                stop: req.stop,
                strategy: req.strategy,
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

/// Lets the page rehydrate the visible chat bubbles, the token stats bar,
/// the sticky facts, and the branch/checkpoint lists on load, so a browser
/// reload after the server restarts shows the same conversation.
async fn history(State(agent): State<Arc<Agent>>) -> Json<HistoryResponse> {
    let tokens = agent.snapshot();
    Json(HistoryResponse {
        messages: agent.visible_history(),
        facts: agent.facts_snapshot(),
        branches: agent.branches_summary(),
        checkpoints: agent.checkpoints_snapshot(),
        keep_last_n: agent.keep_last_n,
        tokens,
    })
}

async fn facts(State(agent): State<Arc<Agent>>) -> Json<BTreeMap<String, String>> {
    Json(agent.facts_snapshot())
}

async fn set_strategy(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<StrategyRequest>,
) -> Json<TokenReport> {
    agent.set_strategy(req.strategy);
    Json(agent.snapshot())
}

async fn list_branches(State(agent): State<Arc<Agent>>) -> Json<Vec<BranchSummary>> {
    Json(agent.branches_summary())
}

async fn switch_branch(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<SwitchRequest>,
) -> Result<Json<Vec<BranchSummary>>, (StatusCode, String)> {
    agent.switch_branch(&req.branch_id).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.branches_summary()))
}

async fn fork_branch(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<ForkRequest>,
) -> Result<Json<BranchSummary>, (StatusCode, String)> {
    agent
        .fork_branch(&req.checkpoint_id, req.name)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

async fn list_checkpoints(State(agent): State<Arc<Agent>>) -> Json<Vec<Checkpoint>> {
    Json(agent.checkpoints_snapshot())
}

async fn create_checkpoint(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<CheckpointRequest>,
) -> Json<Checkpoint> {
    let label = req.label.unwrap_or_else(|| "Checkpoint".to_string());
    Json(agent.create_checkpoint(label))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".to_string());
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
    let state_path = std::env::var("STATE_FILE").unwrap_or_else(|_| "state.json".to_string());
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
    let default_strategy = std::env::var("DEFAULT_STRATEGY")
        .ok()
        .and_then(|v| Strategy::from_env_str(&v))
        .unwrap_or(Strategy::SlidingWindow);

    let agent = Agent::new(
        reqwest::Client::new(),
        AgentConfig {
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            default_model,
            state_path: state_path.clone(),
            context_limit,
            price_per_1m_input,
            price_per_1m_output,
            keep_last_n,
            default_strategy,
        },
    );

    let restored_messages = agent.active_branch_messages().len().saturating_sub(1);
    if restored_messages > 0 {
        println!(
            "Restored {restored_messages} message(s) from {state_path}; default strategy: {:?}.",
            agent.strategy()
        );
    } else {
        println!(
            "No previous state at {state_path}; starting a fresh conversation. Default strategy: {:?}.",
            agent.strategy()
        );
    }
    let agent = Arc::new(agent);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/chat", post(chat))
        .route("/api/reset", post(reset))
        .route("/api/history", get(history))
        .route("/api/facts", get(facts))
        .route("/api/strategy", post(set_strategy))
        .route("/api/branches", get(list_branches))
        .route("/api/branches/switch", post(switch_branch))
        .route("/api/branches/fork", post(fork_branch))
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .with_state(agent);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!("ai-advent-lesson-10-{label}-{}-{}.json", std::process::id(), n))
            .to_string_lossy()
            .into_owned()
    }

    fn test_agent_with_path(state_path: String) -> Agent {
        Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                default_model: "test-model".to_string(),
                state_path,
                context_limit: 1000,
                price_per_1m_input: Some(2.0),
                price_per_1m_output: Some(4.0),
                keep_last_n: 4,
                default_strategy: Strategy::SlidingWindow,
            },
        )
    }

    fn test_agent() -> Agent {
        test_agent_with_path(temp_path("state"))
    }

    fn sample_messages(non_system_count: usize) -> Vec<Message> {
        let mut messages = vec![Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }];
        for i in 0..non_system_count {
            messages.push(Message {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("message {i}"),
            });
        }
        messages
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
    fn sliding_window_keeps_everything_when_within_the_window() {
        let messages = sample_messages(3);
        let ctx = sliding_window_context(&messages, 4);
        assert_eq!(ctx.len(), messages.len());
    }

    #[test]
    fn sliding_window_trims_to_the_last_n_and_keeps_the_system_prompt() {
        let messages = sample_messages(10);
        let ctx = sliding_window_context(&messages, 4);
        assert_eq!(ctx.len(), 5); // system + last 4
        assert_eq!(ctx[0].role, "system");
        assert_eq!(ctx[1].content, "message 6");
        assert_eq!(ctx[4].content, "message 9");
    }

    #[test]
    fn build_context_branching_returns_the_full_history_unchanged() {
        let messages = sample_messages(20);
        let ctx = build_context(&messages, Strategy::Branching, 4, &BTreeMap::new());
        assert_eq!(ctx.len(), messages.len());
    }

    #[test]
    fn build_context_sticky_facts_inserts_the_facts_block_after_the_system_prompt() {
        let messages = sample_messages(10);
        let mut facts = BTreeMap::new();
        facts.insert("goal".to_string(), "build a web app".to_string());
        let ctx = build_context(&messages, Strategy::StickyFacts, 4, &facts);
        assert_eq!(ctx.len(), 6); // system + facts-note + last 4
        assert_eq!(ctx[0].role, "system");
        assert_eq!(ctx[1].role, "system");
        assert!(ctx[1].content.contains("goal: build a web app"));
        assert_eq!(ctx[2].content, "message 6");
    }

    #[test]
    fn build_context_sticky_facts_skips_the_note_when_there_are_no_facts_yet() {
        let messages = sample_messages(10);
        let ctx = build_context(&messages, Strategy::StickyFacts, 4, &BTreeMap::new());
        assert_eq!(ctx.len(), 5); // system + last 4, no facts note
    }

    #[test]
    fn format_facts_block_is_none_when_empty() {
        assert_eq!(format_facts_block(&BTreeMap::new()), None);
    }

    #[test]
    fn format_facts_block_lists_every_key_sorted() {
        let mut facts = BTreeMap::new();
        facts.insert("z_last".to_string(), "z".to_string());
        facts.insert("a_first".to_string(), "a".to_string());
        let block = format_facts_block(&facts).unwrap();
        let a_pos = block.find("a_first").unwrap();
        let z_pos = block.find("z_last").unwrap();
        assert!(a_pos < z_pos);
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
    fn build_facts_prompt_includes_current_facts_and_the_latest_turn() {
        let mut facts = BTreeMap::new();
        facts.insert("goal".to_string(), "ship an mvp".to_string());
        let prompt = build_facts_prompt(&facts, "we also need auth", "noted, adding auth");
        assert!(prompt.contains("ship an mvp"));
        assert!(prompt.contains("we also need auth"));
        assert!(prompt.contains("noted, adding auth"));
    }

    #[test]
    fn new_seeds_a_single_main_branch_when_no_state_file_exists() {
        let agent = test_agent();
        let branches = agent.branches_summary();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].id, "b0");
        assert!(branches[0].is_active);
        assert_eq!(agent.strategy(), Strategy::SlidingWindow);
    }

    #[test]
    fn new_falls_back_to_default_state_on_invalid_json() {
        let state_path = temp_path("state");
        std::fs::write(&state_path, "not json").unwrap();
        let agent = test_agent_with_path(state_path.clone());
        assert_eq!(agent.branches_summary().len(), 1);
        let _ = std::fs::remove_file(&state_path);
    }

    #[test]
    fn new_loads_previously_persisted_state() {
        let state_path = temp_path("state");
        let mut facts = BTreeMap::new();
        facts.insert("goal".to_string(), "ship an mvp".to_string());
        let seeded = PersistedState {
            branches: vec![Branch {
                id: "b0".to_string(),
                name: "Main".to_string(),
                messages: vec![
                    Message {
                        role: "system".to_string(),
                        content: SYSTEM_PROMPT.to_string(),
                    },
                    Message {
                        role: "user".to_string(),
                        content: "hi".to_string(),
                    },
                ],
                parent_id: None,
                forked_at: None,
            }],
            active_branch_id: "b0".to_string(),
            checkpoints: Vec::new(),
            facts,
            strategy: Strategy::StickyFacts,
            next_id: 5,
        };
        std::fs::write(&state_path, serde_json::to_string(&seeded).unwrap()).unwrap();

        let agent = test_agent_with_path(state_path.clone());
        assert_eq!(agent.strategy(), Strategy::StickyFacts);
        assert_eq!(agent.facts_snapshot().get("goal"), Some(&"ship an mvp".to_string()));
        assert_eq!(agent.active_branch_messages().len(), 2);
        assert_eq!(agent.allocate_id("cp"), "cp5"); // continues from the persisted counter

        let _ = std::fs::remove_file(&state_path);
    }

    #[test]
    fn checkpoint_and_fork_creates_an_independent_branch_history() {
        let agent = test_agent();
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            branch.messages.push(Message {
                role: "user".to_string(),
                content: "we need a web app".to_string(),
            });
            branch.messages.push(Message {
                role: "assistant".to_string(),
                content: "got it".to_string(),
            });
        }

        let checkpoint = agent.create_checkpoint("after scope".to_string());
        assert_eq!(checkpoint.message_count, 2);

        // The source branch keeps going after the checkpoint was taken.
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            branch.messages.push(Message {
                role: "user".to_string(),
                content: "actually make it mobile".to_string(),
            });
        }

        let fork = agent.fork_branch(&checkpoint.id, "Mobile idea".to_string()).unwrap();
        assert_eq!(fork.message_count, 2); // only what existed at the checkpoint
        assert!(!fork.is_active); // forking never switches the active branch

        let forked_messages = agent.branch_messages_snapshot(&fork.id);
        assert_eq!(forked_messages.len(), 3); // system + 2
        assert!(!forked_messages.iter().any(|m| m.content.contains("mobile")));

        let main_messages = agent.branch_messages_snapshot("b0");
        assert_eq!(main_messages.len(), 4); // system + 3, unaffected by the fork
    }

    #[test]
    fn two_forks_from_the_same_checkpoint_stay_independent() {
        let agent = test_agent();
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            branch.messages.push(Message {
                role: "user".to_string(),
                content: "gathering requirements".to_string(),
            });
        }
        let checkpoint = agent.create_checkpoint("scope agreed".to_string());

        let fork_a = agent.fork_branch(&checkpoint.id, "Web option".to_string()).unwrap();
        let fork_b = agent.fork_branch(&checkpoint.id, "Mobile option".to_string()).unwrap();
        assert_ne!(fork_a.id, fork_b.id);

        {
            let mut branches = agent.branches.lock().unwrap();
            Agent::find_branch_mut(&mut branches, &fork_a.id)
                .unwrap()
                .messages
                .push(Message {
                    role: "user".to_string(),
                    content: "web-only detail".to_string(),
                });
        }

        let fork_b_messages = agent.branch_messages_snapshot(&fork_b.id);
        assert!(!fork_b_messages.iter().any(|m| m.content.contains("web-only")));
    }

    #[test]
    fn switch_branch_changes_the_active_branch() {
        let agent = test_agent();
        let checkpoint = agent.create_checkpoint("start".to_string());
        let fork = agent.fork_branch(&checkpoint.id, "Branch A".to_string()).unwrap();
        agent.switch_branch(&fork.id).unwrap();
        assert_eq!(agent.active_branch_id(), fork.id);
    }

    #[test]
    fn switch_branch_rejects_an_unknown_id() {
        let agent = test_agent();
        assert!(agent.switch_branch("nope").is_err());
    }

    #[test]
    fn fork_branch_rejects_an_unknown_checkpoint() {
        let agent = test_agent();
        assert!(agent.fork_branch("nope", "x".to_string()).is_err());
    }

    #[test]
    fn reset_clears_branches_checkpoints_and_facts() {
        let state_path = temp_path("state");
        let agent = test_agent_with_path(state_path.clone());
        agent.create_checkpoint("cp".to_string());
        agent.facts.lock().unwrap().insert("goal".to_string(), "x".to_string());
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            branch.messages.push(Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
        }
        agent.set_strategy(Strategy::Branching);

        let report = agent.reset();
        assert_eq!(
            report.raw_history_tokens_after,
            history_tokens(&agent.active_branch_messages())
        );
        assert_eq!(agent.branches_summary().len(), 1);
        assert!(agent.checkpoints_snapshot().is_empty());
        assert!(agent.facts_snapshot().is_empty());
        assert_eq!(agent.strategy(), Strategy::SlidingWindow); // back to the configured default

        let on_disk: PersistedState = serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
        assert_eq!(on_disk.branches.len(), 1);

        let _ = std::fs::remove_file(&state_path);
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
    fn estimate_cost_is_none_without_configured_prices() {
        let agent = Agent::new(
            reqwest::Client::new(),
            AgentConfig {
                endpoint: "http://localhost:0/chat/completions".to_string(),
                api_key: "test-key".to_string(),
                default_model: "test-model".to_string(),
                state_path: temp_path("state"),
                context_limit: 1000,
                price_per_1m_input: None,
                price_per_1m_output: None,
                keep_last_n: 4,
                default_strategy: Strategy::SlidingWindow,
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
    fn visible_history_omits_the_system_prompt() {
        let agent = test_agent();
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            branch.messages.push(Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            });
        }
        let visible = agent.visible_history();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].role, "user");
    }

    #[test]
    fn snapshot_reflects_the_active_strategy() {
        let agent = test_agent(); // keep_last_n = 4, sliding_window default
        {
            let mut branches = agent.branches.lock().unwrap();
            let branch = Agent::find_branch_mut(&mut branches, "b0").unwrap();
            for i in 0..10 {
                branch.messages.push(Message {
                    role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                    content: format!("a reasonably long message body number {i}"),
                });
            }
        }
        let report = agent.snapshot();
        assert!(report.sent_context_tokens < report.uncompacted_context_tokens);
        assert_eq!(report.strategy, Strategy::SlidingWindow);

        agent.set_strategy(Strategy::Branching);
        let report = agent.snapshot();
        assert_eq!(report.sent_context_tokens, report.uncompacted_context_tokens);
        assert_eq!(report.tokens_saved_this_turn, 0);
    }
}

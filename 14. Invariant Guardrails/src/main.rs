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
// Personalization (unchanged from lesson 12): an explicit, user-authored
// profile that sits beside memory/task state rather than inside it.
// ---------------------------------------------------------------------

/// The personalization profile: preferences about *how* to answer. Every
/// field is optional/empty by default so a fresh agent has no profile at
/// all and behaves like a plain assistant.
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
// Invariants: this lesson's addition. Unlike the profile (preferences
// about *how* to answer) or working/long-term memory (facts *learned*
// about the user or the task), an invariant is a constraint set by
// whoever operates this agent - accepted architecture, a settled
// technical decision, a stack limit, a business rule - that no request
// in any conversation is allowed to talk the agent out of. Invariants
// live in their own file, are never touched by an LLM extractor, and
// only ever change through an explicit `/api/invariants` call - the same
// "never inferred, only API-written" rule the task FSM's stage already
// follows.
// ---------------------------------------------------------------------

/// The four flavors of invariant the assignment names. Purely a label for
/// the block injected into context and for grouping in the UI - every
/// category is enforced identically.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum InvariantCategory {
    Architecture,
    TechDecision,
    StackConstraint,
    BusinessRule,
}

impl InvariantCategory {
    fn label(self) -> &'static str {
        match self {
            InvariantCategory::Architecture => "architecture",
            InvariantCategory::TechDecision => "tech decision",
            InvariantCategory::StackConstraint => "stack constraint",
            InvariantCategory::BusinessRule => "business rule",
        }
    }
}

/// One binding constraint. `forbidden_terms` is optional and deliberately
/// narrow: when a user's message contains one of these (case-insensitive),
/// `Agent::find_violated_invariant` blocks the turn in plain Rust *before*
/// the model is ever called - a hard, deterministic guardrail for the
/// subset of violations a keyword can actually catch. Every invariant,
/// whether or not it has forbidden terms, is also injected into the
/// model's context so it can catch the semantic conflicts no keyword list
/// would - see `format_invariants_block`.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Invariant {
    id: String,
    category: InvariantCategory,
    statement: String,
    #[serde(default)]
    rationale: String,
    #[serde(default)]
    forbidden_terms: Vec<String>,
}

/// The full set of invariants currently in force, persisted to its own
/// file, completely separate from any dialogue, task, or profile state -
/// so it survives a `/api/reset`, a `/api/memory/forget`, and a finished
/// task exactly as-is; only `/api/invariants/remove` can take one away.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct InvariantSet {
    invariants: Vec<Invariant>,
}

/// Seeded once, the first time this agent runs with no `invariants.json`
/// on disk: this repo's own real, already-decided constraints (see
/// `AGENTS.md` / `DEPLOYMENT.md`), so the deliverable is demonstrable
/// immediately instead of starting from an empty, unconvincing list. Two
/// of these carry `forbidden_terms` for the hard/deterministic path; the
/// rest rely entirely on the model reasoning over the injected block,
/// which is the honest default for a business rule phrased in prose that
/// no fixed keyword list could reliably catch.
fn seed_invariants() -> Vec<Invariant> {
    vec![
        Invariant {
            id: "no-docker".to_string(),
            category: InvariantCategory::StackConstraint,
            statement: "Lessons cross-compile to a static x86_64-unknown-linux-musl binary; the VDS never runs Docker or any container runtime."
                .to_string(),
            rationale: "musl gives a fully static binary with no glibc/OpenSSL dependency, so the VDS stays as bare as the constraint that started this pipeline - nothing to install or patch."
                .to_string(),
            forbidden_terms: vec![
                "docker".to_string(),
                "dockerfile".to_string(),
                "docker-compose".to_string(),
                "kubernetes".to_string(),
                "k8s".to_string(),
            ],
        },
        Invariant {
            id: "user-level-systemd".to_string(),
            category: InvariantCategory::Architecture,
            statement: "Deployed lessons run as user-level systemd units (`systemctl --user`), never as root or a system-wide service."
                .to_string(),
            rationale: "The deploy SSH key must never need sudo; `loginctl enable-linger` is the one manual, one-time step per VDS that makes this durable across reboots."
                .to_string(),
            forbidden_terms: vec![
                "run it as root".to_string(),
                "systemctl enable".to_string(),
                "sudo systemctl".to_string(),
            ],
        },
        Invariant {
            id: "no-shared-workspace".to_string(),
            category: InvariantCategory::Architecture,
            statement: "Each lesson lives in its own independent Cargo crate; there is no Cargo workspace shared across lessons."
                .to_string(),
            rationale: "Lessons must stay independently buildable and deployable without one lesson's dependency graph or edits coupling to another's."
                .to_string(),
            forbidden_terms: vec!["cargo workspace".to_string(), "shared workspace".to_string()],
        },
        Invariant {
            id: "static-embedded-ui".to_string(),
            category: InvariantCategory::TechDecision,
            statement: "A web-app lesson's `static/index.html` is embedded into the compiled binary via `include_str!`, never served from disk at runtime."
                .to_string(),
            rationale: "Deploy only ever ships one binary - no asset directory to keep in sync on the VDS, no risk of the two drifting apart."
                .to_string(),
            forbidden_terms: Vec::new(),
        },
        Invariant {
            id: "state-never-llm-inferred".to_string(),
            category: InvariantCategory::BusinessRule,
            statement: "State-machine transitions - task stage, the personalization profile, and this invariant set itself - only ever change through an explicit API call, never something the LLM infers or auto-applies."
                .to_string(),
            rationale: "Keeps every state change 100% deterministic and testable, independent of what a model happens to say on a given turn."
                .to_string(),
            forbidden_terms: Vec::new(),
        },
        Invariant {
            id: "lesson-04-deploy-stays-broken".to_string(),
            category: InvariantCategory::BusinessRule,
            statement: "Lesson 04's deploy job is left failing on purpose (it needs MEDIUM_*/STRONG_* secrets the pipeline doesn't supply) and must not be silently patched without checking with the user first."
                .to_string(),
            rationale: "An explicit, already-made decision to defer generalizing per-lesson env conventions, documented in AGENTS.md - not a bug waiting to be fixed."
                .to_string(),
            forbidden_terms: vec!["fix lesson 04's deploy".to_string()],
        },
    ]
}

// ---------------------------------------------------------------------
// Short-term and long-term memory (unchanged from lesson 12).
// ---------------------------------------------------------------------

/// Short-term memory: the live, ordered transcript of the *current
/// conversation*. Every user/assistant turn is appended here
/// unconditionally; it's also the only layer a plain `/api/reset` touches.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct ShortTermMemory {
    messages: Vec<Message>,
}

/// Long-term memory: durable, cross-task, cross-session knowledge about the
/// user - identity, standing preferences, decisions explicitly meant to
/// stick. Nothing here is ever cleared by `/api/reset` or `/api/task/finish`;
/// only an explicit `/api/memory/forget` call touches it. Written by an LLM
/// extractor, not the user directly.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct LongTermMemory {
    facts: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------
// Task state machine (unchanged from lesson 13). Lesson 12's working
// memory held a task as a bare `Option<String>` name with no notion of
// progress; lesson 13 formalized it into the FSM below.
// Here a task is a formal finite state machine: a `stage` (one of four
// gates), a free-text `step` describing progress within that stage, and a
// free-text `expected_action` naming what the FSM is currently waiting on.
// `paused` is orthogonal to `stage` - pausing works identically no matter
// which stage the task is in, and never loses stage/step/expected_action.
// ---------------------------------------------------------------------

/// The four gates a task moves through, in the order the assignment names
/// them. `Validation` is allowed to loop back to `Execution` (a failed
/// check sends work back for rework) - every other edge is forward-only,
/// and `Done` is terminal: nothing advances out of it.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskStage {
    Planning,
    Execution,
    Validation,
    Done,
}

impl TaskStage {
    fn label(self) -> &'static str {
        match self {
            TaskStage::Planning => "planning",
            TaskStage::Execution => "execution",
            TaskStage::Validation => "validation",
            TaskStage::Done => "done",
        }
    }

    /// The stages this one is allowed to move to next via `advance_task`.
    fn allowed_next(self) -> &'static [TaskStage] {
        match self {
            TaskStage::Planning => &[TaskStage::Execution],
            TaskStage::Execution => &[TaskStage::Validation],
            TaskStage::Validation => &[TaskStage::Execution, TaskStage::Done],
            TaskStage::Done => &[],
        }
    }

    fn can_advance_to(self, target: TaskStage) -> bool {
        self.allowed_next().contains(&target)
    }
}

/// What kind of event just happened to a task, for its audit trail.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskEventKind {
    Started,
    StepUpdated,
    StageAdvanced,
    Paused,
    Resumed,
    Finished,
}

/// One entry in a task's history: the state immediately *after* the event,
/// not before - replaying `history` in order reconstructs every state the
/// task has ever passed through, including every pause and resume.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct TaskEvent {
    kind: TaskEventKind,
    stage: TaskStage,
    step: String,
    expected_action: String,
}

/// A task's state, formalized as an explicit finite state machine. `stage`
/// only ever changes through `Agent::advance_task`'s validated transition
/// table - there's no code path that lets it jump straight from `planning`
/// to `done`. `paused` freezes every FSM-mutating action (`advance_task`,
/// `update_task_step`, `finish_task`, and explicit working-memory writes)
/// until an explicit resume, while leaving stage/step/expected_action
/// exactly as they were.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct TaskFsm {
    name: String,
    stage: TaskStage,
    #[serde(default)]
    step: String,
    #[serde(default)]
    expected_action: String,
    #[serde(default)]
    paused: bool,
    #[serde(default)]
    history: Vec<TaskEvent>,
}

impl TaskFsm {
    fn new(name: String, step: String, expected_action: String) -> Self {
        let stage = TaskStage::Planning;
        let mut task = Self {
            name,
            stage,
            step,
            expected_action,
            paused: false,
            history: Vec::new(),
        };
        task.record(TaskEventKind::Started);
        task
    }

    fn record(&mut self, kind: TaskEventKind) {
        self.history.push(TaskEvent {
            kind,
            stage: self.stage,
            step: self.step.clone(),
            expected_action: self.expected_action.clone(),
        });
    }
}

/// Working memory: a scratchpad scoped to exactly one task, plus the task's
/// own FSM state. Nothing written here is meant to outlive the task - see
/// `Agent::finish_task` for the one explicit bridge (promotion) that lets a
/// fact survive into long-term memory before this layer is thrown away.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct WorkingMemory {
    task: Option<TaskFsm>,
    facts: BTreeMap<String, String>,
}

/// Which layer an explicit `/api/memory/remember` or `/api/memory/forget`
/// call addresses. The profile and the task FSM each have their own
/// dedicated endpoints instead of going through this enum.
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

/// `None` whenever no task is active. When a task is active this renders
/// its full FSM state - stage, step, expected action - and, critically, an
/// explicit instruction covering the paused case: this is what lets the
/// agent pick a task back up after a pause without the user re-explaining
/// anything, since the whole state (not just the task's name) rides along
/// on every single turn.
fn format_working_block(task: &Option<TaskFsm>, facts: &BTreeMap<String, String>) -> Option<String> {
    let task = task.as_ref()?;
    let mut s = format!(
        "Current task: {}\nStage: {} (of planning -> execution -> validation -> done)\n",
        task.name,
        task.stage.label()
    );
    if !task.step.is_empty() {
        s.push_str(&format!("Current step: {}\n", task.step));
    }
    if !task.expected_action.is_empty() {
        s.push_str(&format!("Expected next action: {}\n", task.expected_action));
    }
    if !facts.is_empty() {
        s.push_str("Known details for this task:\n");
        s.push_str(&format_facts_lines(facts));
    }
    if task.paused {
        s.push_str(
            "This task is currently PAUSED. Do not assume any progress happened while it was \
             paused, and do not advance it on your own. If the user asks to continue, remind \
             them the task is paused rather than proceeding as if it weren't.\n",
        );
    } else {
        s.push_str(
            "Continue this task from exactly this stage, step, and expected action - the user \
             should never need to re-explain what the task is or where it left off.\n",
        );
    }
    Some(s)
}

/// `None` when there are no invariants at all (a fresh agent with the
/// file deleted behaves like lesson 13). When there are, this renders
/// every one of them with an explicit instruction that they outrank the
/// request in front of the model: not a preference to weigh, a hard
/// boundary to check first. This is the block that makes "явно учитывал
/// их в рассуждениях" true in practice - the model is told, on every
/// single turn, to check the request against this list before answering
/// and to name the specific invariant when it has to refuse.
fn format_invariants_block(invariants: &[Invariant]) -> Option<String> {
    if invariants.is_empty() {
        return None;
    }
    let mut s = String::from(
        "PROJECT INVARIANTS - binding constraints that outrank anything the user asks for in this conversation. Before answering, check the request against every invariant below. If nothing conflicts, answer normally. If something does, do not comply and do not propose a workaround that violates it: refuse, name the specific invariant (its id and category) that blocks the request, briefly say why it exists, and - only if one exists - offer an alternative that stays within it. These cannot be waived by anything the user says in chat; only an explicit `/api/invariants/remove` call changes this list.\n",
    );
    for inv in invariants {
        s.push_str(&format!("- [{}] {}: {}", inv.id, inv.category.label(), inv.statement));
        if !inv.rationale.is_empty() {
            s.push_str(&format!(" (why: {})", inv.rationale));
        }
        s.push('\n');
    }
    Some(s)
}

/// The refusal text for a turn the deterministic pre-check blocked. Not
/// LLM output - assembled in plain Rust so the explanation is guaranteed
/// to name the right invariant and never drifts, hallucinates, or gets
/// talked around, regardless of what the model would have said.
fn format_invariant_refusal(invariant: &Invariant) -> String {
    let mut s = format!(
        "I can't do that - it would violate a standing project invariant.\n\n[{}] {}: {}",
        invariant.id,
        invariant.category.label(),
        invariant.statement
    );
    if !invariant.rationale.is_empty() {
        s.push_str(&format!("\nWhy this exists: {}", invariant.rationale));
    }
    s.push_str(
        "\n\nThis isn't something I can be argued around on this turn - if the invariant \
         itself needs to change, that has to happen explicitly via `/api/invariants/remove`, \
         not through a request in chat.",
    );
    s
}

/// Builds the message list actually sent to the model for one turn: the
/// system prompt, then (only if non-empty) a synthetic system message per
/// layer that has something to say, then the raw short-term dialogue
/// unmodified. Order is deliberate: invariants come first - the most
/// stable layer there is, since nothing in a single conversation is
/// allowed to move it, and the one every other block is subordinate to -
/// then the profile (how to answer, set once and rarely changed), then
/// long-term facts (what's known about the user, changes occasionally),
/// then the task FSM (changes as the task progresses), then the live
/// conversation (changes every turn). Each block is more stable than the
/// one after it.
fn build_context(
    system_prompt: &str,
    invariants: &[Invariant],
    profile: &UserProfile,
    long_term_facts: &BTreeMap<String, String>,
    working_task: &Option<TaskFsm>,
    working_facts: &BTreeMap<String, String>,
    short_term_messages: &[Message],
) -> Vec<Message> {
    let mut context = Vec::with_capacity(short_term_messages.len() + 5);
    context.push(Message {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    });
    if let Some(block) = format_invariants_block(invariants) {
        context.push(Message {
            role: "system".to_string(),
            content: block,
        });
    }
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
// Explicit, narrowly-scoped extraction prompts. There is deliberately no
// extraction prompt for the profile or for task-stage transitions: neither
// is ever written by the LLM, only by an explicit API call - the FSM's
// state changes are 100% deterministic and caller-driven, never inferred.
// ---------------------------------------------------------------------

/// Governs the working-memory extractor: runs only while a task is active
/// and unpaused, and is deliberately blind to anything that isn't scoped to
/// *this* task.
const WORKING_MEMORY_PROMPT: &str = "You maintain the working memory for ONE specific task an assistant is currently helping with. You will be given the TASK, the CURRENT WORKING FACTS as a JSON object, and the LATEST TURN (the user's message and the assistant's reply). Extract only facts needed to complete THIS task: parameters, constraints, choices, and progress specific to it. Do NOT include anything about the user as a person that would still matter after this task is finished (identity, standing preferences, recurring habits) - that belongs to a different memory system and must be left out here. Return the complete, updated facts as a single JSON object of string keys to string values, and nothing else - no prose, no markdown code fences, no commentary. Keep keys short and snake_case, and reuse an existing key when a new message updates the same fact. If nothing new or changed, return the CURRENT WORKING FACTS unchanged.";

/// Governs the long-term extractor: runs on every turn regardless of
/// whether a task is active, and is deliberately blind to task-scoped
/// detail and to style/format/tone preferences, which have a dedicated
/// home in the profile.
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
// loaded/saved independently, so a corrupt file for one layer can never
// take another down with it.
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

/// What a chat turn or a memory inspection reports about the memory
/// system: how many tokens each layer contributed to the last request
/// built (or would contribute right now), plus how big each layer
/// currently is, and the active task's full FSM state (if any).
#[derive(Serialize)]
struct MemoryReport {
    system_prompt_tokens: u32,
    invariants_tokens: u32,
    profile_tokens: u32,
    long_term_tokens: u32,
    working_memory_tokens: u32,
    short_term_tokens: u32,
    sent_context_tokens: u32,
    actual: Option<Usage>,
    profile_is_set: bool,
    invariants_count: usize,
    long_term_facts_count: usize,
    working_facts_count: usize,
    short_term_message_count: usize,
    task: Option<TaskFsm>,
    invariants: Vec<Invariant>,
    context_limit: u32,
    percent_of_limit: f32,
    estimated_cost_usd: Option<f64>,
    /// Set only on a turn the deterministic pre-check blocked before it
    /// ever reached the model - the id of the invariant that blocked it.
    blocked_by_invariant: Option<String>,
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
    invariants_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
}

/// The agent: owns the personalization profile, the short-term/long-term
/// memory layers, and working memory (which now includes the task FSM),
/// each behind its own mutex and persisted to its own file. No method ever
/// locks more than one layer's mutex across an `.await` point, and every
/// mutation is followed by a persist call scoped to exactly the layer that
/// changed.
struct Agent {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    profile_path: String,
    short_term_path: String,
    working_memory_path: String,
    long_term_path: String,
    invariants_path: String,
    context_limit: u32,
    price_per_1m_input: Option<f64>,
    price_per_1m_output: Option<f64>,
    profile: Mutex<UserProfile>,
    short_term: Mutex<ShortTermMemory>,
    working: Mutex<WorkingMemory>,
    long_term: Mutex<LongTermMemory>,
    invariants: Mutex<InvariantSet>,
}

impl Agent {
    fn new(client: reqwest::Client, config: AgentConfig) -> Self {
        let profile = load_or_default::<UserProfile>(&config.profile_path);
        let short_term = load_or_default::<ShortTermMemory>(&config.short_term_path);
        let working = load_or_default::<WorkingMemory>(&config.working_memory_path);
        let long_term = load_or_default::<LongTermMemory>(&config.long_term_path);
        // Seed with this repo's own real invariants the first time there's
        // no invariants.json on disk yet, and persist immediately so the
        // seed is what a restart reloads, not something re-derived from
        // code every boot.
        let mut invariants = load_or_default::<InvariantSet>(&config.invariants_path);
        if invariants.invariants.is_empty() {
            invariants.invariants = seed_invariants();
            save_json(&config.invariants_path, &invariants);
        }
        Self {
            client,
            endpoint: config.endpoint,
            api_key: config.api_key,
            model: config.model,
            profile_path: config.profile_path,
            short_term_path: config.short_term_path,
            working_memory_path: config.working_memory_path,
            long_term_path: config.long_term_path,
            invariants_path: config.invariants_path,
            context_limit: config.context_limit,
            price_per_1m_input: config.price_per_1m_input,
            price_per_1m_output: config.price_per_1m_output,
            profile: Mutex::new(profile),
            short_term: Mutex::new(short_term),
            working: Mutex::new(working),
            long_term: Mutex::new(long_term),
            invariants: Mutex::new(invariants),
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

    fn persist_invariants(&self) {
        save_json(&self.invariants_path, &*self.invariants.lock().unwrap());
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

    fn working_task(&self) -> Option<TaskFsm> {
        self.working.lock().unwrap().task.clone()
    }

    fn working_facts(&self) -> BTreeMap<String, String> {
        self.working.lock().unwrap().facts.clone()
    }

    fn long_term_facts(&self) -> BTreeMap<String, String> {
        self.long_term.lock().unwrap().facts.clone()
    }

    fn invariants(&self) -> Vec<Invariant> {
        self.invariants.lock().unwrap().invariants.clone()
    }

    /// Adds one invariant. Rejects a duplicate id rather than silently
    /// overwriting it - a governance list like this should never let a
    /// same-named invariant quietly change meaning under a caller who
    /// didn't intend to redefine it; remove it explicitly first.
    fn add_invariant(&self, invariant: Invariant) -> Result<(), String> {
        let mut guard = self.invariants.lock().unwrap();
        if guard.invariants.iter().any(|i| i.id == invariant.id) {
            return Err(format!(
                "An invariant with id \"{}\" already exists; remove it first if you mean to replace it.",
                invariant.id
            ));
        }
        guard.invariants.push(invariant);
        drop(guard);
        self.persist_invariants();
        Ok(())
    }

    /// Removes exactly one invariant by id. Rejects an unknown id rather
    /// than silently no-op'ing, so a typo in the id surfaces immediately
    /// instead of leaving a stale invariant in force.
    fn remove_invariant(&self, id: &str) -> Result<(), String> {
        let mut guard = self.invariants.lock().unwrap();
        let before = guard.invariants.len();
        guard.invariants.retain(|i| i.id != id);
        if guard.invariants.len() == before {
            return Err(format!("No invariant with id \"{id}\"."));
        }
        drop(guard);
        self.persist_invariants();
        Ok(())
    }

    /// The hard, deterministic half of enforcement: scans the user's raw
    /// message (case-insensitively) for any invariant's `forbidden_terms`
    /// and returns the first match, *before* anything is sent to the
    /// model. This is the one part of invariant enforcement that's 100%
    /// testable without a live model - the rest depends on the model
    /// actually reading and honoring `format_invariants_block`.
    fn find_violated_invariant(&self, user_message: &str) -> Option<Invariant> {
        let lower = user_message.to_lowercase();
        self.invariants
            .lock()
            .unwrap()
            .invariants
            .iter()
            .find(|inv| inv.forbidden_terms.iter().any(|term| lower.contains(&term.to_lowercase())))
            .cloned()
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

    /// Builds a `MemoryReport` from whatever the profile and memory layers
    /// hold right now, given the token count of whatever request this
    /// report is describing (a real one just sent, or a hypothetical one
    /// for inspection).
    fn compose_report(&self, actual: Option<Usage>, sent_context_tokens: u32) -> MemoryReport {
        self.compose_report_with(actual, sent_context_tokens, None)
    }

    fn compose_report_with(
        &self,
        actual: Option<Usage>,
        sent_context_tokens: u32,
        blocked_by_invariant: Option<String>,
    ) -> MemoryReport {
        let invariants = self.invariants();
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
        let invariants_tokens = format_invariants_block(&invariants)
            .map(|b| estimate_tokens(&b) + MESSAGE_OVERHEAD_TOKENS)
            .unwrap_or(0);
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
            invariants_tokens,
            profile_tokens,
            long_term_tokens,
            working_memory_tokens,
            short_term_tokens,
            sent_context_tokens,
            actual,
            profile_is_set: !profile.is_empty(),
            invariants_count: invariants.len(),
            long_term_facts_count: long_term.len(),
            working_facts_count: working_facts.len(),
            short_term_message_count: short_term_len,
            task: working_task,
            invariants,
            context_limit: self.context_limit,
            percent_of_limit,
            estimated_cost_usd,
            blocked_by_invariant,
        }
    }

    /// A descriptive snapshot for endpoints that aren't mid-turn (memory
    /// inspection, reset, task lifecycle calls, profile changes): what a
    /// request would look like *right now*, with no real call made.
    fn snapshot(&self) -> MemoryReport {
        let invariants = self.invariants();
        let profile = self.profile();
        let long_term = self.long_term_facts();
        let working_task = self.working_task();
        let working_facts = self.working_facts();
        let short_term = self.short_term_messages();
        let sent_messages = build_context(
            SYSTEM_PROMPT,
            &invariants,
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

        let invariants = self.invariants();
        let profile = self.profile();
        let long_term = self.long_term_facts();
        let working_task = self.working_task();
        let working_facts = self.working_facts();
        let short_term = self.short_term_messages();

        let sent_messages = build_context(
            SYSTEM_PROMPT,
            &invariants,
            &profile,
            &long_term,
            &working_task,
            &working_facts,
            &short_term,
        );
        let sent_context_tokens = history_tokens(&sent_messages);

        // The hard invariant check runs BEFORE the model is ever called.
        // If the raw user message trips a forbidden term, this is a
        // conflict no amount of clever prompting could talk the model
        // out of - refuse deterministically, in plain Rust, and never
        // spend a token asking the model to decide. Note this is
        // narrower than full enforcement: it only catches the violations
        // a keyword can name. The invariants block still rides along in
        // `sent_messages` for the model to reason about on requests that
        // conflict semantically without using any forbidden term - see
        // `format_invariants_block`.
        if let Some(violated) = self.find_violated_invariant(user_message) {
            let reply = format_invariant_refusal(&violated);
            {
                let mut st = self.short_term.lock().unwrap();
                st.messages.push(Message {
                    role: "assistant".to_string(),
                    content: reply.clone(),
                });
            }
            self.persist_short_term();
            let memory = self.compose_report_with(None, sent_context_tokens, Some(violated.id.clone()));
            return RespondResult { reply, memory };
        }

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
                // active and not paused - pausing freezes this layer just
                // like it freezes explicit stage transitions, so a paused
                // task's facts can never drift out from under the user.
                if let Some(task) = working_task.as_ref() {
                    if !task.paused {
                        self.update_working_memory(&task.name, user_message, &reply).await;
                    }
                }
                // Long-term extraction runs on every turn, task or no
                // task, paused or not - it's about the user, not the task.
                // The profile is never touched here - it only ever changes
                // via an explicit `/api/profile` call.
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

    /// Strategy for working memory: runs only while a task is active and
    /// unpaused (callers gate on that) and only ever touches
    /// `self.working.facts`. Best-effort: any failure - network, a
    /// non-200, unparseable JSON - just leaves the prior facts in place and
    /// gets retried on the next turn; it never blocks or breaks the chat
    /// reply it rode in on.
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
    /// away. A no-op when working memory is already empty.
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

    /// Starts a new task in the `planning` stage and clears any facts left
    /// over from before (there shouldn't be any, since `finish_task` always
    /// clears them, but this makes the guarantee explicit rather than
    /// assumed). Refuses to start a second task on top of an active one -
    /// the lifecycle boundary has to be crossed explicitly via
    /// `finish_task` first, not implicitly overwritten.
    fn start_task(&self, name: String, step: String, expected_action: String) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        if w.task.is_some() {
            return Err("A task is already active; finish it before starting a new one.".to_string());
        }
        w.task = Some(TaskFsm::new(name, step, expected_action));
        w.facts = BTreeMap::new();
        drop(w);
        self.persist_working();
        Ok(())
    }

    /// Moves the active task to `target`, validated against
    /// `TaskStage::can_advance_to` so a stray request can't drive the FSM
    /// into a state it isn't allowed to reach directly (e.g. `planning`
    /// straight to `done`). Refuses while paused - resume first, so a
    /// stage change is never silently applied to a task the user thinks is
    /// frozen.
    fn advance_task(&self, target: TaskStage, step: String, expected_action: String) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        let task = w.task.as_mut().ok_or_else(|| "No task is currently active.".to_string())?;
        if task.paused {
            return Err("Task is paused; resume it before advancing its stage.".to_string());
        }
        if !task.stage.can_advance_to(target) {
            return Err(format!(
                "Cannot move from \"{}\" to \"{}\".",
                task.stage.label(),
                target.label()
            ));
        }
        task.stage = target;
        task.step = step;
        task.expected_action = expected_action;
        task.record(TaskEventKind::StageAdvanced);
        drop(w);
        self.persist_working();
        Ok(())
    }

    /// Updates step/expected-action detail without changing the stage -
    /// for progress within a stage (e.g. several execution steps in a row)
    /// that doesn't warrant a formal stage transition of its own. Also
    /// refuses while paused, for the same reason `advance_task` does.
    fn update_task_step(&self, step: String, expected_action: String) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        let task = w.task.as_mut().ok_or_else(|| "No task is currently active.".to_string())?;
        if task.paused {
            return Err("Task is paused; resume it before updating its step.".to_string());
        }
        task.step = step;
        task.expected_action = expected_action;
        task.record(TaskEventKind::StepUpdated);
        drop(w);
        self.persist_working();
        Ok(())
    }

    /// Pauses the task in place, whatever stage it's currently at -
    /// pausing is orthogonal to `stage`, not a stage of its own, so it
    /// works identically during planning, execution, or validation.
    fn pause_task(&self) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        let task = w.task.as_mut().ok_or_else(|| "No task is currently active.".to_string())?;
        if task.paused {
            return Err("Task is already paused.".to_string());
        }
        task.paused = true;
        task.record(TaskEventKind::Paused);
        drop(w);
        self.persist_working();
        Ok(())
    }

    /// Resumes a paused task. Stage, step, and expected action are exactly
    /// what they were before the pause - resuming never resets or re-asks
    /// for anything, it just lets FSM-mutating actions run again.
    fn resume_task(&self) -> Result<(), String> {
        let mut w = self.working.lock().unwrap();
        let task = w.task.as_mut().ok_or_else(|| "No task is currently active.".to_string())?;
        if !task.paused {
            return Err("Task is not paused.".to_string());
        }
        task.paused = false;
        task.record(TaskEventKind::Resumed);
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

    /// Ends the active task. Requires the FSM to have already reached
    /// `done` via `advance_task` and to not be paused - finishing is
    /// deliberately not a shortcut around the state machine, so "the task
    /// is finished" always means it was actually walked through
    /// validation, not just that someone called this endpoint. With
    /// `promote: true` (the default from the API), runs the promotion step
    /// first so durable facts survive into long-term memory; with
    /// `promote: false`, working memory is simply discarded.
    async fn finish_task(&self, promote: bool) -> Result<(), String> {
        let task = {
            let w = self.working.lock().unwrap();
            match &w.task {
                None => return Err("No task is currently active.".to_string()),
                Some(t) if t.paused => return Err("Task is paused; resume it before finishing.".to_string()),
                Some(t) if t.stage != TaskStage::Done => {
                    return Err(format!(
                        "Task must reach the \"done\" stage before it can be finished (currently \"{}\").",
                        t.stage.label()
                    ));
                }
                Some(t) => t.clone(),
            }
        };
        if promote {
            self.run_promotion(&task.name).await;
        }
        self.clear_working();
        Ok(())
    }

    /// Explicit, human-directed writes that bypass the LLM extractors
    /// entirely. Working-memory writes require an active, unpaused task -
    /// same gate `update_working_memory` uses, just enforced for the
    /// explicit path too.
    fn remember(&self, layer: MemoryLayer, key: String, value: String) -> Result<(), String> {
        match layer {
            MemoryLayer::ShortTerm => Err(
                "Short-term memory is the raw dialogue, not a key-value store; there's nothing to \"remember\" into it directly.".to_string(),
            ),
            MemoryLayer::Working => {
                let mut w = self.working.lock().unwrap();
                match &w.task {
                    None => return Err("No task is active; start one before writing to working memory.".to_string()),
                    Some(t) if t.paused => {
                        return Err("Task is paused; resume it before writing to working memory.".to_string());
                    }
                    Some(_) => {}
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
    /// also clears the active task (and its whole history) - this is an
    /// unconditional escape hatch, so it works even on a paused task.
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
    /// (the active task and its stage), and long-term memory are untouched
    /// - resetting a conversation does not make the agent forget who it's
    /// talking to, how it was told to talk, or where a task is in its
    /// lifecycle.
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
    invariants: Vec<Invariant>,
    profile: UserProfile,
    short_term: Vec<Message>,
    working_task: Option<TaskFsm>,
    working_facts: BTreeMap<String, String>,
    long_term_facts: BTreeMap<String, String>,
    report: MemoryReport,
}

#[derive(Deserialize)]
struct StartTaskRequest {
    name: String,
    #[serde(default)]
    step: String,
    #[serde(default)]
    expected_action: String,
}

#[derive(Deserialize)]
struct AdvanceTaskRequest {
    stage: TaskStage,
    #[serde(default)]
    step: String,
    #[serde(default)]
    expected_action: String,
}

#[derive(Deserialize)]
struct UpdateTaskStepRequest {
    #[serde(default)]
    step: String,
    #[serde(default)]
    expected_action: String,
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

#[derive(Deserialize)]
struct AddInvariantRequest {
    id: String,
    category: InvariantCategory,
    statement: String,
    #[serde(default)]
    rationale: String,
    #[serde(default)]
    forbidden_terms: Vec<String>,
}

#[derive(Deserialize)]
struct RemoveInvariantRequest {
    id: String,
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
        invariants: agent.invariants(),
        profile: agent.profile(),
        short_term: agent.short_term_messages(),
        working_task: agent.working_task(),
        working_facts: agent.working_facts(),
        long_term_facts: agent.long_term_facts(),
        report: agent.snapshot(),
    })
}

async fn get_invariants(State(agent): State<Arc<Agent>>) -> Json<Vec<Invariant>> {
    Json(agent.invariants())
}

async fn add_invariant(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<AddInvariantRequest>,
) -> Result<Json<Vec<Invariant>>, (StatusCode, String)> {
    let invariant = Invariant {
        id: req.id,
        category: req.category,
        statement: req.statement,
        rationale: req.rationale,
        forbidden_terms: req.forbidden_terms,
    };
    agent.add_invariant(invariant).map_err(|e| (StatusCode::CONFLICT, e))?;
    Ok(Json(agent.invariants()))
}

async fn remove_invariant(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<RemoveInvariantRequest>,
) -> Result<Json<Vec<Invariant>>, (StatusCode, String)> {
    agent.remove_invariant(&req.id).map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(Json(agent.invariants()))
}

async fn reset(State(agent): State<Arc<Agent>>) -> Json<MemoryReport> {
    agent.reset_conversation();
    Json(agent.snapshot())
}

async fn start_task(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<StartTaskRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent
        .start_task(req.name, req.step, req.expected_action)
        .map_err(|e| (StatusCode::CONFLICT, e))?;
    Ok(Json(agent.snapshot()))
}

async fn advance_task(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<AdvanceTaskRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent
        .advance_task(req.stage, req.step, req.expected_action)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.snapshot()))
}

async fn update_task_step(
    State(agent): State<Arc<Agent>>,
    Json(req): Json<UpdateTaskStepRequest>,
) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent
        .update_task_step(req.step, req.expected_action)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.snapshot()))
}

async fn pause_task(State(agent): State<Arc<Agent>>) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent.pause_task().map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(agent.snapshot()))
}

async fn resume_task(State(agent): State<Arc<Agent>>) -> Result<Json<MemoryReport>, (StatusCode, String)> {
    agent.resume_task().map_err(|e| (StatusCode::BAD_REQUEST, e))?;
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
    let invariants_path = std::env::var("INVARIANTS_FILE").unwrap_or_else(|_| "invariants.json".to_string());
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
            invariants_path: invariants_path.clone(),
            context_limit,
            price_per_1m_input,
            price_per_1m_output,
        },
    );

    let task_note = agent
        .working_task()
        .map(|t| {
            format!(
                " for task \"{}\" (stage: {}{})",
                t.name,
                t.stage.label(),
                if t.paused { ", paused" } else { "" }
            )
        })
        .unwrap_or_default();
    let profile_note = if agent.profile().is_empty() {
        "no personalization profile set".to_string()
    } else {
        "personalization profile restored".to_string()
    };
    println!(
        "Restored {} short-term message(s), {} invariant(s), {} working fact(s){task_note}, {} long-term fact(s), {profile_note}.",
        agent.short_term_messages().len(),
        agent.invariants().len(),
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
        .route("/api/task/advance", post(advance_task))
        .route("/api/task/step", post(update_task_step))
        .route("/api/task/pause", post(pause_task))
        .route("/api/task/resume", post(resume_task))
        .route("/api/task/finish", post(finish_task))
        .route("/api/memory/remember", post(remember))
        .route("/api/memory/forget", post(forget))
        .route("/api/profile", get(get_profile).put(set_profile).delete(reset_profile))
        .route("/api/invariants", get(get_invariants).post(add_invariant))
        .route("/api/invariants/remove", post(remove_invariant))
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
            .join(format!("ai-advent-lesson-14-{label}-{}-{}.json", std::process::id(), n))
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
                invariants_path: temp_path("invariants"),
                context_limit: 1000,
                price_per_1m_input: Some(2.0),
                price_per_1m_output: Some(4.0),
            },
        )
    }

    /// Same as `test_agent`, but with an empty invariant set (no seeding)
    /// - for tests about the "no invariants at all" baseline, where the
    /// six seeded, repo-specific invariants would only be noise.
    fn test_agent_no_invariants() -> Agent {
        let agent = test_agent();
        agent.invariants.lock().unwrap().invariants.clear();
        agent
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
    fn format_profile_block_is_none_when_empty() {
        assert_eq!(format_profile_block(&UserProfile::default()), None);
    }

    #[test]
    fn format_profile_block_includes_every_set_field() {
        let block = format_profile_block(&sample_profile()).unwrap();
        assert!(block.contains("concise, no fluff"));
        assert!(block.contains("Never use emojis"));
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

    // -------------------------------------------------------------
    // Task FSM: transition table
    // -------------------------------------------------------------

    #[test]
    fn planning_can_only_advance_to_execution() {
        assert!(TaskStage::Planning.can_advance_to(TaskStage::Execution));
        assert!(!TaskStage::Planning.can_advance_to(TaskStage::Validation));
        assert!(!TaskStage::Planning.can_advance_to(TaskStage::Done));
    }

    #[test]
    fn execution_can_only_advance_to_validation() {
        assert!(TaskStage::Execution.can_advance_to(TaskStage::Validation));
        assert!(!TaskStage::Execution.can_advance_to(TaskStage::Planning));
        assert!(!TaskStage::Execution.can_advance_to(TaskStage::Done));
    }

    #[test]
    fn validation_can_advance_to_execution_for_rework_or_done() {
        assert!(TaskStage::Validation.can_advance_to(TaskStage::Execution));
        assert!(TaskStage::Validation.can_advance_to(TaskStage::Done));
        assert!(!TaskStage::Validation.can_advance_to(TaskStage::Planning));
    }

    #[test]
    fn done_is_terminal() {
        assert!(TaskStage::Done.allowed_next().is_empty());
    }

    #[test]
    fn task_fsm_new_starts_in_planning_with_one_history_entry() {
        let task = TaskFsm::new("Ship the feature".to_string(), "gather requirements".to_string(), "".to_string());
        assert_eq!(task.stage, TaskStage::Planning);
        assert!(!task.paused);
        assert_eq!(task.history.len(), 1);
        assert_eq!(task.history[0].kind, TaskEventKind::Started);
    }

    // -------------------------------------------------------------
    // format_working_block / build_context
    // -------------------------------------------------------------

    #[test]
    fn format_working_block_is_none_without_an_active_task() {
        assert_eq!(format_working_block(&None, &BTreeMap::new()), None);
    }

    #[test]
    fn format_working_block_includes_stage_step_and_expected_action() {
        let task = TaskFsm::new(
            "Plan a party".to_string(),
            "pick a venue".to_string(),
            "waiting for the user to confirm a date".to_string(),
        );
        let block = format_working_block(&Some(task), &BTreeMap::new()).unwrap();
        assert!(block.contains("Plan a party"));
        assert!(block.contains("planning"));
        assert!(block.contains("pick a venue"));
        assert!(block.contains("waiting for the user to confirm a date"));
    }

    #[test]
    fn format_working_block_notes_when_paused() {
        let mut task = TaskFsm::new("Plan a party".to_string(), "pick a venue".to_string(), "".to_string());
        task.paused = true;
        let block = format_working_block(&Some(task), &BTreeMap::new()).unwrap();
        assert!(block.contains("PAUSED"));
    }

    #[test]
    fn format_working_block_includes_facts() {
        let task = TaskFsm::new("Plan a party".to_string(), "".to_string(), "".to_string());
        let mut facts = BTreeMap::new();
        facts.insert("budget".to_string(), "$200".to_string());
        let block = format_working_block(&Some(task), &facts).unwrap();
        assert!(block.contains("budget: $200"));
    }

    fn sample_invariant() -> Invariant {
        Invariant {
            id: "no-python".to_string(),
            category: InvariantCategory::StackConstraint,
            statement: "Lessons are written in Rust, not Python.".to_string(),
            rationale: "Consistency with every other lesson in the repo.".to_string(),
            forbidden_terms: vec!["rewrite it in python".to_string(), "switch to python".to_string()],
        }
    }

    #[test]
    fn build_context_with_no_memory_or_profile_is_just_system_and_dialogue() {
        let dialogue = vec![Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let ctx = build_context(
            SYSTEM_PROMPT,
            &[],
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
    fn build_context_includes_invariants_profile_long_term_and_task_blocks_in_order() {
        let mut long_term = BTreeMap::new();
        long_term.insert("diet".to_string(), "vegetarian".to_string());
        let task = TaskFsm::new("Plan a party".to_string(), "".to_string(), "".to_string());
        let dialogue = vec![Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let ctx = build_context(
            SYSTEM_PROMPT,
            &[sample_invariant()],
            &sample_profile(),
            &long_term,
            &Some(task),
            &BTreeMap::new(),
            &dialogue,
        );
        assert_eq!(ctx.len(), 6);
        assert_eq!(ctx[0].role, "system");
        assert!(ctx[1].content.contains("no-python"));
        assert!(ctx[2].content.contains("concise, no fluff"));
        assert!(ctx[3].content.contains("vegetarian"));
        assert!(ctx[4].content.contains("Plan a party"));
        assert_eq!(ctx[5].content, "hi");
    }

    #[test]
    fn format_invariants_block_is_none_when_empty() {
        assert!(format_invariants_block(&[]).is_none());
    }

    #[test]
    fn format_invariants_block_includes_id_category_statement_and_rationale() {
        let block = format_invariants_block(&[sample_invariant()]).unwrap();
        assert!(block.contains("no-python"));
        assert!(block.contains("stack constraint"));
        assert!(block.contains("Lessons are written in Rust"));
        assert!(block.contains("Consistency with every other lesson"));
    }

    #[test]
    fn seed_invariants_have_unique_non_empty_ids() {
        let seeds = seed_invariants();
        assert!(!seeds.is_empty());
        let mut ids: Vec<&str> = seeds.iter().map(|i| i.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), seeds.len());
        assert!(seeds.iter().all(|i| !i.id.is_empty()));
    }

    #[test]
    fn fresh_agent_is_seeded_with_the_repos_own_invariants() {
        let agent = test_agent();
        let invariants = agent.invariants();
        assert!(invariants.iter().any(|i| i.id == "no-docker"));
        assert!(invariants.iter().any(|i| i.id == "user-level-systemd"));
    }

    #[test]
    fn add_invariant_rejects_a_duplicate_id() {
        let agent = test_agent_no_invariants();
        agent.add_invariant(sample_invariant()).unwrap();
        assert!(agent.add_invariant(sample_invariant()).is_err());
        assert_eq!(agent.invariants().len(), 1);
    }

    #[test]
    fn remove_invariant_rejects_an_unknown_id() {
        let agent = test_agent_no_invariants();
        assert!(agent.remove_invariant("does-not-exist").is_err());
    }

    #[test]
    fn remove_invariant_takes_out_exactly_one() {
        let agent = test_agent_no_invariants();
        agent.add_invariant(sample_invariant()).unwrap();
        agent
            .add_invariant(Invariant {
                id: "other".to_string(),
                category: InvariantCategory::BusinessRule,
                statement: "Something else entirely.".to_string(),
                rationale: String::new(),
                forbidden_terms: Vec::new(),
            })
            .unwrap();
        agent.remove_invariant("no-python").unwrap();
        let remaining = agent.invariants();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "other");
    }

    #[test]
    fn find_violated_invariant_matches_case_insensitively_and_returns_none_when_clean() {
        let agent = test_agent_no_invariants();
        agent.add_invariant(sample_invariant()).unwrap();

        assert!(agent.find_violated_invariant("Can we just REWRITE IT IN PYTHON already?").is_some());
        assert!(agent.find_violated_invariant("Let's add a caching layer instead.").is_none());
    }

    #[tokio::test]
    async fn respond_refuses_deterministically_without_calling_the_model() {
        // test_agent() points at a bogus endpoint (localhost:0) - if this
        // reached call_chat, it would fail with a connection error, not
        // return a clean refusal. Getting a clean refusal back proves the
        // network was never touched.
        let agent = test_agent_no_invariants();
        agent.add_invariant(sample_invariant()).unwrap();

        let result = agent
            .respond("Please switch to Python for this one, it'll be faster.", ChatOptions { temperature: None, max_tokens: None })
            .await;

        assert!(result.reply.contains("no-python"));
        assert!(result.reply.contains("stack constraint"));
        assert_eq!(result.memory.blocked_by_invariant, Some("no-python".to_string()));
        assert!(result.memory.actual.is_none());

        // The blocked turn still shows up in the transcript...
        let messages = agent.short_term_messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].content.contains("no-python"));
    }

    #[tokio::test]
    async fn respond_does_not_block_a_request_that_matches_no_forbidden_term() {
        // A request that doesn't trip any forbidden term falls through to
        // call_chat, which fails fast against the bogus test endpoint -
        // an error reply, not a network hang, and definitely not a
        // refusal that names an invariant.
        let agent = test_agent_no_invariants();
        agent.add_invariant(sample_invariant()).unwrap();

        let result = agent
            .respond("What's a good caching strategy here?", ChatOptions { temperature: None, max_tokens: None })
            .await;

        assert!(!result.reply.contains("no-python"));
        assert_eq!(result.memory.blocked_by_invariant, None);
    }

    #[test]
    fn strip_json_fences_removes_a_json_code_fence() {
        let raw = "```json\n{\"a\": \"b\"}\n```";
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

    // -------------------------------------------------------------
    // Agent: task lifecycle
    // -------------------------------------------------------------

    #[test]
    fn new_starts_with_no_active_task() {
        let agent = test_agent();
        assert!(agent.working_task().is_none());
    }

    #[test]
    fn start_task_begins_in_planning_and_clears_stale_working_facts() {
        let agent = test_agent();
        agent.working.lock().unwrap().facts.insert("stale".to_string(), "value".to_string());
        agent
            .start_task("Plan a party".to_string(), "pick a venue".to_string(), "".to_string())
            .unwrap();
        let task = agent.working_task().unwrap();
        assert_eq!(task.name, "Plan a party");
        assert_eq!(task.stage, TaskStage::Planning);
        assert_eq!(task.step, "pick a venue");
        assert!(agent.working_facts().is_empty());
    }

    #[test]
    fn start_task_rejects_when_a_task_is_already_active() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string(), "".to_string(), "".to_string()).unwrap();
        assert!(agent.start_task("Plan a meeting".to_string(), "".to_string(), "".to_string()).is_err());
    }

    #[test]
    fn advance_task_walks_the_pipeline_in_order() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();

        agent
            .advance_task(TaskStage::Execution, "write the code".to_string(), "".to_string())
            .unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Execution);

        agent
            .advance_task(TaskStage::Validation, "run the tests".to_string(), "".to_string())
            .unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Validation);

        agent
            .advance_task(TaskStage::Done, "shipped".to_string(), "".to_string())
            .unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Done);
    }

    #[test]
    fn advance_task_rejects_skipping_a_stage() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        assert!(agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).is_err());
        assert!(agent.advance_task(TaskStage::Done, "".to_string(), "".to_string()).is_err());
    }

    #[test]
    fn advance_task_allows_validation_back_to_execution_for_rework() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).unwrap();

        agent
            .advance_task(TaskStage::Execution, "fix the failing test".to_string(), "".to_string())
            .unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Execution);
        assert_eq!(agent.working_task().unwrap().step, "fix the failing test");
    }

    #[test]
    fn advance_task_rejects_without_an_active_task() {
        let agent = test_agent();
        assert!(agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).is_err());
    }

    #[test]
    fn update_task_step_updates_detail_without_changing_stage() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "step 1".to_string(), "".to_string()).unwrap();
        agent
            .update_task_step("step 2".to_string(), "waiting on review".to_string())
            .unwrap();
        let task = agent.working_task().unwrap();
        assert_eq!(task.stage, TaskStage::Planning);
        assert_eq!(task.step, "step 2");
        assert_eq!(task.expected_action, "waiting on review");
    }

    // -------------------------------------------------------------
    // Agent: pause / resume - "test pause at any stage"
    // -------------------------------------------------------------

    #[test]
    fn pause_and_resume_round_trip_preserves_stage_step_and_expected_action() {
        let agent = test_agent();
        agent
            .start_task("Ship it".to_string(), "write the code".to_string(), "review needed".to_string())
            .unwrap();
        agent.advance_task(TaskStage::Execution, "write the code".to_string(), "review needed".to_string()).unwrap();

        agent.pause_task().unwrap();
        let paused = agent.working_task().unwrap();
        assert!(paused.paused);
        assert_eq!(paused.stage, TaskStage::Execution);

        agent.resume_task().unwrap();
        let resumed = agent.working_task().unwrap();
        assert!(!resumed.paused);
        assert_eq!(resumed.stage, TaskStage::Execution);
        assert_eq!(resumed.step, "write the code");
        assert_eq!(resumed.expected_action, "review needed");
    }

    #[test]
    fn pause_works_from_every_stage() {
        // planning
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Planning);
        assert!(agent.working_task().unwrap().paused);

        // execution
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Execution);
        assert!(agent.working_task().unwrap().paused);

        // validation
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Validation);
        assert!(agent.working_task().unwrap().paused);

        // done
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Done, "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert_eq!(agent.working_task().unwrap().stage, TaskStage::Done);
        assert!(agent.working_task().unwrap().paused);
    }

    #[test]
    fn pause_task_rejects_double_pause() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert!(agent.pause_task().is_err());
    }

    #[test]
    fn resume_task_rejects_when_not_paused() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        assert!(agent.resume_task().is_err());
    }

    #[test]
    fn advance_task_rejects_while_paused() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert!(agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).is_err());
    }

    #[test]
    fn update_task_step_rejects_while_paused() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert!(agent.update_task_step("anything".to_string(), "".to_string()).is_err());
    }

    #[test]
    fn remember_into_working_rejects_while_paused() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert!(agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .is_err());
    }

    #[test]
    fn task_history_records_start_advance_pause_and_resume_in_order() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        agent.resume_task().unwrap();

        let history = agent.working_task().unwrap().history;
        let kinds: Vec<TaskEventKind> = history.iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TaskEventKind::Started,
                TaskEventKind::StageAdvanced,
                TaskEventKind::Paused,
                TaskEventKind::Resumed,
            ]
        );
    }

    // -------------------------------------------------------------
    // Agent: finish requires reaching "done"
    // -------------------------------------------------------------

    #[tokio::test]
    async fn finish_task_rejects_before_reaching_done() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        assert!(agent.finish_task(false).await.is_err());

        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        assert!(agent.finish_task(false).await.is_err());
    }

    #[tokio::test]
    async fn finish_task_rejects_while_paused_even_at_done() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Done, "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        assert!(agent.finish_task(false).await.is_err());
    }

    #[tokio::test]
    async fn finish_task_without_promotion_clears_working_memory_once_done() {
        let agent = test_agent();
        agent.start_task("Ship it".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.working.lock().unwrap().facts.insert("budget".to_string(), "$200".to_string());
        agent.long_term.lock().unwrap().facts.insert("diet".to_string(), "vegetarian".to_string());

        agent.advance_task(TaskStage::Execution, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Validation, "".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Done, "".to_string(), "".to_string()).unwrap();

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
        agent.start_task("Plan a party".to_string(), "".to_string(), "".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        assert_eq!(agent.working_facts().get("budget"), Some(&"$200".to_string()));
    }

    #[test]
    fn forget_working_clears_facts_and_the_active_task() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string(), "".to_string(), "".to_string()).unwrap();
        agent
            .remember(MemoryLayer::Working, "budget".to_string(), "$200".to_string())
            .unwrap();
        agent.forget(MemoryLayer::Working);
        assert!(agent.working_task().is_none());
        assert!(agent.working_facts().is_empty());
    }

    #[test]
    fn forget_working_clears_even_a_paused_task() {
        let agent = test_agent();
        agent.start_task("Plan a party".to_string(), "".to_string(), "".to_string()).unwrap();
        agent.pause_task().unwrap();
        agent.forget(MemoryLayer::Working);
        assert!(agent.working_task().is_none());
    }

    #[test]
    fn reset_conversation_clears_short_term_but_preserves_profile_and_task_state() {
        let agent = test_agent();
        agent.set_profile(sample_profile());
        agent.short_term.lock().unwrap().messages.push(Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        agent.start_task("Plan a party".to_string(), "pick a venue".to_string(), "".to_string()).unwrap();
        agent.advance_task(TaskStage::Execution, "book the venue".to_string(), "".to_string()).unwrap();

        agent.reset_conversation();

        assert!(agent.short_term_messages().is_empty());
        assert_eq!(agent.profile(), sample_profile());
        let task = agent.working_task().unwrap();
        assert_eq!(task.name, "Plan a party");
        assert_eq!(task.stage, TaskStage::Execution);
        assert_eq!(task.step, "book the venue");
    }

    #[test]
    fn snapshot_reports_zero_working_tokens_when_no_task_is_active() {
        let agent = test_agent();
        let report = agent.snapshot();
        assert_eq!(report.working_memory_tokens, 0);
        assert!(report.task.is_none());
    }

    #[test]
    fn snapshot_includes_working_tokens_and_task_once_a_task_is_active() {
        let agent = test_agent();
        let before = agent.snapshot();
        agent.start_task("Plan a party".to_string(), "pick a venue".to_string(), "".to_string()).unwrap();
        let after = agent.snapshot();
        assert_eq!(before.working_memory_tokens, 0);
        assert!(after.working_memory_tokens > 0);
        assert_eq!(after.task.unwrap().name, "Plan a party");
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
                invariants_path: temp_path("invariants"),
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

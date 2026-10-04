mod index;
mod rerank;
mod scenarios;
mod task;

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{Path, State},
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use index::{ErrorResponse, Hit, Index, builtin_documents, embedder_from_env, env_non_empty};
use rerank::Heuristic;
use scenarios::{SCENARIOS, Scenario, Step};
use task::{Change, TaskState, json_object, parse_update};

/// Stage 1: how many chunks the vector search hands to the reranker.
const DEFAULT_K_BEFORE: usize = 20;
const MAX_K_BEFORE: usize = 40;
/// Stage 2: at most this many chunks go into the prompt.
const DEFAULT_K_AFTER: usize = 5;
const MAX_K_AFTER: usize = 10;
/// The relevance gate from lessons 23 and 24: below it the assistant says "I
/// don't know" and asks for a clarification, without calling the model for
/// an answer.
const DEFAULT_THRESHOLD: f32 = 0.30;
const MAX_MESSAGE_CHARS: usize = 2000;
/// A shorter quote proves nothing ("the port").
const MIN_QUOTE_CHARS: usize = 12;
/// The first reply, and one retry that is told what was wrong with it.
const MAX_ATTEMPTS: usize = 2;
/// How many of the nearest sections an "I don't know" shows as its sources.
const HINTS: usize = 3;
/// How many earlier turns go with every answer request, as real messages.
/// Anything older reaches the model only through the task state.
const HISTORY_TURNS: usize = 6;
/// How many earlier turns the state update sees, to resolve "it" and "that".
const STATE_HISTORY_TURNS: usize = 3;
/// A session stops taking messages after this many turns.
const MAX_TURNS: usize = 200;

const BASE_PROMPT: &str = "You are a chat assistant for the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. You keep a \
conversation with the user over many messages. Answer concisely, in the language of \
the user's latest message. If you don't know something, say so instead of guessing.";

/// How the task state is to be used.
const MEMORY_RULES: &str = "Below is the task state of this conversation. The system \
keeps it for the whole dialogue, so it is current even when the messages that set it \
are no longer shown.
- Answer the latest message in the light of the goal. Read references like \"it\", \
\"my lesson\" or \"that option\" through what the user has clarified.
- Never suggest anything that breaks a constraint. If the documents describe something \
that conflicts with a constraint, say so plainly.
- Use every term in the meaning the task state gives it.
- If the user asks for a recap, restate the goal, what was clarified, the constraints \
and the terms from the task state, then add what the fragments say. Facts from the \
task state need no quote; facts from the documents do.";

/// The answer is a JSON object with its sources and quotes, as in lesson 24.
const ANSWER_RULES: &str = "The latest user message contains numbered fragments of \
the repository's documents, found for this message, then the message itself. Answer \
only from these fragments, the task state and the conversation.

Reply with one JSON object and nothing else:
{\"status\": \"answered\", \"answer\": \"...\", \"sources\": [1, 3], \"quotes\": [{\"fragment\": 1, \"text\": \"...\"}, {\"fragment\": 3, \"text\": \"...\"}]}

- answer: concise, in the language of the latest message. After each fact from a fragment, cite it as [1], [2] and so on.
- sources: the number of every fragment the answer uses. At least one.
- quotes: at least one quote from every source, the words that support the answer. Copy each quote character for character from its fragment, in the fragment's language: one sentence, or one line of a list, table or code, up to 300 characters. Don't translate, paraphrase or join pieces of different sentences.
- Every fact from the documents must be backed by one of the quotes.

If the fragments don't contain the answer, don't guess. Reply instead:
{\"status\": \"unknown\", \"clarification\": \"...\"}
where clarification is a short question to the user, in the language of their message, that would help to find the answer.";

/// The task state update: one call per user message, before the search.
const STATE_PROMPT: &str = "You maintain the task state of a conversation between a \
user and an assistant that answers questions from the documents of the GitHub \
repository CaptainDmitro/AI-Advent-Challenge (a Rust monorepo of AI-course lessons \
with a GitHub Actions deploy pipeline). You get the current task state, the last turns \
of the conversation and the user's latest message. Reply with one JSON object only:
{\"goal\": \"...\", \"goal_changed\": false, \"clarified\": [], \"constraints\": [], \"terms\": [{\"term\": \"...\", \"meaning\": \"...\"}], \"remove\": [], \"search_query\": \"...\"}

- goal: what the user wants to achieve in this whole conversation, one sentence, in the user's language. If the state has no goal yet, derive it from the message. If it has one, repeat it and set goal_changed to false, unless the user explicitly sets a different overall goal: then write the new goal and set goal_changed to true. A single question, an aside or an off-topic message never changes the goal.
- clarified: only NEW details the user has just stated about their situation or task (which lesson, which stack, what they need). Not their questions, and nothing the assistant said.
- constraints: only NEW restrictions or requirements the user has just set (what must or must not be used, limits, budgets).
- terms: only NEW terms the user has just defined, or asked to use in a certain meaning.
- remove: the exact text of earlier items of the state that the user has just withdrawn or replaced.
- Don't repeat items that are already in the state. Keep each item short, in the user's language.
- search_query: the latest message rewritten as a self-contained search query in English, the language of the documents. Resolve references (\"it\", \"that lesson\", \"the port\") with the task state and the conversation, keep exact names, numbers and identifiers, and drop greetings and filler. If the user asks for a recap, search for the goal. If the message has nothing to do with the goal or the repository, just translate it, without adding words from the goal.";

// ---------------------------------------------------------------------------
// The LLM: any OpenAI-compatible /chat/completions.
// ---------------------------------------------------------------------------

struct Llm {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct ChatCompletion {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(default)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
    }
}

impl Llm {
    /// One completion at temperature 0, as a JSON object.
    async fn complete(&self, messages: &[Value]) -> Result<(String, Usage), String> {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": { "type": "json_object" },
        });
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("LLM request failed: {e}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<ErrorResponse>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(format!("LLM API error ({status}): {detail}"));
        }
        let parsed = serde_json::from_str::<ChatCompletion>(&text)
            .map_err(|e| format!("Failed to parse LLM response: {e}"))?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .ok_or("The model returned an empty answer.")?;
        Ok((content, parsed.usage))
    }
}

// ---------------------------------------------------------------------------
// Quotes: is this text really in that chunk? (Lesson 24.)
// ---------------------------------------------------------------------------

/// Text as it is compared: lower case, one space for any run of whitespace,
/// typographic quotes and dashes made plain, and Markdown's `` ` ``, `*` and
/// `|` dropped.
fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        let c = match c {
            '`' | '*' | '|' => continue,
            c if c.is_whitespace() => {
                space = true;
                continue;
            }
            '’' | '‘' => '\'',
            '“' | '”' | '«' | '»' => '"',
            '–' | '—' => '-',
            c => c,
        };
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.extend(c.to_lowercase());
    }
    out
}

/// The quote is in the text word for word, after normalizing both. An
/// ellipsis may skip text: the parts must then occur in order.
fn quote_in(quote: &str, text: &str) -> bool {
    let text = normalize_text(text);
    let quote = quote.replace('…', "...");
    let mut from = 0;
    let mut parts = 0;
    for part in quote.split("...") {
        let part = normalize_text(part);
        let part = part.trim_matches(|c: char| {
            c.is_whitespace() || matches!(c, '"' | '\'' | '.' | ',' | ';' | ':')
        });
        if part.is_empty() {
            continue;
        }
        match text[from..].find(part) {
            Some(i) => {
                from += i + part.len();
                parts += 1;
            }
            None => return false,
        }
    }
    parts > 0
}

/// Every `[n]` and `[n, m]` in the text, in order, without repeats.
fn citations(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for piece in text.split('[').skip(1) {
        let Some((inside, _)) = piece.split_once(']') else {
            continue;
        };
        let numbers: Option<Vec<usize>> = inside
            .split(',')
            .map(|n| n.trim().parse::<usize>().ok())
            .collect();
        for n in numbers.unwrap_or_default() {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// The text without its `[n]` markers: an earlier answer, as the model sees
/// it in the history, where its fragment numbers no longer mean anything.
fn strip_citations(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find('[') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.split_once(']') {
            Some((inside, tail))
                if !inside.trim().is_empty()
                    && inside.split(',').all(|n| n.trim().parse::<usize>().is_ok()) =>
            {
                rest = tail;
            }
            _ => {
                out.push('[');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// The model's reply, checked. (Lesson 24.)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
#[serde(default)]
struct Reply {
    status: String,
    answer: String,
    /// Numbers, but `"2"` and `"[2]"` are read too.
    sources: Vec<Value>,
    quotes: Vec<ReplyQuote>,
    clarification: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct ReplyQuote {
    fragment: Value,
    text: String,
}

/// A reply that passed every check. Fragment numbers are 1-based.
#[derive(Debug, PartialEq)]
struct Verified {
    answer: String,
    sources: Vec<usize>,
    quotes: Vec<(usize, String)>,
}

#[derive(Debug, PartialEq)]
enum Parsed {
    Answered(Verified),
    /// The model says the fragments don't answer the question.
    Unknown { clarification: String },
}

fn fragment_number(v: &Value) -> Option<usize> {
    match v {
        Value::Number(n) => n.as_u64().map(|n| n as usize),
        Value::String(s) => s.trim().trim_matches(|c: char| c == '[' || c == ']').trim().parse().ok(),
        _ => None,
    }
}

/// Checks a reply against the fragments of its prompt. Every problem is
/// listed, so a retry can fix them all at once.
fn check_reply(text: &str, context: &[Hit]) -> Result<Parsed, Vec<String>> {
    let reply: Reply = json_object(text)
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| vec!["the reply is not the JSON object the rules ask for".to_string()])?;
    let status = reply.status.trim().to_lowercase();
    if status == "unknown" {
        return Ok(Parsed::Unknown {
            clarification: reply.clarification.trim().to_string(),
        });
    }
    let n = context.len();
    let mut errors = Vec::new();
    if !status.is_empty() && status != "answered" {
        errors.push(format!("status must be \"answered\" or \"unknown\", not {status:?}"));
    }
    let answer = reply.answer.trim().to_string();
    if answer.is_empty() {
        errors.push("the answer is empty".to_string());
    }

    let mut sources: Vec<usize> = Vec::new();
    for v in &reply.sources {
        match fragment_number(v).filter(|f| (1..=n).contains(f)) {
            Some(f) if !sources.contains(&f) => sources.push(f),
            Some(_) => {}
            None => errors.push(format!("source {v} is not a fragment number from 1 to {n}")),
        }
    }
    if sources.is_empty() {
        errors.push("no sources: list the number of every fragment the answer uses".to_string());
    }

    let mut quotes: Vec<(usize, String)> = Vec::new();
    if reply.quotes.is_empty() {
        errors.push("no quotes: quote every source word for word".to_string());
    }
    for (i, q) in reply.quotes.iter().enumerate() {
        let k = i + 1;
        let quote = q.text.trim();
        let Some(f) = fragment_number(&q.fragment).filter(|f| (1..=n).contains(f)) else {
            errors.push(format!("quote {k} names fragment {}, not a number from 1 to {n}", q.fragment));
            continue;
        };
        if quote.chars().count() < MIN_QUOTE_CHARS {
            errors.push(format!(
                "quote {k} is too short: quote at least {MIN_QUOTE_CHARS} characters"
            ));
        } else if !quote_in(quote, &context[f - 1].chunk.text) {
            errors.push(format!(
                "quote {k} is not word for word in fragment [{f}]: \"{}\"",
                preview(quote, 80)
            ));
        }
        if !sources.contains(&f) {
            errors.push(format!("quote {k} is from fragment [{f}], which isn't in sources"));
        }
        quotes.push((f, quote.to_string()));
    }
    for s in &sources {
        if !reply.quotes.is_empty() && !quotes.iter().any(|(f, _)| f == s) {
            errors.push(format!("source [{s}] has no quote"));
        }
    }
    for c in citations(&answer) {
        if !sources.contains(&c) {
            errors.push(format!("the answer cites [{c}], which isn't in sources"));
        }
    }
    if errors.is_empty() {
        Ok(Parsed::Answered(Verified {
            answer,
            sources,
            quotes,
        }))
    } else {
        Err(errors)
    }
}

fn retry_message(errors: &[String]) -> String {
    format!(
        "Your reply was rejected:\n- {}\n\nReply again with the JSON object only. Copy every \
quote character for character from its fragment, list every fragment you use in sources, \
and if the fragments don't contain the answer, reply with status \"unknown\".",
        errors.join("\n- ")
    )
}

// ---------------------------------------------------------------------------
// "I don't know"
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
enum Status {
    /// An answer with its sources and verified quotes.
    Answered,
    /// "I don't know", with a request to clarify and the nearest sections.
    Unknown,
    /// The search or the model failed: see `error`.
    #[default]
    Error,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
enum UnknownReason {
    /// No chunk reached the relevance threshold, so the model wasn't asked.
    LowRelevance,
    /// The model said the fragments don't contain the answer.
    ModelUnsure,
    /// No reply of the model could be backed by verbatim quotes.
    Unverified,
}

fn is_russian(text: &str) -> bool {
    text.chars()
        .any(|c| matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё'))
}

fn dont_know(message: &str, reason: UnknownReason, best: Option<f32>, threshold: f32) -> String {
    let best = best.unwrap_or(0.0);
    match (is_russian(message), reason) {
        (false, UnknownReason::LowRelevance) => format!(
            "I don't know: nothing in the documents is relevant enough to answer this (best relevance {best:.2}, threshold {threshold:.2})."
        ),
        (false, UnknownReason::ModelUnsure) => {
            "I don't know: the fragments found don't contain the answer.".to_string()
        }
        (false, UnknownReason::Unverified) => "I don't know for sure: the answer couldn't be backed by word-for-word quotes from the documents, so it isn't shown.".to_string(),
        (true, UnknownReason::LowRelevance) => format!(
            "Не знаю: в документах нет достаточно релевантного фрагмента (лучшая релевантность {best:.2}, порог {threshold:.2})."
        ),
        (true, UnknownReason::ModelUnsure) => {
            "Не знаю: в найденных фрагментах нет ответа на этот вопрос.".to_string()
        }
        (true, UnknownReason::Unverified) => "Не знаю наверняка: ответ не удалось подтвердить дословными цитатами из документов, поэтому он не показан.".to_string(),
    }
}

/// The clarification request of an "I don't know": it brings the goal back,
/// so an aside doesn't derail the conversation.
fn default_clarification(message: &str, goal: &str) -> String {
    match (is_russian(message), goal.is_empty()) {
        (true, true) => "Уточните, пожалуйста, вопрос: о каком уроке, файле или части пайплайна деплоя идёт речь?".to_string(),
        (true, false) => format!("Как это связано с нашей целью — «{goal}»? Уточните, пожалуйста, или вернёмся к ней."),
        (false, true) => "Could you clarify the question: which lesson, file or part of the deploy pipeline is it about?".to_string(),
        (false, false) => format!("How does this relate to our goal — \"{goal}\"? Please clarify, or let's get back to it."),
    }
}

// ---------------------------------------------------------------------------
// One turn: what is stored, and what is only shown.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
struct Settings {
    k_before: usize,
    k_after: usize,
    threshold: f32,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            k_before: DEFAULT_K_BEFORE,
            k_after: DEFAULT_K_AFTER,
            threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl Settings {
    /// In range, and never fewer candidates than chunks to keep.
    fn clamped(self) -> Settings {
        let k_after = self.k_after.clamp(1, MAX_K_AFTER);
        Settings {
            k_after,
            k_before: self.k_before.clamp(1, MAX_K_BEFORE).max(k_after),
            threshold: if self.threshold.is_finite() {
                self.threshold.clamp(0.0, 1.0)
            } else {
                DEFAULT_THRESHOLD
            },
        }
    }
}

/// One chunk from the vector search, scored again by the reranker.
#[derive(Serialize, Clone, Debug)]
struct Candidate {
    chunk_id: String,
    source: String,
    section: String,
    tokens: usize,
    cosine_rank: usize,
    cosine: f32,
    coverage: f32,
    heading: f32,
    score: f32,
    /// Position after reranking.
    rank: usize,
    kept: bool,
    /// Why it isn't in the prompt: `below threshold` or `past top-K`.
    dropped: Option<&'static str>,
}

/// A fragment the answer uses.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct SourceRef {
    /// Its number in the prompt: `[n]` in the answer.
    n: usize,
    chunk_id: String,
    source: String,
    section: String,
    /// The reranker's score.
    score: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Quote {
    n: usize,
    chunk_id: String,
    source: String,
    section: String,
    text: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
#[serde(default)]
struct Timings {
    state_ms: f64,
    retrieve_ms: f64,
    rerank_ms: f64,
    llm_ms: f64,
}

/// One user message and what came of it. This is what a session stores.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Turn {
    n: usize,
    user: String,
    /// What was searched: the state update's self-contained rewrite of
    /// `user`, or `user` itself if the update failed.
    query: String,
    status: Status,
    unknown_reason: Option<UnknownReason>,
    /// The answer with its `[n]` markers, or the "I don't know" and the
    /// clarification request.
    text: String,
    clarification: Option<String>,
    sources: Vec<SourceRef>,
    quotes: Vec<Quote>,
    /// Every quote is word for word in the chunk it names.
    quotes_verified: bool,
    /// With "I don't know": the nearest sections, shown as its sources.
    hints: Vec<String>,
    /// The best reranked score: what the relevance gate compares.
    best_score: Option<f32>,
    /// What this message changed in the task state.
    state_change: Change,
    /// The state update failed; the state stayed as it was.
    state_error: Option<String>,
    /// Model calls for the answer: 0 when the gate stopped it.
    attempts: usize,
    usage: Usage,
    timings: Timings,
    error: Option<String>,
}

impl Turn {
    fn start(n: usize, user: &str) -> Turn {
        Turn {
            n,
            user: user.to_string(),
            query: user.to_string(),
            ..Turn::default()
        }
    }

    fn give_up(&mut self, reason: UnknownReason, clarification: Option<String>, goal: &str, threshold: f32) {
        let clarification = clarification
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| default_clarification(&self.user, goal));
        self.status = Status::Unknown;
        self.unknown_reason = Some(reason);
        self.text = format!(
            "{} {clarification}",
            dont_know(&self.user, reason, self.best_score, threshold)
        );
        self.clarification = Some(clarification);
    }
}

/// What a turn sent and got back: shown on the page, never stored.
#[derive(Serialize, Clone, Debug, Default)]
struct Trace {
    state_messages: Vec<Value>,
    state_reply: Option<String>,
    candidates: Vec<Candidate>,
    /// The chunks in the prompt, `[1]` first.
    context: Vec<Hit>,
    /// Everything sent to the model for the answer, retry included.
    messages: Vec<Value>,
    replies: Vec<String>,
    rejected: Vec<Vec<String>>,
}

/// A conversation: the task state and every turn.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Session {
    id: String,
    /// Unix seconds.
    created: u64,
    state: TaskState,
    turns: Vec<Turn>,
}

fn elapsed_ms(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 1000.0).round()
}

fn round3(x: f32) -> f32 {
    (x * 1000.0).round() / 1000.0
}

fn preview(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut short: String = text.chars().take(max).collect();
    short.push('…');
    short
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// Prompts.
// ---------------------------------------------------------------------------

/// The kept chunks as numbered fragments, each headed by where it's from.
fn context_block(sources: &[Hit]) -> String {
    sources
        .iter()
        .map(|h| {
            format!(
                "[{}] {} — {}\n{}",
                h.rank,
                h.chunk.source,
                h.chunk.section,
                h.chunk.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The turns that go back to the model: the last `max` that didn't fail.
fn recent(turns: &[Turn], max: usize) -> Vec<&Turn> {
    let ok: Vec<&Turn> = turns.iter().filter(|t| t.status != Status::Error).collect();
    ok[ok.len().saturating_sub(max)..].to_vec()
}

fn state_json(state: &TaskState) -> String {
    serde_json::to_string_pretty(&json!({
        "goal": state.goal,
        "clarified": state.clarified,
        "constraints": state.constraints,
        "terms": state.terms,
    }))
    .unwrap_or_default()
}

/// The state update request: the state, the last few turns, the message.
fn state_messages(state: &TaskState, turns: &[Turn], message: &str) -> Vec<Value> {
    let dialogue = recent(turns, STATE_HISTORY_TURNS)
        .iter()
        .map(|t| {
            format!(
                "User: {}\nAssistant: {}",
                preview(&t.user, 500),
                preview(&strip_citations(&t.text), 500)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let dialogue = if dialogue.is_empty() { "(none yet)".to_string() } else { dialogue };
    vec![
        json!({ "role": "system", "content": STATE_PROMPT }),
        json!({
            "role": "user",
            "content": format!(
                "Current task state (JSON):\n{}\n\nRecent dialogue:\n{dialogue}\n\nLatest user message:\n{message}",
                state_json(state)
            ),
        }),
    ]
}

/// The answer request: the rules with the task state, the last turns as real
/// messages, then the fragments and the message.
fn answer_messages(state: &TaskState, turns: &[Turn], message: &str, context: &[Hit]) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": format!("{BASE_PROMPT}\n\n{MEMORY_RULES}\n\n{}\n\n{ANSWER_RULES}", state.render()),
    })];
    for t in recent(turns, HISTORY_TURNS) {
        messages.push(json!({ "role": "user", "content": t.user }));
        messages.push(json!({ "role": "assistant", "content": strip_citations(&t.text) }));
    }
    messages.push(json!({
        "role": "user",
        "content": format!("Context:\n\n{}\n\nMessage: {message}", context_block(context)),
    }));
    messages
}

// ---------------------------------------------------------------------------
// Retrieval: search → rerank → threshold → top-K (lesson 23).
// ---------------------------------------------------------------------------

/// Orders the candidates by score (ties: the vector search's order), then
/// keeps those at or above the threshold, at most `k_after` of them.
fn select(candidates: &mut [Candidate], settings: Settings) {
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.cosine_rank.cmp(&b.cosine_rank))
    });
    let mut kept = 0;
    for (i, c) in candidates.iter_mut().enumerate() {
        c.rank = i + 1;
        if c.score < settings.threshold {
            c.dropped = Some("below threshold");
        } else if kept >= settings.k_after {
            c.dropped = Some("past top-K");
        } else {
            c.kept = true;
            kept += 1;
        }
    }
}

/// The kept candidates as numbered hits, in their new order, scored with the
/// reranker's score.
fn kept_hits(candidates: &[Candidate], hits: &[Hit]) -> Vec<Hit> {
    candidates
        .iter()
        .filter(|c| c.kept)
        .filter_map(|c| {
            hits.iter()
                .find(|h| h.chunk.chunk_id == c.chunk_id)
                .map(|h| (c, h))
        })
        .enumerate()
        .map(|(i, (c, h))| Hit {
            rank: i + 1,
            score: c.score,
            chunk: h.chunk.clone(),
        })
        .collect()
}

fn source_ref(context: &[Hit], n: usize) -> SourceRef {
    let h = &context[n - 1];
    SourceRef {
        n,
        chunk_id: h.chunk.chunk_id.clone(),
        source: h.chunk.source.clone(),
        section: h.chunk.section.clone(),
        score: h.score,
    }
}

// ---------------------------------------------------------------------------
// The agent.
// ---------------------------------------------------------------------------

struct Agent {
    llm: Llm,
    index: Index,
    heuristic: Heuristic,
}

/// How the answer loop ended.
enum Outcome {
    Verified(Verified),
    Unsure(String),
    Rejected,
    Failed(String),
}

impl Agent {
    fn new(llm: Llm, index: Index) -> Agent {
        let heuristic = Heuristic::new(&index.chunks);
        Agent {
            llm,
            index,
            heuristic,
        }
    }

    /// Every hit with the heuristic's signals; not yet ordered.
    fn rerank(&self, query: &str, hits: &[Hit]) -> Vec<Candidate> {
        let terms = self.heuristic.terms(query);
        hits.iter()
            .map(|h| {
                let s = self.heuristic.score(&terms, h);
                Candidate {
                    chunk_id: h.chunk.chunk_id.clone(),
                    source: h.chunk.source.clone(),
                    section: h.chunk.section.clone(),
                    tokens: h.chunk.tokens,
                    cosine_rank: h.rank,
                    cosine: h.score,
                    coverage: round3(s.coverage),
                    heading: round3(s.heading),
                    score: round3(s.score),
                    rank: 0,
                    kept: false,
                    dropped: None,
                }
            })
            .collect()
    }

    /// One user message: update the task state, search with the state's
    /// rewrite of the message, answer from what was found with the state and
    /// the recent dialogue, check the quotes. The turn is appended to the
    /// session.
    async fn turn(&self, session: &mut Session, message: &str, settings: Settings) -> (Turn, Trace) {
        let mut turn = Turn::start(session.turns.len() + 1, message);
        let mut trace = Trace::default();
        self.answer(session, &mut turn, &mut trace, settings).await;
        session.turns.push(turn.clone());
        (turn, trace)
    }

    async fn answer(&self, session: &mut Session, turn: &mut Turn, trace: &mut Trace, settings: Settings) {
        let message = turn.user.clone();

        // 1. The task state, and the message as a self-contained query. A
        // failed update leaves the state as it was and searches the message.
        let started = Instant::now();
        let messages = state_messages(&session.state, &session.turns, &message);
        let result = self.llm.complete(&messages).await;
        trace.state_messages = messages;
        match result {
            Ok((text, usage)) => {
                turn.usage += usage;
                match parse_update(&text) {
                    Ok(update) => {
                        turn.state_change = session.state.apply(&update, turn.n);
                        if !update.search_query.is_empty() {
                            turn.query = update.search_query;
                        }
                    }
                    Err(e) => turn.state_error = Some(e),
                }
                trace.state_reply = Some(text);
            }
            Err(e) => turn.state_error = Some(e),
        }
        turn.timings.state_ms = elapsed_ms(started);

        // 2. Retrieval.
        let started = Instant::now();
        let hits = self.index.search(&turn.query, settings.k_before).await;
        turn.timings.retrieve_ms = elapsed_ms(started);
        let hits = match hits {
            Ok(hits) => hits,
            Err(e) => {
                turn.error = Some(e);
                return;
            }
        };
        let started = Instant::now();
        let mut candidates = self.rerank(&turn.query, &hits);
        select(&mut candidates, settings);
        turn.timings.rerank_ms = elapsed_ms(started);
        turn.best_score = candidates.first().map(|c| c.score);
        let context = kept_hits(&candidates, &hits);
        turn.hints = candidates
            .iter()
            .take(HINTS)
            .map(|c| format!("{} — {}", c.source, c.section))
            .collect();
        trace.candidates = candidates;
        trace.context = context.clone();

        // The relevance gate: nothing to quote, so nothing to answer from.
        if context.is_empty() {
            turn.give_up(UnknownReason::LowRelevance, None, &session.state.goal, settings.threshold);
            return;
        }

        // 3. The answer, with a retry that is told what was wrong.
        let mut messages = answer_messages(&session.state, &session.turns, &message, &context);
        let mut outcome = Outcome::Rejected;
        let started = Instant::now();
        for attempt in 1..=MAX_ATTEMPTS {
            turn.attempts = attempt;
            let (text, usage) = match self.llm.complete(&messages).await {
                Ok(reply) => reply,
                Err(e) => {
                    outcome = Outcome::Failed(e);
                    break;
                }
            };
            turn.usage += usage;
            trace.replies.push(text.clone());
            match check_reply(&text, &context) {
                Ok(Parsed::Answered(v)) => {
                    outcome = Outcome::Verified(v);
                    break;
                }
                Ok(Parsed::Unknown { clarification }) => {
                    outcome = Outcome::Unsure(clarification);
                    break;
                }
                Err(errors) => {
                    if attempt < MAX_ATTEMPTS {
                        messages.push(json!({ "role": "assistant", "content": text }));
                        messages.push(json!({ "role": "user", "content": retry_message(&errors) }));
                    }
                    trace.rejected.push(errors);
                }
            }
        }
        turn.timings.llm_ms = elapsed_ms(started);
        trace.messages = messages;

        let goal = &session.state.goal;
        let v = match outcome {
            Outcome::Verified(v) => v,
            Outcome::Failed(e) => {
                turn.error = Some(e);
                return;
            }
            Outcome::Unsure(clarification) => {
                turn.give_up(UnknownReason::ModelUnsure, Some(clarification), goal, settings.threshold);
                return;
            }
            Outcome::Rejected => {
                turn.give_up(UnknownReason::Unverified, None, goal, settings.threshold);
                return;
            }
        };
        turn.status = Status::Answered;
        turn.text = v.answer;
        turn.sources = v.sources.iter().map(|&n| source_ref(&context, n)).collect();
        turn.quotes = v
            .quotes
            .iter()
            .map(|(n, text)| {
                let s = source_ref(&context, *n);
                Quote {
                    n: *n,
                    chunk_id: s.chunk_id,
                    source: s.source,
                    section: s.section,
                    text: text.clone(),
                }
            })
            .collect();
        turn.quotes_verified = !turn.quotes.is_empty()
            && turn
                .quotes
                .iter()
                .all(|q| quote_in(&q.text, &context[q.n - 1].chunk.text));
    }
}

// ---------------------------------------------------------------------------
// Sessions: one JSON file each, so the history survives a restart.
// ---------------------------------------------------------------------------

struct Store {
    /// `None`: in memory only (the eval, and the tests).
    dir: Option<PathBuf>,
    sessions: Mutex<HashMap<String, Arc<tokio::sync::Mutex<Session>>>>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

impl Store {
    fn new(dir: Option<PathBuf>) -> Store {
        Store {
            dir,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// `CHAT_DIR` (default `sessions`); `off` keeps sessions in memory.
    fn from_env() -> Store {
        let dir = env_non_empty("CHAT_DIR").unwrap_or_else(|| "sessions".to_string());
        if dir == "off" {
            return Store::new(None);
        }
        let dir = PathBuf::from(dir);
        match std::fs::create_dir_all(&dir) {
            Ok(()) => Store::new(Some(dir)),
            Err(e) => {
                eprintln!("Warning: can't create {}: {e}; sessions stay in memory.", dir.display());
                Store::new(None)
            }
        }
    }

    /// Ids name files, so only letters, digits, `-` and `_`.
    fn valid_id(id: &str) -> bool {
        (1..=64).contains(&id.len()) && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }

    fn new_id() -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        format!("{nanos:x}{:04x}", NEXT_ID.fetch_add(1, Ordering::Relaxed) & 0xffff)
    }

    fn path(&self, id: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{id}.json")))
    }

    fn insert(&self, session: Session) -> Arc<tokio::sync::Mutex<Session>> {
        let id = session.id.clone();
        let session = Arc::new(tokio::sync::Mutex::new(session));
        self.sessions.lock().unwrap().insert(id, session.clone());
        session
    }

    fn create(&self) -> Arc<tokio::sync::Mutex<Session>> {
        self.insert(Session {
            id: Store::new_id(),
            created: now_secs(),
            ..Session::default()
        })
    }

    /// A session in memory, or else from its file.
    fn get(&self, id: &str) -> Option<Arc<tokio::sync::Mutex<Session>>> {
        if !Store::valid_id(id) {
            return None;
        }
        if let Some(s) = self.sessions.lock().unwrap().get(id) {
            return Some(s.clone());
        }
        let text = std::fs::read_to_string(self.path(id)?).ok()?;
        let mut session: Session = serde_json::from_str(&text).ok()?;
        session.id = id.to_string();
        Some(self.insert(session))
    }

    /// `get`, or a new empty session with this id.
    fn open(&self, id: &str) -> Result<Arc<tokio::sync::Mutex<Session>>, String> {
        if !Store::valid_id(id) {
            return Err(format!("{id:?} is not a valid session id."));
        }
        Ok(self.get(id).unwrap_or_else(|| {
            self.insert(Session {
                id: id.to_string(),
                created: now_secs(),
                ..Session::default()
            })
        }))
    }

    /// Writes the session to a temporary file, then renames it over the old
    /// one, so a crash never leaves half a file.
    fn save(&self, session: &Session) -> Result<(), String> {
        let Some(path) = self.path(&session.id) else {
            return Ok(());
        };
        let text = serde_json::to_string_pretty(session).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| format!("Can't save the session to {}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Scenario checks: does the assistant keep the goal and give its sources?
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct StepCheck {
    /// In scope: answered. Off topic: "I don't know".
    status_ok: bool,
    /// Answered: sources and every quote verified. "I don't know": the
    /// nearest sections are shown instead.
    sources_shown: bool,
    /// In scope: a source of the answer is from an expected file.
    source_file_ok: Option<bool>,
    /// The goal is set and still about what the scenario is about.
    goal_kept: bool,
    /// This message's additions are in the task state.
    notes_ok: Option<bool>,
    missing_notes: Vec<&'static str>,
    facts_found: usize,
    facts_total: usize,
    missing_facts: Vec<&'static str>,
}

fn check_step(scenario: &Scenario, step: &Step, turn: &Turn, state: &TaskState) -> StepCheck {
    let text = turn.text.to_lowercase();
    let goal = state.goal.to_lowercase();
    let all = state.all_text();
    let missing_notes: Vec<&'static str> = step
        .notes
        .iter()
        .filter(|n| !n.keys.iter().any(|k| all.contains(&k.to_lowercase())))
        .map(|n| n.text)
        .collect();
    let missing_facts: Vec<&'static str> = step
        .facts
        .iter()
        .filter(|variants| !variants.iter().any(|v| text.contains(&v.to_lowercase())))
        .map(|variants| variants[0])
        .collect();
    let in_scope = step.in_scope();
    StepCheck {
        status_ok: if in_scope {
            turn.status == Status::Answered
        } else {
            turn.status == Status::Unknown
        },
        sources_shown: match turn.status {
            Status::Answered => !turn.sources.is_empty() && turn.quotes_verified,
            Status::Unknown => !turn.hints.is_empty(),
            Status::Error => false,
        },
        source_file_ok: in_scope.then(|| {
            turn.sources
                .iter()
                .any(|s| step.sources.contains(&s.source.as_str()))
        }),
        goal_kept: !goal.is_empty() && scenario.goal_keys.iter().any(|k| goal.contains(k)),
        notes_ok: (!step.notes.is_empty()).then_some(missing_notes.is_empty()),
        missing_notes,
        facts_found: step.facts.len() - missing_facts.len(),
        facts_total: step.facts.len(),
        missing_facts,
    }
}

// ---------------------------------------------------------------------------
// HTTP API and page.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    store: Arc<Store>,
}

async fn config(State(app): State<AppState>) -> Json<Value> {
    let index = &app.agent.index;
    Json(json!({
        "model": app.agent.llm.model,
        "embedder": index.embedder.id(),
        "documents": index.documents,
        "chunks": index.chunks.len(),
        "defaults": Settings::default(),
        "max_k_before": MAX_K_BEFORE,
        "max_k_after": MAX_K_AFTER,
        "history_turns": HISTORY_TURNS,
        "persistent": app.store.dir.is_some(),
    }))
}

async fn scenarios_api() -> Json<&'static [Scenario]> {
    Json(SCENARIOS)
}

#[derive(Serialize, Default)]
struct SessionView {
    session_id: String,
    state: TaskState,
    turns: Vec<Turn>,
    error: Option<String>,
}

impl SessionView {
    fn of(s: &Session) -> SessionView {
        SessionView {
            session_id: s.id.clone(),
            state: s.state.clone(),
            turns: s.turns.clone(),
            error: None,
        }
    }

    fn missing(id: &str) -> SessionView {
        SessionView {
            session_id: id.to_string(),
            error: Some(format!("No session {id:?}.")),
            ..SessionView::default()
        }
    }
}

async fn new_session(State(app): State<AppState>) -> Json<SessionView> {
    let session = app.store.create();
    let session = session.lock().await;
    let mut view = SessionView::of(&session);
    view.error = app.store.save(&session).err();
    Json(view)
}

async fn get_session(State(app): State<AppState>, Path(id): Path<String>) -> Json<SessionView> {
    let Some(s) = app.store.get(&id) else {
        return Json(SessionView::missing(&id));
    };
    let session = s.lock().await;
    Json(SessionView::of(&session))
}

/// Forgets the dialogue and the task state, keeps the id.
async fn reset_session(State(app): State<AppState>, Path(id): Path<String>) -> Json<SessionView> {
    let Some(s) = app.store.get(&id) else {
        return Json(SessionView::missing(&id));
    };
    let mut session = s.lock().await;
    session.state = TaskState::default();
    session.turns.clear();
    let mut view = SessionView::of(&session);
    view.error = app.store.save(&session).err();
    Json(view)
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct ChatRequest {
    /// Missing: a new session is started.
    session_id: Option<String>,
    /// Ignored when `scenario` and `step` are given.
    message: String,
    /// A scenario step: its message is sent, and the turn is checked.
    scenario: Option<String>,
    /// 0-based.
    step: Option<usize>,
    k_before: Option<usize>,
    k_after: Option<usize>,
    threshold: Option<f32>,
}

impl ChatRequest {
    fn settings(&self) -> Settings {
        let d = Settings::default();
        Settings {
            k_before: self.k_before.unwrap_or(d.k_before),
            k_after: self.k_after.unwrap_or(d.k_after),
            threshold: self.threshold.unwrap_or(d.threshold),
        }
        .clamped()
    }
}

#[derive(Serialize, Default)]
struct ChatResponse {
    session_id: Option<String>,
    settings: Option<Settings>,
    turn: Option<Turn>,
    /// The task state after the turn.
    state: Option<TaskState>,
    check: Option<StepCheck>,
    trace: Option<Trace>,
    /// The turn happened but couldn't be written to disk.
    save_error: Option<String>,
    error: Option<String>,
}

fn chat_error(error: String) -> Json<ChatResponse> {
    Json(ChatResponse {
        error: Some(error),
        ..ChatResponse::default()
    })
}

async fn chat(State(app): State<AppState>, Json(req): Json<ChatRequest>) -> Json<ChatResponse> {
    let settings = req.settings();
    let step = match (req.scenario.as_deref(), req.step) {
        (Some(id), Some(i)) => match scenarios::find(id) {
            Some(scn) => match scn.steps.get(i) {
                Some(step) => Some((scn, step)),
                None => return chat_error(format!("Scenario {id:?} has no step {i}.")),
            },
            None => return chat_error(format!("No scenario {id:?}.")),
        },
        _ => None,
    };
    let message = step.map_or(req.message.trim(), |(_, s)| s.say).to_string();
    if message.is_empty() {
        return chat_error("Type a message first.".to_string());
    }
    if message.chars().count() > MAX_MESSAGE_CHARS {
        return chat_error(format!("Keep the message under {MAX_MESSAGE_CHARS} characters."));
    }
    let session = match req.session_id.as_deref() {
        Some(id) => match app.store.get(id) {
            Some(s) => s,
            None => return chat_error(format!("No session {id:?}.")),
        },
        None => app.store.create(),
    };
    let mut session = session.lock().await;
    if session.turns.len() >= MAX_TURNS {
        return chat_error(format!("This session has {MAX_TURNS} turns already: start a new one."));
    }
    let (turn, trace) = app.agent.turn(&mut session, &message, settings).await;
    let save_error = app.store.save(&session).err();
    Json(ChatResponse {
        session_id: Some(session.id.clone()),
        settings: Some(settings),
        check: step.map(|(scn, st)| check_step(scn, st, &turn, &session.state)),
        state: Some(session.state.clone()),
        turn: Some(turn),
        trace: Some(trace),
        save_error,
        error: None,
    })
}

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/api/config", get(config))
        .route("/api/scenarios", get(scenarios_api))
        .route("/api/session", post(new_session))
        .route("/api/session/{id}", get(get_session))
        .route("/api/session/{id}/reset", post(reset_session))
        .route("/api/chat", post(chat))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// The CLI: `app chat [session]` and `app eval [scenario]`.
// ---------------------------------------------------------------------------

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

fn status_label(turn: &Turn) -> String {
    match (turn.status, turn.unknown_reason) {
        (Status::Answered, _) if turn.attempts > 1 => "answered (retry)".to_string(),
        (Status::Answered, _) => "answered".to_string(),
        (Status::Unknown, Some(r)) => format!(
            "don't know: {}",
            serde_json::to_value(r).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
        ),
        (Status::Unknown, None) => "don't know".to_string(),
        (Status::Error, _) => format!("error: {}", preview(turn.error.as_deref().unwrap_or(""), 60)),
    }
}

/// The answer, then always its sources: the cited chunks, or for an "I don't
/// know" the nearest sections.
fn print_turn(turn: &Turn) {
    println!("\n{}", turn.text);
    if let Some(e) = &turn.error {
        println!("error: {e}");
    }
    if turn.sources.is_empty() {
        let best = turn.best_score.map_or("–".to_string(), |b| format!("{b:.2}"));
        println!("Sources: none reached the threshold (best {best}); nearest:");
        for h in &turn.hints {
            println!("  · {h}");
        }
    } else {
        println!("Sources:");
        for s in &turn.sources {
            println!("  [{}] {} — {} ({}, score {:.2})", s.n, s.source, s.section, s.chunk_id, s.score);
        }
        for q in &turn.quotes {
            println!("  quote [{}]: \"{}\"", q.n, preview(&q.text, 200));
        }
    }
    let c = &turn.state_change;
    if let Some(g) = &c.goal {
        println!("  + goal: {g}");
    }
    for a in &c.added {
        println!("  + {a}");
    }
    for r in &c.removed {
        println!("  − {r}");
    }
    if let Some(e) = &turn.state_error {
        println!("  (task state not updated: {e})");
    }
}

async fn run_chat(agent: &Agent, store: &Store, id: &str) {
    let session = match store.open(id) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };
    let mut session = session.lock().await;
    println!(
        "Session {id}: {} earlier turns. Commands: /state, /history, /reset, /quit",
        session.turns.len()
    );
    if !session.state.is_empty() {
        println!("\n{}", session.state.render());
    }
    let settings = Settings::default();
    loop {
        print!("\nyou> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/quit" | "/exit" => break,
            "/state" => println!("\n{}", session.state.render()),
            "/history" => {
                for t in &session.turns {
                    println!("{:>3}. you: {}\n     bot: {}", t.n, preview(&t.user, 100), preview(&t.text, 140));
                }
            }
            "/reset" => {
                session.state = TaskState::default();
                session.turns.clear();
                println!("Dialogue and task state cleared.");
            }
            message if message.chars().count() > MAX_MESSAGE_CHARS => {
                println!("Keep the message under {MAX_MESSAGE_CHARS} characters.");
                continue;
            }
            message => {
                let (turn, _) = agent.turn(&mut session, message, settings).await;
                print_turn(&turn);
            }
        }
        if let Err(e) = store.save(&session) {
            eprintln!("{e}");
        }
    }
}

/// Runs scenarios in fresh in-memory sessions and prints, per turn, whether
/// the goal survived and the answer came with sources.
async fn run_eval(agent: &Agent, only: Option<&str>) {
    let settings = Settings::default();
    for scenario in SCENARIOS.iter().filter(|s| only.is_none_or(|id| s.id == id)) {
        println!("\n# {} ({} messages)\n", scenario.title, scenario.steps.len());
        println!("| # | message | status | sources shown | expected file | goal kept | state notes | facts | search query |");
        println!("|---|---|---|---|---|---|---|---|---|");
        let mut session = Session::default();
        let mut rows = Vec::new();
        for (i, step) in scenario.steps.iter().enumerate() {
            let (turn, _) = agent.turn(&mut session, step.say, settings).await;
            let c = check_step(scenario, step, &turn, &session.state);
            println!(
                "| {} | {} | {} {} | {} {} | {} | {} | {} | {}/{} | {} |",
                i + 1,
                preview(step.say, 70),
                mark(c.status_ok),
                status_label(&turn),
                mark(c.sources_shown),
                if turn.sources.is_empty() { turn.hints.len() } else { turn.sources.len() },
                c.source_file_ok.map_or("–", mark),
                mark(c.goal_kept),
                c.notes_ok.map_or("–", mark),
                c.facts_found,
                c.facts_total,
                preview(&turn.query, 80),
            );
            rows.push((step, turn, c));
        }

        let n = rows.len();
        let count = |f: &dyn Fn(&Step, &Turn, &StepCheck) -> bool| rows.iter().filter(|(s, t, c)| f(s, t, c)).count();
        let in_scope = count(&|s, _, _| s.in_scope());
        println!("\nTotals:");
        println!(
            "- answered with sources and verified quotes: {}/{in_scope}",
            count(&|s, t, c| s.in_scope() && t.status == Status::Answered && c.sources_shown)
        );
        println!(
            "- off topic → \"I don't know\" with the nearest sections: {}/{}",
            count(&|s, _, c| !s.in_scope() && c.status_ok && c.sources_shown),
            n - in_scope
        );
        println!("- every turn showed sources: {}/{n}", count(&|_, _, c| c.sources_shown));
        println!("- a source from the expected file: {}/{in_scope}", count(&|_, _, c| c.source_file_ok == Some(true)));
        println!("- goal kept after the turn: {}/{n}", count(&|_, _, c| c.goal_kept));
        let goal_changes = count(&|_, t, _| t.state_change.goal.is_some());
        println!(
            "- goal set at turn {} and set {goal_changes} time(s) in all",
            session.state.goal_turn
        );
        let notes: usize = rows.iter().map(|(s, _, _)| s.notes.len()).sum();
        let missing: usize = rows.iter().map(|(_, _, c)| c.missing_notes.len()).sum();
        println!("- clarifications, constraints and terms in the state: {}/{notes}", notes - missing);
        let (found, total) = rows
            .iter()
            .fold((0, 0), |(f, t), (_, _, c)| (f + c.facts_found, t + c.facts_total));
        println!("- expected facts stated: {found}/{total}");
        println!(
            "- state update failures: {}",
            count(&|_, t, _| t.state_error.is_some())
        );

        println!("\nTask state at the end:\n\n{}", session.state.render());
        for (step, turn, c) in &rows {
            println!("\n## {}. {}", turn.n, step.say);
            println!("search: {}", turn.query);
            print_turn(turn);
            if !c.missing_facts.is_empty() {
                println!("  missing facts: {}", c.missing_facts.join(", "));
            }
            if !c.missing_notes.is_empty() {
                println!("  not in the state: {}", c.missing_notes.join(", "));
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let base_url =
        env_non_empty("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.deepseek.com".to_string());
    let model = env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key = env_non_empty("OPENAI_API_KEY").unwrap_or_else(|| {
        eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
        std::process::exit(1);
    });
    let embedder = embedder_from_env().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    let index = Index::build(builtin_documents(), embedder)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Error: indexing failed: {e}");
            std::process::exit(1);
        });
    println!(
        "indexed {} documents into {} chunks ({})",
        index.documents,
        index.chunks.len(),
        index.embedder.id()
    );
    let llm = Llm {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client"),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model,
    };
    let agent = Arc::new(Agent::new(llm, index));

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        // `app eval [deploy|memory]`: run the scenarios, print the table, exit.
        Some("eval") => {
            run_eval(&agent, args.next().as_deref()).await;
            return;
        }
        // `app chat [session]`: the same chat in the terminal.
        Some("chat") => {
            let store = Store::from_env();
            let id = args.next().unwrap_or_else(|| "cli".to_string());
            run_chat(&agent, &store, &id).await;
            return;
        }
        _ => {}
    }

    let store = Arc::new(Store::from_env());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(AppState { agent, store })).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenarios::{REFUSAL, Slot};

    #[test]
    fn quotes_are_matched_word_for_word() {
        let text = "Port is always `4000 + NN`\n(lesson 6 → `4006`). Service name is always\n`ai-advent-lesson-NN`.";
        assert!(quote_in("Port is always 4000 + NN", text));
        assert!(quote_in("port is   ALWAYS `4000 + NN`.", text));
        assert!(quote_in("“Port is always 4000 + NN”", text));
        assert!(quote_in("Port is always … Service name is always", text));
        assert!(!quote_in("Port is always 8080", text));
        assert!(!quote_in("The port is 4000 + NN", text), "a paraphrase is not a quote");
        assert!(!quote_in("Service name is always … Port is always", text), "parts out of order");
        assert!(!quote_in("...", text));

        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert_eq!(
            strip_citations("Port 4000 [1]. Unit [2, 3], see [docs] [x]."),
            "Port 4000 . Unit , see [docs] [x]."
        );
    }

    /// A line of the text worth quoting, cut at a word to at most 160
    /// characters: what a careful model would copy. Lines with `[` are
    /// skipped, so code like `messages[0]` can't look like a citation.
    fn quotable(text: &str) -> Option<String> {
        let line = text
            .lines()
            .map(str::trim)
            .find(|l| {
                l.chars().count() >= 30
                    && !l.starts_with('#')
                    && !l.starts_with("```")
                    && !l.starts_with('|')
                    && !l.contains('[')
            })
            .or_else(|| {
                text.lines()
                    .map(str::trim)
                    .filter(|l| l.chars().count() >= MIN_QUOTE_CHARS && !l.starts_with("```") && !l.contains('['))
                    .max_by_key(|l| l.chars().count())
            })?;
        if line.chars().count() <= 160 {
            return Some(line.to_string());
        }
        let cut: String = line.chars().take(160).collect();
        Some(cut.rsplit_once(' ').map_or(cut.clone(), |(head, _)| head.to_string()))
    }

    #[tokio::test]
    async fn replies_are_checked_against_their_fragments() {
        let index = Index::build(builtin_documents(), index::local()).await.unwrap();
        let context = index
            .search("loginctl enable-linger user services after the SSH session", 3)
            .await
            .unwrap();
        let q1 = quotable(&context[0].chunk.text).unwrap();

        let ok = json!({
            "status": "answered",
            "answer": "Something true [1].",
            "sources": [1],
            "quotes": [{ "fragment": 1, "text": q1 }],
        });
        assert_eq!(
            check_reply(&ok.to_string(), &context),
            Ok(Parsed::Answered(Verified {
                answer: "Something true [1].".to_string(),
                sources: vec![1],
                quotes: vec![(1, q1.clone())],
            }))
        );
        let unknown = json!({ "status": "unknown", "clarification": " Which lesson? " });
        assert_eq!(
            check_reply(&unknown.to_string(), &context),
            Ok(Parsed::Unknown { clarification: "Which lesson?".to_string() })
        );

        let has = |errors: &[String], what: &str| errors.iter().any(|e| e.contains(what));
        let errors = check_reply("Port 4000.", &context).unwrap_err();
        assert!(has(&errors, "JSON"), "{errors:?}");
        let bad = json!({
            "answer": "X [4]",
            "sources": [1, 9],
            "quotes": [
                { "fragment": 1, "text": "Every lesson runs as a Kubernetes pod" },
                { "fragment": 2, "text": "short" },
            ],
        });
        let errors = check_reply(&bad.to_string(), &context).unwrap_err();
        for what in [
            "source 9 is not a fragment number",
            "quote 1 is not word for word in fragment [1]",
            "quote 2 is too short",
            "quote 2 is from fragment [2], which isn't in sources",
            "the answer cites [4]",
        ] {
            assert!(has(&errors, what), "{what}: {errors:?}");
        }
        assert!(retry_message(&errors).contains("quote 2 is too short"));
    }

    #[test]
    fn select_orders_by_score_and_applies_threshold_then_top_k() {
        let c = |cosine_rank: usize, score: f32| Candidate {
            chunk_id: format!("c{cosine_rank}"),
            source: String::new(),
            section: String::new(),
            tokens: 1,
            cosine_rank,
            cosine: 0.0,
            coverage: 0.0,
            heading: 0.0,
            score,
            rank: 0,
            kept: false,
            dropped: None,
        };
        let mut cs = vec![c(1, 0.2), c(2, 0.9), c(3, 0.5), c(4, 0.5), c(5, 0.4)];
        select(&mut cs, Settings { k_after: 2, threshold: 0.3, ..Settings::default() });
        let order: Vec<usize> = cs.iter().map(|c| c.cosine_rank).collect();
        assert_eq!(order, vec![2, 3, 4, 5, 1], "ties keep the cosine order");
        let kept: Vec<bool> = cs.iter().map(|c| c.kept).collect();
        assert_eq!(kept, vec![true, true, false, false, false]);
        assert_eq!(cs[2].dropped, Some("past top-K"));
        assert_eq!(cs[4].dropped, Some("below threshold"));

        let s = Settings { k_before: 3, k_after: 99, threshold: 7.0 }.clamped();
        assert_eq!((s.k_before, s.k_after, s.threshold), (MAX_K_AFTER, MAX_K_AFTER, 1.0));
        let s = Settings { k_before: 0, k_after: 0, threshold: f32::NAN }.clamped();
        assert_eq!((s.k_before, s.k_after, s.threshold), (1, 1, DEFAULT_THRESHOLD));
    }

    #[test]
    fn dont_know_brings_the_goal_back() {
        for reason in [UnknownReason::LowRelevance, UnknownReason::ModelUnsure, UnknownReason::Unverified] {
            let en = dont_know("Why?", reason, Some(0.21), 0.3).to_lowercase();
            let ru = dont_know("Почему?", reason, Some(0.21), 0.3).to_lowercase();
            assert!(REFUSAL.iter().any(|r| en.contains(r)) && REFUSAL.iter().any(|r| ru.contains(r)));
        }
        assert!(default_clarification("Почему?", "").starts_with("Уточните"));
        assert!(default_clarification("Why?", "").starts_with("Could you clarify"));
        assert!(default_clarification("Почему?", "задеплоить урок").contains("«задеплоить урок»"));
        assert!(default_clarification("Why?", "pick a strategy").contains("\"pick a strategy\""));

        let mut turn = Turn::start(3, "What is the capital of Australia?");
        turn.best_score = Some(0.22);
        turn.give_up(UnknownReason::LowRelevance, None, "pick a strategy", 0.3);
        assert_eq!(turn.status, Status::Unknown);
        assert!(turn.text.starts_with("I don't know") && turn.text.contains("0.22"));
        assert!(turn.text.ends_with(turn.clarification.as_deref().unwrap()));
        assert!(turn.text.contains("pick a strategy"));
    }

    #[test]
    fn prompts_carry_the_state_and_only_the_recent_dialogue() {
        let mut state = TaskState::default();
        state.goal = "Deploy lesson 25".to_string();
        state.constraints.push("No Docker".to_string());
        let turns: Vec<Turn> = (1..=10)
            .map(|n| Turn {
                n,
                user: format!("question {n}"),
                text: format!("answer {n} [1]"),
                status: if n == 10 { Status::Error } else { Status::Answered },
                ..Turn::default()
            })
            .collect();
        let m = answer_messages(&state, &turns, "next", &[]);
        let system = m[0]["content"].as_str().unwrap();
        assert!(system.contains("Goal of the dialogue: Deploy lesson 25") && system.contains("- No Docker"));
        assert!(system.contains(MEMORY_RULES) && system.contains(ANSWER_RULES));
        // The last HISTORY_TURNS turns that didn't fail, citations stripped.
        assert_eq!(m.len(), 1 + 2 * HISTORY_TURNS + 1);
        assert_eq!(m[1]["content"], "question 4");
        assert_eq!(m[2]["content"], "answer 4 ");
        assert_eq!(m[m.len() - 3]["content"], "question 9");
        assert!(m.last().unwrap()["content"].as_str().unwrap().ends_with("Message: next"));

        let s = state_messages(&state, &turns, "next");
        assert_eq!(s[0]["content"], STATE_PROMPT);
        let user = s[1]["content"].as_str().unwrap();
        assert!(user.contains("\"goal\": \"Deploy lesson 25\"") && user.contains("No Docker"));
        assert!(user.contains("User: question 9") && !user.contains("question 6"));
        assert!(user.ends_with("Latest user message:\nnext"));
        let empty = state_messages(&TaskState::default(), &[], "hi");
        assert!(empty[1]["content"].as_str().unwrap().contains("Recent dialogue:\n(none yet)"));
    }

    // -----------------------------------------------------------------------
    // A fake model that plays both roles: the state updater, scripted from
    // the scenarios, and the answerer.
    // -----------------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Behaviour {
        /// Quotes the first lines of fragments [1] and [2] exactly, and
        /// answers with nothing but those quotes.
        Good,
        /// Forgets the quotes, then gets them right when told.
        Fixer,
        /// Makes a quote up, every time.
        Liar,
        /// Says the fragments don't answer the question.
        Unsure,
        /// Answers well, but the state update is never JSON.
        BrokenState,
    }

    type Seen = Arc<Mutex<Vec<Value>>>;

    #[derive(Clone)]
    struct Fake {
        seen: Seen,
        behaviour: Behaviour,
    }

    fn find_step(say: &str) -> Option<&'static Step> {
        SCENARIOS.iter().flat_map(|s| s.steps.iter()).find(|s| s.say == say)
    }

    /// What a careful state updater would reply for a scenario message: its
    /// notes and its reference query. Any other message: nothing new, and the
    /// message itself as the query.
    fn state_reply(user: &str) -> String {
        let latest = user.split_once("Latest user message:\n").map_or(user, |(_, m)| m);
        let Some(step) = find_step(latest) else {
            return json!({ "search_query": latest }).to_string();
        };
        let of = |slot: Slot| step.notes.iter().filter(move |n| n.slot == slot).map(|n| n.text);
        json!({
            "goal": of(Slot::Goal).next().unwrap_or(""),
            "goal_changed": false,
            "clarified": of(Slot::Clarified).collect::<Vec<_>>(),
            "constraints": of(Slot::Constraint).collect::<Vec<_>>(),
            "terms": of(Slot::Term)
                .map(|t| {
                    let (term, meaning) = t.split_once(" — ").unwrap();
                    json!({ "term": term, "meaning": meaning })
                })
                .collect::<Vec<_>>(),
            "remove": [],
            "search_query": step.query,
        })
        .to_string()
    }

    /// The fragments of an answer prompt, `[1]` first: the lines under each
    /// `[n] file — section` header.
    fn fragments(user: &str) -> Vec<String> {
        let context = user.strip_prefix("Context:\n\n").unwrap_or(user);
        let context = context.rfind("\n\nMessage: ").map_or(context, |i| &context[..i]);
        let mut out: Vec<String> = Vec::new();
        for line in context.lines() {
            let header = line
                .strip_prefix(&format!("[{}] ", out.len() + 1))
                .is_some_and(|rest| rest.contains(" — "));
            if header {
                out.push(String::new());
            } else if let Some(last) = out.last_mut() {
                last.push_str(line);
                last.push('\n');
            }
        }
        out
    }

    fn good_reply(user: &str) -> String {
        let quotes: Vec<(usize, String)> = fragments(user)
            .iter()
            .enumerate()
            .filter_map(|(i, f)| quotable(f).map(|q| (i + 1, q)))
            .take(2)
            .collect();
        let answer = quotes
            .iter()
            .map(|(n, q)| format!("{q} [{n}]"))
            .collect::<Vec<_>>()
            .join(" ");
        json!({
            "status": "answered",
            "answer": answer,
            "sources": quotes.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            "quotes": quotes.iter().map(|(n, q)| json!({ "fragment": n, "text": q })).collect::<Vec<_>>(),
        })
        .to_string()
    }

    async fn fake_chat(
        State(fake): State<Fake>,
        headers: axum::http::HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        fake.seen.lock().unwrap().push(body.clone());
        let messages = body["messages"].as_array().cloned().unwrap_or_default();
        let content_of = |m: &Value| m["content"].as_str().unwrap_or_default().to_string();
        let system = content_of(&messages[0]);
        let last = content_of(messages.last().unwrap());
        let context = messages
            .iter()
            .rev()
            .map(content_of)
            .find(|c| c.starts_with("Context:"))
            .unwrap_or_default();
        let content = if system == STATE_PROMPT {
            if fake.behaviour == Behaviour::BrokenState {
                "Sure! The goal is clear.".to_string()
            } else {
                state_reply(&last)
            }
        } else {
            match fake.behaviour {
                Behaviour::Good | Behaviour::BrokenState => good_reply(&context),
                Behaviour::Fixer if last.starts_with("Your reply was rejected") => good_reply(&context),
                Behaviour::Fixer => {
                    json!({ "status": "answered", "answer": "See the docs [1].", "sources": [1] }).to_string()
                }
                Behaviour::Liar => json!({
                    "status": "answered",
                    "answer": "Lessons run on Kubernetes, port 8080 [1].",
                    "sources": [1],
                    "quotes": [{ "fragment": 1, "text": "Every lesson runs as a Kubernetes pod on port 8080" }],
                })
                .to_string(),
                Behaviour::Unsure => {
                    json!({ "status": "unknown", "clarification": "Which lesson do you mean?" }).to_string()
                }
            }
        };
        Json(json!({
            "choices": [{ "message": { "role": "assistant", "content": content } }],
            "usage": { "prompt_tokens": 100, "completion_tokens": 9 },
        }))
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// An agent over the real index, talking to a fake model.
    async fn fake_agent(behaviour: Behaviour) -> (Agent, Seen) {
        let seen: Seen = Arc::default();
        let base = spawn(
            Router::new()
                .route("/v1/chat/completions", post(fake_chat))
                .with_state(Fake { seen: seen.clone(), behaviour }),
        )
        .await;
        let agent = Agent::new(
            Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{base}/v1/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            },
            Index::build(builtin_documents(), index::local()).await.unwrap(),
        );
        (agent, seen)
    }

    fn system_of(body: &Value) -> String {
        body["messages"][0]["content"].as_str().unwrap_or_default().to_string()
    }

    #[tokio::test]
    async fn both_long_scenarios_keep_the_goal_and_answer_with_sources() {
        for scenario in SCENARIOS {
            assert!((10..=15).contains(&scenario.steps.len()), "{}", scenario.id);
            let (agent, seen) = fake_agent(Behaviour::Good).await;
            let mut session = Session::default();
            let goal = scenario.steps[0]
                .notes
                .iter()
                .find(|n| n.slot == Slot::Goal)
                .map(|n| n.text)
                .unwrap();
            for (i, step) in scenario.steps.iter().enumerate() {
                let (turn, trace) = agent.turn(&mut session, step.say, Settings::default()).await;
                let c = check_step(scenario, step, &turn, &session.state);
                let id = format!("{} step {}", scenario.id, i + 1);
                assert!(turn.error.is_none() && turn.state_error.is_none(), "{id}: {turn:?}");
                assert_eq!(turn.query, step.query, "{id}: the state update's query is searched");
                assert!(c.goal_kept && c.sources_shown && c.status_ok, "{id}: {c:?} {}", turn.text);
                assert_ne!(c.notes_ok, Some(false), "{id}: {c:?}");
                assert_eq!(session.state.goal, goal, "{id}: the goal never drifts");
                assert_eq!(session.state.goal_turn, 1, "{id}");
                if step.in_scope() {
                    assert_eq!((turn.status, turn.attempts), (Status::Answered, 1), "{id}: {trace:?}");
                    assert!(!turn.sources.is_empty() && turn.quotes_verified, "{id}");
                    assert_eq!(c.source_file_ok, Some(true), "{id}: {:?}", turn.sources);
                    for s in &turn.sources {
                        let h = &trace.context[s.n - 1];
                        assert_eq!((&h.chunk.chunk_id, &h.chunk.source), (&s.chunk_id, &s.source));
                    }
                    // The answer request: the goal in the system prompt, and
                    // at most HISTORY_TURNS earlier turns as messages.
                    let system = trace.messages[0]["content"].as_str().unwrap();
                    assert!(system.contains(&format!("Goal of the dialogue: {goal}")), "{id}");
                    assert_eq!(trace.messages.len(), 1 + 2 * i.min(HISTORY_TURNS) + 1, "{id}");
                } else {
                    assert_eq!(turn.status, Status::Unknown, "{id}");
                    assert_eq!(turn.unknown_reason, Some(UnknownReason::LowRelevance), "{id}");
                    assert_eq!(turn.attempts, 0);
                    assert!(trace.messages.is_empty() && turn.sources.is_empty());
                    assert_eq!(turn.hints.len(), HINTS, "{id}: the nearest sections are its sources");
                    assert!(turn.clarification.as_deref().unwrap().contains(goal), "{id}: {}", turn.text);
                    assert_eq!(c.facts_found, 1, "{id}: {}", turn.text);
                }
            }
            // Every note of the scenario is in the final state.
            let all = session.state.all_text();
            for note in scenario.steps.iter().flat_map(|s| s.notes.iter()) {
                // A term is kept as `term: meaning`.
                let text = note.text.split_once(" — ").map_or(note.text, |(_, meaning)| meaning);
                assert!(all.contains(&text.to_lowercase()), "{}: {note:?} lost", scenario.id);
            }
            assert!(!session.state.constraints.is_empty() && !session.state.terms.is_empty());

            // The first message is long out of the history window by the
            // last turn: the goal reaches the model only through the state.
            let bodies = seen.lock().unwrap();
            let answers: Vec<&Value> = bodies.iter().filter(|b| system_of(b) != STATE_PROMPT).collect();
            let states = bodies.len() - answers.len();
            assert_eq!(states, scenario.steps.len(), "one state update per message");
            assert_eq!(answers.len(), scenario.steps.iter().filter(|s| s.in_scope()).count());
            let last = answers.last().unwrap();
            let sent = last["messages"].to_string();
            assert!(!sent.contains(&json!(scenario.steps[0].say).to_string()), "step 1 is out of the window");
            assert!(system_of(last).contains(goal));
            assert!(bodies.iter().all(|b| b["temperature"] == 0.0 && b["response_format"]["type"] == "json_object"));
        }
    }

    #[tokio::test]
    async fn failures_are_retried_or_end_in_dont_know_and_a_broken_update_keeps_the_state() {
        let first = &SCENARIOS[0].steps[0];
        let settings = Settings::default();

        let (agent, _) = fake_agent(Behaviour::Fixer).await;
        let mut s = Session::default();
        let (t, trace) = agent.turn(&mut s, first.say, settings).await;
        assert_eq!((t.status, t.attempts, trace.rejected.len()), (Status::Answered, 2, 1), "{t:?}");
        assert!(trace.rejected[0].iter().any(|e| e.contains("no quotes")));
        assert!(t.quotes_verified);

        let (agent, _) = fake_agent(Behaviour::Liar).await;
        let mut s = Session::default();
        let (t, _) = agent.turn(&mut s, first.say, settings).await;
        assert_eq!((t.status, t.unknown_reason), (Status::Unknown, Some(UnknownReason::Unverified)));
        assert!(!t.text.contains("Kubernetes") && t.text.starts_with("Не знаю"), "{}", t.text);
        assert!(t.sources.is_empty() && !t.hints.is_empty());
        assert!(t.clarification.as_deref().unwrap().contains(&s.state.goal), "the goal comes back");

        let (agent, _) = fake_agent(Behaviour::Unsure).await;
        let mut s = Session::default();
        let (t, _) = agent.turn(&mut s, first.say, settings).await;
        assert_eq!(t.unknown_reason, Some(UnknownReason::ModelUnsure));
        assert_eq!(t.clarification.as_deref(), Some("Which lesson do you mean?"));

        // A broken update: the state stays as it was, the message itself is
        // searched, and the turn is still answered.
        let (agent, _) = fake_agent(Behaviour::BrokenState).await;
        let mut s = Session::default();
        s.state.goal = "Keep me".to_string();
        let message = "Which port and which systemd service name does a deployed lesson get on the VDS?";
        let (t, _) = agent.turn(&mut s, message, settings).await;
        assert!(t.state_error.is_some() && t.state_change.is_empty());
        assert_eq!((s.state.goal.as_str(), t.query.as_str()), ("Keep me", message));
        assert_eq!(t.status, Status::Answered);
        assert_eq!(s.turns.len(), 1);
    }

    #[tokio::test]
    async fn sessions_survive_a_restart_and_ids_are_checked() {
        let dir = std::env::temp_dir().join(format!("rag-chat-test-{}", Store::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (agent, _) = fake_agent(Behaviour::Good).await;
        let id = {
            let store = Store::new(Some(dir.clone()));
            let session = store.create();
            let mut session = session.lock().await;
            for step in &SCENARIOS[1].steps[..2] {
                agent.turn(&mut session, step.say, Settings::default()).await;
            }
            store.save(&session).unwrap();
            session.id.clone()
        };
        let store = Store::new(Some(dir.clone()));
        let session = store.get(&id).expect("loaded from its file");
        let session = session.lock().await;
        assert_eq!(session.turns.len(), 2);
        assert!(!session.state.goal.is_empty() && !session.state.constraints.is_empty());
        assert!(!session.turns[1].sources.is_empty());
        assert!(!dir.join(format!("{id}.json.tmp")).exists());

        let long = "x".repeat(65);
        for bad in ["", "../etc/passwd", "a b", long.as_str()] {
            assert!(!Store::valid_id(bad) && store.get(bad).is_none(), "{bad:?}");
        }
        assert!(store.open("../x").is_err());
        assert!(store.get("never-made").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn http_api_keeps_sessions_and_returns_sources_and_state() {
        let (agent, _) = fake_agent(Behaviour::Good).await;
        let store = Arc::new(Store::new(None));
        let url = spawn(app(AppState { agent: Arc::new(agent), store })).await;
        let http = reqwest::Client::new();

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>RAG Chat with Task Memory</title>"));
        let cfg: Value = http.get(format!("{url}/api/config")).send().await.unwrap().json().await.unwrap();
        assert_eq!((cfg["model"].as_str(), cfg["persistent"].as_bool()), (Some("test-model"), Some(false)));
        let scns: Value = http.get(format!("{url}/api/scenarios")).send().await.unwrap().json().await.unwrap();
        assert_eq!(scns.as_array().unwrap().len(), SCENARIOS.len());
        assert_eq!(scns[0]["steps"][0]["notes"][0]["slot"], "goal");

        let post = |path: &str, body: Value| {
            let req = http.post(format!("{url}{path}")).json(&body);
            async move { req.send().await.unwrap().json::<Value>().await.unwrap() }
        };

        let created = post("/api/session", json!({})).await;
        let id = created["session_id"].as_str().unwrap().to_string();
        assert!(created["turns"].as_array().unwrap().is_empty());

        let r = post("/api/chat", json!({ "session_id": id, "scenario": "deploy", "step": 0 })).await;
        assert!(r["error"].is_null(), "{r}");
        assert_eq!(r["turn"]["status"], "answered");
        for field in ["chunk_id", "source", "section"] {
            assert!(r["turn"]["sources"][0][field].as_str().is_some_and(|s| !s.is_empty()));
        }
        assert_eq!(r["check"]["goal_kept"], true);
        assert_eq!(r["check"]["notes_ok"], true);
        assert!(r["state"]["goal"].as_str().unwrap().contains("задеплоить"));
        assert!(!r["trace"]["candidates"].as_array().unwrap().is_empty());

        let r = post("/api/chat", json!({ "session_id": id, "message": "Кстати, какая погода будет завтра в Москве?" })).await;
        assert_eq!(r["turn"]["status"], "unknown");
        assert_eq!(r["turn"]["hints"].as_array().unwrap().len(), HINTS);
        assert!(r["check"].is_null());

        let s: Value = http.get(format!("{url}/api/session/{id}")).send().await.unwrap().json().await.unwrap();
        assert_eq!(s["turns"].as_array().unwrap().len(), 2);
        assert_eq!(s["turns"][1]["n"], 2);

        let fresh = post("/api/chat", json!({ "message": "How does lesson 09 compress context?" })).await;
        assert!(fresh["session_id"].as_str().is_some_and(|s| s != id), "no id: a new session");

        let reset = post(&format!("/api/session/{id}/reset"), json!({})).await;
        assert!(reset["turns"].as_array().unwrap().is_empty() && reset["state"]["goal"] == "");

        let errors = [
            post("/api/chat", json!({ "session_id": id, "message": "  " })).await,
            post("/api/chat", json!({ "session_id": "nope", "message": "hi" })).await,
            post("/api/chat", json!({ "scenario": "deploy", "step": 99 })).await,
            post("/api/chat", json!({ "scenario": "x", "step": 0 })).await,
        ];
        for e in &errors {
            assert!(e["error"].as_str().is_some_and(|s| !s.is_empty()), "{e}");
        }
        let missing: Value = http.get(format!("{url}/api/session/nope")).send().await.unwrap().json().await.unwrap();
        assert!(missing["error"].as_str().unwrap().contains("nope"));
    }
}

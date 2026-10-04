mod index;
mod rerank;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use index::{ErrorResponse, Hit, Index, builtin_documents, embedder_from_env, env_non_empty, stems};
use rerank::Heuristic;

/// Stage 1: how many chunks the vector search hands to the reranker.
const DEFAULT_K_BEFORE: usize = 20;
const MAX_K_BEFORE: usize = 40;
/// Stage 2: at most this many chunks go into the prompt.
const DEFAULT_K_AFTER: usize = 5;
const MAX_K_AFTER: usize = 10;
/// The relevance gate. A chunk that scores lower is dropped. When none is
/// left, the assistant says "I don't know" and asks for a clarification,
/// without calling the model. Lesson 23's calibration for the heuristic
/// reranker and the local embedder: on its control set the weakest relevant
/// section scores 0.31 and the best out-of-scope candidate 0.27.
const DEFAULT_THRESHOLD: f32 = 0.30;
const MAX_QUESTION_CHARS: usize = 1000;
/// A shorter quote proves nothing ("the port").
const MIN_QUOTE_CHARS: usize = 12;
/// The first reply, and one retry that is told what was wrong with it.
const MAX_ATTEMPTS: usize = 2;
/// How many of the nearest sections an "I don't know" shows, so the user can
/// see what the documents do cover.
const HINTS: usize = 3;
/// Without the judge, an answer counts as grounded when at least this share
/// of its content terms occurs in its quotes and every number and identifier
/// it names is quoted.
const MIN_SUPPORT: f32 = 0.6;

/// The system prompt of every answer; same as lessons 22 and 23.
const BASE_PROMPT: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. Answer \
concisely, in the language of the question. If you don't know something, say so \
instead of guessing.";

/// The answer is a JSON object with its sources and quotes. Every quote is
/// checked against the fragment it names, so the rules ask for exact copies.
const ANSWER_RULES: &str = "The user message contains numbered fragments of the \
repository's documents, then the question. Answer only from these fragments.

Reply with one JSON object and nothing else:
{\"status\": \"answered\", \"answer\": \"...\", \"sources\": [1, 3], \"quotes\": [{\"fragment\": 1, \"text\": \"...\"}, {\"fragment\": 3, \"text\": \"...\"}]}

- answer: concise, in the language of the question. After each fact, cite the fragment it came from as [1], [2] and so on.
- sources: the number of every fragment the answer uses. At least one.
- quotes: at least one quote from every source, the words that support the answer. Copy each quote character for character from its fragment, in the fragment's language: one sentence, or one line of a list, table or code, up to 300 characters. Don't translate, paraphrase or join pieces of different sentences.
- Every fact in the answer must be backed by one of the quotes.

If the fragments don't contain the answer, don't guess. Reply instead:
{\"status\": \"unknown\", \"clarification\": \"...\"}
where clarification is a short question to the user, in the language of their question, that would help to find the answer.";

/// One extra call checks that the answer means what its quotes say.
const JUDGE_PROMPT: &str = "You check an answer against the quotes it cites. \
Decide whether every statement of the answer is backed by the quotes. A paraphrase \
or a translation of a quote counts as backed; a fact, number or name that no quote \
contains does not. Reply with JSON only: {\"verdict\": \"supported\" | \"partial\" \
| \"unsupported\", \"unsupported\": [\"each statement that no quote backs\"]}";

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
struct ChatResponse {
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
        let parsed = serde_json::from_str::<ChatResponse>(&text)
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
// Quotes: is this text really in that chunk?
// ---------------------------------------------------------------------------

/// Text as it is compared: lower case, one space for any run of whitespace,
/// typographic quotes and dashes made plain, and Markdown's `` ` ``, `*` and
/// `|` dropped (a model quoting `` `4000 + NN` `` usually writes 4000 + NN).
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
/// ellipsis (`...` or `…`) may skip text: the parts must then occur in order.
/// Punctuation at the ends of a part doesn't count, so a quote may end with a
/// full stop where the text goes on.
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

/// The text without its `[n]` markers.
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
// The model's reply, checked.
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

/// A reply that passed every check. Fragment numbers are 1-based, as in the
/// prompt.
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

/// The JSON object in a reply, possibly wrapped in a code fence or prose.
fn json_object(text: &str) -> Option<&str> {
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if start < end => Some(&text[start..=end]),
        _ => None,
    }
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

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
enum Status {
    /// An answer with its sources and verified quotes.
    Answered,
    /// "I don't know", with a request to clarify.
    Unknown,
    /// The search or the model failed: see `error`.
    Error,
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
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

fn dont_know(question: &str, reason: UnknownReason, best: Option<f32>, threshold: f32) -> String {
    let best = best.unwrap_or(0.0);
    match (is_russian(question), reason) {
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

fn default_clarification(question: &str) -> &'static str {
    if is_russian(question) {
        "Уточните, пожалуйста, вопрос: о каком уроке, файле или части пайплайна деплоя идёт речь?"
    } else {
        "Could you clarify the question: which lesson, file or part of the deploy pipeline is it about?"
    }
}

// ---------------------------------------------------------------------------
// The pipeline: search → rerank → relevance gate → answer with quotes →
// check the quotes → check the meaning.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
struct Settings {
    k_before: usize,
    k_after: usize,
    threshold: f32,
    /// Ask the LLM judge whether the answer means what its quotes say.
    judge: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            k_before: DEFAULT_K_BEFORE,
            k_after: DEFAULT_K_AFTER,
            threshold: DEFAULT_THRESHOLD,
            judge: true,
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
            judge: self.judge,
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
#[derive(Serialize, Clone, Debug)]
struct SourceRef {
    /// Its number in the prompt: `[n]` in the answer.
    n: usize,
    chunk_id: String,
    source: String,
    section: String,
    /// The reranker's score.
    score: f32,
}

#[derive(Serialize, Clone, Debug)]
struct Quote {
    n: usize,
    chunk_id: String,
    source: String,
    section: String,
    text: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Verdict {
    Supported,
    Partial,
    Unsupported,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
struct Judgement {
    verdict: Verdict,
    /// The statements no quote backs.
    unsupported: Vec<String>,
}

/// Does the answer stay within its quotes?
#[derive(Serialize, Clone, Debug, Default)]
struct Grounding {
    has_sources: bool,
    has_quotes: bool,
    /// Every quote is word for word in the chunk it names, checked again
    /// from the chunk ids.
    quotes_verified: bool,
    /// The share of the answer's content terms (corpus-rare stems) that its
    /// quotes contain. Low for an answer in another language than its
    /// quotes.
    support: f32,
    /// Numbers and identifiers in the answer that no quote contains.
    unsupported_terms: Vec<String>,
    judge: Option<Judgement>,
    judge_error: Option<String>,
    /// The meaning of the answer matches its quotes: the judge says
    /// "supported", or, without a judge, enough support and nothing
    /// unquoted.
    grounded: bool,
}

#[derive(Serialize, Clone, Copy, Debug, Default)]
struct Timings {
    retrieve_ms: f64,
    rerank_ms: f64,
    llm_ms: f64,
    judge_ms: f64,
}

#[derive(Serialize, Clone, Debug)]
struct Answer {
    status: Status,
    unknown_reason: Option<UnknownReason>,
    /// The answer with its `[n]` markers, or the "I don't know" and the
    /// clarification request.
    text: String,
    /// What the user should clarify, with every "I don't know".
    clarification: Option<String>,
    sources: Vec<SourceRef>,
    quotes: Vec<Quote>,
    /// With "I don't know": the nearest sections, to show what the documents
    /// do cover.
    hints: Vec<String>,
    /// The best reranked score: what the relevance gate compares.
    best_score: Option<f32>,
    /// The chunks in the prompt, `[1]` first.
    context: Vec<Hit>,
    /// Every candidate with its scores, best first.
    candidates: Vec<Candidate>,
    /// Model calls for the answer: 0 when the gate stopped it.
    attempts: usize,
    /// The model's raw replies, and why each rejected one was rejected.
    replies: Vec<String>,
    rejected: Vec<Vec<String>>,
    grounding: Option<Grounding>,
    /// Everything sent to the model for the answer, retry included.
    messages: Vec<Value>,
    usage: Usage,
    timings: Timings,
    error: Option<String>,
}

impl Answer {
    fn start() -> Answer {
        Answer {
            status: Status::Error,
            unknown_reason: None,
            text: String::new(),
            clarification: None,
            sources: Vec::new(),
            quotes: Vec::new(),
            hints: Vec::new(),
            best_score: None,
            context: Vec::new(),
            candidates: Vec::new(),
            attempts: 0,
            replies: Vec::new(),
            rejected: Vec::new(),
            grounding: None,
            messages: Vec::new(),
            usage: Usage::default(),
            timings: Timings::default(),
            error: None,
        }
    }

    fn give_up(
        &mut self,
        question: &str,
        reason: UnknownReason,
        clarification: Option<String>,
        threshold: f32,
    ) {
        let clarification = clarification
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| default_clarification(question).to_string());
        self.status = Status::Unknown;
        self.unknown_reason = Some(reason);
        self.text = format!(
            "{} {clarification}",
            dont_know(question, reason, self.best_score, threshold)
        );
        self.clarification = Some(clarification);
    }
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

fn build_messages(question: &str, context: &[Hit]) -> Vec<Value> {
    vec![
        json!({ "role": "system", "content": format!("{BASE_PROMPT}\n\n{ANSWER_RULES}") }),
        json!({
            "role": "user",
            "content": format!("Context:\n\n{}\n\nQuestion: {question}", context_block(context)),
        }),
    ]
}

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

/// Words of the answer that name something exact: they contain a digit, `_`,
/// `/`, or a hyphen inside a word (`enable-linger`, `x86_64`, `4006`).
fn identifiers(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());
        let exact = word.chars().any(|c| c.is_ascii_digit())
            || word.contains('_')
            || word.contains('/')
            || (word.contains('-') && word.chars().count() > 3);
        if exact && !out.iter().any(|w| w == word) {
            out.push(word.to_string());
        }
    }
    out
}

/// The deterministic part of the meaning check: what the answer says that its
/// quotes don't.
fn grounding(heuristic: &Heuristic, answer: &str, quotes: &[Quote], context: &[Hit]) -> Grounding {
    let answer = strip_citations(answer);
    let quoted = quotes.iter().map(|q| q.text.as_str()).collect::<Vec<_>>().join("\n");
    let quoted_stems: HashSet<String> = stems(&quoted).into_iter().collect();
    let terms = heuristic.terms(&answer);
    let support = if terms.is_empty() {
        1.0
    } else {
        terms.iter().filter(|t| quoted_stems.contains(t.as_str())).count() as f32 / terms.len() as f32
    };
    let quoted_norm = normalize_text(&quoted);
    Grounding {
        has_quotes: !quotes.is_empty(),
        quotes_verified: !quotes.is_empty()
            && quotes.iter().all(|q| {
                context
                    .iter()
                    .find(|h| h.chunk.chunk_id == q.chunk_id)
                    .is_some_and(|h| quote_in(&q.text, &h.chunk.text))
            }),
        support: round3(support),
        unsupported_terms: identifiers(&answer)
            .into_iter()
            .filter(|w| !quoted_norm.contains(&normalize_text(w)))
            .collect(),
        ..Grounding::default()
    }
}

#[derive(Deserialize)]
struct JudgeReply {
    verdict: String,
    #[serde(default)]
    unsupported: Vec<String>,
}

fn parse_judgement(text: &str) -> Result<Judgement, String> {
    let reply: JudgeReply = json_object(text)
        .ok_or("the judge's reply has no JSON object")
        .and_then(|j| serde_json::from_str(j).map_err(|_| "the judge's reply isn't {\"verdict\": ...}"))?;
    let verdict = match reply.verdict.trim().to_lowercase().as_str() {
        "supported" => Verdict::Supported,
        "partial" => Verdict::Partial,
        "unsupported" => Verdict::Unsupported,
        other => return Err(format!("unknown verdict {other:?}")),
    };
    Ok(Judgement {
        verdict,
        unsupported: reply.unsupported,
    })
}

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

    async fn judge(&self, question: &str, answer: &str, quotes: &[Quote]) -> Result<(Judgement, Usage), String> {
        let quotes = quotes
            .iter()
            .map(|q| format!("[{}] \"{}\"", q.n, q.text))
            .collect::<Vec<_>>()
            .join("\n");
        let messages = vec![
            json!({ "role": "system", "content": JUDGE_PROMPT }),
            json!({
                "role": "user",
                "content": format!("Question: {question}\n\nAnswer:\n{answer}\n\nQuotes:\n{quotes}"),
            }),
        ];
        let (text, usage) = self.llm.complete(&messages).await?;
        Ok((parse_judgement(&text)?, usage))
    }

    async fn ask(&self, question: &str, settings: Settings) -> Answer {
        let mut answer = Answer::start();

        // Stage 1: the vector search.
        let started = Instant::now();
        let hits = self.index.search(question, settings.k_before).await;
        answer.timings.retrieve_ms = elapsed_ms(started);
        let hits = match hits {
            Ok(hits) => hits,
            Err(e) => {
                answer.error = Some(e);
                return answer;
            }
        };

        // Stage 2: rerank, threshold, top-K.
        let started = Instant::now();
        let mut candidates = self.rerank(question, &hits);
        select(&mut candidates, settings);
        answer.timings.rerank_ms = elapsed_ms(started);
        answer.best_score = candidates.first().map(|c| c.score);
        answer.context = kept_hits(&candidates, &hits);
        answer.hints = candidates
            .iter()
            .take(HINTS)
            .map(|c| format!("{} — {}", c.source, c.section))
            .collect();
        answer.candidates = candidates;

        // The relevance gate: nothing to quote, so nothing to answer from.
        if answer.context.is_empty() {
            answer.give_up(question, UnknownReason::LowRelevance, None, settings.threshold);
            return answer;
        }

        // Stage 3: the answer, with a retry that is told what was wrong.
        let mut messages = build_messages(question, &answer.context);
        let mut outcome = Outcome::Rejected;
        let started = Instant::now();
        for attempt in 1..=MAX_ATTEMPTS {
            answer.attempts = attempt;
            let (text, usage) = match self.llm.complete(&messages).await {
                Ok(reply) => reply,
                Err(e) => {
                    outcome = Outcome::Failed(e);
                    break;
                }
            };
            answer.usage += usage;
            answer.replies.push(text.clone());
            match check_reply(&text, &answer.context) {
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
                    answer.rejected.push(errors);
                }
            }
        }
        answer.timings.llm_ms = elapsed_ms(started);
        answer.messages = messages;

        let v = match outcome {
            Outcome::Verified(v) => v,
            Outcome::Failed(e) => {
                answer.error = Some(e);
                return answer;
            }
            Outcome::Unsure(clarification) => {
                answer.give_up(question, UnknownReason::ModelUnsure, Some(clarification), settings.threshold);
                return answer;
            }
            Outcome::Rejected => {
                answer.give_up(question, UnknownReason::Unverified, None, settings.threshold);
                return answer;
            }
        };
        let sources: Vec<SourceRef> = v.sources.iter().map(|&n| source_ref(&answer.context, n)).collect();
        let quotes: Vec<Quote> = v
            .quotes
            .iter()
            .map(|(n, text)| {
                let s = source_ref(&answer.context, *n);
                Quote {
                    n: *n,
                    chunk_id: s.chunk_id,
                    source: s.source,
                    section: s.section,
                    text: text.clone(),
                }
            })
            .collect();
        answer.status = Status::Answered;
        answer.text = v.answer;
        answer.sources = sources;
        answer.quotes = quotes;

        // Stage 4: does the answer mean what its quotes say?
        let mut g = grounding(&self.heuristic, &answer.text, &answer.quotes, &answer.context);
        g.has_sources = !answer.sources.is_empty();
        if settings.judge {
            let started = Instant::now();
            match self.judge(question, &answer.text, &answer.quotes).await {
                Ok((judgement, usage)) => {
                    answer.usage += usage;
                    g.judge = Some(judgement);
                }
                Err(e) => g.judge_error = Some(e),
            }
            answer.timings.judge_ms = elapsed_ms(started);
        }
        g.grounded = match &g.judge {
            Some(j) => j.verdict == Verdict::Supported,
            None => g.support >= MIN_SUPPORT && g.unsupported_terms.is_empty(),
        };
        answer.grounding = Some(g);
        answer
    }
}

// ---------------------------------------------------------------------------
// Control questions: lesson 22's ten, and four the documents don't answer.
// ---------------------------------------------------------------------------

struct Case {
    id: &'static str,
    question: &'static str,
    /// What a correct answer has to say, in words.
    expect: &'static str,
    /// The same as checks. Each group is one fact; it's found if the answer
    /// contains any of its variants, ignoring case. None of them is a word of
    /// the question.
    facts: &'static [&'static [&'static str]],
    /// Where the answer is: `(file, section)`. Empty: the documents don't
    /// cover the question, and the right answer is "I don't know".
    sources: &'static [(&'static str, &'static str)],
}

/// The fact of an out-of-scope question: the answer says it doesn't know.
const REFUSAL: &[&str] = &["don't know", "не знаю"];

const DEPLOY_WHERE: (&str, &str) = ("DEPLOYMENT.md", "Deployment › Where things live on the VDS");
const DEPLOY_MUSL: (&str, &str) = (
    "DEPLOYMENT.md",
    "Deployment › Why a static musl binary instead of Docker",
);
const DEPLOY_SECRETS: (&str, &str) = ("DEPLOYMENT.md", "Deployment › Secrets & variables");
const AGENTS_CI: (&str, &str) = (
    "AGENTS.md",
    "Agent instructions for AI-Advent-Challenge › CI/CD pipeline",
);

const CASES: &[Case] = &[
    Case {
        id: "q01",
        question: "Which port and which systemd service name does a deployed lesson get on the VDS?",
        expect: "Port 4000 + the lesson number (lesson 6 → 4006); the service is ai-advent-lesson-NN, a user-level systemd unit.",
        facts: &[&["4000"], &["ai-advent-lesson-"]],
        sources: &[DEPLOY_WHERE],
    },
    Case {
        id: "q02",
        question: "Why is each lesson deployed as a static musl binary instead of a Docker container?",
        expect: "x86_64-unknown-linux-musl is fully statically linked: no glibc version dependency, no OpenSSL (reqwest uses rustls), and no Docker needed on the VDS.",
        facts: &[
            &["glibc"],
            &["openssl", "rustls"],
            &["statically linked", "x86_64-unknown-linux-musl", "fully static"],
        ],
        sources: &[DEPLOY_MUSL],
    },
    Case {
        id: "q03",
        question: "What one-time command must be run on the VDS so the lesson services keep running after the SSH session closes?",
        expect: "sudo loginctl enable-linger <ssh-user>, once per VDS; without it user services stop when the SSH session that started them closes.",
        facts: &[&["enable-linger"]],
        sources: &[DEPLOY_WHERE],
    },
    Case {
        id: "q04",
        question: "Why does the deploy job of lesson 04 fail?",
        expect: "Lesson 04 needs MEDIUM_* and STRONG_* (and optionally WEAK_*) variables that the pipeline doesn't supply; it's left failing on purpose.",
        facts: &[&["medium_"], &["strong_"]],
        sources: &[DEPLOY_SECRETS, AGENTS_CI],
    },
    Case {
        id: "q05",
        question: "Which lessons get deployed on a normal push to master, and how do you redeploy an older lesson?",
        expect: "Only the highest-numbered lesson folder, and only if it depends on axum. An older one is redeployed by a manual \"Run workflow\" (workflow_dispatch) with the lessons input, e.g. 06, 03,06 or all.",
        facts: &[
            &["highest-numbered", "highest numbered", "newest", "latest"],
            &["axum"],
            &["workflow_dispatch", "run workflow"],
        ],
        sources: &[("DEPLOYMENT.md", "Deployment › Which lessons deploy")],
    },
    Case {
        id: "q06",
        question: "Do cargo fmt and cargo clippy block the build-test job in CI, and why?",
        expect: "No: they run with continue-on-error, because lessons 02–05 predate the pipeline and were never run through rustfmt. cargo build and cargo test are the real gate.",
        facts: &[
            &[
                "continue-on-error",
                "non-blocking",
                "not blocking",
                "don't block",
                "do not block",
                "won't block",
            ],
            &["rustfmt"],
        ],
        sources: &[AGENTS_CI],
    },
    Case {
        id: "q07",
        question: "How does the local embedder in the document indexing lesson turn text into a vector?",
        expect: "A hashed TF-IDF vector of 1024 dimensions: words lowercased and cut to their first 5 characters, stems and stem pairs hashed into signed buckets, log-scaled, weighted by IDF, normalized.",
        facts: &[
            &["1024"],
            &["idf"],
            &["hash"],
            &["first 5", "5 char", "five char", "first five"],
        ],
        sources: &[(
            "21. Document Indexing/README.md",
            "21. Document Indexing › What this is › Embeddings",
        )],
    },
    Case {
        id: "q08",
        question: "Which two chunking strategies does the document indexing lesson compare, and what sizes do they use by default?",
        expect: "Fixed windows of 1000 characters with 150 overlap, and structural chunks: one per section, joined under 300 and split over 1800 characters.",
        facts: &[&["fixed"], &["structural"], &["1000"], &["150"], &["1800"]],
        sources: &[(
            "21. Document Indexing/README.md",
            "21. Document Indexing › What this is › Chunking: two strategies",
        )],
    },
    Case {
        id: "q09",
        question: "In the MCP orchestration lesson, how are tools from several MCP servers named, and how is a tool call routed to the right server?",
        expect: "Every tool is shown as <server>__<tool> (e.g. crates__crate_info). The router splits the name on __, finds the connected server and checks its tools/list before sending tools/call.",
        facts: &[&["__"], &["tools/list"], &["crates", "github", "notes"]],
        sources: &[(
            "20. MCP Orchestration/README.md",
            "20. MCP Orchestration › What this is › Choosing the tool and routing the call",
        )],
    },
    Case {
        id: "q10",
        question: "In the invariant guardrails lesson, what are the two layers of enforcement of an invariant?",
        expect: "A deterministic check of forbidden_terms in the user's message that refuses before the model is called, and every invariant injected as a system message into every request, telling the model to check and refuse.",
        facts: &[
            &["forbidden_terms", "forbidden terms"],
            &["before the model", "without calling", "never reaches", "no network"],
            &["every request", "every single request", "system message"],
        ],
        sources: &[(
            "14. Invariant Guardrails/README.md",
            "14. Invariant Guardrails › What this is › Two layers of enforcement",
        )],
    },
    // Out of scope: the right answer is "I don't know" and a clarification
    // request, without a model call.
    Case {
        id: "n01",
        question: "What is the capital of Australia?",
        expect: "Not in the documents: \"I don't know\", and a request to clarify.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n02",
        question: "How does this repository run database migrations with Diesel and PostgreSQL?",
        expect: "No lesson uses a database: \"I don't know\", and a request to clarify.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n03",
        question: "Какая погода будет завтра в Москве?",
        expect: "Не про репозиторий: «не знаю» и просьба уточнить вопрос.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n04",
        question: "How do I configure a Kubernetes ingress for the lessons?",
        expect: "Lessons run as systemd user services, with no Kubernetes: \"I don't know\", and a request to clarify.",
        facts: &[REFUSAL],
        sources: &[],
    },
];

fn find_case(id: &str) -> Option<&'static Case> {
    CASES.iter().find(|c| c.id == id)
}

#[derive(Serialize, Clone, Debug)]
struct CaseInfo {
    id: &'static str,
    question: &'static str,
    expect: &'static str,
    in_scope: bool,
    facts: Vec<Vec<&'static str>>,
    sources: Vec<ExpectedSource>,
}

#[derive(Serialize, Clone, Debug)]
struct ExpectedSource {
    file: &'static str,
    section: &'static str,
}

impl Case {
    fn in_scope(&self) -> bool {
        !self.sources.is_empty()
    }

    fn info(&self) -> CaseInfo {
        CaseInfo {
            id: self.id,
            question: self.question,
            expect: self.expect,
            in_scope: self.in_scope(),
            facts: self.facts.iter().map(|g| g.to_vec()).collect(),
            sources: self
                .sources
                .iter()
                .map(|(file, section)| ExpectedSource {
                    file: *file,
                    section: *section,
                })
                .collect(),
        }
    }
}

#[derive(Serialize, Clone, Debug)]
struct FactCheck {
    fact: &'static str,
    /// The answer states it.
    found: bool,
    /// One of the answer's quotes contains it too.
    quoted: bool,
}

#[derive(Serialize, Clone, Debug)]
struct Score {
    facts: Vec<FactCheck>,
    found: usize,
    total: usize,
    /// Facts both stated and quoted.
    quoted: usize,
    /// In scope: answered. Out of scope: "I don't know".
    status_ok: bool,
    /// In scope: a chunk of the expected section is in the prompt.
    section_retrieved: Option<bool>,
    /// In scope: a source of the answer is from an expected file.
    source_cited: Option<bool>,
}

fn score(case: &Case, answer: &Answer) -> Score {
    let text = answer.text.to_lowercase();
    let quoted_text = answer
        .quotes
        .iter()
        .map(|q| q.text.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    let facts: Vec<FactCheck> = case
        .facts
        .iter()
        .map(|variants| FactCheck {
            fact: variants[0],
            found: variants.iter().any(|v| text.contains(&v.to_lowercase())),
            quoted: variants.iter().any(|v| quoted_text.contains(&v.to_lowercase())),
        })
        .collect();
    let in_scope = case.in_scope();
    Score {
        found: facts.iter().filter(|f| f.found).count(),
        total: facts.len(),
        quoted: facts.iter().filter(|f| f.found && f.quoted).count(),
        facts,
        status_ok: if in_scope {
            answer.status == Status::Answered
        } else {
            answer.status == Status::Unknown
        },
        section_retrieved: in_scope.then(|| {
            answer.context.iter().any(|h| {
                case.sources.iter().any(|(file, section)| {
                    h.chunk.source == *file && h.chunk.section.starts_with(section)
                })
            })
        }),
        source_cited: in_scope.then(|| {
            answer
                .sources
                .iter()
                .any(|s| case.sources.iter().any(|(file, _)| s.source == *file))
        }),
    }
}

// ---------------------------------------------------------------------------
// HTTP API and page.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
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
        "max_attempts": MAX_ATTEMPTS,
        "min_quote_chars": MIN_QUOTE_CHARS,
    }))
}

async fn cases_api() -> Json<Vec<CaseInfo>> {
    Json(CASES.iter().map(Case::info).collect())
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct AskRequest {
    /// The question; ignored when `case_id` is given.
    question: String,
    /// A control question: its text is asked, and the answer is scored.
    case_id: Option<String>,
    k_before: Option<usize>,
    k_after: Option<usize>,
    threshold: Option<f32>,
    judge: Option<bool>,
}

impl AskRequest {
    fn settings(&self) -> Settings {
        let d = Settings::default();
        Settings {
            k_before: self.k_before.unwrap_or(d.k_before),
            k_after: self.k_after.unwrap_or(d.k_after),
            threshold: self.threshold.unwrap_or(d.threshold),
            judge: self.judge.unwrap_or(d.judge),
        }
        .clamped()
    }
}

#[derive(Serialize, Clone, Debug, Default)]
struct AskResponse {
    question: String,
    case: Option<CaseInfo>,
    settings: Settings,
    answer: Option<Answer>,
    score: Option<Score>,
    error: Option<String>,
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<AskResponse> {
    let settings = req.settings();
    let case = match req.case_id.as_deref() {
        Some(id) => match find_case(id) {
            Some(case) => Some(case),
            None => {
                return Json(AskResponse {
                    settings,
                    error: Some(format!("No control question {id:?}.")),
                    ..AskResponse::default()
                });
            }
        },
        None => None,
    };
    let question = case.map_or(req.question.trim(), |c| c.question).to_string();
    let fail = |error: String| AskResponse {
        question: question.clone(),
        settings,
        error: Some(error),
        ..AskResponse::default()
    };
    if question.is_empty() {
        return Json(fail("Type a question first.".to_string()));
    }
    if question.chars().count() > MAX_QUESTION_CHARS {
        return Json(fail(format!(
            "Keep the question under {MAX_QUESTION_CHARS} characters."
        )));
    }
    let answer = app.agent.ask(&question, settings).await;
    Json(AskResponse {
        score: case.map(|c| score(c, &answer)),
        question,
        case: case.map(Case::info),
        settings,
        answer: Some(answer),
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
        .route("/api/cases", get(cases_api))
        .route("/api/ask", post(ask_api))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// `app eval`: every control question, checked, as a table.
// ---------------------------------------------------------------------------

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

async fn run_eval(agent: &Agent, settings: Settings) {
    println!(
        "top-{} by cosine → heuristic reranker → threshold {:.2} → top-{} → answer with quotes (≤{MAX_ATTEMPTS} attempts) → judge: {}\n",
        settings.k_before, settings.k_after, settings.threshold, settings.judge
    );
    println!("| # | question | status | sources | quotes | word for word | meaning matches quotes | facts (quoted) |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut results = Vec::new();
    for case in CASES {
        let answer = agent.ask(case.question, settings).await;
        let s = score(case, &answer);
        let status = match (answer.status, answer.unknown_reason) {
            (Status::Answered, _) if answer.attempts > 1 => "answered (retry)".to_string(),
            (Status::Answered, _) => "answered".to_string(),
            (Status::Unknown, Some(r)) => format!("don't know: {}", serde_json::to_value(r).unwrap().as_str().unwrap_or("")),
            (Status::Unknown, None) => "don't know".to_string(),
            (Status::Error, _) => format!("error: {}", preview(answer.error.as_deref().unwrap_or(""), 50)),
        };
        let g = answer.grounding.as_ref();
        let meaning = match g {
            Some(g) => format!(
                "{} {}, support {:.2}",
                mark(g.grounded),
                g.judge
                    .as_ref()
                    .map_or("no judge".to_string(), |j| format!("{:?}", j.verdict).to_lowercase()),
                g.support
            ),
            None => "–".to_string(),
        };
        println!(
            "| {} | {} | {} {status} | {} | {} | {} | {meaning} | {}/{} ({}) |",
            case.id,
            case.question,
            mark(s.status_ok),
            answer.sources.len(),
            answer.quotes.len(),
            g.map_or("–", |g| mark(g.quotes_verified)),
            s.found,
            s.total,
            s.quoted
        );
        results.push((case, answer, s));
    }

    let in_scope: Vec<_> = results.iter().filter(|(c, _, _)| c.in_scope()).collect();
    let out_scope: Vec<_> = results.iter().filter(|(c, _, _)| !c.in_scope()).collect();
    let count = |f: &dyn Fn(&Answer, &Score) -> bool| in_scope.iter().filter(|(_, a, s)| f(a, s)).count();
    let n = in_scope.len();
    let has = |a: &Answer, f: &dyn Fn(&Grounding) -> bool| a.grounding.as_ref().is_some_and(f);
    println!("\nIn scope ({n} questions):");
    println!("- answered: {}/{n}", count(&|a, _| a.status == Status::Answered));
    println!("- with sources: {}/{n}", count(&|a, _| !a.sources.is_empty()));
    println!("- with quotes: {}/{n}", count(&|a, _| !a.quotes.is_empty()));
    println!("- every quote word for word in its chunk: {}/{n}", count(&|a, _| has(a, &|g| g.quotes_verified)));
    println!("- meaning matches the quotes: {}/{n}", count(&|a, _| has(a, &|g| g.grounded)));
    let (found, total, quoted) = in_scope
        .iter()
        .fold((0, 0, 0), |(f, t, q), (_, _, s)| (f + s.found, t + s.total, q + s.quoted));
    println!("- facts stated: {found}/{total}, of them quoted: {quoted}");
    println!("- answered on the first attempt: {}/{n}", count(&|a, _| a.status == Status::Answered && a.attempts == 1));
    let m = out_scope.len();
    println!("\nOut of scope ({m} questions):");
    println!(
        "- \"I don't know\" with a clarification request: {}/{m}",
        out_scope
            .iter()
            .filter(|(_, a, s)| s.status_ok && a.clarification.is_some())
            .count()
    );
    println!(
        "- without a model call: {}/{m}",
        out_scope.iter().filter(|(_, a, _)| a.attempts == 0).count()
    );

    for (case, a, _) in &results {
        println!("\n## {} {}\nexpected: {}\n\n{}", case.id, case.question, case.expect, a.text);
        if let Some(e) = &a.error {
            println!("error: {e}");
        }
        for s in &a.sources {
            println!("  source [{}] {} — {} ({}, score {:.2})", s.n, s.source, s.section, s.chunk_id, s.score);
        }
        for q in &a.quotes {
            println!("  quote [{}] {}: \"{}\"", q.n, q.chunk_id, q.text);
        }
        if let Some(j) = a.grounding.as_ref().and_then(|g| g.judge.as_ref()) {
            for u in &j.unsupported {
                println!("  not backed by a quote: {u}");
            }
        }
        for (i, errors) in a.rejected.iter().enumerate() {
            println!("  attempt {} rejected: {}", i + 1, errors.join("; "));
        }
        if a.status == Status::Unknown {
            for h in &a.hints {
                println!("  nearest: {h}");
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

    // `app eval [nojudge]`: run the control questions, print the table, exit.
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("eval") {
        let settings = Settings {
            judge: args.next().as_deref() != Some("nojudge"),
            ..Settings::default()
        };
        run_eval(&agent, settings).await;
        return;
    }

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(AppState { agent })).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn quotes_are_matched_word_for_word() {
        let text = "Port is always `4000 + NN`\n(lesson 6 → `4006`). Service name is always\n`ai-advent-lesson-NN`.";
        assert!(quote_in("Port is always 4000 + NN", text));
        assert!(quote_in("port is   ALWAYS `4000 + NN`.", text));
        assert!(quote_in("Service name is always ai-advent-lesson-NN", text));
        assert!(quote_in("“Port is always 4000 + NN”", text));
        assert!(quote_in("Port is always … Service name is always", text));
        assert!(quote_in("Port is always... ai-advent-lesson-NN", text));
        assert!(!quote_in("Port is always 8080", text));
        assert!(!quote_in("The port is 4000 + NN", text), "a paraphrase is not a quote");
        assert!(!quote_in("Service name is always … Port is always", text), "parts out of order");
        assert!(!quote_in("...", text));

        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert_eq!(
            strip_citations("Port 4000 [1]. Unit [2, 3], see [docs] [x]."),
            "Port 4000 . Unit , see [docs] [x]."
        );
        assert_eq!(
            identifiers(&strip_citations("Port 4000 [1] and ai-advent-lesson-NN via a-b, x86_64 or tools/list.")),
            vec!["4000", "ai-advent-lesson-NN", "x86_64", "tools/list"]
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

    async fn deploy_hits() -> Vec<Hit> {
        let index = Index::build(builtin_documents(), index::local()).await.unwrap();
        let hits = index
            .search("loginctl enable-linger user services after the SSH session", 3)
            .await
            .unwrap();
        assert_eq!(hits.len(), 3);
        hits
    }

    #[tokio::test]
    async fn replies_are_checked_against_their_fragments() {
        let context = deploy_hits().await;
        let q1 = quotable(&context[0].chunk.text).unwrap();
        let q2 = quotable(&context[1].chunk.text).unwrap();

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

        // A code fence, no status, and numbers as strings are all fine.
        let fenced = json!({ "answer": "X [2]", "sources": ["[2]"], "quotes": [{ "fragment": "2", "text": q2 }] });
        match check_reply(&format!("```json\n{fenced}\n```"), &context) {
            Ok(Parsed::Answered(v)) => assert_eq!(v.sources, vec![2]),
            other => panic!("{other:?}"),
        }

        let unknown = json!({ "status": "unknown", "clarification": " Which lesson? " });
        assert_eq!(
            check_reply(&unknown.to_string(), &context),
            Ok(Parsed::Unknown { clarification: "Which lesson?".to_string() })
        );

        let has = |errors: &[String], what: &str| errors.iter().any(|e| e.contains(what));
        let errors = check_reply("Port 4000.", &context).unwrap_err();
        assert!(has(&errors, "JSON"), "{errors:?}");

        let errors = check_reply(&json!({ "answer": "X" }).to_string(), &context).unwrap_err();
        assert!(has(&errors, "no sources") && has(&errors, "no quotes"), "{errors:?}");

        let bad = json!({
            "answer": "X [4]",
            "sources": [1, 9],
            "quotes": [
                { "fragment": 1, "text": "Every lesson runs as a Kubernetes pod" },
                { "fragment": 2, "text": "short" },
                { "fragment": 7, "text": q1 },
            ],
        });
        let errors = check_reply(&bad.to_string(), &context).unwrap_err();
        for what in [
            "source 9 is not a fragment number",
            "quote 1 is not word for word in fragment [1]",
            "quote 2 is too short",
            "quote 2 is from fragment [2], which isn't in sources",
            "quote 3 names fragment 7",
            "the answer cites [4]",
        ] {
            assert!(has(&errors, what), "{what}: {errors:?}");
        }
        assert!(retry_message(&errors).contains("quote 2 is too short"));

        let unquoted = json!({
            "answer": "X [1] [2]",
            "sources": [1, 2],
            "quotes": [{ "fragment": 1, "text": q1 }],
        });
        let errors = check_reply(&unquoted.to_string(), &context).unwrap_err();
        assert_eq!(errors, vec!["source [2] has no quote".to_string()]);
    }

    #[test]
    fn dont_know_judgements_and_settings() {
        assert!(is_russian("Почему?") && !is_russian("Why?"));
        for reason in [UnknownReason::LowRelevance, UnknownReason::ModelUnsure, UnknownReason::Unverified] {
            let en = dont_know("Why?", reason, Some(0.21), 0.3).to_lowercase();
            let ru = dont_know("Почему?", reason, Some(0.21), 0.3).to_lowercase();
            assert!(en.contains("i don't know"), "{en}");
            assert!(ru.contains("не знаю"), "{ru}");
            assert!(REFUSAL.iter().any(|r| en.contains(r)) && REFUSAL.iter().any(|r| ru.contains(r)));
        }
        let low = dont_know("Why?", UnknownReason::LowRelevance, Some(0.21), 0.3);
        assert!(low.contains("0.21") && low.contains("0.30"), "{low}");
        assert!(default_clarification("Почему?").starts_with("Уточните"));
        assert!(default_clarification("Why?").starts_with("Could you clarify"));

        let j = parse_judgement("```json\n{\"verdict\": \"Supported\", \"unsupported\": []}\n```").unwrap();
        assert_eq!(j.verdict, Verdict::Supported);
        let j = parse_judgement("{\"verdict\": \"partial\", \"unsupported\": [\"port 8080\"]}").unwrap();
        assert_eq!((j.verdict, j.unsupported.len()), (Verdict::Partial, 1));
        assert!(parse_judgement("{\"verdict\": \"maybe\"}").is_err());
        assert!(parse_judgement("no idea").is_err());

        let s = Settings { k_before: 3, k_after: 99, threshold: 7.0, judge: false }.clamped();
        assert_eq!((s.k_before, s.k_after, s.threshold, s.judge), (MAX_K_AFTER, MAX_K_AFTER, 1.0, false));
        let s = Settings { k_before: 0, k_after: 0, threshold: f32::NAN, judge: true }.clamped();
        assert_eq!((s.k_before, s.k_after, s.threshold), (1, 1, DEFAULT_THRESHOLD));
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
        let settings = Settings { k_after: 2, threshold: 0.3, ..Settings::default() };
        select(&mut cs, settings);
        let order: Vec<usize> = cs.iter().map(|c| c.cosine_rank).collect();
        assert_eq!(order, vec![2, 3, 4, 5, 1], "ties keep the cosine order");
        let kept: Vec<bool> = cs.iter().map(|c| c.kept).collect();
        assert_eq!(kept, vec![true, true, false, false, false]);
        assert_eq!(cs[2].dropped, Some("past top-K"));
        assert_eq!(cs[4].dropped, Some("below threshold"));
    }

    #[test]
    fn control_questions_are_grounded_in_the_corpus() {
        assert_eq!(CASES.len(), 14);
        assert_eq!(CASES.iter().filter(|c| c.in_scope()).count(), 10);
        let ids: Vec<&str> = CASES.iter().map(|c| c.id).collect();
        assert!(ids.iter().all(|id| ids.iter().filter(|x| *x == id).count() == 1));
        let index_docs: Vec<(&str, &str)> = index::BUILTIN.to_vec();
        for case in CASES {
            assert!(!case.facts.is_empty(), "{}", case.id);
            let question = case.question.to_lowercase();
            for variants in case.facts {
                assert!(
                    !variants.iter().any(|v| question.contains(&v.to_lowercase())),
                    "{}: {variants:?} is given away by the question",
                    case.id
                );
            }
            if !case.in_scope() {
                assert_eq!(case.facts, &[REFUSAL], "{}", case.id);
                continue;
            }
            let mut text = String::new();
            for (file, section) in case.sources {
                let (_, body) = index_docs
                    .iter()
                    .find(|(path, _)| path == file)
                    .unwrap_or_else(|| panic!("{}: no file {file}", case.id));
                let heading = section.rsplit(" › ").next().unwrap();
                assert!(
                    body.lines()
                        .any(|l| l.starts_with('#') && l.trim_start_matches('#').trim().starts_with(heading)),
                    "{}: no heading {heading:?} in {file}",
                    case.id
                );
                text.push_str(&body.to_lowercase());
            }
            for variants in case.facts {
                assert!(
                    variants.iter().any(|v| text.contains(&v.to_lowercase())),
                    "{}: {variants:?} is not in its sources",
                    case.id
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // A fake model.
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
    }

    /// Request bodies the fake model has seen.
    type Seen = Arc<Mutex<Vec<Value>>>;

    #[derive(Clone)]
    struct Fake {
        seen: Seen,
        behaviour: Behaviour,
    }

    /// The fragments of an answer prompt, `[1]` first: the lines under each
    /// `[n] file — section` header.
    fn fragments(user: &str) -> Vec<String> {
        let context = user.strip_prefix("Context:\n\n").unwrap_or(user);
        let context = context.rfind("\n\nQuestion: ").map_or(context, |i| &context[..i]);
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
        let reply = json!({
            "status": "answered",
            "answer": answer,
            "sources": quotes.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            "quotes": quotes.iter().map(|(n, q)| json!({ "fragment": n, "text": q })).collect::<Vec<_>>(),
        });
        format!("```json\n{reply}\n```")
    }

    /// "supported" when the answer is nothing but its quotes.
    fn judge_reply(user: &str) -> String {
        let (answer, quotes) = user.split_once("\n\nQuotes:\n").unwrap();
        let answer = answer.split_once("Answer:\n").unwrap().1;
        let mut rest = normalize_text(&strip_citations(answer));
        for line in quotes.lines() {
            let quote = line
                .split_once("] \"")
                .map_or(line, |(_, q)| q.strip_suffix('"').unwrap_or(q));
            rest = rest.replacen(&normalize_text(quote), "", 1);
        }
        let verdict = if rest.chars().any(char::is_alphanumeric) {
            "unsupported"
        } else {
            "supported"
        };
        json!({ "verdict": verdict, "unsupported": [] }).to_string()
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
        let system = messages[0]["content"].as_str().unwrap_or_default();
        let user = messages[1]["content"].as_str().unwrap_or_default();
        let content = if system == JUDGE_PROMPT {
            judge_reply(user)
        } else {
            match fake.behaviour {
                Behaviour::Good => good_reply(user),
                Behaviour::Fixer if messages.len() > 2 => good_reply(user),
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
            "usage": { "prompt_tokens": user.len() / 4, "completion_tokens": 9 },
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

    #[tokio::test]
    async fn every_answer_has_sources_and_verified_quotes_or_says_it_does_not_know() {
        let (agent, seen) = fake_agent(Behaviour::Good).await;
        let settings = Settings::default();
        for case in CASES {
            let a = agent.ask(case.question, settings).await;
            let s = score(case, &a);
            assert!(a.error.is_none(), "{}: {:?}", case.id, a.error);
            assert!(s.status_ok, "{}: {:?} {} {:?}", case.id, a.status, a.text, a.rejected);
            if case.in_scope() {
                assert_eq!(a.status, Status::Answered);
                assert_eq!(a.attempts, 1, "{}: {:?}", case.id, a.rejected);
                assert!(!a.sources.is_empty() && !a.quotes.is_empty(), "{}", case.id);
                assert!(a.context.iter().all(|h| h.score >= DEFAULT_THRESHOLD));
                let g = a.grounding.as_ref().unwrap();
                assert!(
                    g.has_sources && g.has_quotes && g.quotes_verified && g.grounded,
                    "{}: {g:?}",
                    case.id
                );
                assert_eq!(g.judge.as_ref().map(|j| j.verdict), Some(Verdict::Supported), "{}", case.id);
                assert!(g.unsupported_terms.is_empty(), "{}: {g:?}", case.id);
                assert!(g.support >= MIN_SUPPORT, "{}: {g:?}", case.id);
                // Every source and quote points at a chunk of the prompt by
                // its number, chunk id, file and section.
                for src in &a.sources {
                    let h = &a.context[src.n - 1];
                    assert_eq!(
                        (&h.chunk.chunk_id, &h.chunk.source, &h.chunk.section),
                        (&src.chunk_id, &src.source, &src.section)
                    );
                }
                for q in &a.quotes {
                    assert!(a.sources.iter().any(|s| s.n == q.n && s.chunk_id == q.chunk_id));
                    assert!(quote_in(&q.text, &a.context[q.n - 1].chunk.text));
                }
                for n in citations(&a.text) {
                    assert!(a.sources.iter().any(|s| s.n == n), "{}: [{n}]", case.id);
                }
            } else {
                // The relevance gate: "I don't know" and a clarification
                // request, with no model call.
                assert_eq!(a.status, Status::Unknown);
                assert_eq!(a.unknown_reason, Some(UnknownReason::LowRelevance));
                assert_eq!(a.attempts, 0);
                assert!(a.messages.is_empty() && a.context.is_empty());
                assert!(a.sources.is_empty() && a.quotes.is_empty() && a.grounding.is_none());
                assert!(a.best_score.unwrap() < DEFAULT_THRESHOLD, "{}: {:?}", case.id, a.best_score);
                assert_eq!(s.found, 1, "{}: {}", case.id, a.text);
                let clarification = a.clarification.as_deref().unwrap();
                assert!(a.text.ends_with(clarification));
                assert_eq!(a.hints.len(), HINTS);
            }
        }
        // Only in-scope questions reached the model: an answer and a judge
        // call each.
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2 * CASES.iter().filter(|c| c.in_scope()).count());
        assert!(bodies.iter().all(|b| b["model"] == "test-model"
            && b["temperature"] == 0.0
            && b["response_format"]["type"] == "json_object"));
        let system = bodies[0]["messages"][0]["content"].as_str().unwrap();
        assert_eq!(system, format!("{BASE_PROMPT}\n\n{ANSWER_RULES}"));
    }

    #[tokio::test]
    async fn a_rejected_reply_is_retried_and_an_unverifiable_one_becomes_dont_know() {
        let question = CASES[0].question;
        let settings = Settings::default();

        // No quotes at first: the retry is told so, and fixes it.
        let (agent, _) = fake_agent(Behaviour::Fixer).await;
        let a = agent.ask(question, settings).await;
        assert_eq!(a.status, Status::Answered, "{a:?}");
        assert_eq!((a.attempts, a.rejected.len(), a.replies.len()), (2, 1, 2));
        assert!(a.rejected[0].iter().any(|e| e.contains("no quotes")), "{:?}", a.rejected);
        assert_eq!(a.messages.len(), 4);
        assert!(a.messages[3]["content"].as_str().unwrap().contains("rejected"));
        assert!(a.grounding.as_ref().unwrap().quotes_verified);

        // A made-up quote, twice: not shown, "I don't know".
        let (agent, seen) = fake_agent(Behaviour::Liar).await;
        let a = agent.ask(question, settings).await;
        assert_eq!(a.status, Status::Unknown);
        assert_eq!(a.unknown_reason, Some(UnknownReason::Unverified));
        assert_eq!((a.attempts, a.rejected.len()), (MAX_ATTEMPTS, MAX_ATTEMPTS));
        assert!(a.rejected.iter().all(|e| e.iter().any(|e| e.contains("not word for word"))));
        assert!(a.sources.is_empty() && a.quotes.is_empty() && a.grounding.is_none());
        assert!(!a.text.contains("Kubernetes") && a.text.contains("don't know"), "{}", a.text);
        assert!(a.clarification.is_some());
        assert_eq!(seen.lock().unwrap().len(), MAX_ATTEMPTS, "no judge for an unverified answer");

        // The model itself says the fragments don't answer it.
        let (agent, _) = fake_agent(Behaviour::Unsure).await;
        let a = agent.ask(question, settings).await;
        assert_eq!(a.status, Status::Unknown);
        assert_eq!(a.unknown_reason, Some(UnknownReason::ModelUnsure));
        assert_eq!(a.attempts, 1);
        assert_eq!(a.clarification.as_deref(), Some("Which lesson do you mean?"));
        assert!(a.text.starts_with("I don't know") && a.text.ends_with("Which lesson do you mean?"));

        // Without the judge, the deterministic check decides.
        let (agent, seen) = fake_agent(Behaviour::Good).await;
        let a = agent.ask(question, Settings { judge: false, ..settings }).await;
        let g = a.grounding.unwrap();
        assert!(g.judge.is_none() && g.grounded, "{g:?}");
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn http_api_returns_answers_with_sources_and_quotes() {
        let (agent, _) = fake_agent(Behaviour::Good).await;
        let url = spawn(app(AppState { agent: Arc::new(agent) })).await;
        let http = reqwest::Client::new();

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>Citations and Sources</title>"));

        let cfg: Value = http.get(format!("{url}/api/config")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cfg["model"], "test-model");
        assert_eq!(cfg["defaults"]["k_after"], DEFAULT_K_AFTER);
        assert_eq!(cfg["defaults"]["judge"], true);

        let cases: Value = http.get(format!("{url}/api/cases")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cases.as_array().unwrap().len(), CASES.len());
        assert_eq!(cases[10]["in_scope"], false);

        let post = |body: Value| {
            let http = http.clone();
            let url = url.clone();
            async move {
                http.post(format!("{url}/api/ask"))
                    .json(&body)
                    .send()
                    .await
                    .unwrap()
                    .json::<Value>()
                    .await
                    .unwrap()
            }
        };

        let q = post(json!({ "case_id": "q03" })).await;
        assert!(q["error"].is_null(), "{q}");
        let a = &q["answer"];
        assert_eq!(a["status"], "answered");
        let source = &a["sources"][0];
        for field in ["chunk_id", "source", "section"] {
            assert!(source[field].as_str().is_some_and(|s| !s.is_empty()), "{source}");
        }
        assert!(a["quotes"][0]["text"].as_str().is_some_and(|s| !s.is_empty()));
        assert_eq!(a["grounding"]["quotes_verified"], true);
        assert_eq!(a["grounding"]["judge"]["verdict"], "supported");
        assert_eq!(q["score"]["status_ok"], true);

        let n = post(json!({ "case_id": "n03" })).await;
        assert_eq!(n["answer"]["status"], "unknown");
        assert_eq!(n["answer"]["unknown_reason"], "low_relevance");
        assert!(n["answer"]["text"].as_str().unwrap().starts_with("Не знаю"));
        assert!(n["answer"]["clarification"].as_str().unwrap().starts_with("Уточните"));
        assert_eq!(n["score"]["status_ok"], true);

        // A stricter gate turns the same in-scope question into "don't know".
        let strict = post(json!({ "case_id": "q03", "threshold": 0.99 })).await;
        assert_eq!(strict["answer"]["status"], "unknown", "{}", strict["answer"]["best_score"]);
        assert_eq!(strict["score"]["status_ok"], false);

        let free = post(json!({ "question": "What does lesson 05 compare?", "k_after": 99, "judge": false })).await;
        assert_eq!(free["settings"]["k_after"], MAX_K_AFTER);
        assert_eq!(free["settings"]["judge"], false);
        assert!(free["case"].is_null() && free["score"].is_null());

        let empty = post(json!({ "question": "  " })).await;
        assert!(empty["error"].as_str().unwrap().contains("question"));
        let unknown = post(json!({ "case_id": "q99" })).await;
        assert!(unknown["error"].as_str().unwrap().contains("q99"));
    }
}

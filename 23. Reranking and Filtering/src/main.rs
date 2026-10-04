mod index;
mod rerank;

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

use index::{ErrorResponse, Hit, Index, builtin_documents, embedder_from_env, env_non_empty};
use rerank::Heuristic;

/// Stage 1: how many chunks the vector search hands to the reranker.
const DEFAULT_K_BEFORE: usize = 20;
const MAX_K_BEFORE: usize = 40;
/// Stage 2: at most this many chunks go into the prompt. Basic mode takes
/// the same number straight from the vector search, so every mode has the
/// same context budget.
const DEFAULT_K_AFTER: usize = 5;
const MAX_K_AFTER: usize = 10;
/// A chunk that scores lower is dropped, even if that leaves fewer than
/// `k_after`, or none. Calibrated for the heuristic reranker and the local
/// embedder: on the control set the weakest relevant section kept scores
/// 0.31 and the best out-of-scope candidate 0.27.
const DEFAULT_THRESHOLD: f32 = 0.30;
const MAX_QUESTION_CHARS: usize = 1000;
/// The LLM reranker reads this much of each candidate.
const RERANK_PREVIEW_CHARS: usize = 700;

/// The system prompt of every answer; same as lesson 22.
const BASE_PROMPT: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. Answer \
concisely, in the language of the question. If you don't know something, say so \
instead of guessing.";

const RAG_RULES: &str = "The user message contains numbered fragments of the \
repository's documents, then the question. Answer only from these fragments. \
After each fact, cite the fragment it came from as [1], [2] and so on. If the \
fragments don't contain the answer, say that the documents don't cover it.";

/// Improved mode: the question becomes a search query first. The list of
/// documents is appended, so the model can name the right lesson.
const REWRITE_PROMPT: &str = "You turn a user's question into a search query \
for a keyword and embedding search over the documents of the GitHub repository \
CaptainDmitro/AI-Advent-Challenge. The documents are in English, so write the \
query in English whatever the language of the question. Keep every name, number \
and identifier from the question. If it's clear which lesson or document the \
question is about, name it. Add the technical terms the answer is likely to use. \
Don't answer the question, and don't add topics it doesn't ask about. Reply with \
the query only, on one line.";

/// The LLM reranker: one call scores every candidate.
const RERANK_PROMPT: &str = "You rate how relevant fragments of a repository's \
documentation are to a question. Score each fragment from 0 to 10: 10 means it \
directly contains the answer, 5 that it's on the topic but doesn't answer it, 0 \
that it's unrelated. Reply with JSON only, {\"scores\": [s1, s2, ...]}: one \
integer per fragment, in the order given.";

/// What the agent says, without asking the model, when no chunk passed the
/// filter: nothing in the documents, so nothing to ground an answer on.
const NOT_COVERED_EN: &str = "The documents don't cover this question: no fragment passed the relevance filter.";
const NOT_COVERED_RU: &str = "В документах нет ответа на этот вопрос: ни один фрагмент не прошёл фильтр релевантности.";

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
    /// One completion at temperature 0, so the modes are compared on what
    /// they retrieve, not on sampling luck.
    async fn complete(&self, messages: &[Value]) -> Result<(String, Usage), String> {
        let body = json!({ "model": self.model, "messages": messages, "temperature": 0.0 });
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
// The pipeline: three modes, from lesson 22's RAG to the improved one.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// Lesson 22: question → top `k_after` by cosine → LLM.
    Basic,
    /// question → top `k_before` by cosine → rerank → threshold → top
    /// `k_after` → LLM.
    Filtered,
    /// question → rewrite → the same as filtered, searching with the rewrite.
    Improved,
}

const MODES: [Mode; 3] = [Mode::Basic, Mode::Filtered, Mode::Improved];

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Basic => "basic",
            Mode::Filtered => "filtered",
            Mode::Improved => "improved",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum RerankerKind {
    /// In-process: cosine + keyword coverage + heading match.
    Heuristic,
    /// One extra LLM call that scores every candidate 0–10.
    Llm,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq)]
struct Settings {
    k_before: usize,
    k_after: usize,
    threshold: f32,
    reranker: RerankerKind,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            k_before: DEFAULT_K_BEFORE,
            k_after: DEFAULT_K_AFTER,
            threshold: DEFAULT_THRESHOLD,
            reranker: RerankerKind::Heuristic,
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
            reranker: self.reranker,
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
    /// Where the vector search put it, and its cosine similarity.
    cosine_rank: usize,
    cosine: f32,
    /// The heuristic's signals; computed for the LLM reranker too, to compare.
    coverage: f32,
    heading: f32,
    /// The LLM reranker's score, 0..1.
    llm: Option<f32>,
    /// What the order and the threshold use: the heuristic's score, or the
    /// LLM's.
    score: f32,
    /// Position after reranking.
    rank: usize,
    kept: bool,
    /// Why it isn't in the prompt: `below threshold` or `past top-K`.
    dropped: Option<&'static str>,
}

#[derive(Serialize, Clone, Copy, Debug, Default)]
struct Timings {
    rewrite_ms: f64,
    retrieve_ms: f64,
    rerank_ms: f64,
    llm_ms: f64,
}

#[derive(Serialize, Clone, Debug)]
struct Answer {
    mode: Mode,
    text: String,
    /// What the vector search ran with: the question, or its rewrite.
    query: String,
    rewritten: bool,
    /// Exactly what was sent to the model for the answer.
    messages: Vec<Value>,
    /// Filtered and improved: every candidate with its scores, best first.
    candidates: Vec<Candidate>,
    /// The chunks in the prompt, `[1]` first.
    sources: Vec<Hit>,
    /// Set when the LLM reranker failed and the heuristic was used instead.
    note: Option<String>,
    /// No chunk passed the filter, so the model wasn't asked.
    skipped_llm: bool,
    /// Every call of this answer: rewrite, rerank and the answer itself.
    usage: Usage,
    timings: Timings,
    error: Option<String>,
}

impl Answer {
    fn start(mode: Mode, question: &str) -> Answer {
        Answer {
            mode,
            text: String::new(),
            query: question.to_string(),
            rewritten: false,
            messages: Vec::new(),
            candidates: Vec::new(),
            sources: Vec::new(),
            note: None,
            skipped_llm: false,
            usage: Usage::default(),
            timings: Timings::default(),
            error: None,
        }
    }
}

struct Agent {
    llm: Llm,
    index: Index,
    heuristic: Heuristic,
    /// `REWRITE_PROMPT` and the list of documents.
    rewrite_prompt: String,
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

fn is_russian(text: &str) -> bool {
    text.chars()
        .any(|c| matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё'))
}

fn not_covered(question: &str) -> &'static str {
    if is_russian(question) {
        NOT_COVERED_RU
    } else {
        NOT_COVERED_EN
    }
}

/// The model's rewrite as a one-line query: the first non-empty line, without
/// a `Query:` label or quotes. `None` if nothing is left.
fn clean_rewrite(text: &str) -> Option<String> {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let line = ["Query:", "query:", "Search query:"]
        .iter()
        .find_map(|label| line.strip_prefix(label))
        .unwrap_or(line);
    let line = line
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '`' | '«' | '»' | '\''))
        .trim();
    (!line.is_empty()).then(|| line.to_string())
}

#[derive(Deserialize)]
struct RerankReply {
    scores: Vec<f32>,
}

/// `{"scores": [...]}` from the LLM reranker, possibly in a code fence, as
/// 0..1 scores. Exactly one score per fragment, or it's an error.
fn parse_scores(text: &str, n: usize) -> Result<Vec<f32>, String> {
    let (start, end) = match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if start < end => (start, end),
        _ => return Err("the reply has no JSON object".to_string()),
    };
    let reply: RerankReply = serde_json::from_str(&text[start..=end])
        .map_err(|e| format!("the reply isn't {{\"scores\": [...]}}: {e}"))?;
    if reply.scores.len() != n {
        return Err(format!(
            "{} scores for {n} fragments",
            reply.scores.len()
        ));
    }
    Ok(reply
        .scores
        .into_iter()
        .map(|s| round3((s / 10.0).clamp(0.0, 1.0)))
        .collect())
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

/// The same prompt in every mode: only the fragments differ.
fn build_messages(question: &str, sources: &[Hit]) -> Vec<Value> {
    vec![
        json!({ "role": "system", "content": format!("{BASE_PROMPT}\n\n{RAG_RULES}") }),
        json!({
            "role": "user",
            "content": format!("Context:\n\n{}\n\nQuestion: {question}", context_block(sources)),
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

impl Agent {
    fn new(llm: Llm, index: Index) -> Agent {
        let heuristic = Heuristic::new(&index.chunks);
        let documents = index
            .titles
            .iter()
            .map(|(source, title)| {
                if source == title {
                    format!("- {source}")
                } else {
                    format!("- {source}: {title}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        Agent {
            rewrite_prompt: format!("{REWRITE_PROMPT}\n\nThe documents:\n{documents}"),
            llm,
            index,
            heuristic,
        }
    }

    /// The question as a search query. A blank reply keeps the question.
    async fn rewrite(&self, question: &str) -> Result<(String, Usage), String> {
        let messages = vec![
            json!({ "role": "system", "content": self.rewrite_prompt }),
            json!({ "role": "user", "content": question }),
        ];
        let (text, usage) = self.llm.complete(&messages).await?;
        let query = clean_rewrite(&text).unwrap_or_else(|| question.to_string());
        Ok((query, usage))
    }

    /// Every hit with the heuristic's signals; not yet ordered.
    fn rerank_heuristic(&self, query: &str, hits: &[Hit]) -> Vec<Candidate> {
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
                    llm: None,
                    score: round3(s.score),
                    rank: 0,
                    kept: false,
                    dropped: None,
                }
            })
            .collect()
    }

    /// One call scores every candidate against the original question.
    async fn rerank_llm(&self, question: &str, hits: &[Hit]) -> Result<(Vec<f32>, Usage), String> {
        let fragments = hits
            .iter()
            .enumerate()
            .map(|(i, h)| {
                format!(
                    "[{}] {} — {}\n{}",
                    i + 1,
                    h.chunk.source,
                    h.chunk.section,
                    preview(&h.chunk.text, RERANK_PREVIEW_CHARS)
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let messages = vec![
            json!({ "role": "system", "content": RERANK_PROMPT }),
            json!({
                "role": "user",
                "content": format!("Question: {question}\n\nFragments:\n\n{fragments}"),
            }),
        ];
        let (text, usage) = self.llm.complete(&messages).await?;
        Ok((parse_scores(&text, hits.len())?, usage))
    }

    /// The whole pipeline for one mode.
    async fn ask(&self, question: &str, mode: Mode, settings: Settings) -> Answer {
        let mut answer = Answer::start(mode, question);
        let mut usage = Usage::default();

        // Query rewrite.
        if mode == Mode::Improved {
            let started = Instant::now();
            let result = self.rewrite(question).await;
            answer.timings.rewrite_ms = elapsed_ms(started);
            match result {
                Ok((query, u)) => {
                    answer.rewritten = query != question;
                    answer.query = query;
                    usage += u;
                }
                Err(e) => {
                    answer.error = Some(format!("Query rewrite: {e}"));
                    return answer;
                }
            }
        }

        // Stage 1: the vector search.
        let k = match mode {
            Mode::Basic => settings.k_after,
            Mode::Filtered | Mode::Improved => settings.k_before,
        };
        let started = Instant::now();
        let hits = self.index.search(&answer.query, k).await;
        answer.timings.retrieve_ms = elapsed_ms(started);
        let hits = match hits {
            Ok(hits) => hits,
            Err(e) => {
                answer.error = Some(e);
                answer.usage = usage;
                return answer;
            }
        };

        // Stage 2: rerank, then the threshold and top-K.
        if mode == Mode::Basic {
            answer.sources = hits;
        } else {
            let started = Instant::now();
            let mut candidates = self.rerank_heuristic(&answer.query, &hits);
            if settings.reranker == RerankerKind::Llm {
                match self.rerank_llm(question, &hits).await {
                    Ok((scores, u)) => {
                        usage += u;
                        for (c, s) in candidates.iter_mut().zip(scores) {
                            c.llm = Some(s);
                            c.score = s;
                        }
                    }
                    Err(e) => {
                        answer.note = Some(format!(
                            "The LLM reranker failed ({e}), so the heuristic's scores were used."
                        ));
                    }
                }
            }
            select(&mut candidates, settings);
            answer.timings.rerank_ms = elapsed_ms(started);
            answer.sources = kept_hits(&candidates, &hits);
            answer.candidates = candidates;
        }

        // Nothing relevant: say so instead of letting the model improvise.
        if answer.sources.is_empty() {
            answer.skipped_llm = true;
            answer.text = not_covered(question).to_string();
            answer.usage = usage;
            return answer;
        }

        // Stage 3: the answer.
        answer.messages = build_messages(question, &answer.sources);
        let started = Instant::now();
        let result = self.llm.complete(&answer.messages).await;
        answer.timings.llm_ms = elapsed_ms(started);
        match result {
            Ok((text, u)) => {
                answer.text = text;
                usage += u;
            }
            Err(e) => answer.error = Some(e),
        }
        answer.usage = usage;
        answer
    }
}

// ---------------------------------------------------------------------------
// Control questions: lesson 22's ten, plus questions that need a rewrite and
// questions the documents don't answer at all.
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
    /// cover the question, and the right answer is to say so.
    sources: &'static [(&'static str, &'static str)],
}

/// The fact of an out-of-scope question: the answer says the documents don't
/// cover it.
const REFUSAL: &[&str] = &[
    "don't cover",
    "do not cover",
    "doesn't cover",
    "does not cover",
    "not covered",
    "no information",
    "не содерж",
    "нет информации",
    "нет ответа",
    "не описан",
    "не охватыва",
    "don't know",
    "не знаю",
];

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
    // The documents are in English: these need the rewrite to find anything.
    Case {
        id: "r01",
        question: "На каком порту и под каким именем сервиса работает задеплоенный урок?",
        expect: "Порт 4000 + номер урока (урок 6 → 4006), сервис ai-advent-lesson-NN — пользовательский systemd-юнит.",
        facts: &[&["4000"], &["ai-advent-lesson-"]],
        sources: &[DEPLOY_WHERE],
    },
    Case {
        id: "r02",
        question: "Почему не работает деплой четвёртого урока?",
        expect: "Уроку 04 нужны переменные MEDIUM_* и STRONG_* (и необязательные WEAK_*), которых пайплайн не передаёт; это оставлено сознательно.",
        facts: &[&["medium_"], &["strong_"]],
        sources: &[DEPLOY_SECRETS, AGENTS_CI],
    },
    Case {
        id: "r03",
        question: "Что нужно один раз сделать на сервере, чтобы сервисы не падали после выхода из SSH?",
        expect: "Один раз выполнить sudo loginctl enable-linger <ssh-user>; без этого пользовательские сервисы останавливаются, когда закрывается SSH-сессия.",
        facts: &[&["enable-linger"]],
        sources: &[DEPLOY_WHERE],
    },
    Case {
        id: "v01",
        question: "why no docker?",
        expect: "Lessons ship as a fully static x86_64-unknown-linux-musl binary: no glibc dependency, no OpenSSL (rustls), nothing to install on the VDS.",
        facts: &[&["glibc"], &["openssl", "rustls"]],
        sources: &[DEPLOY_MUSL],
    },
    // Out of scope: the right answer is "the documents don't cover it".
    Case {
        id: "n01",
        question: "What is the capital of Australia?",
        expect: "Not in the documents: the answer should say so.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n02",
        question: "How does this repository run database migrations with Diesel and PostgreSQL?",
        expect: "No lesson uses a database: the answer should say the documents don't cover it.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n03",
        question: "Какая погода будет завтра в Москве?",
        expect: "Не про репозиторий: ответ должен сказать, что в документах этого нет.",
        facts: &[REFUSAL],
        sources: &[],
    },
    Case {
        id: "n04",
        question: "How do I configure a Kubernetes ingress for the lessons?",
        expect: "Lessons run as systemd user services, with no Kubernetes: the answer should say the documents don't cover it.",
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
    sources: Vec<SourceRef>,
}

#[derive(Serialize, Clone, Debug)]
struct SourceRef {
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
                .map(|(file, section)| SourceRef {
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
    found: bool,
}

#[derive(Serialize, Clone, Debug)]
struct Score {
    facts: Vec<FactCheck>,
    found: usize,
    total: usize,
    /// found / total.
    fact_score: f64,
    /// In scope: a chunk of an expected file is in the prompt.
    source_retrieved: Option<bool>,
    /// In scope: a chunk of the expected section itself is in the prompt.
    section_retrieved: Option<bool>,
    /// In scope: the share of the prompt's chunks that come from an expected
    /// file. `None` when the prompt has none.
    precision: Option<f64>,
    /// Chunks in the prompt, and their tokens. Out of scope, 0 is right.
    kept: usize,
    context_tokens: usize,
    /// The fragment numbers the answer cites.
    cited: Vec<usize>,
    /// In scope: one of the cited fragments is from an expected file.
    cited_expected: Option<bool>,
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

fn score(case: &Case, answer: &Answer) -> Score {
    let text = answer.text.to_lowercase();
    let facts: Vec<FactCheck> = case
        .facts
        .iter()
        .map(|variants| FactCheck {
            fact: variants[0],
            found: variants.iter().any(|v| text.contains(&v.to_lowercase())),
        })
        .collect();
    let found = facts.iter().filter(|f| f.found).count();
    let total = facts.len();
    let in_scope = case.in_scope();
    let expected_file = |h: &Hit| case.sources.iter().any(|(file, _)| h.chunk.source == *file);
    let kept = answer.sources.len();
    let from_expected = answer.sources.iter().filter(|h| expected_file(h)).count();
    let cited = citations(&answer.text);
    Score {
        found,
        total,
        fact_score: if total == 0 { 0.0 } else { found as f64 / total as f64 },
        facts,
        source_retrieved: in_scope.then_some(from_expected > 0),
        section_retrieved: in_scope.then(|| {
            answer.sources.iter().any(|h| {
                case.sources.iter().any(|(file, section)| {
                    h.chunk.source == *file && h.chunk.section.starts_with(section)
                })
            })
        }),
        precision: (in_scope && kept > 0).then(|| from_expected as f64 / kept as f64),
        kept,
        context_tokens: answer.sources.iter().map(|h| h.chunk.tokens).sum(),
        cited_expected: in_scope.then(|| {
            answer
                .sources
                .iter()
                .any(|h| cited.contains(&h.rank) && expected_file(h))
        }),
        cited,
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
        "weights": {
            "cosine": rerank::W_COSINE,
            "coverage": rerank::W_COVERAGE,
            "heading": rerank::W_HEADING,
            "cosine_full": rerank::COSINE_FULL,
        },
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
    /// A control question: its text is asked, and the answers are scored.
    case_id: Option<String>,
    /// Which modes to run; all three when empty.
    modes: Vec<Mode>,
    k_before: Option<usize>,
    k_after: Option<usize>,
    threshold: Option<f32>,
    reranker: Option<RerankerKind>,
}

impl AskRequest {
    fn settings(&self) -> Settings {
        let d = Settings::default();
        Settings {
            k_before: self.k_before.unwrap_or(d.k_before),
            k_after: self.k_after.unwrap_or(d.k_after),
            threshold: self.threshold.unwrap_or(d.threshold),
            reranker: self.reranker.unwrap_or(d.reranker),
        }
        .clamped()
    }
}

#[derive(Serialize, Clone, Debug)]
struct Graded {
    #[serde(flatten)]
    answer: Answer,
    score: Option<Score>,
}

#[derive(Serialize, Clone, Debug, Default)]
struct AskResponse {
    question: String,
    case: Option<CaseInfo>,
    settings: Settings,
    answers: Vec<Graded>,
    error: Option<String>,
}

/// The requested modes side by side, in the order basic, filtered, improved.
async fn ask(
    agent: &Agent,
    question: &str,
    case: Option<&Case>,
    modes: &[Mode],
    settings: Settings,
) -> Vec<Graded> {
    let run = move |mode: Mode| async move {
        if modes.contains(&mode) {
            Some(agent.ask(question, mode, settings).await)
        } else {
            None
        }
    };
    let (basic, filtered, improved) = tokio::join!(
        run(Mode::Basic),
        run(Mode::Filtered),
        run(Mode::Improved)
    );
    [basic, filtered, improved]
        .into_iter()
        .flatten()
        .map(|answer| Graded {
            score: case.map(|c| score(c, &answer)),
            answer,
        })
        .collect()
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
    let modes: &[Mode] = if req.modes.is_empty() { &MODES } else { &req.modes };
    let answers = ask(&app.agent, &question, case, modes, settings).await;
    Json(AskResponse {
        question,
        case: case.map(Case::info),
        settings,
        answers,
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
// `app eval`: every control question in all three modes, as a table.
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Totals {
    facts: f64,
    sections: usize,
    refused: usize,
    kept: usize,
    tokens: usize,
}

fn cell(graded: &Graded) -> String {
    let a = &graded.answer;
    let s = graded.score.as_ref().expect("control questions are scored");
    if let Some(e) = &a.error {
        return format!("error: {}", preview(e, 60));
    }
    let section = match s.section_retrieved {
        Some(true) => "§✓",
        Some(false) => "§✗",
        None => "–",
    };
    format!("{}/{} · {section} · {} chunks", s.found, s.total, s.kept)
}

/// One summary row: a cell per mode.
fn row(totals: &[Totals; 3], f: impl Fn(&Totals) -> String) -> String {
    totals.iter().map(f).collect::<Vec<_>>().join(" | ")
}

async fn run_eval(agent: &Agent, settings: Settings) {
    println!(
        "top-{} by cosine → {:?} reranker → threshold {:.2} → top-{}\n",
        settings.k_before, settings.reranker, settings.threshold, settings.k_after
    );
    println!("| # | question | basic | filtered | improved |");
    println!("|---|---|---|---|---|");
    let mut totals = [Totals::default(); 3];
    let mut details = Vec::new();
    for case in CASES {
        let graded = ask(agent, case.question, Some(case), &MODES, settings).await;
        for (t, g) in totals.iter_mut().zip(&graded) {
            let s = g.score.as_ref().expect("control questions are scored");
            t.facts += s.fact_score;
            t.sections += usize::from(s.section_retrieved == Some(true));
            t.refused += usize::from(!case.in_scope() && s.found == s.total);
            t.kept += s.kept;
            t.tokens += s.context_tokens;
        }
        let cells: Vec<String> = graded.iter().map(cell).collect();
        println!("| {} | {} | {} |", case.id, case.question, cells.join(" | "));
        details.push((case, graded));
    }
    let n = CASES.len();
    let in_scope = CASES.iter().filter(|c| c.in_scope()).count();
    println!(
        "| | **facts stated** | {} |",
        row(&totals, |t| format!("**{:.0}%**", t.facts / n as f64 * 100.0))
    );
    println!(
        "| | **expected section in the prompt** | {} |",
        row(&totals, |t| format!("{}/{in_scope}", t.sections))
    );
    println!(
        "| | **out of scope, said so** | {} |",
        row(&totals, |t| format!("{}/{}", t.refused, n - in_scope))
    );
    println!(
        "| | **chunks / tokens per prompt** | {} |",
        row(&totals, |t| {
            format!("{:.1} / {:.0}", t.kept as f64 / n as f64, t.tokens as f64 / n as f64)
        })
    );
    for (case, graded) in details {
        println!("\n## {} {}\nexpected: {}", case.id, case.question, case.expect);
        for g in graded {
            let a = &g.answer;
            let rewrite = if a.rewritten {
                format!(" (searched for: {})", a.query)
            } else {
                String::new()
            };
            match &a.error {
                Some(e) => println!("\n### {}: error: {e}", a.mode.label()),
                None => println!("\n### {}{rewrite}\n{}", a.mode.label(), a.text),
            }
            for h in &a.sources {
                println!("  [{}] {:.3} {} — {}", h.rank, h.score, h.chunk.source, h.chunk.section);
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

    // `app eval [llm]`: run the control questions, print the comparison, exit.
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("eval") {
        let reranker = match args.next().as_deref() {
            Some("llm") => RerankerKind::Llm,
            _ => RerankerKind::Heuristic,
        };
        let settings = Settings {
            reranker,
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
    fn citations_rewrites_and_rerank_scores_are_parsed() {
        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert!(citations("no brackets").is_empty());

        assert_eq!(
            clean_rewrite("\n  Query: \"deploy port systemd\"\nsecond line").as_deref(),
            Some("deploy port systemd")
        );
        assert_eq!(clean_rewrite("«порт урока»").as_deref(), Some("порт урока"));
        assert_eq!(clean_rewrite("  \n \"\" "), None);

        assert_eq!(
            parse_scores("```json\n{\"scores\": [10, 3, 0, 12]}\n```", 4).unwrap(),
            vec![1.0, 0.3, 0.0, 1.0]
        );
        assert!(parse_scores("{\"scores\": [1, 2]}", 3).unwrap_err().contains("2 scores for 3"));
        assert!(parse_scores("no idea", 1).is_err());
        assert!(parse_scores("{\"other\": 1}", 1).is_err());

        assert!(is_russian("Почему?") && !is_russian("Why?"));
        assert_eq!(not_covered("Почему?"), NOT_COVERED_RU);
        assert_eq!(not_covered("Why?"), NOT_COVERED_EN);
    }

    #[test]
    fn settings_are_clamped() {
        let s = Settings {
            k_before: 3,
            k_after: 99,
            threshold: 7.0,
            reranker: RerankerKind::Llm,
        }
        .clamped();
        assert_eq!((s.k_before, s.k_after, s.threshold), (MAX_K_AFTER, MAX_K_AFTER, 1.0));
        let s = Settings {
            k_before: 0,
            k_after: 0,
            threshold: f32::NAN,
            reranker: RerankerKind::Heuristic,
        }
        .clamped();
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
            llm: None,
            score,
            rank: 0,
            kept: false,
            dropped: None,
        };
        let mut cs = vec![c(1, 0.2), c(2, 0.9), c(3, 0.5), c(4, 0.5), c(5, 0.4)];
        let settings = Settings {
            k_after: 2,
            threshold: 0.3,
            ..Settings::default()
        };
        select(&mut cs, settings);
        let order: Vec<usize> = cs.iter().map(|c| c.cosine_rank).collect();
        assert_eq!(order, vec![2, 3, 4, 5, 1], "ties keep the cosine order");
        assert_eq!(cs.iter().map(|c| c.rank).collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
        let kept: Vec<bool> = cs.iter().map(|c| c.kept).collect();
        assert_eq!(kept, vec![true, true, false, false, false]);
        assert_eq!(cs[2].dropped, Some("past top-K"));
        assert_eq!(cs[3].dropped, Some("past top-K"));
        assert_eq!(cs[4].dropped, Some("below threshold"));
    }

    #[test]
    fn control_questions_are_grounded_in_the_corpus() {
        assert_eq!(CASES.len(), 18);
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
        // The deterministic "not covered" answers count as refusals.
        for text in [NOT_COVERED_EN, NOT_COVERED_RU] {
            assert!(REFUSAL.iter().any(|v| text.to_lowercase().contains(v)), "{text}");
        }
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// What a good rewriter would answer for the questions that need one. Any
    /// other question comes back unchanged.
    const REWRITES: &[(&str, &str)] = &[
        (
            "На каком порту и под каким именем сервиса работает задеплоенный урок?",
            "deployed lesson port number and systemd service name on the VDS",
        ),
        (
            "Почему не работает деплой четвёртого урока?",
            "why does the deploy of lesson 04 fail, missing secrets and environment variables",
        ),
        (
            "Что нужно один раз сделать на сервере, чтобы сервисы не падали после выхода из SSH?",
            "one-time command on the VDS so user systemd services keep running after the SSH session closes (loginctl linger)",
        ),
        (
            "why no docker?",
            "why are lessons deployed as a static musl binary instead of a Docker container",
        ),
        ("What is the capital of Australia?", "capital city of Australia"),
        (
            "How does this repository run database migrations with Diesel and PostgreSQL?",
            "database migrations Diesel PostgreSQL schema",
        ),
        ("Какая погода будет завтра в Москве?", "weather forecast Moscow tomorrow"),
        (
            "How do I configure a Kubernetes ingress for the lessons?",
            "Kubernetes ingress configuration for deployed lessons",
        ),
    ];

    /// Request bodies the fake model has seen.
    type Seen = Arc<Mutex<Vec<Value>>>;

    /// A fake `/chat/completions` that knows nothing about the repo and
    /// plays three roles, told apart by the system prompt:
    /// - rewriter: the `REWRITES` table;
    /// - reranker: 9 for a fragment from DEPLOYMENT.md, 1 for anything else;
    /// - answerer: echoes the context back (a reader that trusts whatever
    ///   it's given, so a noisy context shows in the answer).
    async fn fake_chat(
        State(seen): State<Seen>,
        headers: axum::http::HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        seen.lock().unwrap().push(body.clone());
        let system = body["messages"][0]["content"].as_str().unwrap_or_default();
        let user = body["messages"][1]["content"].as_str().unwrap_or_default();
        let content = if system.starts_with(REWRITE_PROMPT) {
            REWRITES
                .iter()
                .find(|(q, _)| *q == user)
                .map_or(user.to_string(), |(_, rewrite)| format!("Query: {rewrite}"))
        } else if system == RERANK_PROMPT {
            let scores: Vec<u32> = user
                .lines()
                .filter(|l| {
                    l.strip_prefix('[')
                        .and_then(|rest| rest.split_once("] "))
                        .is_some_and(|(n, _)| n.parse::<usize>().is_ok())
                })
                .map(|l| if l.contains("] DEPLOYMENT.md — ") { 9 } else { 1 })
                .collect();
            json!({ "scores": scores }).to_string()
        } else {
            match user.strip_prefix("Context:") {
                Some(context) => format!("From the documents:{context}"),
                None => "I don't know the details of this repository.".to_string(),
            }
        };
        Json(json!({
            "choices": [{ "message": { "role": "assistant", "content": content } }],
            "usage": { "prompt_tokens": user.len() / 4, "completion_tokens": 9 },
        }))
    }

    async fn test_agent(base: &str, path: &str) -> Agent {
        Agent::new(
            Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{base}{path}"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            },
            Index::build(builtin_documents(), index::local()).await.unwrap(),
        )
    }

    async fn fake_model() -> (String, Seen) {
        let seen: Seen = Arc::default();
        let base = spawn(
            Router::new()
                .route("/v1/chat/completions", post(fake_chat))
                .with_state(seen.clone()),
        )
        .await;
        (base, seen)
    }

    #[derive(Default, Debug)]
    struct Sum {
        facts: f64,
        sections: usize,
        precision: f64,
        precision_n: usize,
        kept: usize,
        tokens: usize,
    }

    #[tokio::test]
    async fn filtering_and_rewriting_beat_basic_rag_on_the_control_set() {
        let (base, seen) = fake_model().await;
        let agent = test_agent(&base, "/v1/chat/completions").await;
        let settings = Settings::default();

        let mut sums: [Sum; 3] = Default::default();
        for case in CASES {
            let graded = ask(&agent, case.question, Some(case), &MODES, settings).await;
            assert_eq!(graded.len(), 3);
            let modes: Vec<Mode> = graded.iter().map(|g| g.answer.mode).collect();
            assert_eq!(modes, MODES);
            for (sum, g) in sums.iter_mut().zip(&graded) {
                let (a, s) = (&g.answer, g.score.as_ref().unwrap());
                assert!(a.error.is_none() && a.note.is_none(), "{} {:?}", case.id, a.mode);
                sum.facts += s.fact_score;
                sum.sections += usize::from(s.section_retrieved == Some(true));
                if let Some(p) = s.precision {
                    sum.precision += p;
                    sum.precision_n += 1;
                }
                sum.kept += s.kept;
                sum.tokens += s.context_tokens;
            }
            let (basic, filtered, improved) = (&graded[0], &graded[1], &graded[2]);

            // Basic is lesson 22: always k_after chunks, never reranked.
            assert_eq!(basic.answer.sources.len(), DEFAULT_K_AFTER, "{}", case.id);
            assert!(basic.answer.candidates.is_empty() && !basic.answer.rewritten);
            assert_eq!(filtered.answer.query, case.question);

            // Filtered and improved: k_before candidates, best first, and the
            // prompt holds exactly the kept ones, renumbered.
            for g in [filtered, improved] {
                let a = &g.answer;
                assert_eq!(a.candidates.len(), DEFAULT_K_BEFORE, "{}", case.id);
                assert!(a.candidates.windows(2).all(|w| w[0].score >= w[1].score));
                let kept: Vec<&Candidate> = a.candidates.iter().filter(|c| c.kept).collect();
                assert!(kept.len() <= DEFAULT_K_AFTER);
                assert!(kept.iter().all(|c| c.score >= DEFAULT_THRESHOLD));
                assert_eq!(kept.len(), a.sources.len());
                for (c, h) in kept.iter().zip(&a.sources) {
                    assert_eq!(c.chunk_id, h.chunk.chunk_id);
                }
                assert_eq!(
                    a.sources.iter().map(|h| h.rank).collect::<Vec<_>>(),
                    (1..=a.sources.len()).collect::<Vec<_>>()
                );
            }

            if case.in_scope() {
                assert_eq!(
                    improved.score.as_ref().unwrap().source_retrieved,
                    Some(true),
                    "{}: {:?}",
                    case.id,
                    improved.answer.query
                );
            } else {
                // Nothing passes the filter, so the model isn't asked and the
                // answer says the documents don't cover it. Basic hands the
                // model 5 unrelated chunks and it repeats them.
                for g in [filtered, improved] {
                    assert!(g.answer.sources.is_empty(), "{}: {:?}", case.id, g.answer.candidates.first());
                    assert!(g.answer.skipped_llm && g.answer.messages.is_empty());
                    assert_eq!(g.score.as_ref().unwrap().found, 1, "{}", case.id);
                }
                assert_eq!(basic.score.as_ref().unwrap().found, 0, "{}", case.id);
            }
            if case.id.starts_with('r') {
                // Russian against English documents: nothing without a rewrite.
                assert!(filtered.answer.sources.is_empty(), "{}", case.id);
                assert!(improved.answer.rewritten, "{}", case.id);
                assert!(improved.answer.usage.prompt_tokens > 0);
            }
        }

        let n = CASES.len() as f64;
        let [basic, filtered, improved] = &sums;
        let facts = |s: &Sum| s.facts / n;
        let precision = |s: &Sum| s.precision / s.precision_n as f64;
        let summary = format!("basic {basic:?}\nfiltered {filtered:?}\nimproved {improved:?}");
        assert!(facts(filtered) > facts(basic), "{summary}");
        assert!(facts(improved) > facts(filtered), "{summary}");
        assert!(facts(improved) >= 0.95, "{summary}");
        assert!(improved.sections > basic.sections, "{summary}");
        assert!(precision(improved) > precision(basic), "{summary}");
        assert!(improved.kept < basic.kept && improved.tokens < basic.tokens, "{summary}");

        // The answer prompt: the same rules in every mode, each kept chunk
        // under its `[n] file — section` header, the question last.
        let probe = agent.ask(CASES[0].question, Mode::Improved, settings).await;
        let system = probe.messages[0]["content"].as_str().unwrap();
        assert_eq!(system, format!("{BASE_PROMPT}\n\n{RAG_RULES}"));
        let user = probe.messages[1]["content"].as_str().unwrap();
        for h in &probe.sources {
            assert!(user.contains(&format!("[{}] {} — {}", h.rank, h.chunk.source, h.chunk.section)));
        }
        assert!(user.ends_with(&format!("Question: {}", CASES[0].question)));

        let bodies = seen.lock().unwrap();
        assert!(bodies.iter().all(|b| b["model"] == "test-model" && b["temperature"] == 0.0));
    }

    #[tokio::test]
    async fn llm_reranker_scores_every_candidate_and_its_scores_decide() {
        let (base, _) = fake_model().await;
        let agent = test_agent(&base, "/v1/chat/completions").await;
        let settings = Settings {
            reranker: RerankerKind::Llm,
            ..Settings::default()
        };
        let a = agent.ask(CASES[0].question, Mode::Filtered, settings).await;
        assert!(a.error.is_none() && a.note.is_none(), "{:?} {:?}", a.error, a.note);
        assert!(a.candidates.iter().all(|c| c.llm == Some(c.score)));
        assert!(!a.sources.is_empty());
        assert!(a.sources.iter().all(|h| h.chunk.source == "DEPLOYMENT.md"));
        assert!(a.usage.prompt_tokens > 0);

        // A reranker that can't be reached: the heuristic's scores are used,
        // and the note says so. (The answer itself fails the same way.)
        let broken = test_agent(&base, "/v1/nothing-here").await;
        let a = broken.ask(CASES[0].question, Mode::Filtered, settings).await;
        assert!(a.note.as_deref().unwrap().contains("heuristic"));
        assert!(a.candidates.iter().all(|c| c.llm.is_none()));
        assert!(!a.sources.is_empty());
        assert!(a.error.as_deref().unwrap().contains("404"));
    }

    #[tokio::test]
    async fn http_api_compares_the_modes_and_reports_errors() {
        let (base, _) = fake_model().await;
        let agent = Arc::new(test_agent(&base, "/v1/chat/completions").await);
        let url = spawn(app(AppState { agent })).await;
        let http = reqwest::Client::new();

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>Reranking and Filtering</title>"));

        let cfg: Value = http.get(format!("{url}/api/config")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cfg["model"], "test-model");
        assert_eq!(cfg["defaults"]["k_before"], DEFAULT_K_BEFORE);
        assert_eq!(cfg["defaults"]["reranker"], "heuristic");

        let cases: Value = http.get(format!("{url}/api/cases")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cases.as_array().unwrap().len(), CASES.len());
        assert_eq!(cases[14]["in_scope"], false);

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

        let all = post(json!({ "case_id": "q03" })).await;
        assert!(all["error"].is_null(), "{all}");
        let modes: Vec<&str> = all["answers"].as_array().unwrap().iter().map(|a| a["mode"].as_str().unwrap()).collect();
        assert_eq!(modes, vec!["basic", "filtered", "improved"]);
        let kept = |i: usize| all["answers"][i]["sources"].as_array().unwrap().len();
        assert!(kept(1) < kept(0), "the filter drops noise: {} vs {}", kept(1), kept(0));
        assert_eq!(all["answers"][1]["score"]["found"], 1);
        assert_eq!(all["answers"][1]["candidates"][0]["kept"], true);
        assert!(all["answers"][1]["candidates"][0]["coverage"].is_number());

        let ru = post(json!({ "case_id": "r03", "modes": ["improved"], "k_after": 99, "threshold": 0.2 })).await;
        assert_eq!(ru["settings"]["k_after"], MAX_K_AFTER);
        assert_eq!(ru["answers"].as_array().unwrap().len(), 1);
        assert_eq!(ru["answers"][0]["rewritten"], true);
        assert!(ru["answers"][0]["query"].as_str().unwrap().contains("linger"));

        let free = post(json!({ "question": "What does lesson 05 compare?", "modes": ["filtered"] })).await;
        assert!(free["case"].is_null());
        assert!(free["answers"][0]["score"].is_null());

        let off = post(json!({ "case_id": "n01", "modes": ["filtered"] })).await;
        assert_eq!(off["answers"][0]["skipped_llm"], true);
        assert_eq!(off["answers"][0]["text"], NOT_COVERED_EN);

        let empty = post(json!({ "question": "  " })).await;
        assert!(empty["error"].as_str().unwrap().contains("question"));
        let unknown = post(json!({ "case_id": "q99" })).await;
        assert!(unknown["error"].as_str().unwrap().contains("q99"));
    }
}

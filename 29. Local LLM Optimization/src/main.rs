// Lesson 28's index, used as is (not copied): the same corpus, chunks and
// local embedder, so retrieval is identical and only the model side changes.
#[path = "../../28. Local RAG/src/index.rs"]
mod index;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use index::{Hit, Index, builtin_documents, embedder_from_env, env_non_empty};

/// The same five chunks for every profile, so retrieval is not a variable.
const TOP_K: usize = 5;
const MAX_RUNS: usize = 5;
const MAX_QUESTION_CHARS: usize = 1000;
const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";

// ---------------------------------------------------------------------------
// Two prompt templates for the same task: answer from numbered fragments and
// cite them.
// ---------------------------------------------------------------------------

/// Lesson 28's prompt (lesson 22's, word for word): every rule in the system
/// prompt, the fragments and the question in one user message.
const BASE_PROMPT: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. Answer \
concisely, in the language of the question. If you don't know something, say so \
instead of guessing.";

const RAG_RULES: &str = "The user message contains numbered fragments of the \
repository's documents, then the question. Answer only from these fragments. \
After each fact, cite the fragment it came from as [1], [2] and so on. If the \
fragments don't contain the answer, say that the documents don't cover it.";

/// The tuned template, written for a 3B model: short numbered rules, an
/// output shape it can copy (bullets, each ending in `[n]`), and an explicit
/// "copy identifiers exactly" against the paraphrasing that cost lesson 28
/// its facts.
const TUNED_SYSTEM: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge, using only numbered fragments of its documents.\n\
Rules:\n\
1. Use only facts from the fragments. If they don't contain the answer, reply: \
The documents don't cover this.\n\
2. Answer in 1 to 5 bullet points of one short sentence each, with only what the \
question asks. End every bullet with the number of the fragment it comes from in \
square brackets, like [2].\n\
3. Copy names, numbers, commands, file names and identifiers exactly as the \
fragments write them. Do not paraphrase or translate them.\n\
4. Answer in the language of the question.";

/// One worked example (a made-up document, so it gives nothing away about the
/// repo): small models follow a shown format far better than a described one.
const EXAMPLE_CONTEXT: &str = "[1] docs/app.md — Storage › Cache\n\
The cache lives in `~/.cache/app`. It is never cleared automatically.\n\n\
[2] docs/app.md — Commands\n\
`app clean --all` deletes the cache and the logs.\n\n\
[3] docs/app.md — Logs\n\
Logs rotate every 7 days.";
const EXAMPLE_QUESTION: &str = "Where does the app keep its cache, and how is it cleared?";
const EXAMPLE_ANSWER: &str = "- The cache lives in `~/.cache/app` and is never cleared automatically [1].\n\
- `app clean --all` clears it, together with the logs [2].";
/// Repeated after the question: the last thing a small model reads is what
/// it follows.
const FORMAT_REMINDER: &str = "Answer in bullet points, each ending with its fragment \
number like [1]. Copy names and numbers exactly.";

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum PromptStyle {
    Lesson28,
    Tuned,
}

/// The retrieved chunks as numbered fragments, each headed by where it's
/// from, so the model can cite them and the reader can check them.
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

fn tuned_user(context: &str, question: &str) -> String {
    format!("Fragments:\n\n{context}\n\nQuestion: {question}\n\n{FORMAT_REMINDER}")
}

fn build_messages(style: PromptStyle, question: &str, sources: &[Hit]) -> Vec<Value> {
    let context = context_block(sources);
    match style {
        PromptStyle::Lesson28 => vec![
            json!({ "role": "system", "content": format!("{BASE_PROMPT}\n\n{RAG_RULES}") }),
            json!({
                "role": "user",
                "content": format!("Context:\n\n{context}\n\nQuestion: {question}"),
            }),
        ],
        PromptStyle::Tuned => vec![
            json!({ "role": "system", "content": TUNED_SYSTEM }),
            json!({ "role": "user", "content": tuned_user(EXAMPLE_CONTEXT, EXAMPLE_QUESTION) }),
            json!({ "role": "assistant", "content": EXAMPLE_ANSWER }),
            json!({ "role": "user", "content": tuned_user(&context, question) }),
        ],
    }
}

// ---------------------------------------------------------------------------
// Profiles: what is sent to Ollama, from lesson 28 as it was to the optimized
// one, one change at a time.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct Profile {
    id: &'static str,
    name: &'static str,
    /// What this profile changes against the one before it.
    change: &'static str,
    model: &'static str,
    prompt: PromptStyle,
    /// `None`: not sent, so Ollama uses its default (0.8).
    temperature: Option<f64>,
    seed: Option<u64>,
    num_ctx: u32,
    /// `None`: no cap on the answer's length.
    num_predict: Option<u32>,
}

const Q4: &str = "qwen2.5:3b";
const TUNED_CTX: u32 = 4096;
const TUNED_PREDICT: u32 = 300;
const SEED: u64 = 42;

const PROFILES: &[Profile] = &[
    Profile {
        id: "baseline",
        name: "Baseline (lesson 28)",
        change: "Lesson 28's request as it was: its prompt, no temperature (Ollama's default \
                 0.8), num_ctx 8192, no cap on the answer.",
        model: Q4,
        prompt: PromptStyle::Lesson28,
        temperature: None,
        seed: None,
        num_ctx: 8192,
        num_predict: None,
    },
    Profile {
        id: "params",
        name: "Tuned parameters",
        change: "Same prompt. Temperature 0 and a fixed seed for repeatable answers; num_ctx \
                 4096 (the longest prompt is ≈2.5k tokens) for a smaller KV cache; \
                 num_predict 300 against runaway answers.",
        model: Q4,
        prompt: PromptStyle::Lesson28,
        temperature: Some(0.0),
        seed: Some(SEED),
        num_ctx: TUNED_CTX,
        num_predict: Some(TUNED_PREDICT),
    },
    Profile {
        id: "optimized",
        name: "Tuned parameters + prompt",
        change: "The tuned template: numbered rules, one worked example, bullets that each \
                 end in [n], \"copy identifiers exactly\", and a format reminder after the \
                 question.",
        model: Q4,
        prompt: PromptStyle::Tuned,
        temperature: Some(0.0),
        seed: Some(SEED),
        num_ctx: TUNED_CTX,
        num_predict: Some(TUNED_PREDICT),
    },
    Profile {
        id: "q8",
        name: "Optimized, Q8_0",
        change: "The optimized profile on the 8-bit quantization of the same model \
                 (default qwen2.5:3b is Q4_K_M).",
        model: "qwen2.5:3b-instruct-q8_0",
        prompt: PromptStyle::Tuned,
        temperature: Some(0.0),
        seed: Some(SEED),
        num_ctx: TUNED_CTX,
        num_predict: Some(TUNED_PREDICT),
    },
    Profile {
        id: "q3",
        name: "Optimized, Q3_K_M",
        change: "The optimized profile on the 3-bit quantization: smaller and lighter, \
                 is it still good enough?",
        model: "qwen2.5:3b-instruct-q3_K_M",
        prompt: PromptStyle::Tuned,
        temperature: Some(0.0),
        seed: Some(SEED),
        num_ctx: TUNED_CTX,
        num_predict: Some(TUNED_PREDICT),
    },
];

fn find_profile(id: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|p| p.id == id)
}

/// Ollama's `options` for a profile. What's `None` isn't sent at all, so
/// Ollama's own default applies, as it did in lesson 28.
fn options(p: &Profile) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("num_ctx".into(), json!(p.num_ctx));
    if let Some(t) = p.temperature {
        o.insert("temperature".into(), json!(t));
    }
    if let Some(s) = p.seed {
        o.insert("seed".into(), json!(s));
    }
    if let Some(n) = p.num_predict {
        o.insert("num_predict".into(), json!(n));
    }
    Value::Object(o)
}

/// The profile's model, or another pulled one picked on the page.
fn model_for(profile: &Profile, model: Option<&str>) -> String {
    model
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or(profile.model)
        .to_string()
}

fn elapsed_ms(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 1000.0).round()
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn nanos_to_ms(nanos: u64) -> f64 {
    round1(nanos as f64 / 1e6)
}

/// Accepts the same shapes Ollama's own `OLLAMA_HOST` does: with or without a scheme.
fn normalize_host(host: &str) -> String {
    let host = host.trim().trim_end_matches('/');
    if host.is_empty() {
        DEFAULT_OLLAMA_HOST.to_string()
    } else if host.contains("://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

/// `llama3.2` and `llama3.2:latest` are the same model to Ollama.
fn same_model(a: &str, b: &str) -> bool {
    let tagged = |name: &str| {
        if name.rsplit('/').next().unwrap_or(name).contains(':') {
            name.to_string()
        } else {
            format!("{name}:latest")
        }
    };
    tagged(a) == tagged(b)
}

// ---------------------------------------------------------------------------
// Ollama: the native API, for its counters and its memory report.
// ---------------------------------------------------------------------------

struct Ollama {
    http: reqwest::Client,
    host: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
    /// `stop`, or `length` when `num_predict` cut the answer.
    #[serde(default)]
    done_reason: String,
    // Durations are in nanoseconds.
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    prompt_eval_duration: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    load_duration: u64,
}

#[derive(Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    content: String,
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
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    details: TagDetails,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct TagDetails {
    parameter_size: String,
    quantization_level: String,
}

/// A pulled model, from `/api/tags`.
#[derive(Serialize, Clone, Debug)]
struct PulledModel {
    name: String,
    /// The model file on disk, in bytes.
    size: u64,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<LoadedModel>,
}

/// A model in memory, from `/api/ps`: weights + KV cache + compute buffers.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
struct LoadedModel {
    name: String,
    #[serde(default)]
    size: u64,
    /// The part of `size` on the GPU (on Apple Silicon: all of it).
    #[serde(default)]
    size_vram: u64,
    /// The context window it was loaded with.
    #[serde(default)]
    context_length: u64,
}

#[derive(Serialize, Clone, Debug, Default)]
struct OllamaStatus {
    host: String,
    version: Option<String>,
    models: Vec<PulledModel>,
    loaded: Vec<LoadedModel>,
    error: Option<String>,
}

/// One generated answer.
struct Completion {
    text: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    gen_tokens_per_sec: f64,
    prefill_ms: f64,
    load_ms: f64,
    truncated: bool,
}

impl Ollama {
    async fn send(&self, request: reqwest::RequestBuilder) -> Result<String, String> {
        let response = request.send().await.map_err(|e| {
            format!(
                "Can't reach Ollama at {}: {e}. Is `ollama serve` running \
                 (or, on the VDS, is `tunnel.sh` up on the Mac)?",
                self.host
            )
        })?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<OllamaError>(&body)
                .map(|e| e.error)
                .unwrap_or(body);
            return Err(format!("Ollama error ({status}): {detail}"));
        }
        Ok(body)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let body = self
            .send(self.http.get(format!("{}{path}", self.host)))
            .await?;
        serde_json::from_str(&body).map_err(|e| format!("Failed to parse {path}: {e}"))
    }

    async fn chat_raw(&self, body: &Value) -> Result<String, String> {
        self.send(self.http.post(format!("{}/api/chat", self.host)).json(body))
            .await
    }

    async fn pulled(&self) -> Result<Vec<PulledModel>, String> {
        let tags = self.get_json::<TagsResponse>("/api/tags").await?;
        Ok(tags
            .models
            .into_iter()
            .map(|m| PulledModel {
                name: m.name,
                size: m.size,
                parameter_size: m.details.parameter_size,
                quantization_level: m.details.quantization_level,
            })
            .collect())
    }

    async fn loaded(&self) -> Result<Vec<LoadedModel>, String> {
        Ok(self.get_json::<PsResponse>("/api/ps").await?.models)
    }

    async fn status(&self) -> OllamaStatus {
        let mut status = OllamaStatus {
            host: self.host.clone(),
            ..OllamaStatus::default()
        };
        let result = async {
            let version = self.get_json::<VersionResponse>("/api/version").await?;
            Ok::<_, String>((version.version, self.pulled().await?, self.loaded().await?))
        }
        .await;
        match result {
            Ok((version, models, loaded)) => {
                status.version = Some(version);
                status.models = models;
                status.loaded = loaded;
            }
            Err(e) => status.error = Some(e),
        }
        status
    }

    /// An empty `messages` with `keep_alive: 0` unloads a model.
    async fn unload(&self, model: &str) -> Result<(), String> {
        self.chat_raw(&json!({ "model": model, "messages": [], "keep_alive": 0, "stream": false }))
            .await
            .map(|_| ())
    }

    /// An empty `messages` loads a model, here with the profile's context
    /// window, so the KV cache is allocated as it will be used.
    async fn load(&self, model: &str, num_ctx: u32) -> Result<(), String> {
        self.chat_raw(&json!({
            "model": model,
            "messages": [],
            "stream": false,
            "options": { "num_ctx": num_ctx },
        }))
        .await
        .map(|_| ())
    }

    async fn complete(
        &self,
        model: &str,
        messages: &[Value],
        options: &Value,
    ) -> Result<Completion, String> {
        let text = self
            .chat_raw(&json!({
                "model": model,
                "messages": messages,
                "stream": false,
                "options": options,
            }))
            .await?;
        let r = serde_json::from_str::<OllamaChatResponse>(&text)
            .map_err(|e| format!("Failed to parse Ollama's response: {e}"))?;
        let content = r.message.content.trim().to_string();
        if content.is_empty() {
            return Err("The model returned an empty answer.".to_string());
        }
        let eval_secs = r.eval_duration as f64 / 1e9;
        Ok(Completion {
            text: content,
            prompt_tokens: r.prompt_eval_count,
            completion_tokens: r.eval_count,
            gen_tokens_per_sec: if eval_secs > 0.0 {
                round1(r.eval_count as f64 / eval_secs)
            } else {
                0.0
            },
            prefill_ms: nanos_to_ms(r.prompt_eval_duration),
            load_ms: nanos_to_ms(r.load_duration),
            truncated: r.done_reason == "length",
        })
    }
}

// ---------------------------------------------------------------------------
// Control questions: lesson 22's ten (as in lesson 28), with what a correct
// answer says and where it comes from.
// ---------------------------------------------------------------------------

struct Case {
    id: &'static str,
    question: &'static str,
    /// What a correct answer has to say, in words.
    expect: &'static str,
    /// The same as checks. Each group is one fact; it's found if the answer
    /// contains any of its variants, ignoring case. None of them is a word of
    /// the question, so repeating the question scores nothing.
    facts: &'static [&'static [&'static str]],
    /// Where the answer is: `(file, section)`.
    sources: &'static [(&'static str, &'static str)],
}

const CASES: &[Case] = &[
    Case {
        id: "q01",
        question: "Which port and which systemd service name does a deployed lesson get on the VDS?",
        expect: "Port 4000 + the lesson number (lesson 6 → 4006); the service is ai-advent-lesson-NN, a user-level systemd unit.",
        facts: &[&["4000"], &["ai-advent-lesson-"]],
        sources: &[("DEPLOYMENT.md", "Deployment › Where things live on the VDS")],
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
        sources: &[("DEPLOYMENT.md", "Deployment › Why a static musl binary instead of Docker")],
    },
    Case {
        id: "q03",
        question: "What one-time command must be run on the VDS so the lesson services keep running after the SSH session closes?",
        expect: "sudo loginctl enable-linger <ssh-user>, once per VDS; without it user services stop when the SSH session that started them closes.",
        facts: &[&["enable-linger"]],
        sources: &[("DEPLOYMENT.md", "Deployment › Where things live on the VDS")],
    },
    Case {
        id: "q04",
        question: "Why does the deploy job of lesson 04 fail?",
        expect: "Lesson 04 needs MEDIUM_* and STRONG_* (and optionally WEAK_*) variables that the pipeline doesn't supply; it's left failing on purpose.",
        facts: &[&["medium_"], &["strong_"]],
        sources: &[
            ("DEPLOYMENT.md", "Deployment › Secrets & variables"),
            ("AGENTS.md", "Agent instructions for AI-Advent-Challenge › CI/CD pipeline"),
        ],
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
        sources: &[("AGENTS.md", "Agent instructions for AI-Advent-Challenge › CI/CD pipeline")],
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
        sources: &[("21. Document Indexing/README.md", "21. Document Indexing › What this is › Embeddings")],
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
];

fn find_case(id: &str) -> Option<&'static Case> {
    CASES.iter().find(|c| c.id == id)
}

#[derive(Serialize, Clone, Debug)]
struct CaseInfo {
    id: &'static str,
    question: &'static str,
    expect: &'static str,
    facts: Vec<Vec<&'static str>>,
    sources: Vec<SourceRef>,
}

#[derive(Serialize, Clone, Debug)]
struct SourceRef {
    file: &'static str,
    section: &'static str,
}

impl Case {
    fn info(&self) -> CaseInfo {
        CaseInfo {
            id: self.id,
            question: self.question,
            expect: self.expect,
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
    /// The fragment numbers the answer cites.
    cited: Vec<usize>,
    /// One of the cited fragments is from an expected file.
    cited_expected: bool,
    /// Cited numbers with no such fragment: a made-up source.
    invalid_cites: Vec<usize>,
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

fn expected_file(case: &Case, hit: &Hit) -> bool {
    case.sources.iter().any(|(file, _)| hit.chunk.source == *file)
}

fn score(case: &Case, text: &str, sources: &[Hit]) -> Score {
    let lower = text.to_lowercase();
    let facts: Vec<FactCheck> = case
        .facts
        .iter()
        .map(|variants| FactCheck {
            fact: variants[0],
            found: variants.iter().any(|v| lower.contains(&v.to_lowercase())),
        })
        .collect();
    let found = facts.iter().filter(|f| f.found).count();
    let total = facts.len();
    let cited = citations(text);
    Score {
        found,
        total,
        fact_score: if total == 0 { 0.0 } else { found as f64 / total as f64 },
        facts,
        cited_expected: sources
            .iter()
            .any(|h| cited.contains(&h.rank) && expected_file(case, h)),
        invalid_cites: cited
            .iter()
            .copied()
            .filter(|n| !sources.iter().any(|h| h.rank == *n))
            .collect(),
        cited,
    }
}

// ---------------------------------------------------------------------------
// Asking: one profile, one question, `runs` times.
// ---------------------------------------------------------------------------

/// One answer.
#[derive(Serialize, Clone, Debug, Default)]
struct Run {
    text: String,
    error: Option<String>,
    /// Wall-clock time of the call, as this server sees it.
    llm_ms: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    /// From Ollama's own counters: generation speed, the time to read the
    /// prompt (prefill) and the model load time.
    gen_tokens_per_sec: f64,
    prefill_ms: f64,
    load_ms: f64,
    /// `num_predict` cut the answer short.
    truncated: bool,
    score: Option<Score>,
}

/// How much the runs of one question agree with each other.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
struct Stability {
    runs: usize,
    errors: usize,
    /// Different texts among the successful runs (whitespace ignored). 1
    /// means the same answer every time.
    distinct_answers: usize,
    /// Control questions: the fact score over the successful runs.
    fact_min: Option<f64>,
    fact_max: Option<f64>,
    fact_mean: Option<f64>,
    ms_mean: f64,
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn stability(runs: &[Run]) -> Stability {
    let ok: Vec<&Run> = runs.iter().filter(|r| r.error.is_none()).collect();
    let mut texts: Vec<String> = ok
        .iter()
        .map(|r| r.text.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    texts.sort();
    texts.dedup();
    let facts: Vec<f64> = ok
        .iter()
        .filter_map(|r| r.score.as_ref().map(|s| s.fact_score))
        .collect();
    let ms: Vec<f64> = ok.iter().map(|r| r.llm_ms).collect();
    Stability {
        runs: runs.len(),
        errors: runs.len() - ok.len(),
        distinct_answers: texts.len(),
        fact_min: facts.iter().copied().reduce(f64::min),
        fact_max: facts.iter().copied().reduce(f64::max),
        fact_mean: mean(&facts),
        ms_mean: mean(&ms).map_or(0.0, f64::round),
    }
}

/// One question through one profile: the retrieval, exactly what was sent,
/// and every run.
#[derive(Serialize, Clone, Debug, Default)]
struct Answer {
    question: String,
    case: Option<CaseInfo>,
    profile: Option<Profile>,
    model: String,
    sources: Vec<Hit>,
    messages: Vec<Value>,
    options: Value,
    retrieve_ms: f64,
    /// Control questions: a chunk of an expected file is among the retrieved.
    source_retrieved: Option<bool>,
    runs: Vec<Run>,
    stability: Stability,
    error: Option<String>,
}

/// Cold-loading a profile's model: its load time and what it takes in memory.
#[derive(Serialize, Clone, Debug, Default)]
struct Warmup {
    profile: String,
    model: String,
    /// Wall-clock time of loading the model from nothing.
    load_ms: f64,
    /// What `/api/ps` reports once it's loaded.
    loaded: Option<LoadedModel>,
    /// The model file and its quantization, from `/api/tags`.
    disk_bytes: Option<u64>,
    quantization: Option<String>,
    error: Option<String>,
}

struct Agent {
    index: Index,
    ollama: Ollama,
}

impl Agent {
    /// question → the local index → top-5 chunks → the profile's prompt and
    /// options → Ollama, `runs` times, one after another.
    async fn ask(
        &self,
        question: &str,
        case: Option<&'static Case>,
        profile: &'static Profile,
        model: &str,
        runs: usize,
    ) -> Answer {
        let mut out = Answer {
            question: question.to_string(),
            case: case.map(Case::info),
            profile: Some(profile.clone()),
            model: model.to_string(),
            options: options(profile),
            ..Answer::default()
        };
        let started = Instant::now();
        let sources = self.index.search(question, TOP_K).await;
        out.retrieve_ms = elapsed_ms(started);
        let sources = match sources {
            Ok(sources) => sources,
            Err(e) => {
                out.error = Some(e);
                return out;
            }
        };
        out.messages = build_messages(profile.prompt, question, &sources);
        if let Some(case) = case {
            out.source_retrieved = Some(sources.iter().any(|h| expected_file(case, h)));
        }
        for _ in 0..runs {
            let started = Instant::now();
            let result = self
                .ollama
                .complete(model, &out.messages, &out.options)
                .await;
            let llm_ms = elapsed_ms(started);
            let mut run = match result {
                Ok(c) => Run {
                    text: c.text,
                    error: None,
                    llm_ms,
                    prompt_tokens: c.prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    gen_tokens_per_sec: c.gen_tokens_per_sec,
                    prefill_ms: c.prefill_ms,
                    load_ms: c.load_ms,
                    truncated: c.truncated,
                    score: None,
                },
                Err(e) => Run {
                    error: Some(e),
                    llm_ms,
                    ..Run::default()
                },
            };
            run.score = case.map(|c| score(c, &run.text, &sources));
            out.runs.push(run);
        }
        out.stability = stability(&out.runs);
        out.sources = sources;
        out
    }

    /// Unloads every model, then loads this one with the profile's
    /// `num_ctx`: the load time is a cold one, and the memory `/api/ps`
    /// reports is this model's alone.
    async fn warm_up(&self, profile: &'static Profile, model: &str) -> Warmup {
        let mut out = Warmup {
            profile: profile.id.to_string(),
            model: model.to_string(),
            ..Warmup::default()
        };
        let result = async {
            for m in self.ollama.loaded().await? {
                self.ollama.unload(&m.name).await?;
            }
            let started = Instant::now();
            self.ollama.load(model, profile.num_ctx).await?;
            let load_ms = elapsed_ms(started);
            Ok::<_, String>((load_ms, self.ollama.loaded().await?, self.ollama.pulled().await?))
        }
        .await;
        match result {
            Ok((load_ms, loaded, pulled)) => {
                out.load_ms = load_ms;
                out.loaded = loaded.into_iter().find(|m| same_model(&m.name, model));
                if let Some(p) = pulled.into_iter().find(|m| same_model(&m.name, model)) {
                    out.disk_bytes = Some(p.size);
                    out.quantization = Some(p.quantization_level);
                }
            }
            Err(e) => out.error = Some(e),
        }
        out
    }
}

/// One profile over the control questions, in numbers.
#[derive(Serialize, Debug, Default, PartialEq)]
struct Summary {
    questions: usize,
    answers: usize,
    errors: usize,
    fact_mean: Option<f64>,
    cited_expected: usize,
    cited_any: usize,
    invalid_citations: usize,
    truncated: usize,
    completion_tokens_mean: Option<f64>,
    prompt_tokens_max: u64,
    ms_mean: Option<f64>,
    gen_tokens_per_sec_mean: Option<f64>,
    prefill_ms_mean: Option<f64>,
    /// Questions answered with the same text on every run.
    same_answer: usize,
    /// Questions whose fact score changed between runs.
    varied_facts: usize,
}

fn summarize(answers: &[Answer]) -> Summary {
    let all: usize = answers.iter().map(|a| a.runs.len()).sum();
    let ok: Vec<&Run> = answers
        .iter()
        .flat_map(|a| &a.runs)
        .filter(|r| r.error.is_none())
        .collect();
    let scores: Vec<&Score> = ok.iter().filter_map(|r| r.score.as_ref()).collect();
    let of = |f: &dyn Fn(&Run) -> f64| mean(&ok.iter().map(|&r| f(r)).collect::<Vec<_>>());
    Summary {
        questions: answers.len(),
        answers: ok.len(),
        errors: all - ok.len(),
        fact_mean: mean(&scores.iter().map(|s| s.fact_score).collect::<Vec<_>>()),
        cited_expected: scores.iter().filter(|s| s.cited_expected).count(),
        cited_any: ok.iter().filter(|r| !citations(&r.text).is_empty()).count(),
        invalid_citations: scores.iter().map(|s| s.invalid_cites.len()).sum(),
        truncated: ok.iter().filter(|r| r.truncated).count(),
        completion_tokens_mean: of(&|r| r.completion_tokens as f64),
        prompt_tokens_max: ok.iter().map(|r| r.prompt_tokens).max().unwrap_or(0),
        ms_mean: of(&|r| r.llm_ms),
        gen_tokens_per_sec_mean: of(&|r| r.gen_tokens_per_sec),
        prefill_ms_mean: of(&|r| r.prefill_ms),
        same_answer: answers
            .iter()
            .filter(|a| !a.runs.is_empty() && a.stability.errors == 0 && a.stability.distinct_answers == 1)
            .count(),
        varied_facts: answers
            .iter()
            .filter(|a| a.stability.fact_min != a.stability.fact_max)
            .count(),
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
        "embedder": index.embedder.id(),
        "documents": index.documents,
        "chunks": index.chunks.len(),
        "top_k": TOP_K,
        "max_runs": MAX_RUNS,
        "profiles": PROFILES,
    }))
}

async fn status(State(app): State<AppState>) -> Json<OllamaStatus> {
    Json(app.agent.ollama.status().await)
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
    profile: String,
    /// Another pulled model instead of the profile's own.
    model: Option<String>,
    runs: Option<usize>,
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Answer> {
    let fail = |error: String| {
        Json(Answer {
            question: req.question.trim().to_string(),
            error: Some(error),
            ..Answer::default()
        })
    };
    let Some(profile) = find_profile(&req.profile) else {
        return fail(format!("No profile {:?}.", req.profile));
    };
    let case = match req.case_id.as_deref() {
        Some(id) => match find_case(id) {
            Some(case) => Some(case),
            None => return fail(format!("No control question {id:?}.")),
        },
        None => None,
    };
    let question = case.map_or(req.question.trim(), |c| c.question).to_string();
    if question.is_empty() {
        return fail("Type a question first.".to_string());
    }
    if question.chars().count() > MAX_QUESTION_CHARS {
        return fail(format!(
            "Keep the question under {MAX_QUESTION_CHARS} characters."
        ));
    }
    let model = model_for(profile, req.model.as_deref());
    let runs = req.runs.unwrap_or(1).clamp(1, MAX_RUNS);
    Json(app.agent.ask(&question, case, profile, &model, runs).await)
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct WarmupRequest {
    profile: String,
    model: Option<String>,
}

async fn warmup_api(State(app): State<AppState>, Json(req): Json<WarmupRequest>) -> Json<Warmup> {
    let Some(profile) = find_profile(&req.profile) else {
        return Json(Warmup {
            error: Some(format!("No profile {:?}.", req.profile)),
            profile: req.profile,
            ..Warmup::default()
        });
    };
    let model = model_for(profile, req.model.as_deref());
    Json(app.agent.warm_up(profile, &model).await)
}

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/api/config", get(config))
        .route("/api/status", get(status))
        .route("/api/cases", get(cases_api))
        .route("/api/ask", post(ask_api))
        .route("/api/warmup", post(warmup_api))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// `app eval`: every profile over every control question, as tables.
// ---------------------------------------------------------------------------

fn pct(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

fn secs(ms: f64) -> String {
    format!("{:.2} s", ms / 1000.0)
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn or_dash<T>(x: Option<T>, f: impl Fn(T) -> String) -> String {
    x.map_or("-".to_string(), f)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// One cell of the per-question table: facts, and whether it cites the source.
fn case_cell(a: &Answer) -> String {
    if let Some(e) = &a.error {
        return format!("error: {e}");
    }
    let s = &a.stability;
    let Some(mean) = s.fact_mean else {
        return format!("error ×{}", s.errors);
    };
    let total = a.runs[0].score.as_ref().map_or(0, |s| s.total) as f64;
    let ok = s.runs - s.errors;
    let cited = a
        .runs
        .iter()
        .filter(|r| r.error.is_none() && r.score.as_ref().is_some_and(|s| s.cited_expected))
        .count();
    let mut cell = format!("{:.1}/{total} · cited {cited}/{ok}", mean * total);
    if s.runs > 1 {
        cell.push_str(&format!(" · {} distinct", s.distinct_answers));
    }
    cell
}

async fn run_eval(agent: &Agent, profiles: &[&'static Profile], runs: usize) {
    let mut columns: Vec<(&'static Profile, Warmup, Vec<Answer>)> = Vec::new();
    for &p in profiles {
        eprintln!("== {} ({})", p.id, p.model);
        let warm = agent.warm_up(p, p.model).await;
        if let Some(e) = &warm.error {
            eprintln!("   warm-up: {e}");
        }
        let mut answers = Vec::new();
        for case in CASES {
            eprintln!("   {} {}", case.id, case.question);
            answers.push(agent.ask(case.question, Some(case), p, p.model, runs).await);
        }
        columns.push((p, warm, answers));
    }
    let sums: Vec<Summary> = columns.iter().map(|(_, _, a)| summarize(a)).collect();

    println!("{runs} run(s) per question, top-{TOP_K} chunks\n");
    let head: Vec<String> = columns.iter().map(|(p, _, _)| format!("`{}`", p.id)).collect();
    println!("| # | {} |", head.join(" | "));
    println!("|---|{}", "---|".repeat(columns.len()));
    for (i, case) in CASES.iter().enumerate() {
        let cells: Vec<String> = columns.iter().map(|(_, _, a)| case_cell(&a[i])).collect();
        println!("| {} | {} |", case.id, cells.join(" | "));
    }

    println!("\n| | {} |", head.join(" | "));
    println!("|---|{}", "---|".repeat(columns.len()));
    let row = |name: &str, cells: Vec<String>| println!("| {name} | {} |", cells.join(" | "));
    let by_profile = |f: &dyn Fn(&Profile, &Warmup, &Summary) -> String| -> Vec<String> {
        columns
            .iter()
            .zip(&sums)
            .map(|((p, w, _), s)| f(*p, w, s))
            .collect()
    };
    row(
        "model · quantization",
        by_profile(&|_, w, _| format!("`{}` {}", w.model, w.quantization.clone().unwrap_or_default())),
    );
    row(
        "prompt · temperature · num_ctx · num_predict",
        by_profile(&|p, _, _| {
            format!(
                "{:?} · {} · {} · {}",
                p.prompt,
                or_dash(p.temperature, |t| t.to_string()),
                p.num_ctx,
                or_dash(p.num_predict, |n| n.to_string())
            )
        }),
    );
    row("**quality**: expected facts stated", by_profile(&|_, _, s| or_dash(s.fact_mean, pct)));
    row(
        "answers citing the expected source",
        by_profile(&|_, _, s| format!("{}/{}", s.cited_expected, s.answers)),
    );
    row(
        "answers with any [n] citation",
        by_profile(&|_, _, s| format!("{}/{}", s.cited_any, s.answers)),
    );
    row(
        "citations of fragments that don't exist",
        by_profile(&|_, _, s| s.invalid_citations.to_string()),
    );
    row(
        "answers cut by num_predict",
        by_profile(&|_, _, s| format!("{}/{}", s.truncated, s.answers)),
    );
    row("**speed**: time per answer", by_profile(&|_, _, s| or_dash(s.ms_mean, secs)));
    row(
        "generation tok/s",
        by_profile(&|_, _, s| or_dash(s.gen_tokens_per_sec_mean, |x| format!("{x:.1}"))),
    );
    row("prefill", by_profile(&|_, _, s| or_dash(s.prefill_ms_mean, secs)));
    row(
        "answer length, tokens",
        by_profile(&|_, _, s| or_dash(s.completion_tokens_mean, |x| format!("{x:.0}"))),
    );
    row(
        "longest prompt, tokens (of num_ctx)",
        by_profile(&|p, _, s| format!("{} / {}", s.prompt_tokens_max, p.num_ctx)),
    );
    row("cold load", by_profile(&|_, w, _| secs(w.load_ms)));
    row(
        "**resources**: in memory (`ollama ps`)",
        by_profile(&|_, w, _| or_dash(w.loaded.as_ref(), |m| gb(m.size))),
    );
    row(
        "of it on the GPU",
        by_profile(&|_, w, _| or_dash(w.loaded.as_ref(), |m| gb(m.size_vram))),
    );
    row("model file", by_profile(&|_, w, _| or_dash(w.disk_bytes, gb)));
    row(
        "**stability**: failed calls",
        by_profile(&|_, _, s| format!("{}/{}", s.errors, s.answers + s.errors)),
    );
    row(
        "questions with one answer across runs",
        by_profile(&|_, _, s| format!("{}/{}", s.same_answer, s.questions)),
    );
    row(
        "questions whose fact score changed",
        by_profile(&|_, _, s| format!("{}/{}", s.varied_facts, s.questions)),
    );

    for (i, case) in CASES.iter().enumerate() {
        println!("\n## {} {}\nexpected: {}", case.id, case.question, case.expect);
        for (p, _, answers) in &columns {
            match answers[i].runs.first() {
                Some(run) if run.error.is_none() => {
                    println!("\n### {} ({})\n{}", p.id, secs(run.llm_ms), run.text)
                }
                Some(run) => println!("\n### {}: error: {}", p.id, run.error.as_deref().unwrap_or("")),
                None => println!("\n### {}: error: {}", p.id, answers[i].error.as_deref().unwrap_or("")),
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let host = normalize_host(&env_non_empty("OLLAMA_HOST").unwrap_or_default());
    let http = reqwest::Client::builder()
        // Generous: a cold 8-bit model on CPU can take minutes.
        .timeout(Duration::from_secs(600))
        .build()
        .expect("failed to build HTTP client");
    let embedder = embedder_from_env(&host).unwrap_or_else(|e| {
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
        "indexed {} documents into {} chunks ({}); Ollama at {host}",
        index.documents,
        index.chunks.len(),
        index.embedder.id()
    );
    let agent = Arc::new(Agent {
        index,
        ollama: Ollama { http, host },
    });

    // `app eval [--runs N] [--profiles baseline,optimized,...]`: run the
    // control questions through each profile, print the comparison, exit.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("eval") {
        let profiles: Vec<&'static Profile> = match flag(&args, "--profiles") {
            Some(list) => list
                .split(',')
                .map(|id| {
                    find_profile(id.trim()).unwrap_or_else(|| {
                        let ids: Vec<&str> = PROFILES.iter().map(|p| p.id).collect();
                        eprintln!("No profile {id:?}. Profiles: {}", ids.join(", "));
                        std::process::exit(2);
                    })
                })
                .collect(),
            None => PROFILES.iter().collect(),
        };
        let runs = flag(&args, "--runs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize)
            .clamp(1, MAX_RUNS);
        run_eval(&agent, &profiles, runs).await;
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
    use axum::http::StatusCode;
    use std::sync::Mutex;

    #[test]
    fn citations_are_parsed() {
        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert_eq!(citations("an array [x] or a[0] index [ 4 ]"), vec![0, 4]);
        assert!(citations("no brackets").is_empty());
    }

    #[test]
    fn normalizes_ollama_host_and_model_names_like_ollama_does() {
        assert_eq!(normalize_host(""), DEFAULT_OLLAMA_HOST);
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(normalize_host("http://gpu-box:11434/"), "http://gpu-box:11434");
        assert!(same_model("llama3.2", "llama3.2:latest"));
        assert!(same_model("qwen2.5:3b", "qwen2.5:3b"));
        assert!(!same_model("qwen2.5:3b", "qwen2.5:3b-instruct-q8_0"));
    }

    #[test]
    fn control_questions_are_grounded_in_the_corpus() {
        assert_eq!(CASES.len(), 10);
        for case in CASES {
            assert!(!case.facts.is_empty() && !case.sources.is_empty(), "{}", case.id);
            let question = case.question.to_lowercase();
            let mut text = String::new();
            for (file, section) in case.sources {
                let (_, body) = index::BUILTIN
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
                assert!(
                    !variants.iter().any(|v| question.contains(&v.to_lowercase())),
                    "{}: {variants:?} is given away by the question",
                    case.id
                );
            }
        }
    }

    /// The worked example is copied by the model. It must not contain any
    /// expected fact, or the tuned prompt would score by leaking answers.
    #[test]
    fn the_worked_example_gives_nothing_away() {
        let example = format!("{EXAMPLE_CONTEXT}\n{EXAMPLE_QUESTION}\n{EXAMPLE_ANSWER}\n{FORMAT_REMINDER}")
            .to_lowercase();
        for case in CASES {
            for variants in case.facts {
                for v in *variants {
                    assert!(!example.contains(&v.to_lowercase()), "{}: {v:?}", case.id);
                }
            }
        }
        // The example answer itself follows the format it asks for.
        assert!(EXAMPLE_ANSWER.lines().all(|l| l.starts_with("- ") && l.ends_with("].")));
    }

    #[test]
    fn profiles_change_one_thing_at_a_time() {
        let ids: Vec<&str> = PROFILES.iter().map(|p| p.id).collect();
        assert_eq!(ids, ["baseline", "params", "optimized", "q8", "q3"]);
        let [base, params, opt, q8, q3] = [0, 1, 2, 3, 4].map(|i| &PROFILES[i]);
        // Baseline is lesson 28: nothing but num_ctx 8192 is set.
        assert_eq!(options(base), json!({ "num_ctx": 8192 }));
        // Tuned parameters: same model and prompt, new options.
        assert_eq!((params.model, params.prompt), (base.model, base.prompt));
        assert_eq!(
            options(params),
            json!({ "num_ctx": 4096, "temperature": 0.0, "seed": 42, "num_predict": 300 })
        );
        // Then only the prompt, then only the model.
        assert_eq!((opt.model, options(opt)), (params.model, options(params)));
        assert_ne!(opt.prompt, params.prompt);
        for p in [q8, q3] {
            assert_eq!((p.prompt, options(p)), (opt.prompt, options(opt)));
            assert_ne!(p.model, opt.model);
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

    /// A host nothing listens on: bind a port, then let it go.
    fn closed_host() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    }

    /// What the fake Ollama has been asked, and what it holds in memory.
    #[derive(Clone, Default)]
    struct Fake {
        chats: Arc<Mutex<Vec<Value>>>,
        /// `load <model> <num_ctx>` / `unload <model>`, in order.
        log: Arc<Mutex<Vec<String>>>,
        /// `(model, num_ctx)` of every loaded model.
        loaded: Arc<Mutex<Vec<(String, u64)>>>,
    }

    /// A perfectly faithful reader that knows nothing about the repo: it
    /// echoes the fragments of the last user message back.
    fn echo(body: &Value) -> String {
        let user = body["messages"]
            .as_array()
            .and_then(|m| m.last())
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default();
        match user
            .strip_prefix("Context:")
            .or_else(|| user.strip_prefix("Fragments:"))
        {
            Some(rest) => format!(
                "From the documents:{}",
                rest.split("\n\nQuestion:").next().unwrap_or(rest)
            ),
            None => "I don't know.".to_string(),
        }
    }

    /// Fake `/api/chat`: an empty `messages` loads or unloads, the model
    /// "missing" isn't pulled, and "chatty" always runs into `num_predict`.
    async fn fake_chat(State(fake): State<Fake>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
        let model = body["model"].as_str().unwrap_or_default().to_string();
        if model == "missing" {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "model \"missing\" not found, try pulling it first" })),
            );
        }
        if body["messages"].as_array().is_some_and(|m| m.is_empty()) {
            let mut loaded = fake.loaded.lock().unwrap();
            loaded.retain(|(m, _)| *m != model);
            let reason = if body["keep_alive"] == 0 {
                fake.log.lock().unwrap().push(format!("unload {model}"));
                "unload"
            } else {
                let ctx = body["options"]["num_ctx"].as_u64().unwrap_or(4096);
                fake.log.lock().unwrap().push(format!("load {model} {ctx}"));
                loaded.push((model.clone(), ctx));
                "load"
            };
            return (
                StatusCode::OK,
                Json(json!({ "model": model, "message": { "role": "assistant", "content": "" }, "done": true, "done_reason": reason })),
            );
        }
        fake.chats.lock().unwrap().push(body.clone());
        let reason = if model == "chatty" { "length" } else { "stop" };
        (
            StatusCode::OK,
            Json(json!({
                "model": model,
                "message": { "role": "assistant", "content": echo(&body) },
                "done": true,
                "done_reason": reason,
                "prompt_eval_count": 1200,
                "prompt_eval_duration": 400_000_000u64,
                "eval_count": 60,
                "eval_duration": 1_500_000_000u64,
                "load_duration": 250_000_000u64,
            })),
        )
    }

    async fn fake_ps(State(fake): State<Fake>) -> Json<Value> {
        let loaded = fake.loaded.lock().unwrap();
        let models: Vec<Value> = loaded
            .iter()
            .map(|(name, ctx)| {
                // Weights plus a KV cache that grows with the context.
                let size = 1_900_000_000 + ctx * 42_000;
                json!({ "name": name, "model": name, "size": size, "size_vram": size, "context_length": ctx })
            })
            .collect();
        Json(json!({ "models": models }))
    }

    async fn fake_ollama(fake: Fake) -> String {
        spawn(
            Router::new()
                .route("/api/chat", post(fake_chat))
                .route("/api/ps", get(fake_ps))
                .route("/api/version", get(|| async { Json(json!({ "version": "0.40.3" })) }))
                .route(
                    "/api/tags",
                    get(|| async {
                        Json(json!({ "models": [
                            { "name": "qwen2.5:3b", "size": 1_929_912_432u64,
                              "details": { "parameter_size": "3.1B", "quantization_level": "Q4_K_M" } },
                            { "name": "qwen2.5:3b-instruct-q8_0", "size": 3_285_475_632u64,
                              "details": { "parameter_size": "3.1B", "quantization_level": "Q8_0" } },
                        ] }))
                    }),
                )
                .with_state(fake),
        )
        .await
    }

    async fn test_agent(host: &str) -> Agent {
        Agent {
            index: Index::build(builtin_documents(), index::local()).await.unwrap(),
            ollama: Ollama {
                http: reqwest::Client::new(),
                host: host.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn each_profile_sends_its_own_prompt_and_options() {
        let fake = Fake::default();
        let agent = test_agent(&fake_ollama(fake.clone()).await).await;
        let case = find_case("q01").unwrap();
        for p in PROFILES {
            let a = agent.ask(case.question, Some(case), p, p.model, 1).await;
            assert!(a.error.is_none() && a.runs[0].error.is_none(), "{}: {a:?}", p.id);
            assert_eq!(a.sources.len(), TOP_K);
            assert_eq!(a.source_retrieved, Some(true));
            // The echo cites real fragments, the expected one among them.
            let score = a.runs[0].score.as_ref().unwrap();
            assert!(score.cited_expected, "{}: {score:?}", p.id);
            assert!(!a.runs[0].truncated);
        }

        let chats = fake.chats.lock().unwrap();
        assert_eq!(chats.len(), PROFILES.len());
        for (body, p) in chats.iter().zip(PROFILES) {
            assert_eq!(body["model"], p.model);
            assert_eq!(body["stream"], false);
            assert_eq!(body["options"], options(p));
        }
        let roles = |body: &Value| -> Vec<String> {
            body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["role"].as_str().unwrap().to_string())
                .collect()
        };
        // Lesson 28's prompt: rules in the system message, then one user message.
        assert_eq!(roles(&chats[0]), ["system", "user"]);
        assert_eq!(chats[0]["messages"], chats[1]["messages"]);
        assert_eq!(chats[0]["messages"][0]["content"], format!("{BASE_PROMPT}\n\n{RAG_RULES}"));
        assert!(chats[0].get("options").unwrap().get("temperature").is_none());

        // The tuned one: rules, a worked example, then the real fragments with
        // the format reminder after the question.
        assert_eq!(roles(&chats[2]), ["system", "user", "assistant", "user"]);
        assert_eq!(chats[2]["messages"][0]["content"], TUNED_SYSTEM);
        assert_eq!(chats[2]["messages"][2]["content"], EXAMPLE_ANSWER);
        let last = chats[2]["messages"][3]["content"].as_str().unwrap();
        assert!(last.starts_with("Fragments:\n\n[1] "));
        assert!(last.ends_with(&format!("Question: {}\n\n{FORMAT_REMINDER}", case.question)));
        // The quantized profiles get exactly the optimized request.
        for i in [3, 4] {
            assert_eq!(chats[i]["messages"], chats[2]["messages"]);
        }
    }

    #[tokio::test]
    async fn warm_up_starts_cold_and_reports_memory() {
        let fake = Fake::default();
        fake.loaded.lock().unwrap().push(("llama3.2:latest".to_string(), 2048));
        let agent = test_agent(&fake_ollama(fake.clone()).await).await;

        let w = agent.warm_up(find_profile("optimized").unwrap(), Q4).await;
        assert!(w.error.is_none(), "{w:?}");
        assert_eq!(w.model, Q4);
        let loaded = w.loaded.unwrap();
        assert_eq!(loaded.context_length, 4096);
        assert_eq!(loaded.size, 1_900_000_000 + 4096 * 42_000);
        assert_eq!(w.disk_bytes, Some(1_929_912_432));
        assert_eq!(w.quantization.as_deref(), Some("Q4_K_M"));

        // The baseline's bigger window takes more memory for the same model.
        let w = agent.warm_up(find_profile("baseline").unwrap(), Q4).await;
        assert_eq!(w.loaded.as_ref().unwrap().context_length, 8192);
        assert!(w.loaded.unwrap().size > loaded.size);

        // A model missing from /api/tags has no file size, but still loads.
        let q3 = find_profile("q3").unwrap();
        let w = agent.warm_up(q3, q3.model).await;
        assert!(w.error.is_none() && w.disk_bytes.is_none() && w.loaded.is_some());

        assert_eq!(
            *fake.log.lock().unwrap(),
            [
                "unload llama3.2:latest",
                "load qwen2.5:3b 4096",
                "unload qwen2.5:3b",
                "load qwen2.5:3b 8192",
                "unload qwen2.5:3b",
                "load qwen2.5:3b-instruct-q3_K_M 4096",
            ]
        );
    }

    #[tokio::test]
    async fn scores_and_summary_count_citations_truncation_and_stability() {
        let fake = Fake::default();
        let agent = test_agent(&fake_ollama(fake.clone()).await).await;
        let case = find_case("q01").unwrap();
        let sources = agent.index.search(case.question, TOP_K).await.unwrap();
        let rank = sources.iter().find(|h| expected_file(case, h)).unwrap().rank;

        let s = score(case, &format!("- Port 4000 + NN [{rank}].\n- A made-up source [7]."), &sources);
        assert_eq!((s.found, s.total), (1, 2));
        assert!(s.cited_expected);
        assert_eq!(s.invalid_cites, [7]);
        let s = score(case, "ai-advent-lesson-NN on port 4000, no citations", &sources);
        assert_eq!(s.fact_score, 1.0);
        assert!(!s.cited_expected && s.cited.is_empty());

        // Three runs at temperature 0 give one answer; "chatty" is cut by
        // num_predict every time; "missing" fails with Ollama's message.
        let opt = find_profile("optimized").unwrap();
        let same = agent.ask(case.question, Some(case), opt, Q4, 3).await;
        assert_eq!((same.stability.runs, same.stability.distinct_answers), (3, 1));
        let chatty = agent.ask(case.question, Some(case), opt, "chatty", 2).await;
        assert!(chatty.runs.iter().all(|r| r.truncated));
        let missing = agent.ask(case.question, Some(case), opt, "missing", 1).await;
        let error = missing.runs[0].error.as_deref().unwrap();
        assert!(error.contains("404") && error.contains("not found"), "{error}");

        let sum = summarize(&[same, chatty, missing]);
        assert_eq!((sum.questions, sum.answers, sum.errors), (3, 5, 1));
        assert_eq!((sum.truncated, sum.cited_expected, sum.cited_any), (2, 5, 5));
        assert_eq!(sum.same_answer, 2);
        assert_eq!(sum.prompt_tokens_max, 1200);
        assert_eq!(sum.gen_tokens_per_sec_mean, Some(40.0));
        assert_eq!(sum.prefill_ms_mean, Some(400.0));
    }

    #[tokio::test]
    async fn http_api_runs_profiles_and_reports_errors() {
        let agent = Arc::new(test_agent(&fake_ollama(Fake::default()).await).await);
        let url = spawn(app(AppState { agent })).await;
        let http = reqwest::Client::new();
        let get_json = |path: &str| {
            let req = http.get(format!("{url}{path}"));
            async move { req.send().await.unwrap().json::<Value>().await.unwrap() }
        };
        let post = |path: &str, body: Value| {
            let req = http.post(format!("{url}{path}")).json(&body);
            async move { req.send().await.unwrap().json::<Value>().await.unwrap() }
        };

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>Local LLM Optimization</title>"));

        let cfg = get_json("/api/config").await;
        assert_eq!(cfg["profiles"].as_array().unwrap().len(), PROFILES.len());
        assert_eq!(cfg["profiles"][0]["id"], "baseline");
        assert!(cfg["profiles"][0]["temperature"].is_null());
        assert_eq!(cfg["profiles"][2]["prompt"], "tuned");
        assert_eq!(cfg["top_k"], TOP_K);
        assert!(cfg["chunks"].as_u64().unwrap() > 100);

        let status = get_json("/api/status").await;
        assert_eq!(status["version"], "0.40.3");
        assert_eq!(status["models"][1]["quantization_level"], "Q8_0");
        assert!(status["error"].is_null());
        assert_eq!(get_json("/api/cases").await.as_array().unwrap().len(), 10);

        let a = post("/api/ask", json!({ "case_id": "q03", "profile": "optimized", "runs": 2 })).await;
        assert!(a["error"].is_null(), "{a}");
        assert_eq!(a["profile"]["id"], "optimized");
        assert_eq!(a["model"], Q4);
        assert_eq!(a["options"]["num_predict"], 300);
        assert_eq!(a["runs"].as_array().unwrap().len(), 2);
        assert_eq!(a["runs"][0]["score"]["found"], 1);
        assert_eq!(a["runs"][0]["score"]["cited_expected"], true);

        let other = post("/api/ask", json!({ "question": "What does lesson 05 compare?", "profile": "q8", "model": "llama3.2" })).await;
        assert_eq!(other["model"], "llama3.2");
        assert!(other["case"].is_null() && other["runs"][0]["score"].is_null());

        let w = post("/api/warmup", json!({ "profile": "q8" })).await;
        assert!(w["error"].is_null(), "{w}");
        assert_eq!(w["quantization"], "Q8_0");
        assert_eq!(w["loaded"]["context_length"], 4096);

        for (path, body, needle) in [
            ("/api/ask", json!({ "case_id": "q01", "profile": "nope" }), "nope"),
            ("/api/ask", json!({ "case_id": "q99", "profile": "baseline" }), "q99"),
            ("/api/ask", json!({ "question": "  ", "profile": "baseline" }), "question"),
            ("/api/warmup", json!({ "profile": "nope" }), "nope"),
        ] {
            let r = post(path, body).await;
            assert!(r["error"].as_str().unwrap().contains(needle), "{r}");
        }

        // Ollama down: readable errors, retrieval still shown.
        let down = test_agent(&closed_host()).await;
        assert!(down.ollama.status().await.error.unwrap().contains("Can't reach Ollama"));
        let base = find_profile("baseline").unwrap();
        let a = down.ask("Which port does a lesson get?", None, base, Q4, 1).await;
        assert_eq!(a.sources.len(), TOP_K);
        assert!(a.runs[0].error.as_deref().unwrap().contains("Can't reach Ollama"));
        let w = down.warm_up(base, Q4).await;
        assert!(w.error.unwrap().contains("Can't reach Ollama"));
    }
}

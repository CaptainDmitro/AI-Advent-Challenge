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

use index::{ErrorResponse, Hit, Index, builtin_documents, embedder_from_env, env_non_empty};

const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 10;
const MAX_RUNS: usize = 5;
const MAX_QUESTION_CHARS: usize = 1000;
const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL: &str = "qwen2.5:3b";
/// Five chunks of up to 1800 characters plus the rules are ≈ 2–3k tokens.
/// Ollama's own default window (2048 or 4096, depending on the version) would
/// silently cut the start of such a prompt, so it is set explicitly.
const DEFAULT_NUM_CTX: u32 = 8192;

/// Lesson 22's RAG prompt, word for word: both models get exactly the same
/// system prompt and the same retrieved context.
const BASE_PROMPT: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. Answer \
concisely, in the language of the question. If you don't know something, say so \
instead of guessing.";

const RAG_RULES: &str = "The user message contains numbered fragments of the \
repository's documents, then the question. Answer only from these fragments. \
After each fact, cite the fragment it came from as [1], [2] and so on. If the \
fragments don't contain the answer, say that the documents don't cover it.";

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

/// One generated answer, whichever model wrote it.
struct Completion {
    text: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    /// Ollama only: its own counters split the call into load, prompt
    /// processing (prefill) and generation.
    gen_tokens_per_sec: Option<f64>,
    prefill_ms: Option<f64>,
    load_ms: Option<f64>,
}

// ---------------------------------------------------------------------------
// The local model: Ollama's native /api/chat (lesson 26).
// ---------------------------------------------------------------------------

struct Ollama {
    http: reqwest::Client,
    host: String,
    model: String,
    num_ctx: u32,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
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
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
}

#[derive(Serialize, Clone, Debug, Default)]
struct OllamaStatus {
    host: String,
    version: Option<String>,
    models: Vec<String>,
    error: Option<String>,
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

    async fn status(&self) -> OllamaStatus {
        let mut status = OllamaStatus {
            host: self.host.clone(),
            ..OllamaStatus::default()
        };
        let result = async {
            let version = self.get_json::<VersionResponse>("/api/version").await?;
            let tags = self.get_json::<TagsResponse>("/api/tags").await?;
            Ok::<_, String>((version.version, tags.models))
        }
        .await;
        match result {
            Ok((version, models)) => {
                status.version = Some(version);
                status.models = models.into_iter().map(|m| m.name).collect();
            }
            Err(e) => status.error = Some(e),
        }
        status
    }

    async fn complete(
        &self,
        model: &str,
        messages: &[Value],
        temperature: f64,
    ) -> Result<Completion, String> {
        let body = json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "options": { "temperature": temperature, "num_ctx": self.num_ctx },
        });
        let text = self
            .send(
                self.http
                    .post(format!("{}/api/chat", self.host))
                    .json(&body),
            )
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
            gen_tokens_per_sec: Some(if eval_secs > 0.0 {
                round1(r.eval_count as f64 / eval_secs)
            } else {
                0.0
            }),
            prefill_ms: Some(nanos_to_ms(r.prompt_eval_duration)),
            load_ms: Some(nanos_to_ms(r.load_duration)),
        })
    }
}

// ---------------------------------------------------------------------------
// The cloud model, for comparison only: any OpenAI-compatible
// /chat/completions. Optional — without OPENAI_API_KEY nothing leaves the
// machine.
// ---------------------------------------------------------------------------

struct CloudLlm {
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

#[derive(Deserialize, Default)]
#[serde(default)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl CloudLlm {
    async fn complete(&self, messages: &[Value], temperature: f64) -> Result<Completion, String> {
        let body = json!({ "model": self.model, "messages": messages, "temperature": temperature });
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Cloud LLM request failed: {e}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<ErrorResponse>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(format!("Cloud LLM API error ({status}): {detail}"));
        }
        let parsed = serde_json::from_str::<ChatResponse>(&text)
            .map_err(|e| format!("Failed to parse the cloud response: {e}"))?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .ok_or("The model returned an empty answer.")?;
        Ok(Completion {
            text: content,
            prompt_tokens: parsed.usage.prompt_tokens,
            completion_tokens: parsed.usage.completion_tokens,
            gen_tokens_per_sec: None,
            prefill_ms: None,
            load_ms: None,
        })
    }
}

// ---------------------------------------------------------------------------
// The pipeline: retrieve once, locally; generate with each model.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Local,
    Cloud,
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

fn build_messages(question: &str, sources: &[Hit]) -> Vec<Value> {
    vec![
        json!({ "role": "system", "content": format!("{BASE_PROMPT}\n\n{RAG_RULES}") }),
        json!({
            "role": "user",
            "content": format!("Context:\n\n{}\n\nQuestion: {question}", context_block(sources)),
        }),
    ]
}

/// How to ask: how many chunks, which models, how many times each.
#[derive(Clone, Debug)]
struct AskOptions {
    k: usize,
    runs: usize,
    temperature: f64,
    backends: Vec<Backend>,
    /// Another pulled Ollama model instead of the default one.
    local_model: Option<String>,
}

/// One answer of one model.
#[derive(Serialize, Clone, Debug, Default)]
struct Run {
    text: String,
    error: Option<String>,
    /// Wall-clock time of the call, as this server sees it.
    llm_ms: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    /// Completion tokens per wall-clock second: the same yardstick for both
    /// models, network and prompt processing included.
    e2e_tokens_per_sec: f64,
    /// Ollama only, from its own counters.
    gen_tokens_per_sec: Option<f64>,
    prefill_ms: Option<f64>,
    load_ms: Option<f64>,
    score: Option<Score>,
}

/// How much the answers of one model agree with each other over several runs
/// of the same question.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
struct Stability {
    runs: usize,
    errors: usize,
    /// Different texts among the successful runs (whitespace ignored). 1
    /// means the same answer every time.
    distinct_answers: usize,
    /// Successful runs that cite at least one fragment as `[n]`.
    cited_runs: usize,
    /// Control questions: the fact score over the successful runs.
    fact_min: Option<f64>,
    fact_max: Option<f64>,
    fact_mean: Option<f64>,
    ms_min: f64,
    ms_max: f64,
    ms_mean: f64,
    e2e_tokens_per_sec_mean: f64,
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
    let tps: Vec<f64> = ok.iter().map(|r| r.e2e_tokens_per_sec).collect();
    Stability {
        runs: runs.len(),
        errors: runs.len() - ok.len(),
        distinct_answers: texts.len(),
        cited_runs: ok.iter().filter(|r| !citations(&r.text).is_empty()).count(),
        fact_min: facts.iter().copied().reduce(f64::min),
        fact_max: facts.iter().copied().reduce(f64::max),
        fact_mean: mean(&facts),
        ms_min: ms.iter().copied().reduce(f64::min).unwrap_or(0.0),
        ms_max: ms.iter().copied().reduce(f64::max).unwrap_or(0.0),
        ms_mean: mean(&ms).map_or(0.0, f64::round),
        e2e_tokens_per_sec_mean: mean(&tps).map_or(0.0, round1),
    }
}

/// Every run of one model on one question.
#[derive(Serialize, Clone, Debug)]
struct BackendResult {
    backend: Backend,
    model: String,
    runs: Vec<Run>,
    stability: Stability,
    /// The model can't be asked at all (no cloud configured).
    error: Option<String>,
}

/// One question: the shared retrieval, then each model's answers to the very
/// same prompt.
#[derive(Serialize, Clone, Debug, Default)]
struct Comparison {
    question: String,
    case: Option<CaseInfo>,
    sources: Vec<Hit>,
    /// Exactly what each model was sent.
    messages: Vec<Value>,
    retrieve_ms: f64,
    /// Control questions: a chunk of an expected file is among the retrieved.
    source_retrieved: Option<bool>,
    /// Control questions: a chunk of the expected section itself is.
    section_retrieved: Option<bool>,
    results: Vec<BackendResult>,
    error: Option<String>,
}

struct Agent {
    index: Index,
    local: Ollama,
    cloud: Option<CloudLlm>,
}

impl Agent {
    /// Local always; cloud too when it's configured.
    fn default_backends(&self) -> Vec<Backend> {
        let mut backends = vec![Backend::Local];
        if self.cloud.is_some() {
            backends.push(Backend::Cloud);
        }
        backends
    }

    /// question → search the local index → chunks + question → each model,
    /// `runs` times. Retrieval happens once, so every model gets byte for byte
    /// the same prompt. The models run side by side; each one's runs go one
    /// after another, so the local model's speed isn't shared between them.
    async fn compare(&self, question: &str, case: Option<&'static Case>, opts: &AskOptions) -> Comparison {
        let mut out = Comparison {
            question: question.to_string(),
            case: case.map(Case::info),
            ..Comparison::default()
        };
        let started = Instant::now();
        let sources = self.index.search(question, opts.k).await;
        out.retrieve_ms = elapsed_ms(started);
        let sources = match sources {
            Ok(sources) => sources,
            Err(e) => {
                out.error = Some(e);
                return out;
            }
        };
        let messages = build_messages(question, &sources);
        if let Some(case) = case {
            out.source_retrieved = Some(sources.iter().any(|h| expected_file(case, h)));
            out.section_retrieved = Some(sources.iter().any(|h| {
                case.sources.iter().any(|(file, section)| {
                    h.chunk.source == *file && h.chunk.section.starts_with(section)
                })
            }));
        }
        let wants = |b: Backend| opts.backends.contains(&b);
        let local = async {
            if wants(Backend::Local) {
                Some(self.run_backend(Backend::Local, &messages, &sources, case, opts).await)
            } else {
                None
            }
        };
        let cloud = async {
            if wants(Backend::Cloud) {
                Some(self.run_backend(Backend::Cloud, &messages, &sources, case, opts).await)
            } else {
                None
            }
        };
        let (local, cloud) = tokio::join!(local, cloud);
        out.results = [local, cloud].into_iter().flatten().collect();
        out.sources = sources;
        out.messages = messages;
        out
    }

    async fn run_backend(
        &self,
        backend: Backend,
        messages: &[Value],
        sources: &[Hit],
        case: Option<&Case>,
        opts: &AskOptions,
    ) -> BackendResult {
        let model = match (backend, &self.cloud) {
            (Backend::Cloud, Some(cloud)) => cloud.model.clone(),
            (Backend::Cloud, None) => {
                return BackendResult {
                    backend,
                    model: String::new(),
                    runs: Vec::new(),
                    stability: stability(&[]),
                    error: Some(
                        "No cloud model: set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, \
                         OPENAI_MODEL) to compare with one."
                            .to_string(),
                    ),
                };
            }
            (Backend::Local, _) => opts
                .local_model
                .clone()
                .unwrap_or_else(|| self.local.model.clone()),
        };
        let mut runs = Vec::with_capacity(opts.runs);
        for _ in 0..opts.runs {
            let started = Instant::now();
            let result = match (backend, &self.cloud) {
                (Backend::Cloud, Some(cloud)) => cloud.complete(messages, opts.temperature).await,
                _ => {
                    self.local
                        .complete(&model, messages, opts.temperature)
                        .await
                }
            };
            let llm_ms = elapsed_ms(started);
            let mut run = match result {
                Ok(c) => Run {
                    e2e_tokens_per_sec: if llm_ms > 0.0 {
                        round1(c.completion_tokens as f64 / (llm_ms / 1000.0))
                    } else {
                        0.0
                    },
                    text: c.text,
                    error: None,
                    llm_ms,
                    prompt_tokens: c.prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    gen_tokens_per_sec: c.gen_tokens_per_sec,
                    prefill_ms: c.prefill_ms,
                    load_ms: c.load_ms,
                    score: None,
                },
                Err(e) => Run {
                    error: Some(e),
                    llm_ms,
                    ..Run::default()
                },
            };
            run.score = case.map(|c| score(c, &run.text, sources));
            runs.push(run);
        }
        BackendResult {
            backend,
            model,
            stability: stability(&runs),
            runs,
            error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Control questions: lesson 22's ten, with what a correct answer says and
// where it comes from.
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
    let agent = &app.agent;
    Json(json!({
        "embedder": agent.index.embedder.id(),
        "documents": agent.index.documents,
        "chunks": agent.index.chunks.len(),
        "top_k": DEFAULT_TOP_K,
        "max_top_k": MAX_TOP_K,
        "max_runs": MAX_RUNS,
        "local_model": agent.local.model,
        "num_ctx": agent.local.num_ctx,
        "cloud_model": agent.cloud.as_ref().map(|c| c.model.clone()),
    }))
}

async fn status(State(app): State<AppState>) -> Json<OllamaStatus> {
    Json(app.agent.local.status().await)
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
    k: Option<usize>,
    /// How many times each model answers, to see how stable it is.
    runs: Option<usize>,
    temperature: Option<f64>,
    /// `["local"]`, `["cloud"]`, both, or nothing for every configured one.
    backends: Option<Vec<Backend>>,
    local_model: Option<String>,
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<Comparison> {
    let case = match req.case_id.as_deref() {
        Some(id) => match find_case(id) {
            Some(case) => Some(case),
            None => {
                return Json(Comparison {
                    error: Some(format!("No control question {id:?}.")),
                    ..Comparison::default()
                });
            }
        },
        None => None,
    };
    let question = case.map_or(req.question.trim(), |c| c.question).to_string();
    let fail = |error: String| Comparison {
        question: question.clone(),
        error: Some(error),
        ..Comparison::default()
    };
    if question.is_empty() {
        return Json(fail("Type a question first.".to_string()));
    }
    if question.chars().count() > MAX_QUESTION_CHARS {
        return Json(fail(format!(
            "Keep the question under {MAX_QUESTION_CHARS} characters."
        )));
    }
    let backends = req
        .backends
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| app.agent.default_backends());
    let opts = AskOptions {
        k: req.k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K),
        runs: req.runs.unwrap_or(1).clamp(1, MAX_RUNS),
        temperature: req.temperature.unwrap_or(0.0).clamp(0.0, 2.0),
        backends,
        local_model: req
            .local_model
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty()),
    };
    Json(app.agent.compare(&question, case, &opts).await)
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
        .with_state(state)
}

// ---------------------------------------------------------------------------
// `app eval`: every control question through every model, as tables.
// ---------------------------------------------------------------------------

fn pct(x: f64) -> String {
    format!("{:.0}%", x * 100.0)
}

fn secs(ms: f64) -> String {
    format!("{:.1} s", ms / 1000.0)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// One cell per model: mean facts, and with several runs the spread too.
fn facts_cell(r: &BackendResult) -> String {
    let s = &r.stability;
    if r.error.is_some() || s.runs == 0 {
        return "-".to_string();
    }
    let Some(mean) = s.fact_mean else {
        return format!("error ×{}", s.errors);
    };
    let total = r.runs[0].score.as_ref().map_or(0, |s| s.total) as f64;
    let mut cell = format!("{:.1}/{total}", mean * total);
    if s.runs > 1 {
        cell.push_str(&format!(" · {} distinct", s.distinct_answers));
    }
    if s.errors > 0 {
        cell.push_str(&format!(" · {} err", s.errors));
    }
    cell
}

async fn run_eval(agent: &Agent, opts: &AskOptions) {
    let mut comparisons = Vec::new();
    for case in CASES {
        eprintln!("{} {}", case.id, case.question);
        comparisons.push(agent.compare(case.question, Some(case), opts).await);
    }
    let backends = &opts.backends;
    let name = |b: Backend| match b {
        Backend::Local => "local",
        Backend::Cloud => "cloud",
    };

    println!(
        "{} run(s) per question, temperature {}, top-{} chunks\n",
        opts.runs, opts.temperature, opts.k
    );
    let mut header = "| # | source retrieved |".to_string();
    let mut rule = "|---|---|".to_string();
    for b in backends {
        header.push_str(&format!(" facts {0} | cites it ({0}) | time {0} |", name(*b)));
        rule.push_str("---|---|---|");
    }
    println!("{header}\n{rule}");
    for c in &comparisons {
        let mut row = format!(
            "| {} | {} |",
            c.case.as_ref().map_or("", |k| k.id),
            match c.source_retrieved {
                Some(true) => "yes",
                Some(false) => "no",
                None => "-",
            }
        );
        for b in backends {
            match c.results.iter().find(|r| r.backend == *b) {
                Some(r) => {
                    let cited = r
                        .runs
                        .iter()
                        .filter(|run| run.score.as_ref().is_some_and(|s| s.cited_expected))
                        .count();
                    row.push_str(&format!(
                        " {} | {cited}/{} | {} |",
                        facts_cell(r),
                        r.runs.len(),
                        secs(r.stability.ms_mean)
                    ));
                }
                None => row.push_str(" - | - | - |"),
            }
        }
        println!("{row}");
    }

    // The summary: quality, speed and stability per model.
    println!("\n| | {} |", backends.iter().map(|b| {
        let model = comparisons
            .iter()
            .flat_map(|c| &c.results)
            .find(|r| r.backend == *b)
            .map_or(String::new(), |r| r.model.clone());
        format!("{} `{model}`", name(*b))
    }).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(backends.len()));
    let per = |f: &dyn Fn(&[&BackendResult]) -> String| {
        backends
            .iter()
            .map(|b| {
                let rs: Vec<&BackendResult> = comparisons
                    .iter()
                    .flat_map(|c| &c.results)
                    .filter(|r| r.backend == *b && r.error.is_none())
                    .collect();
                f(rs.as_slice())
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let all_runs = |rs: &[&BackendResult]| -> Vec<Run> {
        rs.iter().flat_map(|r| r.runs.iter().cloned()).collect()
    };
    println!(
        "| quality: expected facts stated | {} |",
        per(&|rs: &[&BackendResult]| {
            let ok: Vec<f64> = all_runs(rs)
                .iter()
                .filter(|r| r.error.is_none())
                .filter_map(|r| r.score.as_ref().map(|s| s.fact_score))
                .collect();
            mean(&ok).map_or("-".to_string(), pct)
        })
    );
    println!(
        "| quality: answers citing the expected source | {} |",
        per(&|rs: &[&BackendResult]| {
            let runs = all_runs(rs);
            let n = runs.iter().filter(|r| r.error.is_none()).count();
            let cited = runs
                .iter()
                .filter(|r| r.score.as_ref().is_some_and(|s| s.cited_expected))
                .count();
            format!("{cited}/{n}")
        })
    );
    println!(
        "| speed: mean time per answer | {} |",
        per(&|rs: &[&BackendResult]| {
            let ms: Vec<f64> = all_runs(rs)
                .iter()
                .filter(|r| r.error.is_none())
                .map(|r| r.llm_ms)
                .collect();
            mean(&ms).map_or("-".to_string(), secs)
        })
    );
    println!(
        "| speed: completion tokens per wall-clock second | {} |",
        per(&|rs: &[&BackendResult]| {
            let tps: Vec<f64> = all_runs(rs)
                .iter()
                .filter(|r| r.error.is_none())
                .map(|r| r.e2e_tokens_per_sec)
                .collect();
            mean(&tps).map_or("-".to_string(), |x| format!("{x:.1}"))
        })
    );
    println!(
        "| speed: generation tok/s · prefill · load (Ollama's counters) | {} |",
        per(&|rs: &[&BackendResult]| {
            let runs: Vec<Run> = all_runs(rs).into_iter().filter(|r| r.error.is_none()).collect();
            let gen_tps: Vec<f64> = runs.iter().filter_map(|r| r.gen_tokens_per_sec).collect();
            let prefill: Vec<f64> = runs.iter().filter_map(|r| r.prefill_ms).collect();
            let load: Vec<f64> = runs.iter().filter_map(|r| r.load_ms).collect();
            match (mean(&gen_tps), mean(&prefill), mean(&load)) {
                (Some(g), Some(p), Some(l)) => format!("{g:.1} tok/s · {} · {}", secs(p), secs(l)),
                _ => "-".to_string(),
            }
        })
    );
    println!(
        "| stability: failed calls | {} |",
        per(&|rs: &[&BackendResult]| {
            let runs = all_runs(rs);
            format!("{}/{}", runs.iter().filter(|r| r.error.is_some()).count(), runs.len())
        })
    );
    println!(
        "| stability: questions with one answer across all runs | {} |",
        per(&|rs: &[&BackendResult]| {
            let same = rs
                .iter()
                .filter(|r| r.stability.errors == 0 && r.stability.distinct_answers == 1)
                .count();
            format!("{same}/{}", rs.len())
        })
    );
    println!(
        "| stability: questions whose fact score changed between runs | {} |",
        per(&|rs: &[&BackendResult]| {
            let varied = rs
                .iter()
                .filter(|r| r.stability.fact_min != r.stability.fact_max)
                .count();
            format!("{varied}/{}", rs.len())
        })
    );
    println!(
        "| stability: answers that cite no fragment at all | {} |",
        per(&|rs: &[&BackendResult]| {
            let ok: usize = rs.iter().map(|r| r.stability.runs - r.stability.errors).sum();
            let cited: usize = rs.iter().map(|r| r.stability.cited_runs).sum();
            format!("{}/{ok}", ok - cited)
        })
    );

    for c in &comparisons {
        let Some(case) = &c.case else { continue };
        println!("\n## {} {}\nexpected: {}", case.id, case.question, case.expect);
        for h in &c.sources {
            println!("  [{}] {:.3} {} — {}", h.rank, h.score, h.chunk.source, h.chunk.section);
        }
        for r in &c.results {
            for (i, run) in r.runs.iter().enumerate() {
                let label = format!("{} `{}` run {}", name(r.backend), r.model, i + 1);
                match &run.error {
                    Some(e) => println!("\n### {label}: error: {e}"),
                    None => println!("\n### {label} ({})\n{}", secs(run.llm_ms), run.text),
                }
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let ollama_host = normalize_host(&env_non_empty("OLLAMA_HOST").unwrap_or_default());
    let http = reqwest::Client::builder()
        // Generous: on CPU a long answer from a cold model can take minutes.
        .timeout(Duration::from_secs(600))
        .build()
        .expect("failed to build HTTP client");
    let local = Ollama {
        http: http.clone(),
        host: ollama_host.clone(),
        model: env_non_empty("OLLAMA_MODEL").unwrap_or_else(|| DEFAULT_OLLAMA_MODEL.to_string()),
        num_ctx: env_non_empty("OLLAMA_NUM_CTX")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_NUM_CTX),
    };
    let cloud = env_non_empty("OPENAI_API_KEY").map(|api_key| {
        let base_url = env_non_empty("OPENAI_BASE_URL")
            .unwrap_or_else(|| "https://api.deepseek.com".to_string());
        CloudLlm {
            http: http.clone(),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model: env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string()),
        }
    });
    let embedder = embedder_from_env(&ollama_host).unwrap_or_else(|e| {
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
    println!(
        "local model: {} at {} (num_ctx {})",
        local.model, local.host, local.num_ctx
    );
    match &cloud {
        Some(c) => println!("cloud model for comparison: {} at {}", c.model, c.endpoint),
        None => println!("cloud model: off (no OPENAI_API_KEY) — fully local"),
    }
    let agent = Arc::new(Agent { index, local, cloud });

    // `app eval [--runs N] [--temperature T] [--k K] [--only local|cloud]`:
    // run the control questions, print the comparison, exit.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("eval") {
        let backends = match flag(&args, "--only") {
            Some("local") => vec![Backend::Local],
            Some("cloud") => vec![Backend::Cloud],
            _ => agent.default_backends(),
        };
        let opts = AskOptions {
            k: flag(&args, "--k")
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_TOP_K)
                .clamp(1, MAX_TOP_K),
            runs: flag(&args, "--runs")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .clamp(1, MAX_RUNS),
            temperature: flag(&args, "--temperature")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0f64)
                .clamp(0.0, 2.0),
            backends,
            local_model: None,
        };
        run_eval(&agent, &opts).await;
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn citations_are_parsed() {
        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert_eq!(citations("an array [x] or a[0] index [ 4 ]"), vec![0, 4]);
        assert!(citations("no brackets").is_empty());
    }

    #[test]
    fn normalizes_ollama_host_like_ollama_does() {
        assert_eq!(normalize_host(""), DEFAULT_OLLAMA_HOST);
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(normalize_host("http://gpu-box:11434/"), "http://gpu-box:11434");
    }

    #[test]
    fn control_questions_are_grounded_in_the_corpus() {
        assert_eq!(CASES.len(), 10);
        let index_docs: Vec<(&str, &str)> = index::BUILTIN.to_vec();
        for case in CASES {
            assert!(!case.facts.is_empty() && !case.sources.is_empty(), "{}", case.id);
            let question = case.question.to_lowercase();
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
                assert!(
                    !variants.iter().any(|v| question.contains(&v.to_lowercase())),
                    "{}: {variants:?} is given away by the question",
                    case.id
                );
            }
        }
    }

    #[tokio::test]
    async fn retrieval_is_local_and_finds_the_expected_source_for_every_control_question() {
        let index = Index::build(builtin_documents(), index::local()).await.unwrap();
        assert!(index.embedder.id().starts_with("local:"));
        for case in CASES {
            let hits = index.search(case.question, DEFAULT_TOP_K).await.unwrap();
            assert!(
                hits.iter().any(|h| expected_file(case, h)),
                "{}: {:?}",
                case.id,
                hits.iter()
                    .map(|h| format!("{} — {}", h.chunk.source, h.chunk.section))
                    .collect::<Vec<_>>()
            );
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

    /// What a fake model has been sent, and how many calls it has had.
    #[derive(Clone, Default)]
    struct Fake {
        seen: Arc<Mutex<Vec<Value>>>,
        calls: Arc<AtomicUsize>,
    }

    /// A model that knows nothing about the repo: with context it echoes the
    /// context back (a perfectly faithful reader), without it it doesn't know.
    fn echo(body: &Value) -> String {
        let user = body["messages"][1]["content"].as_str().unwrap_or_default();
        match user.strip_prefix("Context:") {
            Some(context) => format!("From the documents:{context}"),
            None => "I don't know the details of this repository.".to_string(),
        }
    }

    /// A fake Ollama. Above temperature 0 every answer is a little different;
    /// the model "flaky" fails every second call.
    async fn fake_ollama_chat(
        State(fake): State<Fake>,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        let n = fake.calls.fetch_add(1, Ordering::SeqCst) + 1;
        fake.seen.lock().unwrap().push(body.clone());
        if body["model"] == "flaky" && n % 2 == 0 {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "model runner has unexpectedly stopped" })),
            );
        }
        let mut content = echo(&body);
        if body["options"]["temperature"].as_f64().unwrap_or(0.0) > 0.0 {
            content.push_str(&format!(" (take {n})"));
        }
        (
            StatusCode::OK,
            Json(json!({
                "model": body["model"],
                "message": { "role": "assistant", "content": content },
                "done": true,
                "prompt_eval_count": 1200,
                "prompt_eval_duration": 400_000_000u64,
                "eval_count": 60,
                "eval_duration": 1_500_000_000u64,
                "load_duration": 250_000_000u64,
                "total_duration": 2_200_000_000u64,
            })),
        )
    }

    async fn fake_ollama(fake: Fake) -> String {
        spawn(
            Router::new()
                .route("/api/chat", post(fake_ollama_chat))
                .route("/api/version", get(|| async { Json(json!({ "version": "0.40.3" })) }))
                .route(
                    "/api/tags",
                    get(|| async { Json(json!({ "models": [{ "name": "qwen2.5:3b" }] })) }),
                )
                .with_state(fake),
        )
        .await
    }

    /// A fake OpenAI-compatible cloud model with the same echo behaviour.
    async fn fake_cloud_chat(
        State(fake): State<Fake>,
        headers: axum::http::HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        assert_eq!(
            headers.get("authorization").and_then(|v| v.to_str().ok()),
            Some("Bearer test-key")
        );
        fake.calls.fetch_add(1, Ordering::SeqCst);
        fake.seen.lock().unwrap().push(body.clone());
        Json(json!({
            "choices": [{ "message": { "role": "assistant", "content": echo(&body) } }],
            "usage": { "prompt_tokens": 1100, "completion_tokens": 50 },
        }))
    }

    async fn fake_cloud(fake: Fake) -> String {
        spawn(
            Router::new()
                .route("/v1/chat/completions", post(fake_cloud_chat))
                .with_state(fake),
        )
        .await
    }

    async fn test_agent(ollama_host: &str, cloud_base: Option<&str>) -> Agent {
        Agent {
            index: Index::build(builtin_documents(), index::local()).await.unwrap(),
            local: Ollama {
                http: reqwest::Client::new(),
                host: ollama_host.to_string(),
                model: DEFAULT_OLLAMA_MODEL.to_string(),
                num_ctx: DEFAULT_NUM_CTX,
            },
            cloud: cloud_base.map(|base| CloudLlm {
                http: reqwest::Client::new(),
                endpoint: format!("{base}/v1/chat/completions"),
                api_key: "test-key".to_string(),
                model: "cloud-model".to_string(),
            }),
        }
    }

    fn opts(backends: &[Backend], runs: usize, temperature: f64) -> AskOptions {
        AskOptions {
            k: DEFAULT_TOP_K,
            runs,
            temperature,
            backends: backends.to_vec(),
            local_model: None,
        }
    }

    #[tokio::test]
    async fn both_models_get_the_same_prompt_and_are_scored_the_same_way() {
        let (ollama_fake, cloud_fake) = (Fake::default(), Fake::default());
        let ollama = fake_ollama(ollama_fake.clone()).await;
        let cloud = fake_cloud(cloud_fake.clone()).await;
        let agent = test_agent(&ollama, Some(&cloud)).await;
        assert_eq!(agent.default_backends(), [Backend::Local, Backend::Cloud]);

        let both = opts(&agent.default_backends(), 1, 0.0);
        let (mut local_sum, mut cloud_sum) = (0.0, 0.0);
        for case in CASES {
            let c = agent.compare(case.question, Some(case), &both).await;
            assert!(c.error.is_none(), "{}", case.id);
            assert_eq!(c.source_retrieved, Some(true), "{}", case.id);
            assert_eq!(c.sources.len(), DEFAULT_TOP_K);
            let system = c.messages[0]["content"].as_str().unwrap();
            assert!(system.starts_with(BASE_PROMPT) && system.ends_with(RAG_RULES));
            let user = c.messages[1]["content"].as_str().unwrap();
            for hit in &c.sources {
                assert!(user.contains(&format!("[{}] {} — {}", hit.rank, hit.chunk.source, hit.chunk.section)));
            }
            assert!(user.ends_with(&format!("Question: {}", case.question)));

            assert_eq!(c.results.len(), 2);
            let (local, cloud) = (&c.results[0], &c.results[1]);
            assert_eq!((local.backend, cloud.backend), (Backend::Local, Backend::Cloud));
            assert_eq!((local.model.as_str(), cloud.model.as_str()), ("qwen2.5:3b", "cloud-model"));
            for r in [local, cloud] {
                assert_eq!(r.runs.len(), 1);
                let run = &r.runs[0];
                assert!(run.error.is_none(), "{}: {:?}", case.id, run.error);
                let score = run.score.as_ref().unwrap();
                assert!(score.cited_expected, "{} {:?}", case.id, r.backend);
                assert_eq!(r.stability.distinct_answers, 1);
            }
            // Ollama's nanosecond counters become speeds and times.
            let run = &local.runs[0];
            assert_eq!(run.prompt_tokens, 1200);
            assert_eq!(run.completion_tokens, 60);
            assert_eq!(run.gen_tokens_per_sec, Some(40.0));
            assert_eq!(run.prefill_ms, Some(400.0));
            assert_eq!(run.load_ms, Some(250.0));
            assert!(run.e2e_tokens_per_sec > 0.0);
            // The cloud reports only tokens; its speed is wall-clock.
            let run = &cloud.runs[0];
            assert_eq!(run.completion_tokens, 50);
            assert_eq!(run.gen_tokens_per_sec, None);
            assert_eq!(run.load_ms, None);
            local_sum += local.runs[0].score.as_ref().unwrap().fact_score;
            cloud_sum += cloud.runs[0].score.as_ref().unwrap().fact_score;
        }
        let n = CASES.len() as f64;
        assert!(local_sum / n >= 0.7, "local facts {:.2}", local_sum / n);
        assert_eq!(local_sum, cloud_sum);

        let local_seen = ollama_fake.seen.lock().unwrap();
        let cloud_seen = cloud_fake.seen.lock().unwrap();
        assert_eq!(local_seen.len(), CASES.len());
        assert_eq!(cloud_seen.len(), CASES.len());
        for (l, c) in local_seen.iter().zip(cloud_seen.iter()) {
            // Byte for byte the same prompt to both.
            assert_eq!(l["messages"], c["messages"]);
            assert_eq!(l["model"], "qwen2.5:3b");
            assert_eq!(l["stream"], false);
            assert_eq!(l["options"]["temperature"], 0.0);
            assert_eq!(l["options"]["num_ctx"], DEFAULT_NUM_CTX);
            assert_eq!(c["model"], "cloud-model");
            assert_eq!(c["temperature"], 0.0);
        }
    }

    #[tokio::test]
    async fn repeated_runs_measure_stability() {
        let fake = Fake::default();
        let ollama = fake_ollama(fake.clone()).await;
        let agent = test_agent(&ollama, None).await;
        assert_eq!(agent.default_backends(), [Backend::Local]);
        let case = find_case("q03").unwrap();

        // Temperature 0: the same answer three times.
        let c = agent.compare(case.question, Some(case), &opts(&[Backend::Local], 3, 0.0)).await;
        let s = &c.results[0].stability;
        assert_eq!((s.runs, s.errors, s.distinct_answers, s.cited_runs), (3, 0, 1, 3));
        assert_eq!(s.fact_min, s.fact_max);
        assert_eq!(s.fact_mean, Some(1.0));

        // Temperature 1: three different answers, the same facts.
        let c = agent.compare(case.question, Some(case), &opts(&[Backend::Local], 3, 1.0)).await;
        let s = &c.results[0].stability;
        assert_eq!((s.runs, s.errors, s.distinct_answers), (3, 0, 3));
        assert_eq!(s.fact_min, Some(1.0));

        // A model that falls over every second call.
        let mut flaky = opts(&[Backend::Local], 4, 0.0);
        flaky.local_model = Some("flaky".to_string());
        let c = agent.compare(case.question, Some(case), &flaky).await;
        let r = &c.results[0];
        assert_eq!(r.model, "flaky");
        assert_eq!((r.stability.runs, r.stability.errors), (4, 2));
        let errors: Vec<&str> = r.runs.iter().filter_map(|run| run.error.as_deref()).collect();
        assert!(errors.iter().all(|e| e.contains("500") && e.contains("unexpectedly stopped")), "{errors:?}");

        // Asking for the cloud without one configured says how to set it up.
        let c = agent.compare(case.question, Some(case), &opts(&[Backend::Local, Backend::Cloud], 1, 0.0)).await;
        assert_eq!(c.results.len(), 2);
        assert!(c.results[1].error.as_deref().unwrap().contains("OPENAI_API_KEY"));
        assert!(c.results[1].runs.is_empty());
        assert!(c.results[0].runs[0].error.is_none());
    }

    #[tokio::test]
    async fn http_api_compares_and_reports_errors() {
        let ollama = fake_ollama(Fake::default()).await;
        let cloud = fake_cloud(Fake::default()).await;
        let agent = Arc::new(test_agent(&ollama, Some(&cloud)).await);
        let url = spawn(app(AppState { agent })).await;
        let http = reqwest::Client::new();
        let get_json = |path: &str| {
            let req = http.get(format!("{url}{path}"));
            async move { req.send().await.unwrap().json::<Value>().await.unwrap() }
        };
        let post = |body: Value| {
            let req = http.post(format!("{url}/api/ask")).json(&body);
            async move { req.send().await.unwrap().json::<Value>().await.unwrap() }
        };

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>Local RAG</title>"));

        let cfg = get_json("/api/config").await;
        assert_eq!(cfg["local_model"], "qwen2.5:3b");
        assert_eq!(cfg["cloud_model"], "cloud-model");
        assert_eq!(cfg["num_ctx"], DEFAULT_NUM_CTX);
        assert!(cfg["embedder"].as_str().unwrap().starts_with("local:"));
        assert!(cfg["chunks"].as_u64().unwrap() > 100);

        let status = get_json("/api/status").await;
        assert_eq!(status["version"], "0.40.3");
        assert_eq!(status["models"][0], "qwen2.5:3b");
        assert!(status["error"].is_null());

        let cases = get_json("/api/cases").await;
        assert_eq!(cases.as_array().unwrap().len(), 10);

        let both = post(json!({ "case_id": "q03", "k": 3, "runs": 2 })).await;
        assert!(both["error"].is_null(), "{both}");
        assert_eq!(both["case"]["id"], "q03");
        assert_eq!(both["sources"].as_array().unwrap().len(), 3);
        assert_eq!(both["source_retrieved"], true);
        assert_eq!(both["results"][0]["backend"], "local");
        assert_eq!(both["results"][1]["backend"], "cloud");
        assert_eq!(both["results"][0]["runs"].as_array().unwrap().len(), 2);
        assert_eq!(both["results"][0]["runs"][0]["score"]["found"], 1);
        assert_eq!(both["results"][0]["stability"]["distinct_answers"], 1);

        let free = post(json!({ "question": "What does lesson 05 compare?", "backends": ["local"] })).await;
        assert_eq!(free["results"].as_array().unwrap().len(), 1);
        assert!(free["case"].is_null());
        assert!(free["results"][0]["runs"][0]["score"].is_null());

        let empty = post(json!({ "question": "  " })).await;
        assert!(empty["error"].as_str().unwrap().contains("question"));
        let unknown = post(json!({ "case_id": "q99" })).await;
        assert!(unknown["error"].as_str().unwrap().contains("q99"));

        // Ollama down: the error is on the local answer, retrieval still shows.
        let down = test_agent(&closed_host(), None).await;
        let status = down.local.status().await;
        assert!(status.error.unwrap().contains("Can't reach Ollama"));
        let c = down
            .compare("Which port does a lesson get?", None, &opts(&[Backend::Local], 1, 0.0))
            .await;
        assert_eq!(c.sources.len(), DEFAULT_TOP_K);
        let r = &c.results[0];
        assert!(r.runs[0].error.as_deref().unwrap().contains("Can't reach Ollama"));
        assert_eq!(r.stability.errors, 1);
    }
}

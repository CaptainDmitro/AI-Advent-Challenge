mod index;

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

const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 10;
const MAX_QUESTION_CHARS: usize = 1000;

/// Both modes get the same system prompt, so the only difference between them
/// is the retrieved context.
const BASE_PROMPT: &str = "You answer questions about the GitHub repository \
CaptainDmitro/AI-Advent-Challenge: a Rust monorepo of lessons from an AI course, \
one folder per lesson, built and deployed by a GitHub Actions pipeline. Answer \
concisely, in the language of the question. If you don't know something, say so \
instead of guessing.";

/// Added to the system prompt in RAG mode only.
const RAG_RULES: &str = "The user message contains numbered fragments of the \
repository's documents, then the question. Answer only from these fragments. \
After each fact, cite the fragment it came from as [1], [2] and so on. If the \
fragments don't contain the answer, say that the documents don't cover it.";

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

impl Llm {
    /// One completion at temperature 0, so the two modes are compared on the
    /// prompt alone, not on sampling luck.
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
// The agent: one question, two modes.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// The question goes to the model as is.
    Plain,
    /// question → relevant chunks → question + chunks → model.
    Rag,
}

struct Agent {
    llm: Llm,
    index: Index,
}

#[derive(Serialize, Clone, Debug)]
struct Answer {
    mode: Mode,
    text: String,
    /// Exactly what was sent to the model. In RAG mode the user message is
    /// the retrieved chunks merged with the question.
    messages: Vec<Value>,
    /// RAG mode: the retrieved chunks, `[1]` first.
    sources: Vec<Hit>,
    usage: Usage,
    retrieve_ms: f64,
    llm_ms: f64,
    error: Option<String>,
}

fn elapsed_ms(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 1000.0).round()
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

fn build_messages(question: &str, mode: Mode, sources: &[Hit]) -> Vec<Value> {
    match mode {
        Mode::Plain => vec![
            json!({ "role": "system", "content": BASE_PROMPT }),
            json!({ "role": "user", "content": question }),
        ],
        Mode::Rag => vec![
            json!({ "role": "system", "content": format!("{BASE_PROMPT}\n\n{RAG_RULES}") }),
            json!({
                "role": "user",
                "content": format!("Context:\n\n{}\n\nQuestion: {question}", context_block(sources)),
            }),
        ],
    }
}

impl Agent {
    /// The whole pipeline. Plain: question → LLM. RAG: question → search the
    /// index for the `k` closest chunks → merge them with the question → LLM.
    async fn ask(&self, question: &str, mode: Mode, k: usize) -> Answer {
        let started = Instant::now();
        let sources = match mode {
            Mode::Plain => Ok(Vec::new()),
            Mode::Rag => self.index.search(question, k).await,
        };
        let retrieve_ms = elapsed_ms(started);
        let sources = match sources {
            Ok(sources) => sources,
            Err(e) => {
                return Answer {
                    mode,
                    text: String::new(),
                    messages: Vec::new(),
                    sources: Vec::new(),
                    usage: Usage::default(),
                    retrieve_ms,
                    llm_ms: 0.0,
                    error: Some(e),
                };
            }
        };
        let messages = build_messages(question, mode, &sources);
        let started = Instant::now();
        let result = self.llm.complete(&messages).await;
        let llm_ms = elapsed_ms(started);
        let (text, usage, error) = match result {
            Ok((text, usage)) => (text, usage, None),
            Err(e) => (String::new(), Usage::default(), Some(e)),
        };
        Answer {
            mode,
            text,
            messages,
            sources,
            usage,
            retrieve_ms,
            llm_ms,
            error,
        }
    }
}

// ---------------------------------------------------------------------------
// Control questions: what a correct answer says, and where it comes from.
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
    /// Where the answer is: `(file, section)`. A RAG answer should be built
    /// from these.
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
    /// RAG: a chunk of an expected file is among the retrieved ones.
    source_retrieved: Option<bool>,
    /// RAG: a chunk of the expected section itself is among them.
    section_retrieved: Option<bool>,
    /// RAG: the fragment numbers the answer cites.
    cited: Vec<usize>,
    /// RAG: one of the cited fragments is from an expected file.
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
    let expected_file = |h: &Hit| case.sources.iter().any(|(file, _)| h.chunk.source == *file);
    let rag = answer.mode == Mode::Rag;
    let cited = if rag { citations(&answer.text) } else { Vec::new() };
    Score {
        found,
        total,
        fact_score: if total == 0 { 0.0 } else { found as f64 / total as f64 },
        facts,
        source_retrieved: rag.then(|| answer.sources.iter().any(expected_file)),
        section_retrieved: rag.then(|| {
            answer.sources.iter().any(|h| {
                case.sources
                    .iter()
                    .any(|(file, section)| h.chunk.source == *file && h.chunk.section.starts_with(section))
            })
        }),
        cited_expected: rag.then(|| {
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
        "top_k": DEFAULT_TOP_K,
        "max_top_k": MAX_TOP_K,
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
    /// `plain`, `rag`, or nothing for both, side by side.
    mode: Option<Mode>,
    k: Option<usize>,
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
    answers: Vec<Graded>,
    error: Option<String>,
}

async fn ask(agent: &Agent, question: &str, case: Option<&Case>, mode: Option<Mode>, k: usize) -> Vec<Graded> {
    let answers = match mode {
        Some(mode) => vec![agent.ask(question, mode, k).await],
        None => {
            let (plain, rag) = tokio::join!(
                agent.ask(question, Mode::Plain, k),
                agent.ask(question, Mode::Rag, k)
            );
            vec![plain, rag]
        }
    };
    answers
        .into_iter()
        .map(|answer| Graded {
            score: case.map(|c| score(c, &answer)),
            answer,
        })
        .collect()
}

async fn ask_api(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<AskResponse> {
    let case = match req.case_id.as_deref() {
        Some(id) => match find_case(id) {
            Some(case) => Some(case),
            None => {
                return Json(AskResponse {
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
    let k = req.k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let answers = ask(&app.agent, &question, case, req.mode, k).await;
    Json(AskResponse {
        question,
        case: case.map(Case::info),
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
// `app eval`: every control question in both modes, as a table.
// ---------------------------------------------------------------------------

fn mark(flag: Option<bool>) -> &'static str {
    match flag {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    }
}

async fn run_eval(agent: &Agent, k: usize) {
    println!("| # | question | facts without RAG | facts with RAG | source retrieved | cites it |");
    println!("|---|---|---|---|---|---|");
    let (mut plain_sum, mut rag_sum) = (0.0, 0.0);
    let mut details = Vec::new();
    for case in CASES {
        let graded = ask(agent, case.question, Some(case), None, k).await;
        let (plain, rag) = (&graded[0], &graded[1]);
        let (ps, rs) = (plain.score.as_ref().unwrap(), rag.score.as_ref().unwrap());
        plain_sum += ps.fact_score;
        rag_sum += rs.fact_score;
        println!(
            "| {} | {} | {}/{} | {}/{} | {} | {} |",
            case.id,
            case.question,
            ps.found,
            ps.total,
            rs.found,
            rs.total,
            mark(rs.source_retrieved),
            mark(rs.cited_expected)
        );
        details.push((case, graded));
    }
    let n = CASES.len() as f64;
    println!(
        "| | **average** | **{:.0}%** | **{:.0}%** | | |",
        plain_sum / n * 100.0,
        rag_sum / n * 100.0
    );
    for (case, graded) in details {
        println!("\n## {} {}\nexpected: {}", case.id, case.question, case.expect);
        for g in graded {
            let a = &g.answer;
            let label = match a.mode {
                Mode::Plain => "without RAG",
                Mode::Rag => "with RAG",
            };
            match &a.error {
                Some(e) => println!("\n### {label}: error: {e}"),
                None => println!("\n### {label}\n{}", a.text),
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
    let agent = Arc::new(Agent { llm, index });

    // `app eval`: run the control questions, print the comparison, exit.
    if std::env::args().nth(1).as_deref() == Some("eval") {
        run_eval(&agent, DEFAULT_TOP_K).await;
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
    fn citations_are_parsed() {
        assert_eq!(citations("Port 4000 [1]. Unit [2, 3] and [1]."), vec![1, 2, 3]);
        assert_eq!(citations("an array [x] or a[0] index [ 4 ]"), vec![0, 4]);
        assert!(citations("no brackets").is_empty());
    }

    #[test]
    fn control_questions_are_grounded_in_the_corpus() {
        assert_eq!(CASES.len(), 10);
        let docs = builtin_documents();
        let index_docs: Vec<(&str, &str)> = index::BUILTIN.to_vec();
        assert_eq!(docs.len(), index_docs.len());
        for case in CASES {
            assert!(!case.facts.is_empty() && !case.sources.is_empty(), "{}", case.id);
            let question = case.question.to_lowercase();
            let mut text = String::new();
            for (file, section) in case.sources {
                let (_, body) = index_docs
                    .iter()
                    .find(|(path, _)| path == file)
                    .unwrap_or_else(|| panic!("{}: no file {file}", case.id));
                // The section's own heading is in the file.
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
    async fn retrieval_finds_the_expected_source_for_every_control_question() {
        let index = Index::build(builtin_documents(), index::local()).await.unwrap();
        for case in CASES {
            let hits = index.search(case.question, DEFAULT_TOP_K).await.unwrap();
            assert!(
                hits.iter()
                    .any(|h| case.sources.iter().any(|(file, _)| h.chunk.source == *file)),
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

    /// Request bodies the fake model has seen.
    type Seen = Arc<Mutex<Vec<Value>>>;

    /// A fake `/chat/completions` that knows nothing about the repo: with
    /// context it echoes the context back (a perfectly faithful reader),
    /// without it it says it doesn't know.
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
        let user = body["messages"][1]["content"].as_str().unwrap_or_default();
        let content = match user.strip_prefix("Context:") {
            Some(context) => format!("From the documents:{context}"),
            None => "I don't know the details of this repository.".to_string(),
        };
        Json(json!({
            "choices": [{ "message": { "role": "assistant", "content": content } }],
            "usage": { "prompt_tokens": user.len() / 4, "completion_tokens": 9 },
        }))
    }

    async fn test_agent(base: &str) -> Agent {
        Agent {
            llm: Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{base}/v1/chat/completions"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            },
            index: Index::build(builtin_documents(), index::local()).await.unwrap(),
        }
    }

    #[tokio::test]
    async fn rag_merges_chunks_with_the_question_and_beats_plain_on_the_control_set() {
        let seen: Seen = Arc::default();
        let base = spawn(
            Router::new()
                .route("/v1/chat/completions", post(fake_chat))
                .with_state(seen.clone()),
        )
        .await;
        let agent = test_agent(&base).await;

        let (mut plain_sum, mut rag_sum) = (0.0, 0.0);
        for case in CASES {
            let graded = ask(&agent, case.question, Some(case), None, DEFAULT_TOP_K).await;
            assert_eq!(graded.len(), 2);
            let (plain, rag) = (&graded[0], &graded[1]);
            assert_eq!(plain.answer.mode, Mode::Plain);
            assert_eq!(rag.answer.mode, Mode::Rag);
            assert!(plain.answer.error.is_none() && rag.answer.error.is_none());

            assert_eq!(plain.answer.messages.len(), 2);
            assert_eq!(plain.answer.messages[1]["content"], case.question);
            assert!(plain.answer.sources.is_empty());

            let system = rag.answer.messages[0]["content"].as_str().unwrap();
            assert!(system.starts_with(BASE_PROMPT) && system.ends_with(RAG_RULES));
            let user = rag.answer.messages[1]["content"].as_str().unwrap();
            assert_eq!(rag.answer.sources.len(), DEFAULT_TOP_K);
            for hit in &rag.answer.sources {
                assert!(user.contains(&format!("[{}] {} — {}", hit.rank, hit.chunk.source, hit.chunk.section)));
                assert!(user.contains(hit.chunk.text.trim()));
            }
            assert!(user.ends_with(&format!("Question: {}", case.question)));

            let (ps, rs) = (plain.score.as_ref().unwrap(), rag.score.as_ref().unwrap());
            assert_eq!(ps.found, 0, "{}", case.id);
            assert_eq!(ps.source_retrieved, None);
            assert_eq!(rs.source_retrieved, Some(true), "{}", case.id);
            assert_eq!(rs.cited_expected, Some(true), "{}", case.id);
            plain_sum += ps.fact_score;
            rag_sum += rs.fact_score;
        }
        let n = CASES.len() as f64;
        assert_eq!(plain_sum, 0.0);
        assert!(rag_sum / n >= 0.7, "RAG facts {:.2}", rag_sum / n);

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2 * CASES.len());
        assert!(bodies.iter().all(|b| b["model"] == "test-model" && b["temperature"] == 0.0));
    }

    #[tokio::test]
    async fn http_api_answers_in_both_modes_and_reports_errors() {
        let seen: Seen = Arc::default();
        let base = spawn(
            Router::new()
                .route("/v1/chat/completions", post(fake_chat))
                .with_state(seen.clone()),
        )
        .await;
        let agent = Arc::new(test_agent(&base).await);
        let url = spawn(app(AppState { agent })).await;
        let http = reqwest::Client::new();

        let page = http.get(&url).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>First RAG Query</title>"));

        let cfg: Value = http.get(format!("{url}/api/config")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cfg["model"], "test-model");
        assert!(cfg["chunks"].as_u64().unwrap() > 100);

        let cases: Value = http.get(format!("{url}/api/cases")).send().await.unwrap().json().await.unwrap();
        assert_eq!(cases.as_array().unwrap().len(), 10);
        assert_eq!(cases[0]["sources"][0]["file"], "DEPLOYMENT.md");

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

        let both = post(json!({ "case_id": "q03", "k": 3 })).await;
        assert!(both["error"].is_null(), "{both}");
        assert_eq!(both["case"]["id"], "q03");
        assert_eq!(both["answers"][0]["mode"], "plain");
        assert_eq!(both["answers"][1]["mode"], "rag");
        assert_eq!(both["answers"][1]["sources"].as_array().unwrap().len(), 3);
        assert_eq!(both["answers"][1]["sources"][0]["rank"], 1);
        assert!(both["answers"][1]["sources"][0]["chunk_id"].is_string());
        assert_eq!(both["answers"][1]["score"]["found"], 1);
        assert_eq!(both["answers"][0]["score"]["found"], 0);

        let free = post(json!({ "question": "What does lesson 05 compare?", "mode": "rag" })).await;
        assert_eq!(free["answers"].as_array().unwrap().len(), 1);
        assert!(free["case"].is_null());
        assert!(free["answers"][0]["score"].is_null());

        let empty = post(json!({ "question": "  " })).await;
        assert!(empty["error"].as_str().unwrap().contains("question"));
        let unknown = post(json!({ "case_id": "q99" })).await;
        assert!(unknown["error"].as_str().unwrap().contains("q99"));

        // A broken model is reported per answer; retrieval still shows.
        let broken = Agent {
            llm: Llm {
                http: reqwest::Client::new(),
                endpoint: format!("{base}/v1/nothing-here"),
                api_key: "test-key".to_string(),
                model: "test-model".to_string(),
            },
            index: Index::build(builtin_documents(), index::local()).await.unwrap(),
        };
        let answer = broken.ask("Which port does a lesson get?", Mode::Rag, 2).await;
        assert!(answer.error.as_deref().unwrap().contains("404"));
        assert_eq!(answer.sources.len(), 2);
    }
}

mod agent;
mod github;
mod pipeline;
mod redact;
mod report;
mod verdict;
mod vds;

use std::io::Write as _;
use std::path::PathBuf;
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
use tokio::sync::Mutex;

use agent::Diagnosis;
use github::{GitHub, Issue, Run};
use redact::Redactor;
use report::{IssueAction, Summary};
use vds::Vds;

/// More subjects than this in one run (a `lessons=all` redeploy gone wrong)
/// are listed from the run's data alone, without the model.
const MAX_SUBJECTS: usize = 6;

const USAGE: &str = "Usage:
  deploy-doctor [serve]                       the page (PORT, default 3000)
  deploy-doctor diagnose --run-id N           diagnose a CI/CD run and post the issues
                         [--dry-run]          print only, post nothing
                         [--out FILE]         also write the Markdown to FILE";

pub fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// At most `max` characters, with `…` when something was cut.
pub fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    format!("{cut}…")
}

// ---------------------------------------------------------------------------
// The LLM.
// ---------------------------------------------------------------------------

struct Llm {
    http: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    /// Kept as raw JSON so `tool_calls` (and provider extras such as
    /// DeepSeek's `reasoning_content`) go back to the model exactly as they came.
    message: Value,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

impl Llm {
    async fn chat(&self, messages: &[Value], tools: &[Value]) -> Result<Value, String> {
        let Some(api_key) = &self.api_key else {
            return Err("OPENAI_API_KEY is not set.".to_string());
        };
        let mut body = json!({ "model": self.model, "messages": messages, "temperature": 0.1 });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(api_key)
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
        serde_json::from_str::<ChatCompletionResponse>(&text)
            .map_err(|e| format!("Failed to parse LLM response: {e}"))?
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| "The model returned no choices.".to_string())
    }
}

// ---------------------------------------------------------------------------
// A diagnosis of one run.
// ---------------------------------------------------------------------------

struct Settings {
    model: String,
    /// `owner/name`.
    repo: String,
    language: String,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<Llm>,
    github: Arc<GitHub>,
    vds: Arc<Vds>,
    redactor: Arc<Redactor>,
    settings: Arc<Settings>,
    /// One diagnosis at a time: they share the VDS and the issues.
    lock: Arc<Mutex<()>>,
}

#[derive(Serialize, Default)]
struct DiagnoseReply {
    run: Option<Run>,
    jobs: usize,
    failed_jobs: usize,
    diagnoses: Vec<Diagnosis>,
    /// Keys whose jobs all succeeded in this run.
    green: Vec<String>,
    issues: Vec<IssueAction>,
    posted: bool,
    markdown: String,
    error: Option<String>,
    post_error: Option<String>,
    ms: u64,
}

async fn diagnose(app: &AppState, run_id: u64, post: bool) -> DiagnoseReply {
    let _busy = app.lock.lock().await;
    let started = Instant::now();
    let mut reply = DiagnoseReply::default();
    if let Err(e) = run_diagnosis(app, run_id, post, &mut reply).await {
        reply.error = Some(e);
    }
    reply.ms = started.elapsed().as_millis() as u64;
    reply.markdown = report::summary_markdown(&Summary {
        run: reply.run.as_ref(),
        diagnoses: &reply.diagnoses,
        green: &reply.green,
        issues: &reply.issues,
        error: reply.error.as_deref(),
        post_error: reply.post_error.as_deref(),
        model: &app.settings.model,
    });
    reply
}

async fn run_diagnosis(app: &AppState, run_id: u64, post: bool, reply: &mut DiagnoseReply) -> Result<(), String> {
    let run = app.github.run(run_id).await?;
    let jobs = app.github.jobs(run_id).await?;
    reply.run = Some(run.clone());
    reply.jobs = jobs.len();
    reply.failed_jobs = jobs.iter().filter(|job| pipeline::failed(job)).count();
    let (subjects, green) = pipeline::subjects(&jobs);
    reply.green = green;
    for (i, subject) in subjects.iter().enumerate() {
        let diagnosis = if i < MAX_SUBJECTS {
            agent::diagnose_subject(app, &run, &jobs, subject).await
        } else {
            Diagnosis::skipped(
                subject,
                &format!("more than {MAX_SUBJECTS} subjects in one run, so this one was not investigated"),
            )
        };
        reply.diagnoses.push(diagnosis);
    }
    if post {
        match post_issues(app, &run, &reply.diagnoses, &reply.green).await {
            Ok(actions) => {
                reply.issues = actions;
                reply.posted = true;
            }
            Err(e) => reply.post_error = Some(e),
        }
    }
    Ok(())
}

/// Comments on the issue, then closes it.
async fn recover(app: &AppState, issue: &Issue, body: &str) -> Result<(), String> {
    app.github.comment(issue.number, body).await?;
    app.github.close(issue.number).await
}

/// One open issue per broken subject: created the first time, commented on
/// after that, closed when the subject is healthy again. A known issue is
/// opened once and not commented on while it stays the same.
async fn post_issues(app: &AppState, run: &Run, diagnoses: &[Diagnosis], green: &[String]) -> Result<Vec<IssueAction>, String> {
    if !app.github.has_token() {
        return Err("GITHUB_TOKEN is not set, so no issue was touched (use --dry-run to only print).".to_string());
    }
    let open = app.github.issues("open").await?;
    let find = |key: &str| -> Option<Issue> {
        let marker = report::marker(key);
        open.iter().find(|issue| issue.has_marker(&marker)).cloned()
    };
    let mut actions = Vec::new();
    for d in diagnoses {
        let key = d.subject.key.as_str();
        let body = report::diagnosis_markdown(run, d, &app.settings.model);
        let result = match (d.verdict.status.as_str(), find(key)) {
            ("broken", Some(issue)) => app
                .github
                .comment(issue.number, &body)
                .await
                .map(|url| IssueAction::new(key, "commented", Some(issue.number), Some(url))),
            ("broken" | "known_issue", None) => app
                .github
                .create_issue(&report::issue_title(d), &body)
                .await
                .map(|issue| IssueAction::new(key, "created", Some(issue.number), Some(issue.html_url))),
            ("known_issue", Some(issue)) => Ok(IssueAction::new(key, "unchanged", Some(issue.number), Some(issue.html_url))),
            ("healthy", Some(issue)) => {
                let note = format!("✅ Recovered in run [{}]({}).\n\n{body}", run.id, run.html_url);
                recover(app, &issue, &note)
                    .await
                    .map(|()| IssueAction::new(key, "closed", Some(issue.number), Some(issue.html_url.clone())))
            }
            _ => Ok(IssueAction::new(key, "none", None, None)),
        };
        actions.push(result.unwrap_or_else(|e| IssueAction::failed(key, e)));
    }
    for key in green {
        if let Some(issue) = find(key.as_str()) {
            let note = format!(
                "{}\n✅ All of {key}'s jobs succeeded in run [{}]({}).",
                report::marker(key),
                run.id,
                run.html_url
            );
            let action = match recover(app, &issue, &note).await {
                Ok(()) => IssueAction::new(key, "closed", Some(issue.number), Some(issue.html_url)),
                Err(e) => IssueAction::failed(key, e),
            };
            actions.push(action);
        }
    }
    Ok(actions)
}

// ---------------------------------------------------------------------------
// HTTP.
// ---------------------------------------------------------------------------

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// `{"run_id": 123}` (a number or a string). The page never posts.
async fn diagnose_api(State(app): State<AppState>, Json(req): Json<Value>) -> Json<DiagnoseReply> {
    let run_id = req["run_id"]
        .as_u64()
        .or_else(|| req["run_id"].as_str().and_then(|s| s.trim().parse().ok()));
    Json(match run_id {
        Some(id) => diagnose(&app, id, false).await,
        None => DiagnoseReply {
            error: Some("`run_id` must be a CI/CD run id.".to_string()),
            ..DiagnoseReply::default()
        },
    })
}

async fn runs_api(State(app): State<AppState>) -> Json<Value> {
    Json(match app.github.recent_runs().await {
        Ok(runs) => json!({ "runs": runs, "error": null }),
        Err(e) => json!({ "runs": [], "error": e }),
    })
}

async fn issues_api(State(app): State<AppState>) -> Json<Value> {
    Json(match app.github.issues("all").await {
        Ok(issues) => json!({ "issues": issues, "error": null }),
        Err(e) => json!({ "issues": [], "error": e }),
    })
}

async fn status_api(State(app): State<AppState>) -> Json<Value> {
    Json(json!({
        "repo": app.settings.repo,
        "model": app.settings.model,
        "language": app.settings.language,
        "llm_key": app.llm.api_key.is_some(),
        "github_token": app.github.has_token(),
        "vds": app.vds.describe(),
        "workflow": github::CI_WORKFLOW,
        "max_rounds": agent::MAX_ROUNDS,
    }))
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/api/status", get(status_api))
        .route("/api/runs", get(runs_api))
        .route("/api/issues", get(issues_api))
        .route("/api/diagnose", post(diagnose_api))
        .with_state(state)
}

fn http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build HTTP client")
}

// ---------------------------------------------------------------------------
// The command line: what the GitHub Action runs.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
struct Cli {
    run_id: u64,
    dry_run: bool,
    out: Option<PathBuf>,
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut run_id = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--run-id" => {
                let value = it.next().ok_or("--run-id needs a value.")?;
                run_id = Some(
                    value
                        .trim()
                        .parse::<u64>()
                        .map_err(|_| format!("--run-id needs a run id, not `{value}`."))?,
                );
            }
            "--dry-run" => cli.dry_run = true,
            "--out" => cli.out = Some(PathBuf::from(it.next().ok_or("--out needs a value.")?)),
            other => return Err(format!("Unknown argument `{other}`.")),
        }
    }
    cli.run_id = run_id.ok_or("--run-id is required.")?;
    Ok(cli)
}

fn append(path: &str, text: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{text}")
}

/// `deploy-doctor diagnose ...`: printed, appended to the Actions job
/// summary, and posted as issues. The exit code is 1 only when the doctor
/// itself failed (no run, no model, no posting), never because of what it found.
async fn run_cli(state: AppState, cli: Cli) {
    let reply = diagnose(&state, cli.run_id, !cli.dry_run).await;
    println!("\n{}", reply.markdown);
    if let Some(path) = &cli.out
        && let Err(e) = std::fs::write(path, &reply.markdown)
    {
        eprintln!("Can't write {}: {e}", path.display());
    }
    if let Some(summary) = env_non_empty("GITHUB_STEP_SUMMARY")
        && let Err(e) = append(&summary, &reply.markdown)
    {
        eprintln!("Can't write the job summary: {e}");
    }
    for d in &reply.diagnoses {
        eprintln!(
            "{}: {} ({} tool calls, {} rounds, {} ms){}",
            d.subject.key,
            d.verdict.status,
            d.calls.len(),
            d.rounds,
            d.ms,
            d.error.as_ref().map(|e| format!(", error: {e}")).unwrap_or_default()
        );
    }
    for a in &reply.issues {
        eprintln!(
            "issue for {}: {}{}",
            a.key,
            a.action,
            a.url.as_ref().map(|u| format!(" {u}")).unwrap_or_default()
        );
    }
    if let Some(e) = &reply.error {
        eprintln!("The diagnosis failed: {e}");
    }
    if let Some(e) = &reply.post_error {
        eprintln!("Posting failed: {e}");
    }
    let failed = reply.error.is_some()
        || reply.post_error.is_some()
        || reply.diagnoses.iter().any(|d| d.error.is_some())
        || reply.issues.iter().any(|a| a.error.is_some());
    if failed {
        std::process::exit(1);
    }
}

fn vds_from_env() -> Vds {
    let host = env_non_empty("VDS_HOST");
    let default = if host.is_some() { "ssh" } else { "local" };
    let mode = env_non_empty("VDS_MODE").unwrap_or_else(|| default.to_string());
    match mode.trim() {
        "ssh" => match (host, env_non_empty("VDS_USER"), env_non_empty("VDS_SSH_KEY_PATH")) {
            (Some(host), Some(user), Some(key)) => Vds::Ssh { host, user, key },
            _ => Vds::Off("VDS_MODE=ssh needs VDS_HOST, VDS_USER and VDS_SSH_KEY_PATH.".to_string()),
        },
        "local" => Vds::Local,
        "off" => Vds::Off("VDS_MODE=off: the server is not checked.".to_string()),
        other => Vds::Off(format!("Unknown VDS_MODE `{other}`: use ssh, local or off.")),
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match args.first().map(String::as_str) {
        Some("diagnose") => match parse_cli(&args[1..]) {
            Ok(cli) => Some(cli),
            Err(e) => {
                eprintln!("Error: {e}\n\n{USAGE}");
                std::process::exit(2);
            }
        },
        Some("help" | "--help" | "-h") => {
            println!("{USAGE}");
            return;
        }
        Some("serve") | None => None,
        Some(other) => {
            eprintln!("Unknown command `{other}`.\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    let base_url =
        env_non_empty("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.deepseek.com".to_string());
    let model = env_non_empty("OPENAI_MODEL").unwrap_or_else(|| "deepseek-v4-flash".to_string());
    // Without a key the doctor still runs: every verdict is the fallback.
    let api_key = env_non_empty("OPENAI_API_KEY");
    if api_key.is_none() {
        eprintln!("Warning: OPENAI_API_KEY is not set, so there will be no AI diagnosis.");
    }
    let token = env_non_empty("GITHUB_TOKEN");
    // GitHub Actions sets GITHUB_REPOSITORY and GITHUB_API_URL.
    let repo = env_non_empty("GITHUB_REPOSITORY")
        .unwrap_or_else(|| "CaptainDmitro/AI-Advent-Challenge".to_string());
    let github_api = env_non_empty("GITHUB_API_URL").unwrap_or_else(|| github::DEFAULT_API.to_string());
    let language = env_non_empty("REPORT_LANGUAGE").unwrap_or_else(|| "English".to_string());
    let vds = vds_from_env();
    let secrets: Vec<String> = [api_key.clone(), token.clone()].into_iter().flatten().collect();

    let state = AppState {
        llm: Arc::new(Llm {
            http: http_client(Duration::from_secs(180)),
            endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key,
            model: model.clone(),
        }),
        github: Arc::new(GitHub::new(
            http_client(Duration::from_secs(60)),
            &github_api,
            token,
            repo.clone(),
        )),
        vds: Arc::new(vds),
        redactor: Arc::new(Redactor::new(secrets)),
        settings: Arc::new(Settings {
            model,
            repo,
            language,
        }),
        lock: Arc::new(Mutex::new(())),
    };
    println!(
        "repository {}, VDS {}, GitHub token: {}",
        state.settings.repo,
        state.vds.describe(),
        if state.github.has_token() { "yes" } else { "no" }
    );

    if let Some(cli) = cli {
        run_cli(state, cli).await;
        return;
    }

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Path as UrlPath, Query};
    use axum::routing::patch;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;
    use vds::{FakeVds, Output};

    const TOKEN: &str = "ghs_testtoken1234567890";
    const LLM_SECRET: &str = "sk-test-secret-0123456789abcdef";

    #[test]
    fn the_command_line_is_parsed() {
        fn args(list: &[&str]) -> Vec<String> {
            list.iter().map(|s| s.to_string()).collect()
        }
        assert_eq!(
            parse_cli(&args(&["--run-id", " 42 ", "--dry-run", "--out", "x.md"])).unwrap(),
            Cli {
                run_id: 42,
                dry_run: true,
                out: Some(PathBuf::from("x.md"))
            }
        );
        assert_eq!(parse_cli(&args(&[])).unwrap_err(), "--run-id is required.");
        assert!(parse_cli(&args(&["--run-id", ""])).unwrap_err().contains("not ``"));
        assert!(parse_cli(&args(&["--post"])).unwrap_err().contains("Unknown argument"));
        assert_eq!(clip("abcdef", 3), "abc…");
        assert_eq!(clip("abc", 3), "abc");
    }

    // -----------------------------------------------------------------------
    // Fakes: GitHub, the LLM and the VDS.
    // -----------------------------------------------------------------------

    struct FakeIssue {
        number: u64,
        title: String,
        body: String,
        state: String,
        labels: Vec<String>,
        comments: Vec<String>,
    }

    #[derive(Default)]
    struct FakeGitHub {
        run: Value,
        jobs: Vec<Value>,
        logs: Vec<(u64, String)>,
        readme: String,
        issues: Vec<FakeIssue>,
    }

    type Shared = Arc<StdMutex<FakeGitHub>>;

    fn issue_json(issue: &FakeIssue) -> Value {
        json!({
            "number": issue.number, "title": issue.title, "body": issue.body, "state": issue.state,
            "html_url": format!("https://github.test/issues/{}", issue.number),
            "created_at": "2026-10-19T08:00:00Z", "updated_at": "2026-10-19T08:00:00Z"
        })
    }

    fn open_issue(number: u64, body: &str) -> FakeIssue {
        FakeIssue {
            number,
            title: format!("issue {number}"),
            body: body.to_string(),
            state: "open".to_string(),
            labels: Vec::new(),
            comments: Vec::new(),
        }
    }

    fn fake_github(gh: Shared) -> Router {
        let (run, runs, jobs, logs, file) = (gh.clone(), gh.clone(), gh.clone(), gh.clone(), gh.clone());
        let (list, create, close, comment) = (gh.clone(), gh.clone(), gh.clone(), gh);
        Router::new()
            .route(
                "/repos/{owner}/{name}/actions/runs/{id}",
                get(move || {
                    let gh = run.clone();
                    async move { Json(gh.lock().unwrap().run.clone()) }
                }),
            )
            .route(
                "/repos/{owner}/{name}/actions/workflows/{workflow}/runs",
                get(move || {
                    let gh = runs.clone();
                    async move { Json(json!({ "workflow_runs": [gh.lock().unwrap().run.clone()] })) }
                }),
            )
            .route(
                "/repos/{owner}/{name}/actions/runs/{id}/jobs",
                get(move || {
                    let gh = jobs.clone();
                    async move {
                        let jobs = gh.lock().unwrap().jobs.clone();
                        Json(json!({ "total_count": jobs.len(), "jobs": jobs }))
                    }
                }),
            )
            .route(
                "/repos/{owner}/{name}/actions/jobs/{id}/logs",
                get(move |UrlPath((_, _, id)): UrlPath<(String, String, u64)>| {
                    let gh = logs.clone();
                    async move {
                        let gh = gh.lock().unwrap();
                        gh.logs
                            .iter()
                            .find(|(job, _)| *job == id)
                            .map(|(_, log)| log.clone())
                            .unwrap_or_default()
                    }
                }),
            )
            .route(
                "/repos/{owner}/{name}/contents/{*path}",
                get(move || {
                    let gh = file.clone();
                    async move { gh.lock().unwrap().readme.clone() }
                }),
            )
            .route(
                "/repos/{owner}/{name}/issues",
                get(move |Query(q): Query<HashMap<String, String>>| {
                    let gh = list.clone();
                    async move {
                        let state = q.get("state").cloned().unwrap_or_else(|| "open".to_string());
                        let gh = gh.lock().unwrap();
                        let issues: Vec<Value> = gh
                            .issues
                            .iter()
                            .filter(|i| state == "all" || i.state == state)
                            .map(issue_json)
                            .collect();
                        Json(json!(issues))
                    }
                })
                .post(move |Json(body): Json<Value>| {
                    let gh = create.clone();
                    async move {
                        let mut gh = gh.lock().unwrap();
                        let number = gh.issues.len() as u64 + 1;
                        let labels: Vec<String> = body["labels"]
                            .as_array()
                            .map(|l| l.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                            .unwrap_or_default();
                        let mut issue = open_issue(number, body["body"].as_str().unwrap_or_default());
                        issue.title = body["title"].as_str().unwrap_or_default().to_string();
                        issue.labels = labels;
                        let json = issue_json(&issue);
                        gh.issues.push(issue);
                        Json(json)
                    }
                }),
            )
            .route(
                "/repos/{owner}/{name}/issues/{number}",
                patch(
                    move |UrlPath((_, _, number)): UrlPath<(String, String, u64)>, Json(body): Json<Value>| {
                        let gh = close.clone();
                        async move {
                            let mut gh = gh.lock().unwrap();
                            let issue = gh.issues.iter_mut().find(|i| i.number == number).unwrap();
                            issue.state = body["state"].as_str().unwrap_or("open").to_string();
                            Json(issue_json(issue))
                        }
                    },
                ),
            )
            .route(
                "/repos/{owner}/{name}/issues/{number}/comments",
                post(
                    move |UrlPath((_, _, number)): UrlPath<(String, String, u64)>, Json(body): Json<Value>| {
                        let gh = comment.clone();
                        async move {
                            let mut gh = gh.lock().unwrap();
                            let issue = gh.issues.iter_mut().find(|i| i.number == number).unwrap();
                            issue.comments.push(body["body"].as_str().unwrap_or_default().to_string());
                            Json(json!({ "id": issue.comments.len(), "html_url": format!("https://github.test/issues/{number}#c") }))
                        }
                    },
                ),
            )
    }

    /// A scripted OpenAI-compatible LLM: the n-th answer of a conversation
    /// is `script[n]` (the last one repeats).
    fn fake_llm(script: Vec<Value>, requests: Arc<StdMutex<Vec<Value>>>) -> Router {
        let script = Arc::new(script);
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let (script, requests) = (script.clone(), requests.clone());
                async move {
                    let answered = body["messages"]
                        .as_array()
                        .map(|m| m.iter().filter(|m| m["role"] == "assistant").count())
                        .unwrap_or(0);
                    requests.lock().unwrap().push(body);
                    let message = script[answered.min(script.len() - 1)].clone();
                    Json(json!({ "choices": [{ "message": message }] }))
                }
            }),
        )
    }

    fn calls(list: &[(&str, Value)]) -> Value {
        let tool_calls: Vec<Value> = list
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                json!({ "id": format!("call_{i}"), "type": "function",
                        "function": { "name": name, "arguments": args.to_string() } })
            })
            .collect();
        json!({ "role": "assistant", "content": null, "tool_calls": tool_calls })
    }

    fn out(code: i32, stdout: &str, stderr: &str) -> Output {
        Output {
            code,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
        }
    }

    fn fake_vds(answers: Vec<(&str, Output)>) -> (Vds, Arc<StdMutex<FakeVds>>) {
        let fake = Arc::new(StdMutex::new(FakeVds {
            answers: answers.into_iter().map(|(k, o)| (k.to_string(), o)).collect(),
            scripts: Vec::new(),
        }));
        (Vds::Fake(fake.clone()), fake)
    }

    fn job(id: u64, name: &str, conclusion: &str, failed_step: Option<&str>) -> Value {
        let mut steps = vec![json!({ "name": "Set up job", "status": "completed", "conclusion": "success", "number": 1 })];
        if let Some(step) = failed_step {
            steps.push(json!({ "name": step, "status": "completed", "conclusion": "failure", "number": 2 }));
        }
        json!({ "id": id, "name": name, "status": "completed", "conclusion": conclusion,
                "html_url": format!("https://github.test/job/{id}"), "steps": steps })
    }

    fn run_json(conclusion: &str) -> Value {
        json!({ "id": 9001, "name": "CI/CD", "display_title": "Add lesson 35", "event": "push",
                "status": "completed", "conclusion": conclusion, "head_sha": "abcdef1234567890",
                "head_branch": "master", "html_url": "https://github.test/runs/9001",
                "run_attempt": 1, "created_at": "2026-10-19T08:00:00Z" })
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn test_state(llm_base: &str, github_base: &str, vds: Vds) -> AppState {
        AppState {
            llm: Arc::new(Llm {
                http: http_client(Duration::from_secs(10)),
                endpoint: format!("{llm_base}/chat/completions"),
                api_key: Some("test-key".to_string()),
                model: "test-model".to_string(),
            }),
            github: Arc::new(GitHub::new(
                http_client(Duration::from_secs(10)),
                github_base,
                Some(TOKEN.to_string()),
                "octo/demo".to_string(),
            )),
            vds: Arc::new(vds),
            redactor: Arc::new(Redactor::new(vec![TOKEN.to_string()])),
            settings: Arc::new(Settings {
                model: "test-model".to_string(),
                repo: "octo/demo".to_string(),
                language: "English".to_string(),
            }),
            lock: Arc::new(Mutex::new(())),
        }
    }

    fn failed_deploy_run() -> FakeGitHub {
        FakeGitHub {
            run: run_json("failure"),
            jobs: vec![
                job(1, "discover", "success", None),
                job(2, "build-test (34. Project File Assistant)", "success", None),
                job(3, "build-test (35. Deploy Doctor)", "success", None),
                job(4, "deploy (35. Deploy Doctor)", "failure", Some("Verify service is up")),
            ],
            logs: vec![(
                4,
                format!(
                    "\u{feff}2026-10-19T08:01:00.0000000Z ##[group]Run set -euo pipefail\n\
                     2026-10-19T08:01:09.0000000Z Service ai-advent-lesson-35 did not become active in time.\n\
                     2026-10-19T08:01:09.1000000Z ##[error]Process completed with exit code 1.\n\
                     2026-10-19T08:01:09.2000000Z leaked {TOKEN}\n"
                ),
            )],
            readme: "# 35. Deploy Doctor\n\n## Configuration\n\n| `OPENAI_API_KEY` | yes |\n".to_string(),
            issues: vec![
                open_issue(1, "<!-- deploy-doctor:lesson-34 -->\nbroken build"),
                open_issue(2, "Someone's bug report"),
            ],
        }
    }

    const BROKEN: &str = r#"{"status": "broken", "failed_stage": "deploy",
        "root_cause": "OPENAI_API_KEY is empty in ~/apps/lesson-35/.env, so the app exits on start. The repository secret is missing.",
        "evidence": ["service_logs: Error: OPENAI_API_KEY environment variable is not set.", "inspect_env: OPENAI_API_KEY: EMPTY"],
        "suggested_fix": "Add the OPENAI_API_KEY repository secret and re-run CI/CD.",
        "commands_to_run": ["systemctl --user status ai-advent-lesson-35"]}"#;

    fn broken_deploy_script() -> Vec<Value> {
        vec![
            calls(&[("service_status", json!({})), ("probe", json!({ "path": "/" }))]),
            calls(&[
                ("service_logs", json!({ "lines": 50 })),
                ("inspect_env", json!({})),
                ("job_log", json!({ "job_id": 4, "grep": "error" })),
            ]),
            // Not accepted: a known issue has to say where it is documented.
            calls(&[(
                "submit_verdict",
                json!({ "status": "known_issue", "root_cause": "x", "evidence": ["y"] }),
            )]),
            calls(&[("submit_verdict", serde_json::from_str(BROKEN).unwrap())]),
        ]
    }

    fn broken_vds() -> (Vds, Arc<StdMutex<FakeVds>>) {
        fake_vds(vec![
            ("systemctl --user show", out(0, "ActiveState=activating\nSubState=auto-restart\nNRestarts=7", "")),
            (
                "journalctl",
                out(0, &format!("Error: OPENAI_API_KEY environment variable is not set.\nAuthorization: Bearer {LLM_SECRET}"), ""),
            ),
            (".env", out(0, "PORT=4035\nOPENAI_API_KEY=\nOPENAI_MODEL=deepseek-v4-flash\n", "")),
            ("curl", out(7, "000 0.000s", "curl: (7) Failed to connect to 127.0.0.1 port 4035")),
        ])
    }

    #[tokio::test]
    async fn a_failed_deploy_is_investigated_and_reported_as_one_issue() {
        let gh: Shared = Arc::new(StdMutex::new(failed_deploy_run()));
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(broken_deploy_script(), requests.clone())).await;
        let github = spawn(fake_github(gh.clone())).await;
        let (vds, fake) = broken_vds();
        let state = test_state(&llm, &github, vds);

        let reply = diagnose(&state, 9001, true).await;
        assert_eq!(reply.error, None);
        assert_eq!(reply.post_error, None);
        assert_eq!((reply.jobs, reply.failed_jobs), (4, 1));
        assert_eq!(reply.green, ["lesson-34", "pipeline"]);
        assert_eq!(reply.diagnoses.len(), 1);
        let d = &reply.diagnoses[0];
        assert_eq!(d.subject.key, "lesson-35");
        assert_eq!(d.error, None);
        assert_eq!(d.verdict.status, "broken");
        assert!(d.verdict.ai);
        assert_eq!(d.rounds, 4);
        let tools: Vec<&str> = d.calls.iter().map(|c| c.tool.as_str()).collect();
        assert_eq!(
            tools,
            ["service_status", "probe", "service_logs", "inspect_env", "job_log", "submit_verdict", "submit_verdict"]
        );
        assert!(d.calls[0].result.contains("ActiveState=activating"));
        assert!(d.calls[1].result.contains("exit code 7"));
        assert!(d.calls[2].result.contains("Bearer ***"));
        assert!(d.calls[3].result.contains("OPENAI_API_KEY: EMPTY"));
        assert!(d.calls[3].result.contains("PORT = 4035"));
        assert!(d.calls[4].result.contains("did not become active in time"));
        assert!(d.calls[4].result.contains("leaked ***"));
        assert!(d.calls[5].is_error && d.calls[5].result.contains("doc_reference"));
        assert!(!d.calls[6].is_error);

        // The probes were built from the lesson, never from the model.
        let scripts = fake.lock().unwrap().scripts.clone();
        assert_eq!(scripts.len(), 4);
        assert!(scripts.iter().all(|s| s.starts_with("export XDG_RUNTIME_DIR")));
        assert!(scripts[1].contains("'http://127.0.0.1:4035/'"));
        assert!(scripts[2].contains("-u ai-advent-lesson-35 -n 50"));

        // What the model saw: DEPLOYMENT.md, the run, the VDS tools, and no secret.
        let requests = requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 4);
        let first = &requests[0];
        assert!(first["messages"][0]["content"].as_str().unwrap().contains("Lesson 04"));
        let user = first["messages"][1]["content"].as_str().unwrap();
        assert!(user.contains("- [job 4] deploy (35. Deploy Doctor): failure"));
        assert!(user.contains("step 2 \"Verify service is up\": failure"));
        assert!(user.contains("Service `ai-advent-lesson-35`, port 4035. Deployed in this run: yes."));
        let offered: Vec<&str> = first["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert!(offered.contains(&"inspect_env"));
        for request in &requests {
            let text = request.to_string();
            assert!(!text.contains(LLM_SECRET) && !text.contains(TOKEN));
        }

        // One new issue for lesson 35; lesson 34's issue closed; the other untouched.
        assert_eq!(
            reply.issues,
            [
                IssueAction::new("lesson-35", "created", Some(3), Some("https://github.test/issues/3".into())),
                IssueAction::new("lesson-34", "closed", Some(1), Some("https://github.test/issues/1".into())),
            ]
        );
        {
            let gh = gh.lock().unwrap();
            let created = &gh.issues[2];
            assert_eq!(
                created.title,
                "Deploy Doctor: lesson 35 — OPENAI_API_KEY is empty in ~/apps/lesson-35/.env, so the app exits on start"
            );
            assert!(created.body.starts_with("<!-- deploy-doctor:lesson-35 -->\n### 🔴 Broken"));
            assert!(created.body.contains("Add the OPENAI_API_KEY repository secret"));
            assert!(created.body.contains("7 tool calls in 4 rounds"));
            assert_eq!(created.labels, ["deploy-doctor"]);
            assert_eq!(gh.issues[0].state, "closed");
            assert!(gh.issues[0].comments[0].contains("All of lesson-34's jobs succeeded in run [9001]"));
            assert_eq!(gh.issues[1].state, "open");
            assert!(gh.issues[1].comments.is_empty());
        }
        assert!(reply.markdown.contains("## 🩺 Deploy Doctor"));
        assert!(reply.markdown.contains("- lesson-35: created [#3](https://github.test/issues/3)"));

        // The same run again: a comment on the open issue, not a second issue.
        let again = diagnose(&state, 9001, true).await;
        assert_eq!(again.issues[0].action, "commented");
        let gh = gh.lock().unwrap();
        assert_eq!(gh.issues.len(), 3);
        assert_eq!(gh.issues[2].comments.len(), 1);
    }

    #[tokio::test]
    async fn a_healthy_deploy_closes_the_open_issue() {
        let gh: Shared = Arc::new(StdMutex::new(FakeGitHub {
            run: run_json("success"),
            jobs: vec![
                job(1, "discover", "success", None),
                job(2, "build-test (35. Deploy Doctor)", "success", None),
                job(3, "deploy (35. Deploy Doctor)", "success", None),
            ],
            issues: vec![open_issue(1, "<!-- deploy-doctor:lesson-35 -->\nOPENAI_API_KEY is empty")],
            ..FakeGitHub::default()
        }));
        let script = vec![
            calls(&[("service_status", json!({})), ("probe", json!({}))]),
            calls(&[(
                "submit_verdict",
                json!({ "status": "healthy", "failed_stage": "none", "root_cause": "",
                        "evidence": ["service_status: ActiveState=active", "probe: 200"],
                        "suggested_fix": "" }),
            )]),
        ];
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(script, requests)).await;
        let github = spawn(fake_github(gh.clone())).await;
        let (vds, _) = fake_vds(vec![
            ("systemctl --user show", out(0, "ActiveState=active\nSubState=running", "")),
            ("curl", out(0, "200 0.004s", "")),
        ]);
        let state = test_state(&llm, &github, vds);

        let reply = diagnose(&state, 9001, true).await;
        assert_eq!(reply.diagnoses[0].verdict.status, "healthy");
        assert!(reply.diagnoses[0].calls[1].result.contains("GET /: exit code 0\n200 0.004s"));
        assert_eq!(reply.issues.len(), 1);
        assert_eq!(reply.issues[0].action, "closed");
        let gh = gh.lock().unwrap();
        assert_eq!(gh.issues[0].state, "closed");
        assert!(gh.issues[0].comments[0].starts_with("✅ Recovered in run [9001]"));
        assert!(gh.issues[0].comments[0].contains("🟢 Healthy"));
    }

    #[tokio::test]
    async fn without_the_llm_a_build_failure_is_still_reported() {
        let gh: Shared = Arc::new(StdMutex::new(FakeGitHub {
            run: run_json("failure"),
            jobs: vec![
                job(1, "discover", "success", None),
                job(2, "build-test (33. Support Assistant)", "failure", Some("cargo test")),
                job(3, "deploy", "skipped", None),
            ],
            ..FakeGitHub::default()
        }));
        let github = spawn(fake_github(gh.clone())).await;
        // Nothing listens on port 9.
        let state = test_state("http://127.0.0.1:9", &github, Vds::Off("not in this test".into()));

        let reply = diagnose(&state, 9001, true).await;
        assert_eq!(reply.diagnoses.len(), 1);
        let d = &reply.diagnoses[0];
        assert!(d.error.as_deref().unwrap().starts_with("LLM request failed"));
        assert!(!d.verdict.ai);
        assert_eq!(d.verdict.status, "broken");
        assert_eq!(d.verdict.failed_stage, "test");
        assert_eq!(d.verdict.evidence, ["build-test (33. Support Assistant) › cargo test: failure"]);
        assert_eq!(reply.issues[0].action, "created");
        let gh = gh.lock().unwrap();
        assert!(gh.issues[0].title.starts_with("Deploy Doctor: lesson 33 — AI diagnosis unavailable"));
        assert!(gh.issues[0].body.contains("Open the failed job's log: https://github.test/job/2"));
    }

    #[tokio::test]
    async fn the_page_diagnoses_without_posting() {
        let gh: Shared = Arc::new(StdMutex::new(failed_deploy_run()));
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let llm = spawn(fake_llm(broken_deploy_script(), requests)).await;
        let github = spawn(fake_github(gh.clone())).await;
        let (vds, _) = broken_vds();
        let base = spawn(app(test_state(&llm, &github, vds))).await;
        let client = reqwest::Client::new();

        let reply: Value = client
            .post(format!("{base}/api/diagnose"))
            .json(&json!({ "run_id": "9001" }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(reply["error"], Value::Null);
        assert_eq!(reply["posted"], false);
        assert_eq!(reply["diagnoses"][0]["verdict"]["status"], "broken");
        assert_eq!(reply["diagnoses"][0]["subject"]["label"], "lesson 35 (`35. Deploy Doctor`)");
        assert_eq!(reply["issues"], json!([]));
        {
            let gh = gh.lock().unwrap();
            assert_eq!(gh.issues.len(), 2);
            assert!(gh.issues.iter().all(|i| i.state == "open" && i.comments.is_empty()));
        }

        let reply: Value = client
            .post(format!("{base}/api/diagnose"))
            .json(&json!({}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(reply["error"].as_str().unwrap().contains("run_id"));

        let get = |path: &str| {
            let (client, url) = (client.clone(), format!("{base}{path}"));
            async move { client.get(url).send().await.unwrap() }
        };
        let status: Value = get("/api/status").await.json().await.unwrap();
        assert_eq!(status["repo"], "octo/demo");
        assert_eq!(status["vds"], "fake");
        assert_eq!(status["github_token"], true);
        let runs: Value = get("/api/runs").await.json().await.unwrap();
        assert_eq!(runs["runs"][0]["id"], 9001);
        let issues: Value = get("/api/issues").await.json().await.unwrap();
        assert_eq!(issues["issues"].as_array().unwrap().len(), 1);
        assert_eq!(issues["issues"][0]["number"], 1);
        let page = get("/").await.text().await.unwrap();
        assert!(page.contains("<title>Deploy Doctor</title>"));
    }
}

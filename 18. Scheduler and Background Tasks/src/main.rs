use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use reqwest::header::{ACCEPT, USER_AGENT};
use rmcp::{
    ErrorData, RoleClient, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolRequestParams, CallToolResult, ContentBlock, Implementation, ServerCapabilities,
        ServerConfig, Tool,
    },
    service::RunningService,
    tool, tool_handler, tool_router,
    transport::{
        StreamableHttpClientTransport,
        streamable_http_client::StreamableHttpClientTransportConfig,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const GITHUB_API: &str = "https://api.github.com";
/// How many of the newest commits one collection run asks GitHub for.
const FETCH_PER_RUN: u32 = 30;
const MAX_MESSAGE_CHARS: usize = 200;
/// Upper bound on LLM <-> tool round trips for one question.
const MAX_TOOL_ROUNDS: usize = 5;
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

// Limits that keep a public, always-on instance within GitHub's unauthenticated
// rate limit (60 requests/hour) and keep the JSON file small.
const DEFAULT_WATCH_MINUTES: u32 = 60;
const MAX_INTERVAL_MINUTES: u32 = 7 * 24 * 60;
const MAX_WATCHED_REPOS: usize = 5;
const MAX_PENDING_REMINDERS: usize = 20;
const MAX_REMINDER_CHARS: usize = 300;
const DEFAULT_SUMMARY_HOURS: u32 = 24;
const MAX_SUMMARY_HOURS: u32 = 30 * 24;
const COMMITS_PER_REPO_IN_SUMMARY: usize = 10;

// Retention: older entries are dropped from the JSON file.
const KEEP_COMMITS_PER_REPO: usize = 200;
const KEEP_RUNS: usize = 500;
const KEEP_REMINDERS: usize = 100;
const KEEP_DIGESTS: usize = 50;

const QUIET_DIGEST: &str =
    "Nothing to report: no repositories are being watched and no reminders fired in this window.";

// ---------------------------------------------------------------------------
// Time, without pulling in a date crate.
// ---------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ`, the same shape GitHub uses, so the
/// two compare correctly as plain strings.
fn iso(ts: u64) -> String {
    let secs = ts % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}Z",
        civil_from_days((ts / 86_400) as i64),
        secs / 3_600,
        secs % 3_600 / 60,
        secs % 60
    )
}

/// Days since 1970-01-01 -> proleptic Gregorian date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

fn relative(at: u64, now: u64) -> String {
    if at <= now {
        return "due now".to_string();
    }
    let secs = at - now;
    if secs < 120 * 60 {
        format!("in {} min", secs.div_ceil(60))
    } else {
        format!("in {:.1} h", secs as f64 / 3_600.0)
    }
}

// ---------------------------------------------------------------------------
// Persistent state: everything the scheduler knows lives in one JSON file.
// ---------------------------------------------------------------------------

/// What a job does when it runs. Stored as `{"type": "collect_commits", ...}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum JobKind {
    /// Periodic data collection: fetch the newest commits, keep the unseen ones.
    CollectCommits { repository: String },
    /// Deferred, one-shot: fires once at its time.
    Reminder { text: String },
    /// The regular summary: the agent aggregates the collected data over MCP
    /// and the LLM writes a digest from it.
    Digest { window_hours: u32 },
}

impl JobKind {
    fn name(&self) -> &'static str {
        match self {
            Self::CollectCommits { .. } => "collect_commits",
            Self::Reminder { .. } => "reminder",
            Self::Digest { .. } => "digest",
        }
    }

    fn subject(&self) -> String {
        match self {
            Self::CollectCommits { repository } => repository.clone(),
            Self::Reminder { text } => text.clone(),
            Self::Digest { window_hours } => format!("last {window_hours}h"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::CollectCommits { repository } => format!("collect new commits of {repository}"),
            Self::Reminder { text } => format!("reminder {text:?}"),
            Self::Digest { window_hours } => format!("write a digest of the last {window_hours}h"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Job {
    id: u64,
    kind: JobKind,
    /// `Some` = periodic, rescheduled after every run; `None` = one-shot.
    every_secs: Option<u64>,
    /// Unix seconds. The scheduler runs the job on the first tick at or after it.
    next_run: u64,
    created_at: u64,
    active: bool,
    #[serde(default)]
    cancelled: bool,
    run_count: u64,
    last_run: Option<u64>,
    last_ok: Option<bool>,
    last_message: Option<String>,
}

/// One commit, trimmed down to what's worth handing to an LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CommitSummary {
    sha: String,
    date: String,
    author: String,
    message: String,
    url: String,
    /// When our collector first saw it (unix seconds).
    #[serde(default)]
    collected_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunRecord {
    job_id: u64,
    kind: String,
    subject: String,
    at: u64,
    ok: bool,
    message: String,
    ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FiredReminder {
    job_id: u64,
    text: String,
    fired_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Digest {
    at: u64,
    window_hours: u32,
    /// The `get_summary` text the digest was written from.
    summary: String,
    text: Option<String>,
    error: Option<String>,
    used_llm: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct StoreData {
    next_id: u64,
    jobs: Vec<Job>,
    /// `owner/repo` -> collected commits, newest first.
    commits: BTreeMap<String, Vec<CommitSummary>>,
    reminders: Vec<FiredReminder>,
    runs: Vec<RunRecord>,
    digests: Vec<Digest>,
}

fn keep_last<T>(items: &mut Vec<T>, n: usize) {
    if items.len() > n {
        items.drain(..items.len() - n);
    }
}

impl StoreData {
    fn add_job(&mut self, kind: JobKind, every_secs: Option<u64>, next_run: u64, now: u64) -> Job {
        self.next_id += 1;
        let job = Job {
            id: self.next_id,
            kind,
            every_secs,
            next_run,
            created_at: now,
            active: true,
            cancelled: false,
            run_count: 0,
            last_run: None,
            last_ok: None,
            last_message: None,
        };
        self.jobs.push(job.clone());
        job
    }

    fn count_active(&self, kind: &str) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.active && j.kind.name() == kind)
            .count()
    }

    fn is_active(&self, id: u64) -> bool {
        self.jobs.iter().any(|j| j.id == id && j.active)
    }

    fn due_jobs(&self, now: u64) -> Vec<Job> {
        let mut due: Vec<Job> = self
            .jobs
            .iter()
            .filter(|j| j.active && j.next_run <= now)
            .cloned()
            .collect();
        due.sort_by_key(|j| j.next_run);
        due
    }

    /// Records a run and reschedules the job: periodic jobs move to
    /// `now + every` (missed runs while the process was down collapse into
    /// one), one-shot jobs are done.
    fn finish_run(
        &mut self,
        job_id: u64,
        kind: &JobKind,
        now: u64,
        ok: bool,
        message: String,
        ms: u64,
    ) -> RunRecord {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.run_count += 1;
            job.last_run = Some(now);
            job.last_ok = Some(ok);
            job.last_message = Some(message.clone());
            match job.every_secs {
                Some(every) => job.next_run = now + every,
                None => job.active = false,
            }
        }
        let record = RunRecord {
            job_id,
            kind: kind.name().to_string(),
            subject: kind.subject(),
            at: now,
            ok,
            message,
            ms,
        };
        self.runs.push(record.clone());
        keep_last(&mut self.runs, KEEP_RUNS);
        record
    }

    fn cancel_job(&mut self, id: u64) -> Result<Job, String> {
        let job = self
            .jobs
            .iter_mut()
            .find(|j| j.id == id)
            .ok_or_else(|| format!("There is no job #{id}."))?;
        if matches!(job.kind, JobKind::Digest { .. }) {
            return Err(format!(
                "Job #{id} is the built-in digest job that keeps the agent reporting 24/7; \
                 it can't be cancelled."
            ));
        }
        if !job.active {
            let state = if job.cancelled { "cancelled" } else { "finished" };
            return Err(format!("Job #{id} is already {state}."));
        }
        job.active = false;
        job.cancelled = true;
        Ok(job.clone())
    }

    /// Stores the commits not seen before; returns how many were new.
    fn add_commits(&mut self, repository: &str, commits: Vec<CommitSummary>, now: u64) -> usize {
        let stored = self.commits.entry(repository.to_string()).or_default();
        let known: HashSet<String> = stored.iter().map(|c| c.sha.clone()).collect();
        let mut new = 0;
        for mut commit in commits {
            if known.contains(&commit.sha) {
                continue;
            }
            commit.collected_at = now;
            stored.push(commit);
            new += 1;
        }
        stored.sort_by(|a, b| b.date.cmp(&a.date));
        stored.truncate(KEEP_COMMITS_PER_REPO);
        new
    }

    fn fire_reminder(&mut self, job_id: u64, text: &str, now: u64) {
        self.reminders.push(FiredReminder {
            job_id,
            text: text.to_string(),
            fired_at: now,
        });
        keep_last(&mut self.reminders, KEEP_REMINDERS);
    }

    fn push_digest(&mut self, digest: Digest) {
        self.digests.push(digest);
        keep_last(&mut self.digests, KEEP_DIGESTS);
    }
}

/// The JSON file plus its in-memory copy. Every change goes through `write`,
/// which saves the whole file before returning, so a restart (or a redeploy)
/// never loses a scheduled job.
#[derive(Debug)]
struct Store {
    path: PathBuf,
    data: Mutex<StoreData>,
}

impl Store {
    fn open(path: PathBuf) -> Store {
        let data = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                let backup = path.with_extension("json.corrupt");
                eprintln!(
                    "{} is not valid ({e}); moving it to {} and starting empty.",
                    path.display(),
                    backup.display()
                );
                let _ = std::fs::rename(&path, backup);
                StoreData::default()
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => StoreData::default(),
            Err(e) => {
                eprintln!("Can't read {}: {e}; starting empty.", path.display());
                StoreData::default()
            }
        };
        Store {
            path,
            data: Mutex::new(data),
        }
    }

    fn read<R>(&self, f: impl FnOnce(&StoreData) -> R) -> R {
        f(&self.data.lock().unwrap())
    }

    fn write<R>(&self, f: impl FnOnce(&mut StoreData) -> R) -> R {
        let mut data = self.data.lock().unwrap();
        let result = f(&mut data);
        if let Err(e) = persist(&self.path, &data) {
            eprintln!("Failed to save {}: {e}", self.path.display());
        }
        result
    }
}

/// Write-then-rename, so a crash mid-write never leaves a half-written file.
fn persist(path: &Path, data: &StoreData) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(data)?)?;
    std::fs::rename(&tmp, path)
}

// ---------------------------------------------------------------------------
// The aggregated result: what `get_summary` returns and the digest is built on.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AuthorCount {
    author: String,
    commits: usize,
}

#[derive(Debug, Serialize)]
struct RepoSummary {
    repository: String,
    watched: bool,
    commits_in_window: usize,
    authors: Vec<AuthorCount>,
    /// Newest commits in the window, at most `COMMITS_PER_REPO_IN_SUMMARY`.
    commits: Vec<CommitSummary>,
    collections: usize,
    failed_collections: usize,
    last_collected: Option<String>,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReminderView {
    job_id: u64,
    text: String,
    fired_at: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    window_hours: u32,
    from: String,
    to: String,
    repositories: Vec<RepoSummary>,
    reminders_fired: Vec<ReminderView>,
    active_jobs: usize,
    runs: usize,
    failed_runs: usize,
    upcoming: Vec<String>,
}

fn summarize(data: &StoreData, now: u64, hours: u32) -> Summary {
    let from_ts = now.saturating_sub(u64::from(hours) * 3_600);
    let from = iso(from_ts);

    let watched: BTreeSet<&str> = data
        .jobs
        .iter()
        .filter(|j| j.active)
        .filter_map(|j| match &j.kind {
            JobKind::CollectCommits { repository } => Some(repository.as_str()),
            _ => None,
        })
        .collect();
    // Watched repositories, plus ones no longer watched that still had
    // activity inside the window.
    let mut names = watched.clone();
    for (repository, commits) in &data.commits {
        if commits.iter().any(|c| c.date >= from) {
            names.insert(repository.as_str());
        }
    }

    let repositories = names
        .into_iter()
        .map(|repository| {
            let commits: Vec<CommitSummary> = data
                .commits
                .get(repository)
                .map(|all| all.iter().filter(|c| c.date >= from).cloned().collect())
                .unwrap_or_default();
            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
            for c in &commits {
                *counts.entry(c.author.as_str()).or_default() += 1;
            }
            let mut authors: Vec<AuthorCount> = counts
                .into_iter()
                .map(|(author, commits)| AuthorCount {
                    author: author.to_string(),
                    commits,
                })
                .collect();
            authors.sort_by(|a, b| b.commits.cmp(&a.commits).then(a.author.cmp(&b.author)));

            let collections: Vec<&RunRecord> = data
                .runs
                .iter()
                .filter(|r| {
                    r.kind == "collect_commits" && r.subject == repository && r.at >= from_ts
                })
                .collect();
            RepoSummary {
                repository: repository.to_string(),
                watched: watched.contains(repository),
                commits_in_window: commits.len(),
                authors,
                commits: commits
                    .into_iter()
                    .take(COMMITS_PER_REPO_IN_SUMMARY)
                    .collect(),
                collections: collections.len(),
                failed_collections: collections.iter().filter(|r| !r.ok).count(),
                last_collected: collections
                    .iter()
                    .filter(|r| r.ok)
                    .map(|r| r.at)
                    .max()
                    .map(iso),
                last_error: collections
                    .iter()
                    .rev()
                    .find(|r| !r.ok)
                    .map(|r| r.message.clone()),
            }
        })
        .collect();

    let reminders_fired = data
        .reminders
        .iter()
        .rev()
        .filter(|r| r.fired_at >= from_ts)
        .map(|r| ReminderView {
            job_id: r.job_id,
            text: r.text.clone(),
            fired_at: iso(r.fired_at),
        })
        .collect();

    let runs: Vec<&RunRecord> = data.runs.iter().filter(|r| r.at >= from_ts).collect();
    let mut active: Vec<&Job> = data.jobs.iter().filter(|j| j.active).collect();
    active.sort_by_key(|j| j.next_run);

    Summary {
        window_hours: hours,
        from,
        to: iso(now),
        repositories,
        reminders_fired,
        active_jobs: active.len(),
        runs: runs.len(),
        failed_runs: runs.iter().filter(|r| !r.ok).count(),
        upcoming: active
            .iter()
            .take(3)
            .map(|j| format!("#{} {} at {}", j.id, j.kind.describe(), iso(j.next_run)))
            .collect(),
    }
}

/// The text the LLM reads: compact, one fact per line.
fn format_summary(s: &Summary) -> String {
    let mut out = format!(
        "Activity from {} to {} (last {}h).\n",
        s.from, s.to, s.window_hours
    );
    if s.repositories.is_empty() {
        out.push_str("\nRepositories: none are being watched.\n");
    } else {
        out.push_str("\nRepositories:\n");
        for r in &s.repositories {
            let authors = r
                .authors
                .iter()
                .map(|a| {
                    if a.commits > 1 {
                        format!("{} x{}", a.author, a.commits)
                    } else {
                        a.author.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "- {}{}: {} commit(s) in the window{}. Collected {} time(s), {} failed{}.\n",
                r.repository,
                if r.watched { "" } else { " (no longer watched)" },
                r.commits_in_window,
                if authors.is_empty() {
                    String::new()
                } else {
                    format!(" by {authors}")
                },
                r.collections,
                r.failed_collections,
                r.last_collected
                    .as_ref()
                    .map(|t| format!("; last successful collection {t}"))
                    .unwrap_or_default()
            ));
            for c in &r.commits {
                out.push_str(&format!(
                    "    - {} | {} | {} | {}\n",
                    c.sha, c.date, c.author, c.message
                ));
            }
            if r.commits_in_window > r.commits.len() {
                out.push_str(&format!(
                    "    - ... and {} more\n",
                    r.commits_in_window - r.commits.len()
                ));
            }
            if let Some(e) = &r.last_error {
                out.push_str(&format!("    Last error: {e}\n"));
            }
        }
    }
    if s.reminders_fired.is_empty() {
        out.push_str("\nReminders fired: none.\n");
    } else {
        out.push_str(&format!("\nReminders fired ({}):\n", s.reminders_fired.len()));
        for r in &s.reminders_fired {
            out.push_str(&format!("- {}: {}\n", r.fired_at, r.text));
        }
    }
    out.push_str(&format!(
        "\nJobs: {} active; {} run(s) in the window, {} failed.\n",
        s.active_jobs, s.runs, s.failed_runs
    ));
    for u in &s.upcoming {
        out.push_str(&format!("- next: {u}\n"));
    }
    out
}

fn format_job(job: &Job, now: u64) -> String {
    let schedule = match job.every_secs {
        Some(secs) => format!("every {} min", secs / 60),
        None => "once".to_string(),
    };
    let state = if job.cancelled {
        "cancelled".to_string()
    } else if !job.active {
        "done".to_string()
    } else {
        format!("next run {} ({})", iso(job.next_run), relative(job.next_run, now))
    };
    let last = match (job.last_ok, &job.last_message) {
        (Some(ok), Some(message)) => {
            format!(", last: {} - {message}", if ok { "ok" } else { "failed" })
        }
        _ => String::new(),
    };
    format!(
        "#{} {} | {schedule} | {state} | {} run(s){last}",
        job.id,
        job.kind.describe(),
        job.run_count
    )
}

// ---------------------------------------------------------------------------
// The collected data source: GitHub's public commits endpoint (as in lesson 17).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GitHub {
    client: reqwest::Client,
    api_base: String,
    token: Option<String>,
}

#[derive(Deserialize)]
struct GhCommit {
    sha: String,
    html_url: String,
    commit: GhCommitDetail,
    /// The linked GitHub account; `null` when the commit email isn't tied to one.
    author: Option<GhUser>,
}

#[derive(Deserialize)]
struct GhCommitDetail {
    message: String,
    author: Option<GhGitAuthor>,
}

#[derive(Deserialize)]
struct GhGitAuthor {
    name: Option<String>,
    date: Option<String>,
}

#[derive(Deserialize)]
struct GhUser {
    login: String,
}

#[derive(Deserialize)]
struct GhError {
    message: String,
}

impl From<GhCommit> for CommitSummary {
    fn from(c: GhCommit) -> Self {
        let git_author = c.commit.author.as_ref();
        let first_line = c.commit.message.lines().next().unwrap_or_default();
        let mut message: String = first_line.chars().take(MAX_MESSAGE_CHARS).collect();
        if first_line.chars().count() > MAX_MESSAGE_CHARS {
            message.push('…');
        }
        CommitSummary {
            sha: c.sha.chars().take(7).collect(),
            date: git_author.and_then(|a| a.date.clone()).unwrap_or_default(),
            author: c
                .author
                .map(|u| u.login)
                .or_else(|| git_author.and_then(|a| a.name.clone()))
                .unwrap_or_else(|| "unknown".to_string()),
            message,
            url: c.html_url,
            collected_at: 0,
        }
    }
}

impl GitHub {
    async fn list_commits(&self, repository: &str, limit: u32) -> Result<Vec<CommitSummary>, String> {
        let url = format!(
            "{}/repos/{repository}/commits",
            self.api_base.trim_end_matches('/')
        );
        let mut request = self
            .client
            .get(&url)
            .query(&[("per_page", limit.to_string())])
            .header(ACCEPT, "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            // GitHub rejects requests without a User-Agent.
            .header(USER_AGENT, "ai-advent-scheduler");
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("GitHub request failed: {e}"))?;
        let status = response.status();
        let rate_limited = response
            .headers()
            .get("x-ratelimit-remaining")
            .is_some_and(|v| v.as_bytes() == b"0");
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            let detail = serde_json::from_str::<GhError>(&body)
                .map(|e| e.message)
                .unwrap_or(body);
            return Err(match status.as_u16() {
                404 => format!("Repository {repository} was not found on GitHub."),
                409 => format!("Repository {repository} is empty: it has no commits yet."),
                403 | 429 if rate_limited => "GitHub API rate limit exceeded; \
                     the next scheduled run will try again."
                    .to_string(),
                _ => format!("GitHub API error ({status}): {detail}"),
            });
        }

        let commits: Vec<GhCommit> =
            serde_json::from_str(&body).map_err(|e| format!("Unexpected GitHub response: {e}"))?;
        Ok(commits.into_iter().map(CommitSummary::from).collect())
    }
}

fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 100
        && name != "."
        && name != ".."
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

// ---------------------------------------------------------------------------
// The MCP server: tools that schedule background jobs and read their results.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct WatchRepoArgs {
    /// Repository owner: a GitHub user or organization, e.g. "tokio-rs".
    owner: String,
    /// Repository name, e.g. "axum".
    repo: String,
    /// How often to collect new commits, in minutes. Default 60. The server enforces a minimum interval and says so if this is too small.
    every_minutes: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RemindMeArgs {
    /// What to remind about, e.g. "Call the dentist".
    text: String,
    /// Delay before the reminder fires, in whole minutes from now. 0-10080 (one week).
    in_minutes: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListJobsArgs {
    /// Also include finished and cancelled jobs. Default false.
    include_inactive: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CancelJobArgs {
    /// Id of the job to cancel, as returned by `watch_repo`, `remind_me` or `list_jobs`.
    job_id: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetSummaryArgs {
    /// Size of the window to aggregate, in hours back from now. 1-720, default 24.
    hours: Option<u32>,
}

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong *inside* a tool (bad argument, a limit, unknown job) is a
/// tool-level error the model can explain, not a JSON-RPC protocol error.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

#[derive(Debug, Clone)]
struct SchedulerServer {
    store: Arc<Store>,
    min_interval_minutes: u32,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl SchedulerServer {
    fn new(store: Arc<Store>, min_interval_minutes: u32) -> Self {
        Self {
            store,
            min_interval_minutes,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "watch_repo",
        description = "Start periodic data collection for a public GitHub repository: a background \
                       job fetches its newest commits every N minutes and saves the ones it hasn't \
                       seen yet. The first collection runs right away. Calling it again for the \
                       same repository changes the interval instead of adding a second job."
    )]
    async fn watch_repo(
        &self,
        Parameters(args): Parameters<WatchRepoArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let owner = args.owner.trim().to_ascii_lowercase();
        let repo = args.repo.trim().to_ascii_lowercase();
        if !is_valid_name(&owner) || !is_valid_name(&repo) {
            return tool_error(format!(
                "`owner` and `repo` must be plain GitHub names (letters, digits, '-', '_', '.'), \
                 got {owner:?} / {repo:?}."
            ));
        }
        let minutes = args.every_minutes.unwrap_or(DEFAULT_WATCH_MINUTES);
        if minutes < self.min_interval_minutes || minutes > MAX_INTERVAL_MINUTES {
            return tool_error(format!(
                "`every_minutes` must be between {} and {MAX_INTERVAL_MINUTES}, got {minutes}.",
                self.min_interval_minutes
            ));
        }
        let repository = format!("{owner}/{repo}");
        let kind = JobKind::CollectCommits {
            repository: repository.clone(),
        };
        let every = u64::from(minutes) * 60;
        let now = unix_now();

        let outcome = self.store.write(|d| {
            if let Some(job) = d.jobs.iter_mut().find(|j| j.active && j.kind == kind) {
                job.every_secs = Some(every);
                job.next_run = job.next_run.min(now + every);
                return Ok((job.clone(), false));
            }
            let watched = d.count_active("collect_commits");
            if watched >= MAX_WATCHED_REPOS {
                return Err(format!(
                    "Already watching {watched} repositories, the maximum. \
                     Cancel one with `cancel_job` first."
                ));
            }
            // `next_run = now`: the first collection happens on the next tick.
            Ok((d.add_job(kind.clone(), Some(every), now, now), true))
        });

        match outcome {
            Ok((job, created)) => {
                let text = if created {
                    format!(
                        "Now watching {repository}: job #{} collects new commits every {minutes} \
                         min. The first collection runs on the scheduler's next tick.",
                        job.id
                    )
                } else {
                    format!(
                        "Already watching {repository} (job #{}); it now runs every {minutes} min.",
                        job.id
                    )
                };
                tool_ok(text, json!({ "created": created, "job": job }))
            }
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "remind_me",
        description = "Schedule a one-time reminder that fires after a delay. When it fires it is \
                       saved and shown to the user, and it appears in the next digest."
    )]
    async fn remind_me(
        &self,
        Parameters(args): Parameters<RemindMeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let text = args.text.trim().to_string();
        if text.is_empty() || text.chars().count() > MAX_REMINDER_CHARS {
            return tool_error(format!(
                "`text` must be 1-{MAX_REMINDER_CHARS} characters long."
            ));
        }
        if args.in_minutes > MAX_INTERVAL_MINUTES {
            return tool_error(format!(
                "`in_minutes` must be at most {MAX_INTERVAL_MINUTES} (one week), got {}.",
                args.in_minutes
            ));
        }
        let now = unix_now();
        let due = now + u64::from(args.in_minutes) * 60;

        let outcome = self.store.write(|d| {
            let pending = d.count_active("reminder");
            if pending >= MAX_PENDING_REMINDERS {
                return Err(format!(
                    "There are already {pending} pending reminders, the maximum."
                ));
            }
            Ok(d.add_job(JobKind::Reminder { text: text.clone() }, None, due, now))
        });

        match outcome {
            Ok(job) => tool_ok(
                format!(
                    "Reminder #{} set for {} ({}): {text}",
                    job.id,
                    iso(due),
                    relative(due, now)
                ),
                json!({ "job": job }),
            ),
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "list_jobs",
        description = "List the scheduled background jobs: their ids, what they do, their schedule, \
                       the next run time, and the result of the last run."
    )]
    async fn list_jobs(
        &self,
        Parameters(args): Parameters<ListJobsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let include_inactive = args.include_inactive.unwrap_or(false);
        let now = unix_now();
        let jobs: Vec<Job> = self.store.read(|d| {
            d.jobs
                .iter()
                .filter(|j| include_inactive || j.active)
                .cloned()
                .collect()
        });
        let text = if jobs.is_empty() {
            "No jobs.".to_string()
        } else {
            let mut out = format!("{} job(s), current time {}:\n", jobs.len(), iso(now));
            for job in &jobs {
                out.push_str(&format!("- {}\n", format_job(job, now)));
            }
            out
        };
        tool_ok(text, json!({ "now": iso(now), "jobs": jobs }))
    }

    #[tool(
        name = "cancel_job",
        description = "Cancel a scheduled job by id, e.g. stop watching a repository or drop a \
                       pending reminder. Use `list_jobs` to find the id."
    )]
    async fn cancel_job(
        &self,
        Parameters(args): Parameters<CancelJobArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.store.write(|d| d.cancel_job(args.job_id)) {
            Ok(job) => tool_ok(
                format!("Cancelled job #{}: {}.", job.id, job.kind.describe()),
                json!({ "job": job }),
            ),
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "get_summary",
        description = "Aggregated result of the background jobs over the last N hours: for every \
                       watched repository, the commits collected in the window, who authored \
                       them, and how collection went; the reminders that fired; and job run \
                       statistics with what's scheduled next."
    )]
    async fn get_summary(
        &self,
        Parameters(args): Parameters<GetSummaryArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let hours = args
            .hours
            .unwrap_or(DEFAULT_SUMMARY_HOURS)
            .clamp(1, MAX_SUMMARY_HOURS);
        let summary = self.store.read(|d| summarize(d, unix_now(), hours));
        tool_ok(
            format_summary(&summary),
            serde_json::to_value(&summary).unwrap_or_default(),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SchedulerServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("scheduler", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Background jobs that keep running after the conversation ends: periodic \
                 collection of GitHub commits, one-time reminders, and an aggregated summary \
                 of everything they collected.",
            )
    }
}

fn mcp_service(
    store: Arc<Store>,
    min_interval_minutes: u32,
) -> StreamableHttpService<SchedulerServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(SchedulerServer::new(store.clone(), min_interval_minutes)),
        Default::default(),
        // Meant to be reached on the VDS by IP (see lesson 17), so any Host.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

// ---------------------------------------------------------------------------
// The agent: chats with the tools, and writes the periodic digest.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Agent {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    mcp_url: String,
    digest_language: String,
}

#[derive(Serialize, Debug)]
struct ToolInfo {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

impl From<&Tool> for ToolInfo {
    fn from(tool: &Tool) -> Self {
        ToolInfo {
            name: tool.name.to_string(),
            description: tool.description.as_ref().map(|d| d.to_string()),
            input_schema: Value::Object((*tool.input_schema).clone()),
        }
    }
}

#[derive(Serialize, Debug)]
struct McpInfo {
    url: String,
    server: Option<String>,
    tools: Vec<ToolInfo>,
}

/// One `tools/call` the agent made on the model's behalf.
#[derive(Serialize, Debug)]
struct ToolCallTrace {
    round: usize,
    name: String,
    arguments: Value,
    is_error: bool,
    result_text: String,
    ms: u128,
}

#[derive(Serialize, Debug, Default)]
struct AgentRun {
    mcp: Option<McpInfo>,
    tool_calls: Vec<ToolCallTrace>,
    llm_rounds: usize,
    answer: Option<String>,
    error: Option<String>,
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

fn system_prompt(now: &str) -> String {
    format!(
        "You are a scheduling assistant that keeps working 24/7 in the background. Your MCP \
         server has tools for background jobs: `watch_repo` starts periodic collection of a \
         GitHub repository's commits, `remind_me` schedules a one-time reminder, `list_jobs` and \
         `cancel_job` manage jobs, and `get_summary` returns the aggregated data the jobs have \
         collected. Always use the tools instead of guessing, and never claim a job exists \
         unless a tool confirmed it. To stop or change something, call `list_jobs` first to find \
         the job id. The current time is {now} (UTC). If a tool returns an error, explain it \
         plainly. Reply in the same language the user writes in."
    )
}

fn digest_prompt(language: &str) -> String {
    format!(
        "You write the periodic digest of a scheduling agent that runs 24/7. The user message is \
         aggregated data from the `get_summary` tool: commits collected from watched GitHub \
         repositories, reminders that fired, and background job statistics. Write a short digest \
         in plain text (no markdown headings, at most about 12 lines): for each repository, how \
         many commits landed, who was most active and the notable changes by short SHA; which \
         reminders fired; and any failed collections. Use only the data given and never invent \
         anything. Skip sections that have nothing in them. Write in {language}."
    )
}

/// An MCP tool, re-described as an OpenAI-style function (see lesson 17).
fn openai_tool(tool: &Tool) -> Value {
    let mut parameters = (*tool.input_schema).clone();
    parameters.remove("$schema");
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description.as_deref().unwrap_or_default(),
            "parameters": parameters,
        }
    })
}

fn json_object(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    }
}

fn tool_result_text(result: &CallToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        result
            .structured_content
            .as_ref()
            .map(Value::to_string)
            .unwrap_or_default()
    } else {
        text
    }
}

async fn call_mcp_tool(
    client: &RunningService<RoleClient, ()>,
    round: usize,
    name: &str,
    raw_arguments: &str,
) -> ToolCallTrace {
    let started = Instant::now();
    let arguments: Value = serde_json::from_str(raw_arguments)
        .unwrap_or_else(|_| Value::String(raw_arguments.to_string()));
    let outcome = match &arguments {
        Value::Object(map) => client
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(map.clone()))
            .await
            .map_err(|e| format!("tools/call failed: {e}")),
        _ => Err(format!(
            "The model sent arguments that aren't a JSON object: {raw_arguments}"
        )),
    };
    let (is_error, result_text) = match outcome {
        Ok(result) => (result.is_error.unwrap_or(false), tool_result_text(&result)),
        Err(e) => (true, e),
    };
    ToolCallTrace {
        round,
        name: name.to_string(),
        arguments,
        is_error,
        result_text,
        ms: started.elapsed().as_millis(),
    }
}

impl Agent {
    async fn connect(&self) -> Result<RunningService<RoleClient, ()>, String> {
        let transport = StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(self.mcp_url.clone()),
        );
        ().serve(transport)
            .await
            .map_err(|e| format!("MCP handshake with {} failed: {e}", self.mcp_url))
    }

    /// A chat turn: the LLM decides which scheduler tools to call.
    async fn run(&self, question: &str) -> AgentRun {
        let mut run = AgentRun::default();
        let outcome = tokio::time::timeout(RUN_TIMEOUT, async {
            let client = self.connect().await?;
            let result = self.converse(&client, question, &mut run).await;
            let _ = client.cancel().await;
            result
        })
        .await;
        match outcome {
            Ok(Ok(answer)) => run.answer = Some(answer),
            Ok(Err(e)) => run.error = Some(e),
            Err(_) => run.error = Some(format!("Timed out after {}s", RUN_TIMEOUT.as_secs())),
        }
        run
    }

    async fn converse(
        &self,
        client: &RunningService<RoleClient, ()>,
        question: &str,
        run: &mut AgentRun,
    ) -> Result<String, String> {
        let tools = client
            .list_all_tools()
            .await
            .map_err(|e| format!("tools/list failed: {e}"))?;
        run.mcp = Some(McpInfo {
            url: self.mcp_url.clone(),
            server: client.peer_info().and_then(|info| {
                let info = serde_json::to_value(&*info).ok()?;
                let server = &info["serverInfo"];
                Some(format!(
                    "{} v{}",
                    server["name"].as_str()?,
                    server["version"].as_str()?
                ))
            }),
            tools: tools.iter().map(ToolInfo::from).collect(),
        });
        let llm_tools: Vec<Value> = tools.iter().map(openai_tool).collect();

        let mut messages = vec![
            json!({ "role": "system", "content": system_prompt(&iso(unix_now())) }),
            json!({ "role": "user", "content": question }),
        ];

        for round in 1..=MAX_TOOL_ROUNDS {
            run.llm_rounds = round;
            let message = self.call_llm(&messages, &llm_tools).await?;
            let tool_calls = message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if tool_calls.is_empty() {
                return message_text(&message)
                    .ok_or_else(|| "The model returned an empty answer.".to_string());
            }

            messages.push(message);
            for call in tool_calls {
                let id = call["id"].as_str().unwrap_or_default().to_string();
                let name = call["function"]["name"].as_str().unwrap_or_default();
                let raw_arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
                let trace = call_mcp_tool(client, round, name, raw_arguments).await;
                println!(
                    "tools/call {name} {raw_arguments} -> {} in {} ms",
                    if trace.is_error { "error" } else { "ok" },
                    trace.ms
                );
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": trace.result_text,
                }));
                run.tool_calls.push(trace);
            }
        }
        Err(format!(
            "Gave up after {MAX_TOOL_ROUNDS} rounds of tool calls without a final answer."
        ))
    }

    /// The regular summary. Unlike a chat turn this is deterministic: the
    /// agent itself calls `get_summary` over MCP, then the LLM only writes
    /// the digest text from that aggregated result.
    async fn digest(&self, window_hours: u32, now: u64) -> Digest {
        let mut digest = Digest {
            at: now,
            window_hours,
            summary: String::new(),
            text: None,
            error: None,
            used_llm: false,
        };
        let outcome =
            tokio::time::timeout(RUN_TIMEOUT, self.write_digest(window_hours, &mut digest)).await;
        match outcome {
            Ok(Ok(text)) => digest.text = Some(text),
            Ok(Err(e)) => digest.error = Some(e),
            Err(_) => digest.error = Some(format!("Timed out after {}s", RUN_TIMEOUT.as_secs())),
        }
        digest
    }

    async fn write_digest(&self, window_hours: u32, digest: &mut Digest) -> Result<String, String> {
        let client = self.connect().await?;
        let result = client
            .call_tool(
                CallToolRequestParams::new("get_summary")
                    .with_arguments(json_object(json!({ "hours": window_hours }))),
            )
            .await;
        let _ = client.cancel().await;
        let result = result.map_err(|e| format!("tools/call get_summary failed: {e}"))?;
        let summary = tool_result_text(&result);
        if result.is_error == Some(true) {
            return Err(format!("get_summary failed: {summary}"));
        }
        digest.summary = summary.clone();

        // Nothing collected and nothing fired: no point paying for an LLM call.
        let quiet = result.structured_content.as_ref().is_some_and(|s| {
            s["repositories"].as_array().is_some_and(|a| a.is_empty())
                && s["reminders_fired"].as_array().is_some_and(|a| a.is_empty())
        });
        if quiet {
            return Ok(QUIET_DIGEST.to_string());
        }

        digest.used_llm = true;
        let messages = [
            json!({ "role": "system", "content": digest_prompt(&self.digest_language) }),
            json!({ "role": "user", "content": summary }),
        ];
        let message = self.call_llm(&messages, &[]).await?;
        message_text(&message).ok_or_else(|| "The model returned an empty digest.".to_string())
    }

    async fn call_llm(&self, messages: &[Value], tools: &[Value]) -> Result<Value, String> {
        let mut body = json!({ "model": self.model, "messages": messages });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
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
        serde_json::from_str::<ChatCompletionResponse>(&text)
            .map_err(|e| format!("Failed to parse LLM response: {e}"))?
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| "The model returned no choices.".to_string())
    }
}

fn message_text(message: &Value) -> Option<String> {
    let text = message.get("content")?.as_str()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

// ---------------------------------------------------------------------------
// The scheduler: a background loop that runs whatever is due.
// ---------------------------------------------------------------------------

struct Scheduler {
    store: Arc<Store>,
    github: GitHub,
    agent: Arc<Agent>,
    /// One tick at a time: the background loop and "run now" can never run
    /// the same due job twice.
    tick_lock: tokio::sync::Mutex<()>,
}

impl Scheduler {
    fn new(store: Arc<Store>, github: GitHub, agent: Arc<Agent>) -> Self {
        Self {
            store,
            github,
            agent,
            tick_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Runs every job due at `now`, oldest first, and records each run.
    async fn tick(&self, now: u64) -> Vec<RunRecord> {
        let _guard = self.tick_lock.lock().await;
        let mut records = Vec::new();
        for job in self.store.read(|d| d.due_jobs(now)) {
            // It may have been cancelled while an earlier job was running.
            if !self.store.read(|d| d.is_active(job.id)) {
                continue;
            }
            let started = Instant::now();
            let (ok, message) = match self.execute(&job, now).await {
                Ok(message) => (true, message),
                Err(e) => (false, e),
            };
            let ms = started.elapsed().as_millis() as u64;
            println!(
                "[scheduler] job #{} {} ({}): {} - {message}",
                job.id,
                job.kind.name(),
                job.kind.subject(),
                if ok { "ok" } else { "failed" }
            );
            records.push(
                self.store
                    .write(|d| d.finish_run(job.id, &job.kind, now, ok, message, ms)),
            );
        }
        records
    }

    async fn execute(&self, job: &Job, now: u64) -> Result<String, String> {
        match &job.kind {
            JobKind::CollectCommits { repository } => {
                let commits = self.github.list_commits(repository, FETCH_PER_RUN).await?;
                let fetched = commits.len();
                let new = self
                    .store
                    .write(|d| d.add_commits(repository, commits, now));
                Ok(format!("{new} new commit(s) out of {fetched} fetched"))
            }
            JobKind::Reminder { text } => {
                self.store.write(|d| d.fire_reminder(job.id, text, now));
                Ok(format!("Reminder: {text}"))
            }
            JobKind::Digest { window_hours } => {
                let digest = self.agent.digest(*window_hours, now).await;
                let outcome = match (&digest.text, &digest.error) {
                    (_, Some(e)) => Err(e.clone()),
                    (Some(text), None) if !digest.used_llm => Ok(text.clone()),
                    (Some(text), None) => Ok(format!(
                        "Digest written ({} chars)",
                        text.chars().count()
                    )),
                    (None, None) => Err("The digest came back empty.".to_string()),
                };
                self.store.write(|d| d.push_digest(digest));
                outcome
            }
        }
    }
}

/// The built-in job that makes the agent report on its own, 24/7. Created on
/// first start, then kept in sync with `DIGEST_EVERY_MINUTES` /
/// `DIGEST_WINDOW_HOURS` on every restart.
fn ensure_digest_job(store: &Store, every_minutes: u32, window_hours: u32, now: u64) -> Job {
    let every = u64::from(every_minutes) * 60;
    store.write(|d| {
        if let Some(job) = d
            .jobs
            .iter_mut()
            .find(|j| j.active && matches!(j.kind, JobKind::Digest { .. }))
        {
            job.kind = JobKind::Digest { window_hours };
            if job.every_secs != Some(every) {
                job.every_secs = Some(every);
                job.next_run = job.next_run.min(now + every);
            }
            return job.clone();
        }
        d.add_job(JobKind::Digest { window_hours }, Some(every), now + every, now)
    })
}

fn spawn_scheduler_loop(scheduler: Arc<Scheduler>, every: Duration) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(every);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            // The first tick fires immediately: anything that came due while
            // the process was down runs right after startup.
            interval.tick().await;
            scheduler.tick(unix_now()).await;
        }
    });
}

// ---------------------------------------------------------------------------
// HTTP: the web UI, its JSON API, and the MCP endpoint, all on one port.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Settings {
    tick_seconds: u64,
    digest_every_minutes: u32,
    digest_window_hours: u32,
    min_interval_minutes: u32,
    data_file: String,
    model: String,
    mcp_url: String,
}

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    store: Arc<Store>,
    scheduler: Arc<Scheduler>,
    settings: Arc<Settings>,
}

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

async fn ask(State(app): State<AppState>, Json(req): Json<AskRequest>) -> Json<AgentRun> {
    let question = req.question.trim();
    if question.is_empty() {
        return Json(AgentRun {
            error: Some("Ask something first.".to_string()),
            ..Default::default()
        });
    }
    let run = app.agent.run(question).await;
    if let Some(e) = &run.error {
        eprintln!("Question failed: {e}");
    }
    Json(run)
}

#[derive(Serialize)]
struct StateResponse {
    now: u64,
    settings: Settings,
    jobs: Vec<Job>,
    runs: Vec<RunRecord>,
    reminders: Vec<FiredReminder>,
    digests: Vec<Digest>,
    commits_stored: BTreeMap<String, usize>,
}

async fn get_state(State(app): State<AppState>) -> Json<StateResponse> {
    let now = unix_now();
    Json(app.store.read(|d| {
        let mut jobs: Vec<Job> = d.jobs.iter().filter(|j| j.active).cloned().collect();
        jobs.sort_by_key(|j| j.next_run);
        jobs.extend(d.jobs.iter().rev().filter(|j| !j.active).take(10).cloned());
        StateResponse {
            now,
            settings: (*app.settings).clone(),
            jobs,
            runs: d.runs.iter().rev().take(40).cloned().collect(),
            reminders: d.reminders.iter().rev().take(20).cloned().collect(),
            digests: d.digests.iter().rev().take(10).cloned().collect(),
            commits_stored: d.commits.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
        }
    }))
}

/// Makes a job due right now and runs a tick, instead of waiting for it.
async fn run_now(
    State(app): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Json<Value> {
    let now = unix_now();
    let found = app.store.write(|d| {
        match d.jobs.iter_mut().find(|j| j.id == id && j.active) {
            Some(job) => {
                job.next_run = now;
                true
            }
            None => false,
        }
    });
    if !found {
        return Json(json!({ "error": format!("Job #{id} is not active.") }));
    }
    let runs = app.scheduler.tick(now).await;
    Json(json!({ "runs": runs }))
}

async fn cancel(
    State(app): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<u64>,
) -> Json<Value> {
    Json(match app.store.write(|d| d.cancel_job(id)) {
        Ok(job) => json!({ "job": job }),
        Err(e) => json!({ "error": e }),
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: AppState) -> Router {
    let mcp = mcp_service(state.store.clone(), state.settings.min_interval_minutes);
    Router::new()
        .route("/", get(index))
        .route("/api/state", get(get_state))
        .route("/api/ask", post(ask))
        .route("/api/jobs/{id}/run", post(run_now))
        .route("/api/jobs/{id}/cancel", post(cancel))
        .with_state(state)
        .nest_service("/mcp", mcp)
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn env_number(name: &str, default: u64) -> u64 {
    match env_non_empty(name) {
        None => default,
        Some(value) => value.trim().parse().unwrap_or_else(|_| {
            eprintln!("Ignoring {name}={value:?}: not a whole number, using {default}.");
            default
        }),
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
    let mcp_url =
        env_non_empty("MCP_SERVER_URL").unwrap_or_else(|| format!("http://127.0.0.1:{port}/mcp"));
    let max_interval = u64::from(MAX_INTERVAL_MINUTES);
    let tick_seconds = env_number("TICK_SECONDS", 15).clamp(1, 3_600);
    let digest_every_minutes = env_number("DIGEST_EVERY_MINUTES", 60).clamp(1, max_interval) as u32;
    let digest_window_hours =
        env_number("DIGEST_WINDOW_HOURS", 24).clamp(1, u64::from(MAX_SUMMARY_HOURS)) as u32;
    let min_interval_minutes =
        env_number("MIN_INTERVAL_MINUTES", 10).clamp(1, max_interval) as u32;
    let digest_language = env_non_empty("DIGEST_LANGUAGE").unwrap_or_else(|| "English".to_string());
    // Relative to the working directory: the lesson folder under `cargo run`,
    // `~/apps/lesson-18` on the VDS (the systemd unit's WorkingDirectory).
    let data_file = env_non_empty("DATA_FILE").unwrap_or_else(|| "scheduler-data.json".to_string());

    let store = Arc::new(Store::open(PathBuf::from(&data_file)));
    let digest_job = ensure_digest_job(&store, digest_every_minutes, digest_window_hours, unix_now());
    println!(
        "Loaded {} job(s) from {data_file}; digest is job #{}, every {digest_every_minutes} min, next at {}",
        store.read(|d| d.jobs.iter().filter(|j| j.active).count()),
        digest_job.id,
        iso(digest_job.next_run)
    );

    let github = GitHub {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client"),
        api_base: GITHUB_API.to_string(),
        token: env_non_empty("GITHUB_TOKEN"),
    };
    let agent = Arc::new(Agent {
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .expect("failed to build HTTP client"),
        endpoint: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        api_key,
        model: model.clone(),
        mcp_url: mcp_url.clone(),
        digest_language,
    });
    let scheduler = Arc::new(Scheduler::new(store.clone(), github, agent.clone()));
    let state = AppState {
        agent,
        store,
        scheduler: scheduler.clone(),
        settings: Arc::new(Settings {
            tick_seconds,
            digest_every_minutes,
            digest_window_hours,
            min_interval_minutes,
            data_file,
            model,
            mcp_url,
        }),
    };

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    // Started after the port is bound, so a digest that is due at startup can
    // already reach this process's own `/mcp` endpoint.
    spawn_scheduler_loop(scheduler, Duration::from_secs(tick_seconds));
    println!("Scheduler ticking every {tick_seconds}s. Listening on http://localhost:{port}");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson18-{}-{name}-{}.json",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn temp_store(name: &str) -> Arc<Store> {
        Arc::new(Store::open(temp_path(name)))
    }

    fn github_at(api_base: &str) -> GitHub {
        GitHub {
            client: reqwest::Client::new(),
            api_base: api_base.to_string(),
            token: None,
        }
    }

    /// Two commits made an hour and two hours before `now`.
    fn commits_fixture(now: u64) -> Value {
        json!([
            {
                "sha": "abc1234def5678900000000000000000000000000",
                "html_url": "https://github.com/octo/demo/commit/abc1234def",
                "commit": {
                    "message": "Add scheduler\n\nA longer body that should not reach the model.",
                    "author": { "name": "Octo Cat", "date": iso(now - 3_600) }
                },
                "author": { "login": "octocat" }
            },
            {
                "sha": "9876543fedcba00000000000000000000000000000",
                "html_url": "https://github.com/octo/demo/commit/9876543fed",
                "commit": {
                    "message": "Initial commit",
                    "author": { "name": "Someone", "date": iso(now - 7_200) }
                },
                "author": null
            }
        ])
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Stands in for api.github.com and counts how often it was called.
    fn fake_github(body: Value, hits: Arc<AtomicU64>) -> Router {
        Router::new().route(
            "/repos/{owner}/{repo}/commits",
            get(move || {
                let body = body.clone();
                let hits = hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    Json(body)
                }
            }),
        )
    }

    async fn duplex_client(server: SchedulerServer) -> RunningService<RoleClient, ()> {
        let (server_io, client_io) = tokio::io::duplex(64 * 1024);
        tokio::spawn(async move {
            let running = server.serve(server_io).await.expect("server handshake");
            let _ = running.waiting().await;
        });
        ().serve(client_io).await.expect("client handshake")
    }

    async fn call(
        client: &RunningService<RoleClient, ()>,
        name: &'static str,
        arguments: Value,
    ) -> CallToolResult {
        client
            .call_tool(CallToolRequestParams::new(name).with_arguments(json_object(arguments)))
            .await
            .unwrap()
    }

    fn test_agent(llm_base: &str, mcp_url: String) -> Arc<Agent> {
        Arc::new(Agent {
            http: reqwest::Client::new(),
            endpoint: format!("{llm_base}/chat/completions"),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
            mcp_url,
            digest_language: "English".to_string(),
        })
    }

    /// The real app, `/mcp` included, on a random port.
    async fn spawn_app(store: Arc<Store>, github_base: &str, llm_base: &str) -> AppState {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mcp_url = format!("http://{addr}/mcp");
        let agent = test_agent(llm_base, mcp_url.clone());
        let scheduler = Arc::new(Scheduler::new(
            store.clone(),
            github_at(github_base),
            agent.clone(),
        ));
        let state = AppState {
            agent,
            store,
            scheduler,
            settings: Arc::new(Settings {
                tick_seconds: 15,
                digest_every_minutes: 60,
                digest_window_hours: 24,
                min_interval_minutes: 10,
                data_file: "test.json".to_string(),
                model: "test-model".to_string(),
                mcp_url,
            }),
        };
        let router = app(state.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        state
    }

    #[tokio::test]
    async fn tools_are_registered_with_described_parameters() {
        let client = duplex_client(SchedulerServer::new(temp_store("tools"), 10)).await;
        let tools = client.list_all_tools().await.unwrap();
        client.cancel().await.unwrap();

        let mut names: Vec<&str> = tools.iter().map(|t| &*t.name).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["cancel_job", "get_summary", "list_jobs", "remind_me", "watch_repo"]
        );
        for tool in &tools {
            assert!(tool.description.as_deref().is_some_and(|d| !d.is_empty()));
            let schema = Value::Object((*tool.input_schema).clone());
            for (property, spec) in schema["properties"].as_object().unwrap() {
                assert!(
                    spec["description"].as_str().is_some_and(|d| !d.is_empty()),
                    "{}.{property} has no description",
                    tool.name
                );
            }
        }
        let watch = tools.iter().find(|t| t.name == "watch_repo").unwrap();
        let schema = Value::Object((*watch.input_schema).clone());
        let mut required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        required.sort_unstable();
        assert_eq!(required, ["owner", "repo"]);
    }

    #[tokio::test]
    async fn watch_repo_is_saved_to_json_and_survives_a_restart() {
        let path = temp_path("persist");
        let store = Arc::new(Store::open(path.clone()));
        let client = duplex_client(SchedulerServer::new(store, 10)).await;

        let created = call(
            &client,
            "watch_repo",
            json!({ "owner": "Tokio-RS", "repo": "axum", "every_minutes": 30 }),
        )
        .await;
        assert_ne!(created.is_error, Some(true), "{}", tool_result_text(&created));
        assert!(tool_result_text(&created).contains("job #1"));

        // Same repository again: the interval changes, no second job appears.
        let updated = call(
            &client,
            "watch_repo",
            json!({ "owner": "tokio-rs", "repo": "axum", "every_minutes": 15 }),
        )
        .await;
        assert!(tool_result_text(&updated).contains("Already watching"));
        client.cancel().await.unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"collect_commits\""), "{raw}");

        // A fresh process reading the same file sees the job.
        let reopened = Store::open(path);
        let jobs = reopened.read(|d| d.jobs.clone());
        assert_eq!(jobs.len(), 1);
        assert_eq!(
            jobs[0].kind,
            JobKind::CollectCommits {
                repository: "tokio-rs/axum".to_string()
            }
        );
        assert_eq!(jobs[0].every_secs, Some(15 * 60));
        assert!(jobs[0].active);
    }

    #[tokio::test]
    async fn tool_arguments_and_limits_are_enforced() {
        let store = temp_store("limits");
        let digest = ensure_digest_job(&store, 60, 24, unix_now());
        let client = duplex_client(SchedulerServer::new(store, 10)).await;

        let too_often = call(
            &client,
            "watch_repo",
            json!({ "owner": "octo", "repo": "demo", "every_minutes": 1 }),
        )
        .await;
        assert_eq!(too_often.is_error, Some(true));
        assert!(tool_result_text(&too_often).contains("between 10 and"));

        let bad_name = call(&client, "watch_repo", json!({ "owner": "../x", "repo": "demo" })).await;
        assert_eq!(bad_name.is_error, Some(true));

        for i in 0..MAX_WATCHED_REPOS {
            let ok = call(&client, "watch_repo", json!({ "owner": "octo", "repo": format!("r{i}") })).await;
            assert_ne!(ok.is_error, Some(true), "{}", tool_result_text(&ok));
        }
        let one_too_many = call(&client, "watch_repo", json!({ "owner": "octo", "repo": "extra" })).await;
        assert_eq!(one_too_many.is_error, Some(true));
        assert!(tool_result_text(&one_too_many).contains("maximum"));

        let far = call(&client, "remind_me", json!({ "text": "x", "in_minutes": 999_999 })).await;
        assert_eq!(far.is_error, Some(true));

        let builtin = call(&client, "cancel_job", json!({ "job_id": digest.id })).await;
        assert_eq!(builtin.is_error, Some(true));
        assert!(tool_result_text(&builtin).contains("built-in"));

        let unknown = call(&client, "cancel_job", json!({ "job_id": 999 })).await;
        assert_eq!(unknown.is_error, Some(true));
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn scheduler_runs_due_collections_and_keeps_only_new_commits() {
        let now = unix_now();
        let hits = Arc::new(AtomicU64::new(0));
        let base = spawn(fake_github(commits_fixture(now), hits.clone())).await;
        let store = temp_store("collect");
        let job = store.write(|d| {
            d.add_job(
                JobKind::CollectCommits {
                    repository: "octo/demo".to_string(),
                },
                Some(600),
                now + 60,
                now,
            )
        });
        let scheduler = Scheduler::new(
            store.clone(),
            github_at(&base),
            test_agent("http://127.0.0.1:9", "http://127.0.0.1:9/mcp".to_string()),
        );

        assert!(scheduler.tick(now).await.is_empty(), "not due yet");
        assert_eq!(hits.load(Ordering::SeqCst), 0);

        let runs = scheduler.tick(now + 60).await;
        assert_eq!(runs.len(), 1);
        assert!(runs[0].ok, "{}", runs[0].message);
        assert_eq!(runs[0].message, "2 new commit(s) out of 2 fetched");
        let after = store.read(|d| d.jobs[0].clone());
        assert_eq!(after.id, job.id);
        assert_eq!(after.run_count, 1);
        assert_eq!(after.next_run, now + 60 + 600, "rescheduled one interval later");

        // Next period: GitHub returns the same commits, none of them are new.
        let runs = scheduler.tick(now + 660).await;
        assert_eq!(runs[0].message, "0 new commit(s) out of 2 fetched");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        let stored = store.read(|d| d.commits["octo/demo"].clone());
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].sha, "abc1234");
        assert_eq!(stored[0].message, "Add scheduler");
        assert_eq!(stored[0].collected_at, now + 60);
    }

    #[tokio::test]
    async fn reminder_fires_once_at_its_time() {
        let now = unix_now();
        let store = temp_store("reminder");
        let client = duplex_client(SchedulerServer::new(store.clone(), 10)).await;
        let set = call(&client, "remind_me", json!({ "text": "Drink water", "in_minutes": 2 })).await;
        client.cancel().await.unwrap();
        assert_ne!(set.is_error, Some(true), "{}", tool_result_text(&set));

        let scheduler = Scheduler::new(
            store.clone(),
            github_at("http://127.0.0.1:9"),
            test_agent("http://127.0.0.1:9", "http://127.0.0.1:9/mcp".to_string()),
        );
        assert!(scheduler.tick(now).await.is_empty());

        let runs = scheduler.tick(now + 180).await;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].kind, "reminder");
        assert_eq!(runs[0].message, "Reminder: Drink water");
        assert!(scheduler.tick(now + 360).await.is_empty(), "one-shot");

        let (fired, job) = store.read(|d| (d.reminders.clone(), d.jobs[0].clone()));
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].fired_at, now + 180);
        assert!(!job.active && !job.cancelled);
    }

    #[test]
    fn summary_aggregates_only_the_window() {
        let now = 1_790_380_800; // 2026-09-26T00:00:00Z
        let commit = |sha: &str, author: &str, age_secs: u64| CommitSummary {
            sha: sha.to_string(),
            date: iso(now - age_secs),
            author: author.to_string(),
            message: format!("change {sha}"),
            url: String::new(),
            collected_at: now,
        };
        let run = |subject: &str, age_secs: u64, ok: bool, message: &str| RunRecord {
            job_id: 1,
            kind: "collect_commits".to_string(),
            subject: subject.to_string(),
            at: now - age_secs,
            ok,
            message: message.to_string(),
            ms: 1,
        };
        let mut data = StoreData::default();
        data.add_job(
            JobKind::CollectCommits {
                repository: "octo/demo".to_string(),
            },
            Some(600),
            now + 600,
            now,
        );
        data.commits.insert(
            "octo/demo".to_string(),
            vec![
                commit("aaa0001", "octocat", 3_600),
                commit("aaa0002", "octocat", 7_200),
                commit("aaa0003", "alice", 3_600 * 3),
                commit("old0001", "bob", 3_600 * 30),
            ],
        );
        // Not watched any more and nothing recent: left out.
        data.commits
            .insert("old/repo".to_string(), vec![commit("old0002", "bob", 3_600 * 40)]);
        data.runs = vec![
            run("octo/demo", 3_600 * 40, true, "outside the window"),
            run("octo/demo", 100, true, "2 new commit(s)"),
            run("octo/demo", 50, false, "GitHub API rate limit exceeded"),
        ];
        data.reminders = vec![
            FiredReminder { job_id: 7, text: "old".to_string(), fired_at: now - 3_600 * 50 },
            FiredReminder { job_id: 8, text: "Stand-up".to_string(), fired_at: now - 10 },
        ];

        let s = summarize(&data, now, 24);
        assert_eq!(s.from, "2026-09-25T00:00:00Z");
        assert_eq!(s.repositories.len(), 1);
        let repo = &s.repositories[0];
        assert_eq!(repo.repository, "octo/demo");
        assert!(repo.watched);
        assert_eq!(repo.commits_in_window, 3);
        assert_eq!(repo.authors[0].author, "octocat");
        assert_eq!(repo.authors[0].commits, 2);
        assert_eq!(repo.collections, 2);
        assert_eq!(repo.failed_collections, 1);
        assert_eq!(repo.last_error.as_deref(), Some("GitHub API rate limit exceeded"));
        assert_eq!(s.reminders_fired.len(), 1);
        assert_eq!(s.reminders_fired[0].text, "Stand-up");
        assert_eq!((s.runs, s.failed_runs), (2, 1));
        assert_eq!(s.active_jobs, 1);

        let text = format_summary(&s);
        assert!(text.contains("octo/demo: 3 commit(s) in the window by octocat x2, alice"), "{text}");
        assert!(text.contains("aaa0001"), "{text}");
        assert!(!text.contains("old0001"), "{text}");
        assert!(text.contains("Stand-up"), "{text}");
    }

    /// Stands in for the LLM when it writes a digest: quotes the line about
    /// the newest commit back, so the test can see the data went through.
    fn fake_digest_llm(requests: Arc<Mutex<Vec<Value>>>) -> Router {
        Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let requests = requests.clone();
                async move {
                    requests.lock().unwrap().push(body.clone());
                    let summary = body["messages"][1]["content"].as_str().unwrap_or_default();
                    let line = summary
                        .lines()
                        .find(|l| l.contains("abc1234"))
                        .unwrap_or("nothing")
                        .trim()
                        .to_string();
                    Json(json!({
                        "choices": [{ "message": { "role": "assistant", "content": format!("DIGEST {line}") } }]
                    }))
                }
            }),
        )
    }

    #[tokio::test]
    async fn digest_job_aggregates_over_mcp_and_the_llm_writes_it() {
        let now = unix_now();
        let github_base = spawn(fake_github(commits_fixture(now), Arc::new(AtomicU64::new(0)))).await;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let llm_base = spawn(fake_digest_llm(requests.clone())).await;
        let store = temp_store("digest");
        store.write(|d| {
            d.add_job(
                JobKind::CollectCommits {
                    repository: "octo/demo".to_string(),
                },
                Some(3_600),
                now,
                now,
            )
        });
        let digest_job = ensure_digest_job(&store, 60, 24, now);
        let app = spawn_app(store.clone(), &github_base, &llm_base).await;

        // First tick: only the collection is due.
        let runs = app.scheduler.tick(now).await;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].kind, "collect_commits");

        // An hour later both are due; the digest runs through the real `/mcp`.
        let runs = app.scheduler.tick(now + 3_600).await;
        let digest_run = runs.iter().find(|r| r.kind == "digest").expect("digest ran");
        assert!(digest_run.ok, "{}", digest_run.message);

        let digest = store.read(|d| d.digests.last().cloned()).unwrap();
        assert!(digest.used_llm);
        assert!(digest.error.is_none(), "{:?}", digest.error);
        assert!(digest.summary.contains("octo/demo: 2 commit(s)"), "{}", digest.summary);
        let text = digest.text.unwrap();
        assert!(text.starts_with("DIGEST") && text.contains("abc1234"), "{text}");
        assert!(text.contains("Add scheduler"), "{text}");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].get("tools").is_none(), "the digest call offers no tools");
        let job = store.read(|d| d.jobs.iter().find(|j| j.id == digest_job.id).cloned()).unwrap();
        assert_eq!(job.next_run, now + 2 * 3_600);
    }

    #[tokio::test]
    async fn quiet_digest_skips_the_llm() {
        let now = unix_now();
        let store = temp_store("quiet");
        ensure_digest_job(&store, 60, 24, now);
        // No LLM is listening at this address: calling it would fail the run.
        let app = spawn_app(store.clone(), "http://127.0.0.1:9", "http://127.0.0.1:9").await;

        let runs = app.scheduler.tick(now + 3_600).await;
        assert_eq!(runs.len(), 1);
        assert!(runs[0].ok, "{}", runs[0].message);
        let digest = store.read(|d| d.digests[0].clone());
        assert!(!digest.used_llm);
        assert_eq!(digest.text.as_deref(), Some(QUIET_DIGEST));
    }

    /// Stands in for the LLM in a chat turn: asks for `watch_repo`, then
    /// answers with the tool's result.
    fn fake_chat_llm() -> Router {
        Router::new().route(
            "/chat/completions",
            post(|Json(body): Json<Value>| async move {
                let tool_result = body["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|m| m["role"] == "tool")
                    .and_then(|m| m["content"].as_str())
                    .map(str::to_string);
                let message = match tool_result {
                    None => json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "watch_repo",
                                "arguments": "{\"owner\":\"octo\",\"repo\":\"demo\",\"every_minutes\":30}"
                            }
                        }]
                    }),
                    Some(result) => json!({ "role": "assistant", "content": format!("Done. {result}") }),
                };
                Json(json!({ "choices": [{ "message": message }] }))
            }),
        )
    }

    #[tokio::test]
    async fn agent_schedules_a_job_through_the_mcp_tool() {
        let llm_base = spawn(fake_chat_llm()).await;
        let store = temp_store("chat");
        let app = spawn_app(store.clone(), "http://127.0.0.1:9", &llm_base).await;

        let run = app.agent.run("Watch octo/demo every 30 minutes").await;

        assert!(run.error.is_none(), "{:?}", run.error);
        assert_eq!(run.mcp.as_ref().unwrap().tools.len(), 5);
        assert_eq!(run.tool_calls.len(), 1);
        assert_eq!(run.tool_calls[0].name, "watch_repo");
        assert!(!run.tool_calls[0].is_error, "{}", run.tool_calls[0].result_text);
        assert!(run.answer.unwrap().contains("job #1"));

        let jobs = store.read(|d| d.jobs.clone());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].every_secs, Some(30 * 60));
    }

    #[test]
    fn iso_formats_unix_seconds() {
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(1_790_380_800), "2026-09-26T00:00:00Z");
        assert_eq!(iso(1_790_380_800 + 3_661), "2026-09-26T01:01:01Z");
        assert_eq!(civil_from_days(11_017), "2000-03-01");
    }
}

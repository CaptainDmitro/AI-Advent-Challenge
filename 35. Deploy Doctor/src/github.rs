//! The GitHub REST API: a CI/CD run and its jobs, a job's log, a file at a
//! commit, and the doctor's issues. Issues are found by the hidden marker in
//! their body, not by label, so one is found even if the label never stuck.

use reqwest::{Method, RequestBuilder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::report::MARKER_PREFIX;

pub const DEFAULT_API: &str = "https://api.github.com";
pub const LABEL: &str = "deploy-doctor";
/// The workflow whose runs the doctor diagnoses.
pub const CI_WORKFLOW: &str = "ci-cd.yml";
const JSON: &str = "application/vnd.github+json";
const RAW: &str = "application/vnd.github.raw+json";
const PER_PAGE: usize = 100;
const MAX_PAGES: usize = 5;
const RECENT_RUNS: usize = 10;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Run {
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub display_title: String,
    pub event: String,
    pub status: String,
    pub conclusion: Option<String>,
    pub head_sha: String,
    #[serde(default)]
    pub head_branch: Option<String>,
    pub html_url: String,
    #[serde(default)]
    pub run_attempt: u64,
    #[serde(default)]
    pub created_at: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Job {
    pub id: u64,
    pub name: String,
    pub status: String,
    pub conclusion: Option<String>,
    #[serde(default)]
    pub html_url: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Step {
    pub name: String,
    pub status: String,
    pub conclusion: Option<String>,
    pub number: u64,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Issue {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub body: Option<String>,
    pub state: String,
    pub html_url: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub closed_at: Option<String>,
    /// Present when the "issue" is a pull request.
    #[serde(default, skip_serializing)]
    pull_request: Option<Value>,
}

impl Issue {
    pub fn is_doctors(&self) -> bool {
        self.pull_request.is_none()
            && self
                .body
                .as_deref()
                .is_some_and(|b| b.contains(MARKER_PREFIX))
    }

    pub fn has_marker(&self, marker: &str) -> bool {
        self.body.as_deref().is_some_and(|b| b.contains(marker))
    }
}

#[derive(Deserialize)]
struct Jobs {
    jobs: Vec<Job>,
}

#[derive(Deserialize)]
struct Runs {
    workflow_runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Comment {
    html_url: String,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

pub struct GitHub {
    http: reqwest::Client,
    api: String,
    token: Option<String>,
    /// `owner/name`.
    pub repo: String,
}

impl GitHub {
    pub fn new(http: reqwest::Client, api: &str, token: Option<String>, repo: String) -> GitHub {
        GitHub {
            http,
            api: api.trim_end_matches('/').to_string(),
            token,
            repo,
        }
    }

    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    fn request(&self, method: Method, path: &str, accept: &str) -> RequestBuilder {
        let request = self
            .http
            .request(method, format!("{}{path}", self.api))
            .header("Accept", accept)
            .header("User-Agent", "deploy-doctor")
            .header("X-GitHub-Api-Version", "2022-11-28");
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn send_text(&self, request: RequestBuilder, what: &str) -> Result<String, String> {
        let response = request
            .send()
            .await
            .map_err(|e| format!("GitHub request ({what}) failed: {e}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let detail = serde_json::from_str::<ApiError>(&text)
                .map(|e| e.message)
                .unwrap_or(text);
            return Err(format!("GitHub API error ({what}, {status}): {detail}"));
        }
        Ok(text)
    }

    async fn send<T: DeserializeOwned>(&self, request: RequestBuilder, what: &str) -> Result<T, String> {
        let text = self.send_text(request, what).await?;
        serde_json::from_str(&text).map_err(|e| format!("Unexpected GitHub response ({what}): {e}"))
    }

    fn need_token(&self, what: &str) -> Result<(), String> {
        match self.token {
            Some(_) => Ok(()),
            None => Err(format!("{what} needs GITHUB_TOKEN, which is not set here.")),
        }
    }

    pub async fn run(&self, id: u64) -> Result<Run, String> {
        let path = format!("/repos/{}/actions/runs/{id}", self.repo);
        self.send(self.request(Method::GET, &path, JSON), &format!("run {id}"))
            .await
    }

    /// Every job of the run's latest attempt, with its steps.
    pub async fn jobs(&self, run: u64) -> Result<Vec<Job>, String> {
        let mut all = Vec::new();
        for page in 1..=MAX_PAGES {
            let path = format!(
                "/repos/{}/actions/runs/{run}/jobs?per_page={PER_PAGE}&page={page}",
                self.repo
            );
            let jobs: Jobs = self
                .send(self.request(Method::GET, &path, JSON), "jobs")
                .await?;
            let last = jobs.jobs.len() < PER_PAGE;
            all.extend(jobs.jobs);
            if last {
                break;
            }
        }
        Ok(all)
    }

    pub async fn recent_runs(&self) -> Result<Vec<Run>, String> {
        let path = format!(
            "/repos/{}/actions/workflows/{CI_WORKFLOW}/runs?per_page={RECENT_RUNS}",
            self.repo
        );
        let runs: Runs = self
            .send(self.request(Method::GET, &path, JSON), "recent runs")
            .await?;
        Ok(runs.workflow_runs)
    }

    /// A job's full log. GitHub answers with a redirect to blob storage;
    /// reqwest follows it and drops the token on the way (another host).
    pub async fn job_log(&self, job: u64) -> Result<String, String> {
        self.need_token("Reading a job's log")?;
        let path = format!("/repos/{}/actions/jobs/{job}/logs", self.repo);
        self.send_text(self.request(Method::GET, &path, JSON), &format!("log of job {job}"))
            .await
    }

    /// A file's text at a commit.
    pub async fn file(&self, path: &str, rev: &str) -> Result<String, String> {
        let url = format!(
            "/repos/{}/contents/{}?ref={}",
            self.repo,
            encode(path),
            encode(rev)
        );
        self.send_text(self.request(Method::GET, &url, RAW), path).await
    }

    /// The doctor's issues (`open`, `closed` or `all`), newest activity first.
    pub async fn issues(&self, state: &str) -> Result<Vec<Issue>, String> {
        let mut found = Vec::new();
        for page in 1..=MAX_PAGES {
            let path = format!(
                "/repos/{}/issues?state={state}&sort=updated&per_page={PER_PAGE}&page={page}",
                self.repo
            );
            let issues: Vec<Issue> = self
                .send(self.request(Method::GET, &path, JSON), "issues")
                .await?;
            let last = issues.len() < PER_PAGE;
            found.extend(issues.into_iter().filter(Issue::is_doctors));
            if last {
                break;
            }
        }
        Ok(found)
    }

    pub async fn create_issue(&self, title: &str, body: &str) -> Result<Issue, String> {
        self.need_token("Creating an issue")?;
        let path = format!("/repos/{}/issues", self.repo);
        let request = self
            .request(Method::POST, &path, JSON)
            .json(&json!({ "title": title, "body": body, "labels": [LABEL] }));
        self.send(request, "creating an issue").await
    }

    /// Adds a comment and returns its URL.
    pub async fn comment(&self, issue: u64, body: &str) -> Result<String, String> {
        self.need_token("Commenting")?;
        let path = format!("/repos/{}/issues/{issue}/comments", self.repo);
        let request = self
            .request(Method::POST, &path, JSON)
            .json(&json!({ "body": body }));
        let comment: Comment = self.send(request, "commenting").await?;
        Ok(comment.html_url)
    }

    pub async fn close(&self, issue: u64) -> Result<(), String> {
        self.need_token("Closing an issue")?;
        let path = format!("/repos/{}/issues/{issue}", self.repo);
        let request = self
            .request(Method::PATCH, &path, JSON)
            .json(&json!({ "state": "closed", "state_reason": "completed" }));
        let _: Issue = self.send(request, "closing an issue").await?;
        Ok(())
    }
}

/// Percent-encodes a path or a ref for a URL, keeping `/`.
fn encode(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_encoded() {
        assert_eq!(
            encode("35. Deploy Doctor/README.md"),
            "35.%20Deploy%20Doctor/README.md"
        );
        assert_eq!(encode("a&b?c"), "a%26b%3Fc");
    }

    #[test]
    fn only_the_doctors_issues_count() {
        let issue = |body: Option<&str>, pull: bool| Issue {
            number: 1,
            title: "t".into(),
            body: body.map(str::to_string),
            state: "open".into(),
            html_url: String::new(),
            created_at: String::new(),
            updated_at: String::new(),
            closed_at: None,
            pull_request: pull.then(|| json!({})),
        };
        assert!(issue(Some("<!-- deploy-doctor:lesson-35 -->\nx"), false).is_doctors());
        assert!(!issue(Some("<!-- deploy-doctor:lesson-35 -->"), true).is_doctors());
        assert!(!issue(Some("a bug"), false).is_doctors());
        assert!(!issue(None, false).is_doctors());
    }
}

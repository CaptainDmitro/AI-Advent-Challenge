//! The GitHub REST API: a pull request's metadata, the open pull requests,
//! and the review comment, which is created once and then edited in place
//! on every new push (found by its hidden marker).

use reqwest::{Method, RequestBuilder};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;

use crate::review::MARKER;

pub const DEFAULT_API: &str = "https://api.github.com";
const PER_PAGE: usize = 100;
const MAX_PAGES: usize = 10;
const MAX_OPEN_PULLS: usize = 30;

#[derive(Serialize, Clone, Debug)]
pub struct PullRequest {
    pub number: u64,
    pub title: String,
    pub body: String,
    pub url: String,
    pub author: String,
    pub draft: bool,
    pub base_ref: String,
    pub base_sha: String,
    pub head_ref: String,
    pub head_sha: String,
    /// `owner/name` of the repository the head branch is in (a fork's).
    pub head_repo: Option<String>,
}

/// What happened to the review comment.
#[derive(Serialize, Clone, Debug)]
pub struct Posted {
    /// `created` or `updated`.
    pub action: String,
    pub id: u64,
    pub url: String,
}

#[derive(Deserialize)]
struct ApiUser {
    login: String,
}

#[derive(Deserialize)]
struct ApiRepo {
    full_name: String,
}

#[derive(Deserialize)]
struct ApiRef {
    #[serde(rename = "ref")]
    name: String,
    sha: String,
    repo: Option<ApiRepo>,
}

#[derive(Deserialize)]
struct ApiPull {
    number: u64,
    title: String,
    body: Option<String>,
    html_url: String,
    user: ApiUser,
    #[serde(default)]
    draft: bool,
    base: ApiRef,
    head: ApiRef,
}

#[derive(Deserialize)]
struct ApiComment {
    id: u64,
    body: Option<String>,
    html_url: String,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

impl From<ApiPull> for PullRequest {
    fn from(p: ApiPull) -> PullRequest {
        PullRequest {
            number: p.number,
            title: p.title,
            body: p.body.unwrap_or_default(),
            url: p.html_url,
            author: p.user.login,
            draft: p.draft,
            base_ref: p.base.name,
            base_sha: p.base.sha,
            head_ref: p.head.name,
            head_sha: p.head.sha,
            head_repo: p.head.repo.map(|r| r.full_name),
        }
    }
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

    pub fn api(&self) -> &str {
        &self.api
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        let request = self
            .http
            .request(method, format!("{}{path}", self.api))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "ai-code-review")
            .header("X-GitHub-Api-Version", "2022-11-28");
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn send<T: DeserializeOwned>(&self, request: RequestBuilder, what: &str) -> Result<T, String> {
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
        serde_json::from_str(&text).map_err(|e| format!("Unexpected GitHub response ({what}): {e}"))
    }

    pub async fn pull(&self, number: u64) -> Result<PullRequest, String> {
        let path = format!("/repos/{}/pulls/{number}", self.repo);
        let pull: ApiPull = self
            .send(self.request(Method::GET, &path), &format!("PR #{number}"))
            .await?;
        Ok(pull.into())
    }

    pub async fn open_pulls(&self) -> Result<Vec<PullRequest>, String> {
        let path = format!(
            "/repos/{}/pulls?state=open&per_page={MAX_OPEN_PULLS}",
            self.repo
        );
        let pulls: Vec<ApiPull> = self
            .send(self.request(Method::GET, &path), "open PRs")
            .await?;
        Ok(pulls.into_iter().map(PullRequest::from).collect())
    }

    /// The review comment already on the PR, if there is one.
    async fn find_comment(&self, pr: u64) -> Result<Option<ApiComment>, String> {
        for page in 1..=MAX_PAGES {
            let path = format!(
                "/repos/{}/issues/{pr}/comments?per_page={PER_PAGE}&page={page}",
                self.repo
            );
            let comments: Vec<ApiComment> = self
                .send(self.request(Method::GET, &path), "PR comments")
                .await?;
            let last_page = comments.len() < PER_PAGE;
            if let Some(found) = comments
                .into_iter()
                .find(|c| c.body.as_deref().is_some_and(|b| b.contains(MARKER)))
            {
                return Ok(Some(found));
            }
            if last_page {
                break;
            }
        }
        Ok(None)
    }

    /// Posts the review: edits the earlier review comment, or adds one.
    pub async fn upsert_comment(&self, pr: u64, body: &str) -> Result<Posted, String> {
        if self.token.is_none() {
            return Err("GITHUB_TOKEN is not set, so the review can't be posted.".to_string());
        }
        let (comment, action): (ApiComment, &str) = match self.find_comment(pr).await? {
            Some(existing) => {
                let path = format!("/repos/{}/issues/comments/{}", self.repo, existing.id);
                let request = self
                    .request(Method::PATCH, &path)
                    .json(&json!({ "body": body }));
                (self.send(request, "updating the comment").await?, "updated")
            }
            None => {
                let path = format!("/repos/{}/issues/{pr}/comments", self.repo);
                let request = self
                    .request(Method::POST, &path)
                    .json(&json!({ "body": body }));
                (self.send(request, "adding the comment").await?, "created")
            }
        };
        Ok(Posted {
            action: action.to_string(),
            id: comment.id,
            url: comment.html_url,
        })
    }
}

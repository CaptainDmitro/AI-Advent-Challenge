//! The project's git checkout: where it is, getting it, and running `git` in
//! it. Lesson 31's module, plus what a review needs: resolving revisions,
//! fetching a pull request's commits, the diff between two commits, and a
//! temporary worktree at the head being reviewed. The checkout's own branch
//! and files are never changed.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::RwLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

pub const DEFAULT_REPO_URL: &str = "https://github.com/CaptainDmitro/AI-Advent-Challenge.git";
const GIT_TIMEOUT: Duration = Duration::from_secs(20);
const CLONE_TIMEOUT: Duration = Duration::from_secs(600);
const FETCH_TIMEOUT: Duration = Duration::from_secs(180);
/// Lines of context around each change in the diff the model sees.
const DIFF_CONTEXT: &str = "-U5";
const MAX_REF_CHARS: usize = 100;
const MAX_PATH_CHARS: usize = 300;

/// Runs `git` with `args`, in `dir` when given. No prompts (a clone that
/// would ask for credentials fails instead of hanging), C locale (stable
/// output), killed on timeout. Errors are readable sentences.
async fn run(dir: Option<&Path>, args: &[&str], timeout: Duration) -> Result<String, String> {
    let mut command = Command::new("git");
    if let Some(dir) = dir {
        command.arg("-C").arg(dir);
    }
    command
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            "git is not installed on this machine.".to_string()
        } else {
            format!("Can't run git: {e}")
        }
    })?;
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| {
            format!(
                "`git {}` took longer than {}s.",
                args.first().copied().unwrap_or_default(),
                timeout.as_secs()
            )
        })?
        .map_err(|e| format!("git failed: {e}"))?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Err(format!(
        "`git {}` failed: {}",
        args.join(" "),
        if stderr.is_empty() {
            output.status.to_string()
        } else {
            stderr
        }
    ))
}

/// Where the checkout is in its life.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RepoState {
    /// The first start: `git clone` is running.
    Cloning,
    Ready,
    /// No checkout: not cloned (yet), no `git`, or the clone failed.
    Unavailable { reason: String },
}

#[derive(Debug)]
pub struct Repo {
    pub dir: PathBuf,
    pub url: String,
    state: RwLock<RepoState>,
}

/// What `git_branch` reports.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BranchInfo {
    /// The current branch, or `HEAD` when detached.
    pub branch: String,
    pub detached: bool,
    /// HEAD's short hash.
    pub head: String,
    /// The branch it tracks, e.g. `origin/master`.
    pub upstream: Option<String>,
    /// Every local branch.
    pub branches: Vec<String>,
}

impl Repo {
    pub fn new(dir: PathBuf, url: String) -> Repo {
        let state = if dir.join(".git").exists() {
            RepoState::Ready
        } else {
            RepoState::Unavailable {
                reason: format!("There is no git checkout at {} yet.", dir.display()),
            }
        };
        Repo {
            dir,
            url,
            state: RwLock::new(state),
        }
    }

    pub fn state(&self) -> RepoState {
        self.state.read().unwrap().clone()
    }

    fn set(&self, state: RepoState) {
        *self.state.write().unwrap() = state;
    }

    /// Makes sure there is a checkout: one that exists is used as it is
    /// (never pulled or reset: it may be on a branch someone switched to);
    /// otherwise, when `clone` is on, it is cloned from `url`. The clone goes
    /// into `<dir>.partial` and is renamed when complete, so a clone killed
    /// halfway never looks like a checkout.
    pub async fn ensure(&self, clone: bool) {
        if self.dir.join(".git").exists() {
            self.set(RepoState::Ready);
            return;
        }
        if !clone {
            self.set(RepoState::Unavailable {
                reason: format!(
                    "There is no git checkout at {} and cloning is off.",
                    self.dir.display()
                ),
            });
            return;
        }
        self.set(RepoState::Cloning);
        println!("cloning {} into {}", self.url, self.dir.display());
        let partial = PathBuf::from(format!("{}.partial", self.dir.display()));
        let _ = std::fs::remove_dir_all(&partial);
        let partial_str = partial.display().to_string();
        let result = run(
            None,
            &["clone", "--quiet", "--", self.url.as_str(), partial_str.as_str()],
            CLONE_TIMEOUT,
        )
        .await
        .and_then(|_| {
            std::fs::rename(&partial, &self.dir)
                .map_err(|e| format!("Can't move the clone into place: {e}"))
        });
        match result {
            Ok(()) => {
                println!("cloned into {}", self.dir.display());
                self.set(RepoState::Ready);
            }
            Err(e) => {
                eprintln!("clone failed: {e}");
                let _ = std::fs::remove_dir_all(&partial);
                self.set(RepoState::Unavailable { reason: e });
            }
        }
    }

    /// `git <args>` in the checkout, or why there is no checkout to run it in.
    pub async fn git(&self, args: &[&str]) -> Result<String, String> {
        self.git_with_timeout(args, GIT_TIMEOUT).await
    }

    async fn git_with_timeout(&self, args: &[&str], timeout: Duration) -> Result<String, String> {
        match self.state() {
            RepoState::Ready => run(Some(&self.dir), args, timeout).await,
            RepoState::Cloning => {
                Err("The repository is still being cloned; try again in a moment.".to_string())
            }
            RepoState::Unavailable { reason } => Err(reason),
        }
    }

    pub async fn branch(&self) -> Result<BranchInfo, String> {
        let head = self.git(&["rev-parse", "--short", "HEAD"]).await?;
        let branch = self.git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
        let branch = branch.trim().to_string();
        let upstream = self
            .git(&[
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ])
            .await
            .ok()
            .map(|u| u.trim().to_string())
            .filter(|u| !u.is_empty());
        let branches = self
            .git(&["for-each-ref", "--format=%(refname:short)", "refs/heads"])
            .await?
            .lines()
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .map(str::to_string)
            .collect();
        Ok(BranchInfo {
            detached: branch == "HEAD",
            branch,
            head: head.trim().to_string(),
            upstream,
            branches,
        })
    }

    /// The full commit hash `rev` names (a branch, tag, hash or `HEAD~2`).
    pub async fn rev_parse(&self, rev: &str) -> Result<String, String> {
        let rev = valid_ref(rev)?;
        let spec = format!("{rev}^{{commit}}");
        self.git(&["rev-parse", "--verify", "--quiet", spec.as_str()])
            .await
            .map(|sha| sha.trim().to_string())
            .map_err(|_| format!("There is no commit `{rev}` in the repository."))
    }

    /// Whether the checkout has this commit (a PR's head may not be fetched yet).
    pub async fn has_commit(&self, sha: &str) -> bool {
        let Ok(sha) = valid_ref(sha) else {
            return false;
        };
        let spec = format!("{sha}^{{commit}}");
        self.git(&["cat-file", "-e", spec.as_str()]).await.is_ok()
    }

    /// `git fetch origin <refspecs>`: brings in a pull request's commits
    /// (`refs/pull/N/head`) and its base branch. Only refs under
    /// `refs/review/` and `refs/remotes/origin/` change; the checked-out
    /// branch and the working tree don't.
    pub async fn fetch(&self, refspecs: &[String]) -> Result<(), String> {
        let mut args = vec!["fetch", "--quiet", "--no-tags", "origin"];
        args.extend(refspecs.iter().map(String::as_str));
        self.git_with_timeout(&args, FETCH_TIMEOUT).await.map(|_| ())
    }

    /// What the head changed since it left the base: `git diff base...head`,
    /// with rename detection and stable prefixes and quoting whatever the
    /// user's git config says.
    pub async fn diff(&self, base: &str, head: &str) -> Result<String, String> {
        let range = format!("{}...{}", valid_ref(base)?, valid_ref(head)?);
        self.git_with_timeout(
            &[
                "-c",
                "core.quotePath=false",
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--find-renames",
                DIFF_CONTEXT,
                "--src-prefix=a/",
                "--dst-prefix=b/",
                range.as_str(),
            ],
            FETCH_TIMEOUT,
        )
        .await
    }

    /// A detached worktree at `sha` in `dir`: the files exactly as they are
    /// at the head being reviewed, without touching the main checkout.
    pub async fn add_worktree(&self, dir: &Path, sha: &str) -> Result<(), String> {
        let sha = valid_ref(sha)?;
        let path = dir.display().to_string();
        self.git_with_timeout(
            &["worktree", "add", "--detach", "--force", path.as_str(), sha],
            FETCH_TIMEOUT,
        )
        .await
        .map(|_| ())
    }

    pub async fn remove_worktree(&self, dir: &Path) {
        let path = dir.display().to_string();
        if let Err(e) = self
            .git(&["worktree", "remove", "--force", path.as_str()])
            .await
        {
            eprintln!("removing the worktree {path} failed: {e}");
            let _ = std::fs::remove_dir_all(dir);
        }
        let _ = self.git(&["worktree", "prune"]).await;
    }

    /// A path inside the checkout that really is inside it, following
    /// symlinks, and isn't in `.git`.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        let rel = repo_path(rel)?;
        let root = self
            .dir
            .canonicalize()
            .map_err(|e| format!("Can't open the checkout: {e}"))?;
        let full = root
            .join(&rel)
            .canonicalize()
            .map_err(|_| format!("There is no `{rel}` in the repository."))?;
        if !full.starts_with(&root) || full.starts_with(root.join(".git")) {
            return Err(format!("`{rel}` is outside the repository's files."));
        }
        Ok(full)
    }
}

/// A path the tools accept: relative, `/`-separated, no `..`, not inside
/// `.git`, not starting with `-` (so git can't read it as an option).
/// `""`, `.` and `./` mean the repository root. A trailing `/` is dropped.
pub fn repo_path(input: &str) -> Result<String, String> {
    let trimmed = input.trim().trim_end_matches('/');
    let trimmed = trimmed.strip_prefix("./").unwrap_or(trimmed);
    if trimmed.is_empty() || trimmed == "." {
        return Ok(String::new());
    }
    if trimmed.chars().count() > MAX_PATH_CHARS || trimmed.starts_with('-') {
        return Err(format!("`{input}` is not a valid path."));
    }
    let path = Path::new(trimmed);
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_str().unwrap_or_default().to_string()),
            Component::CurDir => {}
            _ => {
                return Err(format!(
                    "`{input}` must be a path relative to the repository root, without `..`."
                ));
            }
        }
    }
    if parts.first().is_some_and(|p| p == ".git") {
        return Err("The `.git` directory is not readable through this server.".to_string());
    }
    Ok(parts.join("/"))
}

/// A revision the tools accept: a branch, tag, hash or `HEAD~2`-style
/// expression. Nothing that starts with `-` (an option), no `..` (ranges are
/// built by the tool itself), no spaces or shell-ish characters.
pub fn valid_ref(input: &str) -> Result<&str, String> {
    let r = input.trim();
    let ok = !r.is_empty()
        && r.chars().count() <= MAX_REF_CHARS
        && !r.starts_with('-')
        && !r.contains("..")
        && r.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-~^@".contains(c));
    if ok {
        Ok(r)
    } else {
        Err(format!(
            "`{input}` is not a valid revision (a branch, tag or commit, e.g. `master` or `HEAD~3`)."
        ))
    }
}

/// `NN. Title` checkouts: under `cargo run` the working directory is the
/// lesson folder inside a clone of the repo, so its parent *is* the
/// checkout. Elsewhere (the VDS) the default is `repo`, cloned on first start.
pub fn default_repo_dir() -> PathBuf {
    let parent = Path::new("..");
    if parent.join(".git").exists() && parent.join("AGENTS.md").exists() {
        parent
            .canonicalize()
            .unwrap_or_else(|_| parent.to_path_buf())
    } else {
        PathBuf::from("repo")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_stay_inside_the_repository() {
        assert_eq!(repo_path("docs").unwrap(), "docs");
        assert_eq!(repo_path("./docs/").unwrap(), "docs");
        assert_eq!(repo_path("").unwrap(), "");
        assert_eq!(repo_path(".").unwrap(), "");
        assert_eq!(
            repo_path("20. MCP Orchestration/src/main.rs").unwrap(),
            "20. MCP Orchestration/src/main.rs"
        );
        for bad in [
            "../etc/passwd",
            "docs/../../x",
            "/etc/passwd",
            ".git/config",
            ".git",
            "--output=x",
        ] {
            assert!(repo_path(bad).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn refs_cannot_be_options_or_ranges() {
        for good in ["master", "origin/master", "HEAD~3", "v1.0", "a1b2c3d", "HEAD^"] {
            assert_eq!(valid_ref(good).unwrap(), good);
        }
        for bad in [
            "",
            "--output=/tmp/x",
            "-p",
            "master..HEAD",
            "a b",
            "x;rm",
            "$(id)",
        ] {
            assert!(valid_ref(bad).is_err(), "{bad} was accepted");
        }
    }

    #[tokio::test]
    async fn a_missing_checkout_is_reported_not_run() {
        let dir = std::env::temp_dir().join(format!("lesson32-none-{}", std::process::id()));
        let repo = Repo::new(dir, DEFAULT_REPO_URL.to_string());
        assert!(matches!(repo.state(), RepoState::Unavailable { .. }));
        repo.ensure(false).await;
        let err = repo.branch().await.unwrap_err();
        assert!(err.contains("cloning is off"), "{err}");
    }
}

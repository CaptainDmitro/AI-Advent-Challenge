//! The `project` MCP server: read-only access to the git checkout over
//! Streamable HTTP at `/mcp/project`. Built like lesson 20's servers (rmcp
//! macros, a tool error is a result the model can read, not a protocol
//! error). Any MCP client can use it, not only this lesson's assistant.

use std::sync::Arc;

use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::git::{Repo, repo_path, valid_ref};

const DEFAULT_LOG: u32 = 10;
const MAX_LOG: u32 = 30;
const DEFAULT_FILES: u32 = 200;
const MAX_FILES: u32 = 500;
const MAX_STATUS_LINES: usize = 200;
const DEFAULT_READ_LINES: u32 = 200;
const MAX_READ_LINES: u32 = 5000;
const MAX_READ_BYTES: usize = 40 * 1024;
const MAX_DIFF_CHARS: usize = 12_000;

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong inside a tool (a bad argument, no checkout, a git error) is
/// a tool-level error the model can read and react to.
fn tool_error(message: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::error(vec![ContentBlock::text(message)]))
}

fn in_range(name: &str, value: Option<u32>, default: u32, max: u32) -> Result<u32, String> {
    let value = value.unwrap_or(default);
    if (1..=max).contains(&value) {
        Ok(value)
    } else {
        Err(format!("`{name}` must be 1-{max}, got {value}."))
    }
}

/// Shortens `text` to `max` characters for the read_file header.
fn truncate_text(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

/// The first `max` characters, cut at a char boundary.
fn clip(text: &str, max: usize) -> (String, bool) {
    if text.chars().count() <= max {
        return (text.to_string(), false);
    }
    (text.chars().take(max).collect(), true)
}

#[derive(Debug, Deserialize, JsonSchema)]
struct LogArgs {
    /// How many commits to show, newest first: 1-30, default 10.
    limit: Option<u32>,
    /// Only commits that touched this file or folder, e.g. "docs" or "AGENTS.md".
    path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListFilesArgs {
    /// A folder to list, relative to the repository root, e.g. "docs" or "20. MCP Orchestration". Empty for the whole repository.
    path: Option<String>,
    /// Only paths that contain this text (case-insensitive), e.g. "README" or ".rs".
    contains: Option<String>,
    /// How many paths to return: 1-500, default 200.
    limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadFileArgs {
    /// A file path relative to the repository root, e.g. "docs/architecture.md" or "31. Developer Assistant/Cargo.toml".
    path: String,
    /// The first line to return, 1-based. Default 1.
    start_line: Option<u32>,
    /// How many lines to return: 1-5000, default 200.
    max_lines: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiffArgs {
    /// Compare the current branch with this branch or commit (what the branch changed since it left `base`, e.g. "master"). Leave empty for the uncommitted changes in the working tree.
    base: Option<String>,
    /// Only this file or folder.
    path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectServer {
    repo: Arc<Repo>,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl ProjectServer {
    pub fn new(repo: Arc<Repo>) -> Self {
        Self {
            repo,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "git_branch",
        description = "The project's current git branch (or a detached HEAD), HEAD's short \
                       commit hash, the upstream branch it tracks, and every local branch."
    )]
    async fn git_branch(&self) -> Result<CallToolResult, ErrorData> {
        match self.repo.branch().await {
            Ok(info) => {
                let mut text = if info.detached {
                    format!("Detached HEAD at {}", info.head)
                } else {
                    format!("Current branch: {} (HEAD {})", info.branch, info.head)
                };
                if let Some(upstream) = &info.upstream {
                    text.push_str(&format!(", tracking {upstream}"));
                }
                text.push_str(&format!(".\nLocal branches: {}.", info.branches.join(", ")));
                tool_ok(text, json!(info))
            }
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "git_status",
        description = "The working tree's state: the branch, how far it is ahead of or behind \
                       its upstream, and every modified, added, deleted or untracked file."
    )]
    async fn git_status(&self) -> Result<CallToolResult, ErrorData> {
        let out = match self.repo.git(&["status", "--porcelain=v1", "--branch"]).await {
            Ok(out) => out,
            Err(e) => return tool_error(e),
        };
        let mut lines = out.lines();
        let header = lines
            .next()
            .unwrap_or_default()
            .trim_start_matches("## ")
            .to_string();
        let changes: Vec<Value> = lines
            .filter(|l| l.len() > 3)
            .map(|l| json!({ "status": l[..2].trim(), "path": l[3..].trim_matches('"') }))
            .collect();
        let shown = changes.len().min(MAX_STATUS_LINES);
        let text = if changes.is_empty() {
            format!("On {header}. The working tree is clean.")
        } else {
            let mut text = format!("On {header}. {} changed files:", changes.len());
            for c in &changes[..shown] {
                text.push_str(&format!(
                    "\n{:>2} {}",
                    c["status"].as_str().unwrap_or_default(),
                    c["path"].as_str().unwrap_or_default()
                ));
            }
            text
        };
        tool_ok(
            text,
            json!({ "branch": header, "clean": changes.is_empty(), "changes": &changes[..shown] }),
        )
    }

    #[tool(
        name = "git_log",
        description = "Recent commits on the current branch, newest first: short hash, date, \
                       author and subject. Optionally only those that touched one path."
    )]
    async fn git_log(
        &self,
        Parameters(args): Parameters<LogArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = match in_range("limit", args.limit, DEFAULT_LOG, MAX_LOG) {
            Ok(limit) => limit.to_string(),
            Err(e) => return tool_error(e),
        };
        let path = match args.path.as_deref().map(repo_path).transpose() {
            Ok(path) => path.filter(|p| !p.is_empty()),
            Err(e) => return tool_error(e),
        };
        let mut git_args = vec![
            "log",
            "-n",
            limit.as_str(),
            "--date=short",
            "--pretty=format:%h%x09%ad%x09%an%x09%s",
        ];
        if let Some(path) = &path {
            git_args.extend(["--", path.as_str()]);
        }
        let out = match self.repo.git(&git_args).await {
            Ok(out) => out,
            Err(e) => return tool_error(e),
        };
        let commits: Vec<Value> = out
            .lines()
            .filter_map(|line| {
                let mut f = line.splitn(4, '\t');
                Some(json!({
                    "hash": f.next()?, "date": f.next()?, "author": f.next()?, "subject": f.next()?,
                }))
            })
            .collect();
        let mut text = format!(
            "{} most recent commits{}:",
            commits.len(),
            path.as_ref().map(|p| format!(" touching `{p}`")).unwrap_or_default()
        );
        for c in &commits {
            text.push_str(&format!(
                "\n{} {} {}: {}",
                c["hash"].as_str().unwrap_or_default(),
                c["date"].as_str().unwrap_or_default(),
                c["author"].as_str().unwrap_or_default(),
                c["subject"].as_str().unwrap_or_default()
            ));
        }
        tool_ok(text, json!({ "path": path, "commits": commits }))
    }

    #[tool(
        name = "list_files",
        description = "Files tracked in the repository (git ls-files), optionally inside one \
                       folder or matching a text, with a count of files per top-level folder. \
                       Use it to see how the project is laid out or to find a file."
    )]
    async fn list_files(
        &self,
        Parameters(args): Parameters<ListFilesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = match in_range("limit", args.limit, DEFAULT_FILES, MAX_FILES) {
            Ok(limit) => limit as usize,
            Err(e) => return tool_error(e),
        };
        let folder = match repo_path(args.path.as_deref().unwrap_or_default()) {
            Ok(folder) => folder,
            Err(e) => return tool_error(e),
        };
        let out = match self.repo.git(&["ls-files", "-z"]).await {
            Ok(out) => out,
            Err(e) => return tool_error(e),
        };
        let contains = args
            .contains
            .as_deref()
            .map(|c| c.trim().to_lowercase())
            .filter(|c| !c.is_empty());
        let prefix = format!("{folder}/");
        let files: Vec<&str> = out
            .split('\0')
            .filter(|f| !f.is_empty())
            .filter(|f| folder.is_empty() || f.starts_with(&prefix))
            .filter(|f| contains.as_ref().is_none_or(|c| f.to_lowercase().contains(c)))
            .collect();
        if files.is_empty() {
            return tool_error(format!(
                "No tracked files{}{}.",
                if folder.is_empty() {
                    String::new()
                } else {
                    format!(" in `{folder}`")
                },
                contains
                    .as_ref()
                    .map(|c| format!(" matching `{c}`"))
                    .unwrap_or_default()
            ));
        }
        // Files per first path segment below the folder: the layout at a glance.
        let mut groups: Vec<(String, usize)> = Vec::new();
        for f in &files {
            let f: &str = f;
            let rest = if folder.is_empty() { f } else { &f[prefix.len()..] };
            let top = match rest.split_once('/') {
                Some((dir, _)) => format!("{dir}/"),
                None => rest.to_string(),
            };
            match groups.iter_mut().find(|(g, _)| *g == top) {
                Some((_, n)) => *n += 1,
                None => groups.push((top, 1)),
            }
        }
        let shown = &files[..files.len().min(limit)];
        let mut text = format!(
            "{} tracked files in {}{}, showing {}.\nBy top level: {}\n",
            files.len(),
            if folder.is_empty() {
                "the repository".to_string()
            } else {
                format!("`{folder}`")
            },
            contains
                .as_ref()
                .map(|c| format!(" matching `{c}`"))
                .unwrap_or_default(),
            shown.len(),
            groups
                .iter()
                .map(|(g, n)| format!("{g} ({n})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for f in shown {
            text.push('\n');
            text.push_str(f);
        }
        tool_ok(
            text,
            json!({
                "path": folder,
                "total": files.len(),
                "files": shown,
                "groups": groups.iter().map(|(g, n)| json!({ "name": g, "files": n })).collect::<Vec<_>>(),
            }),
        )
    }

    #[tool(
        name = "read_file",
        description = "Read a text file from the checkout (as it is on disk, including \
                       uncommitted edits), a range of lines at a time."
    )]
    async fn read_file(
        &self,
        Parameters(args): Parameters<ReadFileArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        if let Err(e) = self.repo.git(&["rev-parse", "--git-dir"]).await {
            return tool_error(e);
        }
        let max_lines = match in_range("max_lines", args.max_lines, DEFAULT_READ_LINES, MAX_READ_LINES)
        {
            Ok(n) => n as usize,
            Err(e) => return tool_error(e),
        };
        let start = args.start_line.unwrap() as usize;
        let full = match self.repo.resolve(&args.path) {
            Ok(full) => full,
            Err(e) => return tool_error(e),
        };
        let rel = repo_path(&args.path).unwrap_or_default();
        if !full.is_file() {
            return tool_error(format!(
                "`{rel}` is a folder; use `list_files` to see what is in it."
            ));
        }
        let bytes = match std::fs::read(&full) {
            Ok(bytes) => bytes,
            Err(e) => return tool_error(format!("Can't read `{rel}`: {e}")),
        };
        let Ok(content) = String::from_utf8(bytes) else {
            return tool_error(format!("`{rel}` is not a text file."));
        };
        let total = content.lines().count();
        if start > total.max(1) {
            return tool_error(format!("`{rel}` has only {total} lines."));
        }
        let mut text = String::new();
        let mut end = start - 1;
        for line in content.lines().skip(start).take(max_lines) {
            if text.len() + line.len() > MAX_READ_BYTES {
                break;
            }
            text.push_str(line);
            text.push('\n');
            end += 1;
        }
        let header = truncate_text(&format!("{rel}, lines {start}-{end} of {total}:"), 200);
        tool_ok(
            format!("{header}\n\n{text}"),
            json!({
                "path": rel, "start_line": start, "end_line": end, "total_lines": total,
                "truncated": end < total, "content": text,
            }),
        )
    }

    #[tool(
        name = "git_diff",
        description = "What changed: without `base`, the uncommitted changes in the working \
                       tree; with `base`, everything the current branch changed since it \
                       diverged from `base` (git diff base...HEAD). A per-file summary, then \
                       the patch (clipped when long)."
    )]
    async fn git_diff(
        &self,
        Parameters(args): Parameters<DiffArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let base = match args.base.as_deref().map(str::trim).filter(|b| !b.is_empty()) {
            Some(base) => match valid_ref(base) {
                Ok(base) => Some(base.to_string()),
                Err(e) => return tool_error(e),
            },
            None => None,
        };
        let path = match args.path.as_deref().map(repo_path).transpose() {
            Ok(path) => path.filter(|p| !p.is_empty()),
            Err(e) => return tool_error(e),
        };
        let spec = match &base {
            Some(base) => format!("{base}...HEAD"),
            None => "HEAD".to_string(),
        };
        let mut tail = vec![spec.as_str()];
        if let Some(path) = &path {
            tail.extend(["--", path.as_str()]);
        }
        let stat = match self
            .repo
            .git(&[&["diff", "--stat"][..], &tail[..]].concat())
            .await
        {
            Ok(out) => out,
            Err(e) => return tool_error(e),
        };
        let patch = match self.repo.git(&[&["diff"][..], &tail[..]].concat()).await {
            Ok(out) => out,
            Err(e) => return tool_error(e),
        };
        let what = match &base {
            Some(base) => format!("Changes on this branch since it left `{base}`"),
            None => "Uncommitted changes".to_string(),
        };
        if patch.trim().is_empty() {
            return tool_ok(
                format!("{what}: none."),
                json!({ "base": base, "path": path, "stat": "", "diff": "", "truncated": false }),
            );
        }
        let (diff, truncated) = clip(&patch, MAX_DIFF_CHARS);
        tool_ok(
            format!(
                "{what}:\n{}\n\n{diff}{}",
                stat.trim_end(),
                if truncated {
                    "\n… (the patch is clipped; ask for one `path` to see the rest)"
                } else {
                    ""
                }
            ),
            json!({ "base": base, "path": path, "stat": stat, "diff": diff, "truncated": truncated }),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ProjectServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("project", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only access to the project's git checkout: `git_branch` (current branch), \
                 `git_status`, `git_log`, `git_diff`, `list_files` and `read_file`.",
            )
    }
}

pub fn project_service(repo: Arc<Repo>) -> StreamableHttpService<ProjectServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(ProjectServer::new(repo.clone())),
        Default::default(),
        // Reached on the VDS by IP, like lesson 20's servers, so any Host.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

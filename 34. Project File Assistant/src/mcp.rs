//! The `files` MCP server: the project's files over Streamable HTTP at
//! `/mcp/files`. Built like lessons 31 and 33 (rmcp macros, a tool error is a
//! result the model can read, not a protocol error). Four tools read and
//! search, two write; every write answers with the diff it made. Any MCP
//! client can use it, not only this lesson's assistant.

use std::sync::Arc;

use regex::{Regex, RegexBuilder};
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

use crate::diff::{FileDiff, file_diff, patch, summary, tree_diff};
use crate::workspace::{FileInfo, Workspace, normalize};

const MAX_READ_LINES: u32 = 400;
const DEFAULT_MATCHES: u32 = 50;
const MAX_MATCHES: u32 = 200;
/// A matching line is clipped to this many characters.
const MAX_LINE_CHARS: usize = 200;
const MAX_PATTERN_CHARS: usize = 300;

fn tool_ok(text: String, structured: Value) -> Result<CallToolResult, ErrorData> {
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Anything wrong inside a tool (a bad path, text not found, a failed write)
/// is a tool-level error the model can read and react to.
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

/// A folder argument: empty, `.` and `/` mean the whole project.
fn folder(dir: Option<&str>) -> Result<Option<String>, String> {
    match dir.map(str::trim) {
        None | Some("" | "." | "./" | "/") => Ok(None),
        Some(dir) => normalize(dir).map(Some),
    }
}

fn under(path: &str, prefix: Option<&str>) -> bool {
    prefix.is_none_or(|p| path == p || path.starts_with(&format!("{p}/")))
}

fn clip_line(line: &str) -> String {
    let line = line.trim();
    if line.chars().count() <= MAX_LINE_CHARS {
        line.to_string()
    } else {
        let mut out: String = line.chars().take(MAX_LINE_CHARS).collect();
        out.push('…');
        out
    }
}

/// What a write tool answers: what happened, and the diff.
fn write_result(d: FileDiff) -> Result<CallToolResult, ErrorData> {
    let text = if d.diff.is_empty() {
        format!("`{}` is unchanged: the new content is the same as the old.", d.path)
    } else {
        format!(
            "{} `{}` (+{} -{} lines), saved:\n```diff\n{}```",
            if d.status == "added" { "Created" } else { "Updated" },
            d.path,
            d.added,
            d.removed,
            d.diff
        )
    };
    tool_ok(text, json!(d))
}

// ---------------------------------------------------------------------------
// Tool arguments.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct ListArgs {
    /// Only the files under this folder, e.g. "app/routes". The whole project when left out.
    dir: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadArgs {
    /// The file, relative to the project root, e.g. "app/routes/notes.py".
    path: String,
    /// The first line to return, 1-based. Default 1.
    start_line: Option<u32>,
    /// The last line to return, inclusive. Default: up to 400 lines after start_line.
    end_line: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    /// The text to look for, e.g. "require_auth". A regular expression when `regex` is true.
    pattern: String,
    /// Treat `pattern` as a regular expression (Rust regex syntax), e.g. "@router\\.(get|post)".
    regex: Option<bool>,
    /// Match case exactly. Default false.
    case_sensitive: Option<bool>,
    /// Only files under this folder, or this one file, e.g. "app" or "docs/api.md".
    path: Option<String>,
    /// Only files with this extension, e.g. "py" or "md".
    extension: Option<String>,
    /// How many matching lines to return: 1-200, default 50.
    max_results: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WriteArgs {
    /// The file to create or replace, relative to the project root, e.g. "docs/adr/0002-token-expiry.md".
    path: String,
    /// The whole new content of the file.
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditArgs {
    /// The file to change, relative to the project root.
    path: String,
    /// The exact text to replace, copied from read_file without the line numbers. It must occur once, unless replace_all is true.
    old_text: String,
    /// The text to put in its place.
    new_text: String,
    /// Replace every occurrence instead of exactly one.
    replace_all: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DiffArgs {
    /// Only the changes under this folder or in this file.
    path: Option<String>,
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FilesServer {
    ws: Arc<Workspace>,
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl FilesServer {
    pub fn new(ws: Arc<Workspace>) -> Self {
        Self {
            ws,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "list_files",
        description = "The project's text files with their sizes in lines, sorted by path, \
                       optionally only under one folder. Start here to see what the project has."
    )]
    async fn list_files(
        &self,
        Parameters(args): Parameters<ListArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let dir = match folder(args.dir.as_deref()) {
            Ok(dir) => dir,
            Err(e) => return tool_error(e),
        };
        let files: Vec<FileInfo> = match self.ws.list(dir.as_deref()) {
            Ok(files) => files,
            Err(e) => return tool_error(e),
        };
        let mut text = format!(
            "{} files in {}:",
            files.len(),
            dir.as_deref().map_or("the project".to_string(), |d| format!("`{d}`"))
        );
        for f in &files {
            text.push_str(&format!("\n- {} ({} lines)", f.path, f.lines));
        }
        tool_ok(
            text,
            json!({ "dir": dir, "total": files.len(), "files": files }),
        )
    }

    #[tool(
        name = "read_file",
        description = "A file's content with line numbers (`  12 | text`), at most 400 lines \
                       per call; use start_line/end_line for a part of a long file."
    )]
    async fn read_file(
        &self,
        Parameters(args): Parameters<ReadArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let (path, text) = match self.ws.read(&args.path) {
            Ok(file) => file,
            Err(e) => return tool_error(e),
        };
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len() as u32;
        let start = args.start_line.unwrap_or(1).max(1);
        if total > 0 && start > total {
            return tool_error(format!("`{path}` has only {total} lines."));
        }
        let last = start.saturating_add(MAX_READ_LINES - 1).min(total);
        let end = if total == 0 {
            0
        } else {
            args.end_line.map_or(last, |e| e.clamp(start, last))
        };
        let shown = if total == 0 {
            &lines[..]
        } else {
            &lines[(start - 1) as usize..end as usize]
        };
        let mut out = if total == 0 {
            format!("`{path}` is empty.")
        } else {
            format!("`{path}`, lines {start}-{end} of {total}:")
        };
        for (i, line) in shown.iter().enumerate() {
            out.push_str(&format!("\n{:>4} | {line}", start as usize + i));
        }
        let truncated = end < total;
        if truncated {
            out.push_str(&format!(
                "\n… {} more lines: read_file with start_line={}.",
                total - end,
                end + 1
            ));
        }
        tool_ok(
            out,
            json!({
                "path": path,
                "start_line": start,
                "end_line": end,
                "total_lines": total,
                "truncated": truncated,
                "content": shown.join("\n"),
            }),
        )
    }

    #[tool(
        name = "search_files",
        description = "Search every text file of the project (or one folder, or one \
                       extension) for a text or a regular expression. Returns each matching \
                       line as `path:line: text`, and how many matches each file has."
    )]
    async fn search_files(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let pattern = args.pattern.trim();
        if pattern.is_empty() {
            return tool_error("The pattern is empty.".to_string());
        }
        if pattern.chars().count() > MAX_PATTERN_CHARS {
            return tool_error(format!(
                "The pattern is longer than {MAX_PATTERN_CHARS} characters."
            ));
        }
        let limit = match in_range("max_results", args.max_results, DEFAULT_MATCHES, MAX_MATCHES)
        {
            Ok(limit) => limit as usize,
            Err(e) => return tool_error(e),
        };
        let is_regex = args.regex.unwrap_or(false);
        let source = if is_regex {
            pattern.to_string()
        } else {
            regex::escape(pattern)
        };
        let re: Regex = match RegexBuilder::new(&source)
            .case_insensitive(!args.case_sensitive.unwrap_or(false))
            .size_limit(1 << 20)
            .build()
        {
            Ok(re) => re,
            Err(e) => return tool_error(format!("Invalid regular expression: {e}")),
        };
        let prefix = match folder(args.path.as_deref()) {
            Ok(prefix) => prefix,
            Err(e) => return tool_error(e),
        };
        let extension = args
            .extension
            .as_deref()
            .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|e| !e.is_empty());
        let files = match self.ws.list(None) {
            Ok(files) => files,
            Err(e) => return tool_error(e),
        };
        let mut searched = 0;
        let mut total = 0;
        let mut per_file: Vec<Value> = Vec::new();
        let mut matches: Vec<Value> = Vec::new();
        for file in &files {
            if !under(&file.path, prefix.as_deref()) {
                continue;
            }
            if let Some(ext) = &extension
                && !file.path.to_ascii_lowercase().ends_with(&format!(".{ext}"))
            {
                continue;
            }
            let Ok((_, text)) = self.ws.read(&file.path) else {
                continue;
            };
            searched += 1;
            let mut count = 0;
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    count += 1;
                    if matches.len() < limit {
                        matches.push(json!({
                            "path": file.path,
                            "line": i + 1,
                            "text": clip_line(line),
                        }));
                    }
                }
            }
            if count > 0 {
                total += count;
                per_file.push(json!({ "path": file.path, "matches": count }));
            }
        }
        let scope = match (&prefix, &extension) {
            (Some(p), Some(e)) => format!("`{p}` (*.{e})"),
            (Some(p), None) => format!("`{p}`"),
            (None, Some(e)) => format!("the project (*.{e})"),
            (None, None) => "the project".to_string(),
        };
        let mut text = if total == 0 {
            format!("No matches for `{pattern}` in {searched} files of {scope}.")
        } else {
            format!(
                "`{pattern}`: {total} matches in {} of {searched} files of {scope}{}:",
                per_file.len(),
                if matches.len() < total {
                    format!(" (showing the first {})", matches.len())
                } else {
                    String::new()
                }
            )
        };
        for m in &matches {
            text.push_str(&format!(
                "\n{}:{}: {}",
                m["path"].as_str().unwrap_or_default(),
                m["line"],
                m["text"].as_str().unwrap_or_default()
            ));
        }
        tool_ok(
            text,
            json!({
                "pattern": pattern,
                "regex": is_regex,
                "searched": searched,
                "total": total,
                "files": per_file,
                "matches": matches,
            }),
        )
    }

    #[tool(
        name = "write_file",
        description = "WRITE. Create a new file or replace a file's whole content. Saved at \
                       once; returns the unified diff of the change. Prefer edit_file for a \
                       small change to an existing file."
    )]
    async fn write_file(
        &self,
        Parameters(args): Parameters<WriteArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.ws.write(&args.path, &args.content) {
            Ok((path, before)) => write_result(file_diff(&path, before.as_deref(), Some(&args.content))),
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "edit_file",
        description = "WRITE. Replace an exact piece of text in an existing file (old_text → \
                       new_text). old_text must occur exactly once unless replace_all is true; \
                       copy it from read_file without the line numbers. Saved at once; returns \
                       the unified diff of the change."
    )]
    async fn edit_file(
        &self,
        Parameters(args): Parameters<EditArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        if args.old_text.is_empty() {
            return tool_error("old_text is empty; use write_file to create a file.".to_string());
        }
        if args.old_text == args.new_text {
            return tool_error("old_text and new_text are the same.".to_string());
        }
        let (path, text) = match self.ws.read(&args.path) {
            Ok(file) => file,
            Err(e) => return tool_error(e),
        };
        let count = text.matches(args.old_text.as_str()).count();
        let replace_all = args.replace_all.unwrap_or(false);
        if count == 0 {
            return tool_error(format!(
                "old_text was not found in `{path}`. Read the file again and copy the text \
                 exactly, without the line numbers."
            ));
        }
        if count > 1 && !replace_all {
            return tool_error(format!(
                "old_text occurs {count} times in `{path}`. Add surrounding lines to make it \
                 unique, or set replace_all to change all {count}."
            ));
        }
        let new = if replace_all {
            text.replace(args.old_text.as_str(), &args.new_text)
        } else {
            text.replacen(args.old_text.as_str(), &args.new_text, 1)
        };
        match self.ws.write(&path, &new) {
            Ok(_) => write_result(file_diff(&path, Some(&text), Some(&new))),
            Err(e) => tool_error(e),
        }
    }

    #[tool(
        name = "get_diff",
        description = "Every change in the project since the baseline (the demo project as \
                       shipped, or the folder as it was when the service started), as a \
                       unified diff with a list of changed files."
    )]
    async fn get_diff(
        &self,
        Parameters(args): Parameters<DiffArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let prefix = match folder(args.path.as_deref()) {
            Ok(prefix) => prefix,
            Err(e) => return tool_error(e),
        };
        let now = match self.ws.snapshot() {
            Ok(now) => now,
            Err(e) => return tool_error(e),
        };
        let diffs: Vec<FileDiff> = tree_diff(self.ws.baseline(), &now)
            .into_iter()
            .filter(|d| under(&d.path, prefix.as_deref()))
            .collect();
        let all = patch(&diffs);
        let text = if diffs.is_empty() {
            "No changes since the baseline.".to_string()
        } else {
            format!("{}\n```diff\n{all}```", summary(&diffs))
        };
        let files: Vec<Value> = diffs
            .iter()
            .map(|d| json!({ "path": d.path, "status": d.status, "added": d.added, "removed": d.removed }))
            .collect();
        tool_ok(text, json!({ "files": files, "diff": all }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for FilesServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("files", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "A project's text files. Read: `list_files`, `read_file`, `search_files`, \
                 `get_diff`. Write: `write_file`, `edit_file` (both answer with the diff).",
            )
    }
}

pub fn files_service(ws: Arc<Workspace>) -> StreamableHttpService<FilesServer, LocalSessionManager> {
    StreamableHttpService::new(
        move || Ok(FilesServer::new(ws.clone())),
        Default::default(),
        // Reached on the VDS by IP, like lessons 31 and 33, so any Host.
        StreamableHttpServerConfig::default().disable_allowed_hosts(),
    )
}

# API and data schemas

The HTTP and MCP interfaces the lessons expose, and the shapes of the data
they pass around. Every web lesson serves its page at `GET /` and its JSON
API under `/api/...` on port `4000 + NN` when deployed (`3000` locally).

## Lesson 31: developer assistant

### HTTP

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the chat page |
| `POST` | `/api/help` | `{"message": "/help <question>"}` | a help reply (below) |
| `GET` | `/api/status` | | repo, branch, index, MCP server and model |
| `POST` | `/api/reindex` | | rebuilds the documentation index, returns its info |
| `POST` | `/mcp/project` | MCP JSON-RPC | the `project` MCP server (Streamable HTTP) |

`message` accepts three forms:

- `/help`: an overview of the commands, the current branch and the index.
  No LLM call.
- `/help <question>`: an answer from the documentation and the MCP tools.
- plain text: the same as `/help <text>`.

Any other `/command` returns `error: "Unknown command ..."`. Errors come back
as a string in `error` with HTTP 200, like every web lesson except 30.

### The help reply

```json
{
  "message": "/help How do I add a new lesson?",
  "command": "help",
  "question": "How do I add a new lesson?",
  "search_query": "How do I add a new lesson?",
  "answer": "Create the next folder `NN. Title` ... [1] ...",
  "error": null,
  "branch": {
    "branch": "master", "detached": false, "head": "97c33fa",
    "upstream": "origin/master", "branches": ["master"]
  },
  "sources": [
    { "n": 1, "chunk_id": "docs/conventions.md#000", "source": "docs/conventions.md",
      "section": "Conventions › Adding a new lesson", "score": 0.71, "cited": true, "text": "..." }
  ],
  "candidates": [
    { "chunk_id": "...", "source": "...", "section": "...", "cosine": 0.42, "score": 0.71, "passed": true }
  ],
  "calls": [
    { "step": 1, "round": 0, "tool": "git_branch", "arguments": {}, "auto": true,
      "is_error": false, "result": "Current branch: master (HEAD 97c33fa) ...", "structured": {}, "ms": 9 }
  ],
  "tools": ["git_branch", "git_status", "git_log", "list_files", "read_file", "git_diff"],
  "mcp_url": "http://127.0.0.1:4031/mcp/project",
  "mcp_error": null,
  "llm_rounds": 1,
  "index": { "origin": { "kind": "checkout", "signature": "<sha>:<hash>" },
             "documents": 36, "chunks": 400, "sources": ["README.md", "..."],
             "embedder": "local:hashed-tfidf-1024", "built_ms": 40, "error": null },
  "ms": 2400
}
```

- `branch` comes from the MCP server's `git_branch` structured result, never
  from the model.
- `sources` are the fragments that passed the relevance threshold (at most
  5), numbered as the model saw them; `cited` is true when the answer has
  `[n]`.
- `calls[0]` is always the automatic `git_branch` (`auto: true`,
  `round: 0`); the rest are the model's own tool calls, by LLM round.
- `index.origin.kind` is `checkout` (built from the git checkout at that
  signature) or `builtin` (the docs compiled into the binary).

### MCP server `project`

Read-only tools over the git checkout. Every tool returns text for the model
plus `structuredContent`. A bad argument or a git failure is a tool result
with `isError: true`, not a protocol error.

| Tool | Arguments | Structured result |
|---|---|---|
| `git_branch` | — | `{branch, detached, head, upstream, branches}` |
| `git_status` | — | `{branch, clean, changes: [{status, path}]}` |
| `git_log` | `limit?` (1–30, default 10), `path?` | `{path, commits: [{hash, date, author, subject}]}` |
| `list_files` | `path?` (folder), `contains?`, `limit?` (1–500, default 200) | `{path, total, files, groups: [{name, files}]}` |
| `read_file` | `path`, `start_line?`, `max_lines?` (1–400, default 200) | `{path, start_line, end_line, total_lines, truncated, content}` |
| `git_diff` | `base?` (branch/commit), `path?` | `{base, path, stat, diff, truncated}` |

Paths are relative to the repository root, `/`-separated, without `..`, and
never inside `.git`. A revision may not start with `-` or contain `..`.
`git_diff` without `base` shows the uncommitted changes (`git diff HEAD`);
with `base` it shows what the branch changed since it left `base`
(`git diff base...HEAD`).

## Lesson 32: AI code review

### HTTP

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the review page |
| `GET` | `/api/pulls` | | `{pulls: [PullRequest], error}`: the repo's open PRs from the GitHub API |
| `POST` | `/api/review` | `{"pr": 12}` or `{"base": "master", "head": "HEAD"}` | a review (below); the page never posts |
| `GET` | `/api/status` | | repo, branch, GitHub repo and token, model, threshold, pinned docs |
| `POST` | `/mcp/project` | MCP JSON-RPC | lesson 31's `project` server over the service's checkout |

### Command line

`ai-code-review review [--pr N | --base REV --head REV] [--repo-dir DIR] [--post] [--out FILE]`
prints the Markdown review, appends it to `$GITHUB_STEP_SUMMARY` when set,
and with `--post` creates or edits the PR comment. Exit code 1 only when the
review or the posting failed, never because of findings. This is what
`.github/workflows/ai-review.yml` runs.

### The review reply

```json
{
  "target": { "pr": 12, "title": "...", "description": "...", "author": "dev", "url": "...",
              "base_ref": "master", "head_ref": "feature/x", "base_sha": "<40 hex>", "head_sha": "<40 hex>" },
  "checkout": "worktree",
  "files": [{ "path": "07. X/src/main.rs", "old_path": null, "status": "modified", "binary": false,
              "additions": 3, "deletions": 1, "skipped": null }],
  "stats": { "files": 2, "reviewed": 2, "additions": 7, "deletions": 1 },
  "summary": "...",
  "findings": [{ "category": "bug", "severity": "high", "file": "07. X/src/main.rs", "line": 8,
                 "title": "...", "details": "...", "suggestion": "...", "sources": [1, 4],
                 "line_outside_diff": null }],
  "dropped": [{ "reason": "`src/other.rs` is not a file in this diff", "raw": {} }],
  "queries": [{ "file": "07. X/src/main.rs", "hunk": "+3,11", "text": "Retry Lesson attempt MAX_RETRIES" }],
  "fragments": [{ "n": 1, "chunk_id": "docs/conventions.md#000", "source": "docs/conventions.md",
                  "section": "Conventions › Code", "kind": "doc", "start_line": 34, "end_line": 54,
                  "score": 1.0, "pinned": true, "found_for": null, "cited": true, "text": "..." }],
  "candidates": [{ "chunk_id": "...", "source": "...", "section": "...", "kind": "code", "score": 0.41,
                   "found_for": "...", "passed": true, "excluded": null }],
  "calls": [{ "step": 1, "batch": 1, "round": 1, "tool": "read_file", "arguments": {"path": "..."},
              "is_error": false, "result": "...", "ms": 4 }],
  "tools": ["git_branch", "git_status", "git_log", "list_files", "read_file", "git_diff"],
  "mcp_error": null, "batches": 1, "not_reviewed": [], "llm_rounds": 2,
  "index": { "documents": 152, "chunks": 2100, "doc_chunks": 400, "code_chunks": 1700,
             "embedder": "local:hashed-tfidf-1024", "built_ms": 300 },
  "markdown": "<!-- ai-code-review -->\n## 🤖 AI code review ...",
  "posted": { "action": "updated", "id": 123, "url": "https://github.com/.../pull/12#issuecomment-123" },
  "post_error": null, "error": null, "model": "deepseek-v4-flash", "ms": 21000
}
```

- `category` is `bug`, `architecture` or `recommendation`; `severity` is
  `high`, `medium` or `low`.
- A finding's `line` is always one the diff shows on the new side;
  `line_outside_diff` keeps a line the model gave that the diff doesn't
  show (the finding then applies to the whole file).
- `sources` are fragment numbers; `fragments[0..]` start with the pinned
  `docs/conventions.md` chunks.
- `checkout` is `checkout` when the head was already checked out (the
  Action) and `worktree` when a temporary worktree at the head was used.
- The PR comment starts with `<!-- ai-code-review -->`; the next run edits
  the comment with that marker instead of adding one.

## Lesson 33: support assistant

### HTTP

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the support page |
| `POST` | `/api/ask` | `{"message": "...", "ticket_id": "T-1001"}` (`ticket_id` optional) | an answer (below) |
| `GET` | `/api/tickets` | | `{tickets, error}`: `list_tickets` through the MCP server |
| `GET` | `/api/context?ticket_id=T-1001` | | `{ticket, user, hints, calls, error}`: the automatic `get_ticket` + `get_user` |
| `GET` | `/api/status` | | knowledge base, MCP server and tools, CRM file and counts, model |
| `POST` | `/api/reset` | | `{ok, error}`: puts `data/crm.json` back |
| `POST` | `/mcp/crm` | MCP JSON-RPC | the `crm` MCP server (Streamable HTTP) |

### The answer

```json
{
  "message": "Почему не работает авторизация?",
  "ticket_id": "T-1003",
  "search_query": "why does sign-in not work sign-in SSO_REQUIRED",
  "answer": "Your company turned on single sign-on ... [1]",
  "error": null,
  "ticket": { "id": "T-1003", "user_id": "U-103", "subject": "...", "status": "open", "priority": "high",
              "channel": "email", "created_at": "...", "updated_at": "...", "tags": ["sign-in"],
              "messages": [{ "at": "...", "from": "customer", "text": "..." }], "notes": [] },
  "user": { "id": "U-103", "name": "Chen Wei", "email": "chen.wei@bigcorp.com", "plan": "team",
            "security": { "email_verified": true, "two_factor": null,
                          "sso": { "provider": "Okta", "domain": "bigcorp.com", "enforced": true, "enforced_since": "..." },
                          "failed_logins": 0, "locked": false, "locked_until": null },
            "login_events": [{ "at": "...", "method": "password", "client": "web", "result": "failed", "code": "SSO_REQUIRED" }],
            "login_events_total": 4, "tickets": [{ "id": "T-1003", "subject": "...", "status": "open" }] },
  "context_hints": ["sign-in", "SSO_REQUIRED"],
  "sources": [{ "n": 1, "chunk_id": "kb/sign-in.md#005", "source": "kb/sign-in.md",
                "section": "Signing in to Notely › SSO_REQUIRED: your organization requires single sign-on",
                "score": 0.56, "cited": true, "text": "..." }],
  "candidates": [{ "chunk_id": "...", "source": "...", "section": "...", "cosine": 0.23, "score": 0.56, "passed": true }],
  "calls": [
    { "step": 1, "round": 0, "tool": "get_ticket", "arguments": { "ticket_id": "T-1003" }, "auto": true,
      "is_error": false, "result": "Ticket T-1003: ...", "structured": {}, "ms": 3 },
    { "step": 2, "round": 0, "tool": "get_user", "arguments": { "user_id": "U-103" }, "auto": true, "...": "..." }
  ],
  "tools": ["get_ticket", "get_user", "get_login_events", "list_tickets", "search_users", "add_ticket_note", "set_ticket_status"],
  "changed": false,
  "mcp_url": "http://127.0.0.1:4033/mcp/crm",
  "mcp_error": null,
  "llm_rounds": 1,
  "ms": 3100
}
```

- `ticket` and `user` come from the MCP server's structured results, never
  from the model; `user.login_events` are the last 5, newest first.
- `context_hints` are the ticket's tags and the error codes of the
  customer's recent failed sign-ins; they are appended to `search_query`.
  A question with Cyrillic in it is searched by its English rewrite.
- `calls` start with the automatic `get_ticket` and `get_user` (`auto:
  true`, `round: 0`) when a ticket is selected; the rest are the model's own.
- `changed` is true when `add_ticket_note` or `set_ticket_status` succeeded.

### MCP server `crm`

Every tool returns text for the model plus `structuredContent`. An unknown
id or a bad argument is a tool result with `isError: true`. Ids are
normalized: `1001`, `#1001` and `t1001` are `T-1001`; `101` is `U-101`.

| Tool | Arguments | Structured result |
|---|---|---|
| `get_ticket` | `ticket_id` | the ticket (as in `ticket` above) |
| `get_user` | `user_id?` or `email?` | the user (as in `user` above) |
| `get_login_events` | `user_id`, `limit?` (1–50, default 10), `failed_only?` | `{user_id, events}` |
| `list_tickets` | `status?` (`open`/`pending`/`resolved`/`closed`), `user_id?`, `limit?` (1–100, default 20) | `{total, tickets: [{id, subject, status, priority, channel, tags, updated_at, user_id, customer, email, plan}]}` |
| `search_users` | `query` | `{users: [{id, name, email, plan, company}]}` |
| `add_ticket_note` | `ticket_id`, `text` (≤ 2000 characters) | `{ticket_id, note: {at, author, text}, notes}` |
| `set_ticket_status` | `ticket_id`, `status`, `note?` | `{ticket_id, from, to}` |

The two write tools save `crm.json` at once (write-then-rename); notes are
signed `assistant`.

## Lesson 34: project file assistant

### HTTP

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | | the assistant page |
| `POST` | `/api/ask` | `{"message": "...", "dry_run": false}` (`dry_run` optional) | a task reply (below) |
| `GET` | `/api/files` | | `{files: [{path, bytes, lines}], changes: [{path, status}], error}`: the tree and what changed since the baseline |
| `GET` | `/api/file?path=docs/api.md` | | `{path, content, error}` |
| `GET` | `/api/diff` | | `{summary, patch, files: [FileDiff], error}`: every change since the baseline |
| `GET` | `/api/diff.patch` | | the same patch as `text/x-diff` |
| `GET` | `/api/status` | | project folder, demo or not, file count, MCP server and tools, model |
| `POST` | `/api/reset` | | `{ok, restored, error}`: puts `sample/` back (demo project only) |
| `POST` | `/mcp/files` | MCP JSON-RPC | the `files` MCP server (Streamable HTTP) |

### The task reply

```json
{
  "message": "Обнови docs/api.md по коду",
  "dry_run": false,
  "answer": "Added `DELETE /notes/{note_id}` and `POST /notes/{note_id}/share` ... Files changed: docs/api.md.",
  "error": null,
  "calls": [
    { "step": 1, "round": 0, "tool": "list_files", "arguments": {}, "auto": true,
      "is_error": false, "result": "13 files in the project: ...", "structured": {}, "ms": 2 },
    { "step": 2, "round": 1, "tool": "search_files", "arguments": { "pattern": "@router\\.", "regex": true }, "auto": false, "...": "..." }
  ],
  "tools": ["list_files", "read_file", "search_files", "write_file", "edit_file", "get_diff"],
  "changes": [{ "path": "docs/api.md", "status": "modified", "added": 14, "removed": 1,
                "diff": "--- a/docs/api.md\n+++ b/docs/api.md\n@@ -34,7 +34,7 @@ ..." }],
  "diff": "--- a/docs/api.md\n+++ b/docs/api.md\n...",
  "summary": "1 file changed: docs/api.md (modified, +14 -1)",
  "changed": true,
  "reverted": false,
  "mcp_url": "http://127.0.0.1:4034/mcp/files",
  "mcp_error": null,
  "llm_rounds": 5,
  "ms": 21000
}
```

- When the `files` server is reachable, `calls[0]` is the automatic
  `list_files` (`auto: true`, `round: 0`) and the rest are the model's own
  tool calls, by LLM round; when it isn't, `calls` is empty and `mcp_error`
  says why.
- `changes`, `diff` and `summary` compare every text file before the task
  with after it, read from disk, not from the model. `status` is `added`,
  `modified` or `deleted` (`write_file`/`edit_file` may also answer
  `unchanged`); `diff` is a unified diff (`a/`, `b/` prefixes,
  `/dev/null` for a new file) that `git apply` accepts.
- With `dry_run: true` the changes are rolled back after the task and
  `reverted` is true; the diff is still returned.

### MCP server `files`

Every tool returns text for the model plus `structuredContent`. A bad path,
an ambiguous edit or an invalid regex is a tool result with `isError: true`.

| Tool | Arguments | Structured result |
|---|---|---|
| `list_files` | `dir?` | `{dir, total, files: [{path, bytes, lines}]}` |
| `read_file` | `path`, `start_line?`, `end_line?` (at most 400 lines per call) | `{path, start_line, end_line, total_lines, truncated, content}` |
| `search_files` | `pattern`, `regex?`, `case_sensitive?`, `path?` (folder or file), `extension?`, `max_results?` (1–200, default 50) | `{pattern, regex, searched, total, files: [{path, matches}], matches: [{path, line, text}]}` |
| `get_diff` | `path?` | `{files: [{path, status, added, removed}], diff}` |
| `write_file` | `path`, `content` | `FileDiff`: `{path, status, added, removed, diff}` |
| `edit_file` | `path`, `old_text`, `new_text`, `replace_all?` | `FileDiff` |

Paths are relative to the project root, `/`-separated, without `..`, never
hidden (`.env`, `.git`) or under `target`/`node_modules`/`venv`/`__pycache__`,
and only text extensions; symlinks are not followed out of the project.
Files are at most 256 KB of UTF-8 and written with write-then-rename.
`edit_file`'s `old_text` must occur exactly once unless `replace_all`. The
baseline of `get_diff` is `sample/` for the demo project, or the folder as it
was at start for `PROJECT_DIR`.

## MCP in other lessons

| Lesson | Endpoint(s) | Tools |
|---|---|---|
| 17 | `/mcp` | `list_recent_commits` |
| 18 | `/mcp` | `watch_repo`, `remind_me`, `list_jobs`, `cancel_job`, `get_summary` |
| 19 | `/mcp` | `search`, `summarize`, `save_to_file` |
| 20 | `/mcp/crates`, `/mcp/github`, `/mcp/notes` | `search_crates`, `crate_info`; `search_repositories`, `repo_info`; `write_note`, `read_note`, `list_notes` |
| 31 | `/mcp/project` | `git_branch`, `git_status`, `git_log`, `list_files`, `read_file`, `git_diff` |
| 32 | `/mcp/project`, and a private one per review over the PR's head | the same 6 as lesson 31 |
| 33 | `/mcp/crm` | `get_ticket`, `get_user`, `get_login_events`, `list_tickets`, `search_users`, `add_ticket_note`, `set_ticket_status` |
| 34 | `/mcp/files` | `list_files`, `read_file`, `search_files`, `get_diff`, `write_file`, `edit_file` |

All use Streamable HTTP and accept any `Host` header, so they can be reached
by IP on the VDS, e.g. with `npx @modelcontextprotocol/inspector`.

## Lesson 30: OpenAI-compatible service

`POST /v1/chat/completions` and `GET /v1/models` with
`Authorization: Bearer <key>`, OpenAI request and response shapes, and real
HTTP error statuses (401, 413, 429, 503, ...). See that lesson's README for
the full table.

## Shared data shapes

### Chat completions (outgoing)

Lessons call `POST {OPENAI_BASE_URL}/chat/completions` with

```json
{ "model": "deepseek-v4-flash",
  "messages": [{ "role": "system", "content": "..." }, { "role": "user", "content": "..." }],
  "tools": [{ "type": "function", "function": { "name": "...", "description": "...", "parameters": {} } }] }
```

and read `choices[0].message`. A message with `tool_calls` is sent back
unchanged, followed by one `{"role": "tool", "tool_call_id", "content"}` per
call. MCP tools become functions by copying the tool's `inputSchema` into
`parameters` (lesson 20 prefixes the name with `<server>__`).

### Index chunk (lessons 21–25, 28, 29, 31, 32)

```json
{
  "chunk_id": "DEPLOYMENT.md#001",
  "source": "DEPLOYMENT.md",
  "title": "Deployment",
  "section": "Deployment › Why a static musl binary instead of Docker",
  "tokens": 97,
  "text": "## Why a static musl binary instead of Docker\n\n..."
}
```

Lesson 32's chunks also have `kind` (`doc` for Markdown, `code` for
everything else) and `start_line`/`end_line` (1-based, inclusive).
`chunk_id` is `<source>#<ordinal>`. `section` is the heading path for
Markdown (`File title › Heading › Subheading`) or the item for Rust
(`impl Registry`, `fn checksum`). `tokens` is an estimate: about 4 ASCII
characters or 2 other characters per token.

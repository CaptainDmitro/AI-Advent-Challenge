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

## MCP in other lessons

| Lesson | Endpoint(s) | Tools |
|---|---|---|
| 17 | `/mcp` | `list_recent_commits` |
| 18 | `/mcp` | `watch_repo`, `remind_me`, `list_jobs`, `cancel_job`, `get_summary` |
| 19 | `/mcp` | `search`, `summarize`, `save_to_file` |
| 20 | `/mcp/crates`, `/mcp/github`, `/mcp/notes` | `search_crates`, `crate_info`; `search_repositories`, `repo_info`; `write_note`, `read_note`, `list_notes` |
| 31 | `/mcp/project` | `git_branch`, `git_status`, `git_log`, `list_files`, `read_file`, `git_diff` |

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

### Index chunk (lessons 21–25, 28, 29, 31)

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

`chunk_id` is `<source>#<ordinal>`. `section` is the heading path for
Markdown (`File title › Heading › Subheading`) or the item for Rust
(`impl Registry`, `fn checksum`). `tokens` is an estimate: about 4 ASCII
characters or 2 other characters per token.

# 17. First MCP Tool

## Задание

🔥 День 17. Первый инструмент MCP

Реализуйте свой MCP-сервер вокруг любого API (например: Яндекс.Трекер, Git, CRM, mock API)

Сделайте:

👉 регистрацию инструмента
👉 описание входных параметров
👉 возврат результата

Подключите инструмент к своему агенту и:

👉 вызовите его из приложения
👉 получите и используйте результат

Результат:

Агент делает вызов к MCP-инструменту и получает результат

Формат:

Видео + Код

## Demo

https://github.com/user-attachments/assets/36e784bc-e617-4248-bdaa-b34ed335ba7f

## What this is

Lesson 16 was the client half of MCP: connect and list tools. This lesson adds
the other half. It has its **own MCP server** with one tool,
`list_recent_commits`, which wraps the public GitHub REST API. It also has a
minimal **agent** that finds the tool over MCP, lets the LLM decide when to
call it, runs `tools/call`, and answers from what came back.

Both halves live in one binary on one port:

```
browser ──POST /api/ask──▶ Agent ──initialize / tools/list / tools/call──▶ /mcp (CommitsServer)
                              │         (Streamable HTTP, real network)           │
                              ▼                                                  ▼
                        LLM (DeepSeek)                                  api.github.com
```

The agent doesn't call the tool as a Rust function. It connects to
`http://127.0.0.1:$PORT/mcp` over Streamable HTTP, the same way any other MCP
client would. Keeping it to one binary is what lets the existing deploy
pipeline ship it unchanged.

### The server: registration, input schema, result

This uses `rmcp`'s server macros:

```rust
/// Input of the `list_recent_commits` tool. ... each doc comment becomes that
/// property's `description`, and non-`Option` fields become `required`.
#[derive(Deserialize, JsonSchema)]
struct ListCommitsArgs {
    /// Repository owner: a GitHub user or organization, e.g. "rust-lang".
    owner: String,
    /// Repository name, e.g. "rust".
    repo: String,
    /// How many commits to return, newest first. 1-30, default 10.
    limit: Option<u32>,
    path: Option<String>,    // only commits touching this file/dir
    branch: Option<String>,  // branch, tag or SHA
    since: Option<String>,   // "YYYY-MM-DD" or ISO 8601
}

#[tool_router(router = tool_router)]
impl CommitsServer {
    #[tool(name = "list_recent_commits", description = "List the most recent commits of a public GitHub repository ...")]
    async fn list_recent_commits(&self, Parameters(args): Parameters<ListCommitsArgs>)
        -> Result<CallToolResult, ErrorData> { ... }
}

#[tool_handler(router = self.tool_router)]   // wires tools/list + tools/call
impl ServerHandler for CommitsServer { fn get_info(&self) -> ServerConfig { ... } }
```

- **Registration.** `#[tool]` on a method is the whole registration.
  `#[tool_router]` collects the tools, and `#[tool_handler]` routes the
  protocol's `tools/list` and `tools/call` to them.
- **Input parameters.** The tool's `inputSchema` is generated from
  `ListCommitsArgs`. The doc comments become per-property `description`s, and
  `owner`/`repo` are `required`. This is exactly what the LLM reads when it
  fills in the arguments. The input is also validated and normalized before
  anything reaches GitHub: owner and repo must be plain names, so there's no
  path injection; `limit` is clamped to 1-30; a bare date becomes a timestamp;
  and a `""` sent for an optional field is dropped.
- **Result.** A `CallToolResult` carries two views of the same data:
  - `content`: one compact text line per commit,
    `sha | date | author | first line of message`, which is what the LLM reads;
  - `structuredContent`: the same commits as JSON, for programmatic clients
    and the UI.

  Failures such as an unknown repo (404), an empty repo (409), a rate limit,
  or bad arguments are returned as **tool-level errors** (`isError: true`)
  with a readable message, not as JSON-RPC protocol errors. The model can then
  explain what went wrong instead of getting an opaque failure.

The endpoint is mounted with `axum::Router::nest_service("/mcp", ...)`. rmcp's
Streamable HTTP server accepts only `Host: localhost` by default, as DNS
rebinding protection. This server is meant to be reached on the VDS by IP and
only exposes public, read-only data, so that check is turned off
(`disable_allowed_hosts()`). That also means **lesson 16's inspector can
connect to it**. Point it at `http://<VDS host>:4017/mcp` to see the tool and
its schema from the outside.

### The agent: call the tool, get the result, use it

For each question, `Agent::run`:

1. connects to the MCP server and runs `tools/list`;
2. turns every MCP tool into an OpenAI-style function. The MCP `inputSchema`
   already is JSON Schema, so it becomes `parameters` as-is, minus `$schema`;
3. calls the LLM with those `tools`. If the reply contains `tool_calls`, it runs
   each one as an MCP `tools/call` and appends the result as a `role: "tool"`
   message. It repeats this for at most 5 rounds;
4. returns the model's final answer, built from the tool's result.

The assistant message with `tool_calls` goes back into the conversation as the
raw JSON the provider sent. That way provider-specific fields such as
DeepSeek's `reasoning_content` are passed back as-is. The system prompt
includes today's date, so "this week" can become a correct `since`.

The UI shows the whole chain: the MCP server and its tools with their schemas,
then each `tools/call` with the arguments the LLM chose and the exact result
text it got back, then the final answer.

### Verification

`cargo test` checks both halves with no outside network access. Everything
runs in-process or on `127.0.0.1`:

- `tool_is_registered_with_described_parameters`: `tools/list` returns exactly
  `list_recent_commits`. Its schema has the six properties, all with
  descriptions, and `required = [owner, repo]`.
- `call_tool_returns_commits_from_github`: `tools/call` against a fake GitHub
  API returns the formatted text plus `structuredContent`. It also checks the
  query that was sent upstream: `per_page`, normalized `since`, and an empty
  `path` that was dropped.
- `missing_repository_is_a_tool_error` and
  `invalid_owner_is_rejected_before_calling_github` cover the error paths.
- `agent_calls_the_mcp_tool_and_uses_the_result` is the end-to-end test. It
  runs the real app with `/mcp` on a random port, a fake GitHub, and a
  scripted fake LLM. The LLM first asks for `list_recent_commits`, then builds
  its answer from the tool result it receives. The test asserts that the agent
  discovered the tool over HTTP and passed it to the LLM correctly, made the
  `tools/call`, and fed the result back under the right `tool_call_id`. It
  also asserts that the final answer contains data that exists only in the
  tool's result.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

Or with curl:

```bash
curl -s -X POST localhost:3000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What changed in CaptainDmitro/AI-Advent-Challenge this week?"}' | jq

# Or talk to the MCP server directly with any MCP client, e.g. the inspector:
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://localhost:3000/mcp
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `MCP_SERVER_URL` | no | `http://127.0.0.1:$PORT/mcp` (this process's own server) |
| `GITHUB_TOKEN` | no | unset: unauthenticated GitHub API, 60 requests/hour per IP |
| `PORT` | no | `3000` |

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4017`, and
its MCP endpoint is at `http://<VDS host>:4017/mcp`. The pipeline doesn't pass
`GITHUB_TOKEN`, so the deployed instance uses the unauthenticated limit. That's
plenty for a demo.

## Conclusion

_(to fill in after recording the demo)_

# 16. MCP Connection

## Задание

🔥 День 16. Подключение MCP

Установите MCP SDK / клиент (или поднимите MCP-сервер, если используете локальный вариант)

Сделайте минимальный код, который:

👉 устанавливает MCP-соединение
👉 получает от MCP список доступных инструментов

Проверьте:

👉 соединение устанавливается
👉 список инструментов корректно возвращается

Результат:

Код, который подключается к MCP и выводит список доступных инструментов

Формат:

Видео + Код

## Demo

https://github.com/user-attachments/assets/ceb6447c-8398-4b5a-9d81-5a892951b3b9

## What this is

The first lesson that talks to something other than an LLM: a minimal
[Model Context Protocol](https://modelcontextprotocol.io) **client**. It
connects to a remote MCP server, performs the protocol handshake, and lists
the tools that server exposes. There's no LLM call in this lesson. It just
sets up the connection, and a later lesson can hand these tools to the agent.

### SDK

It uses [`rmcp`](https://crates.io/crates/rmcp), the official Rust MCP SDK,
with only the features this needs:

```toml
rmcp = { version = "3.4.1", default-features = false, features = [
    "client",                                     # the client role
    "transport-streamable-http-client-reqwest",   # Streamable HTTP transport
    "reqwest",                                    # ...on reqwest + rustls
] }
```

The `reqwest` feature matters for deployment: it builds reqwest with rustls
rather than native-tls, so the binary is still a fully static musl build with
no OpenSSL. Every other lesson relies on that too.

### Connect and list tools

All of the MCP logic lives in one function, `inspect`:

```rust
let client = ().serve(transport).await?;   // initialize -> notifications/initialized
let info   = client.peer_info();            // server name/version, protocol, capabilities
let tools  = client.list_all_tools().await?; // tools/list, following nextCursor pages
client.cancel().await;                       // close the session
```

`()` is the SDK's no-op `ClientHandler`. This client only sends requests, so
it has nothing to say back when a server asks for sampling, roots, and so on.
`inspect` is generic over the transport. In production it gets a
`StreamableHttpClientTransport` pointed at the server URL, with an optional
Bearer token. In tests it gets one end of an in-memory pipe.

### Why a remote server (DeepWiki) and not a local one

The deploy target is a bare VDS with no Node or Python. Most "local" MCP
servers are `npx`/`uvx` packages started as a child process over stdio, so
they can't run there. Instead the default is
[DeepWiki](https://mcp.deepwiki.com), a public MCP server over Streamable HTTP
that needs no auth. It exposes three tools: `read_wiki_structure`,
`read_wiki_contents`, and `ask_wiki_question`. The URL is editable on the page, so
you can point the page at any other Streamable HTTP MCP server. If that server
requires auth, fill in the token field.

### Verification

Both checks from the assignment are covered by `cargo test`, which runs in CI
with no network access:

- `connects_and_lists_tools` starts a tiny MCP server with two tools
  (`echo`, `add`) in the same process, using the SDK's server half over a
  `tokio::io::duplex` pipe. It runs the same `inspect` the web app uses and
  asserts three things: the handshake succeeds, the server's name, version,
  instructions, and `tools` capability come back, and `tools/list` returns
  exactly those two tools with their descriptions and input schemas intact.
- `handshake_failure_is_reported_as_error` covers a peer that disconnects
  before answering `initialize`. It checks that this becomes a readable error
  rather than a hang or a panic.
- `rejects_non_http_url` covers basic URL validation.

The web UI is the live version of the same check. Press **Connect & list
tools** to see the connection status, the server's handshake info, and every
tool with its parameters and full JSON input schema.

### API

- `GET /api/config` returns `{ "default_url": "..." }`.
- `POST /api/connect` takes `{ "url": "...", "token": "..." }`, where both
  fields are optional and `url` falls back to the default. It returns
  `{ url, result: { server, tools, handshake_ms, list_tools_ms } }` or
  `{ url, error }`. A failed connection is reported in the JSON body with
  HTTP 200, the same as the other lessons.

## Run

```bash
cargo run
```

Then open http://localhost:3000.

Or with curl:

```bash
curl -s -X POST localhost:3000/api/connect \
  -H 'Content-Type: application/json' -d '{}' | jq '.result.tools[].name'
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `MCP_SERVER_URL` | no | `https://mcp.deepwiki.com/mcp` |
| `PORT` | no | `3000` |

No LLM credentials are needed for this lesson.

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4016`.

## Conclusion

_(to fill in after recording the demo)_

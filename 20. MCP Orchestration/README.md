# 20. MCP Orchestration

## Задание

🔥 День 20. Orchestration MCP

Зарегистрируйте несколько MCP-серверов.

Сделайте так, чтобы:

👉 агент выбирал нужный инструмент
👉 корректно маршрутизировал запросы
👉 выполнял длинный флоу взаимодействия

Проверьте:

👉 сценарий, в котором используются инструменты с разных серверов
👉 корректность выбора и порядка вызовов

Результат:

Длинный флоу взаимодействия с несколькими MCP-серверами и инструментами

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

Lessons 17–19 had one MCP server. This lesson has **three separate MCP
servers**, each with its own endpoint, its own `serverInfo` and instructions,
and its own tools. There is also an **orchestrator** agent. It registers all
three servers, picks the right tool for each step, routes every call to the
server that owns that tool, and runs a long flow that crosses all of them.

| Server | Endpoint | Tools | Talks to |
|---|---|---|---|
| `crates` | `/mcp/crates` | `search_crates(query, limit?)`, `crate_info(name)` | crates.io API |
| `github` | `/mcp/github` | `search_repositories(query, limit?)`, `repo_info(repo)` | GitHub REST API |
| `notes` | `/mcp/notes` | `write_note(filename, content)`, `read_note(filename)`, `list_notes()` | a directory on disk |

```
                      ┌────────────── registry (MCP_SERVERS) ──────────────┐
 browser ─/api/ask─▶  │ crates=…/mcp/crates  github=…/mcp/github  notes=…  │
                      └───────┬───────────────────┬──────────────────┬─────┘
                              │ initialize + tools/list, per server  │
                              ▼                   ▼                  ▼
  Orchestrator ──────▶ crates__search_crates  github__repo_info  notes__write_note
   (LLM + router)      crates__crate_info     github__search_…   notes__read_note
        │                     │                   │              notes__list_notes
        │  tools/call ────────┴──── routed by the `<server>__` prefix ─┘
        ▼
   flow checks: several servers? routed to the owner? right order? note intact?
```

### Registering several servers

The registry is a list of `name=url` pairs. By default it lists this binary's
own three endpoints. Any other Streamable HTTP MCP server can be added or
swapped in through `MCP_SERVERS`:

```bash
MCP_SERVERS="crates=http://127.0.0.1:3000/mcp/crates,github=http://127.0.0.1:3000/mcp/github,notes=http://127.0.0.1:3000/mcp/notes,deepwiki=https://mcp.deepwiki.com/mcp"
```

For every run the orchestrator opens a separate MCP client session to each
server (`initialize`, then `tools/list`). The three built-in servers share one
process only so the lesson deploys as one binary on one port. The
orchestrator still reaches them over HTTP by URL, the same way it would reach
remote servers. A server that can't be reached is marked **unavailable** and
left out, and the run continues with the others.

### Choosing the tool and routing the call

- **Namespacing.** Every tool is shown to the LLM as `<server>__<tool>`, e.g.
  `crates__crate_info`. Its description starts with `[crates server] …`. Two
  servers can therefore have tools with the same name without clashing.
- **A system prompt built from the servers.** The system prompt is put
  together from what each server said about itself in the handshake: its
  `instructions` and its tools. It also names any server that is down. A
  newly registered server gets described to the model without any code
  change.
- **The router.** `Registry::route` splits the name on `__`, finds the
  connected server with that name, and checks that the server's `tools/list`
  really contains the tool. Only then does it send `tools/call` over that
  server's session. An unknown server (`weather__forecast`), a tool on the
  wrong server (`github__crate_info`), or a name without a prefix is never
  sent anywhere. The refusal goes back to the model as the tool result, so the
  model can correct itself.
- **A long flow.** Up to 16 LLM rounds are allowed. One round can ask for
  several independent calls, such as two `crate_info` calls at once. Every
  result goes back to the model before it picks the next step.

A typical run for *"compare the top 3 crates for building MCP servers:
crates.io stats plus their GitHub repos' activity, save a table to
rust-mcp.md and read it back"*:

```
round 1  crates · search_crates("mcp server")
round 2  crates · crate_info(rmcp)  crates · crate_info(…)  crates · crate_info(…)
round 3  github · repo_info(modelcontextprotocol/rust-sdk)  github · repo_info(…)  …
round 4  notes  · write_note(rust-mcp.md, <table>)
round 5  notes  · read_note(rust-mcp.md)
round 6  answer
```

### Checking the choice and the order of calls

After every run, `flow_checks` checks the recorded calls. It uses only what
the servers returned and what they list, not what the model says it did:

1. **Tools from several MCP servers**: successful calls reached at least two
   servers. The detail shows which tools each server answered.
2. **Each call routed to the server that owns the tool**: the server that ran
   each call lists that tool in its own `tools/list`. Calls that couldn't be
   routed are named.
3. **Each step uses what an earlier step found**: this is the order check.
   `crate_info(name)`, `repo_info(repo)` and `read_note(filename)` may only
   name something that appears in the user's request or in the result of an
   **earlier** call. The detail shows where each one came from, for example
   `step 5 repo_info(octo/beta) ← step 3 crate_info`. A repository the model
   invented, or a note read before it was written, fails this check.
4. **`<file>`: written after the data, read back intact**: the note was
   written only after data had been collected from another server. If it was
   read back later, it has the same checksum.
5. **Every tool call succeeded.**

The UI shows the registry (each server with its status and tools) and a route
strip of the whole flow, colored by server. It also shows a timeline grouped
by LLM round with each call's server, arguments and result, the checks, the
answer, and the saved note read from disk.

### Verification

`cargo test` needs no outside network. It uses a fake crates.io, a fake
GitHub, a scripted LLM, and the real app with its three `/mcp/*` endpoints on
`127.0.0.1`:

- `each_server_is_its_own_mcp_endpoint_with_its_own_tools`: three
  connections and three different `serverInfo`s, each with instructions. Each
  server has exactly its own tools, and every parameter is described. The LLM
  sees exactly 7 namespaced functions with valid schemas. `notes` validates
  its input, writes, reads back and lists.
- `registry_routes_by_prefix_and_refuses_the_rest`: `crates__crate_info` goes
  to `crates` and `github__repo_info` goes to `github`. `github__crate_info`,
  `weather__forecast` and a name without a prefix are refused. A real routed
  call returns the crate's `github_repo`, license and yanked count.
- `agent_runs_a_long_flow_across_three_servers`: 7 calls over 6 rounds
  (round 2 has two parallel `crate_info`s), in the exact order
  search_crates → crate_info ×2 → repo_info ×2 → write_note → read_note. The
  calls are routed to crates ×3 → github ×2 → notes ×2 and all 5 checks pass.
  The order check traces each argument back to the step it came from. The
  note is on disk, and the model received every result.
- `unknown_tools_and_invented_arguments_are_flagged`: an unknown server, an
  invented repository and a note that was never written are reported to the
  model and fail the routing, order and success checks. The run doesn't
  crash.
- `an_unreachable_server_is_reported_and_the_rest_still_work`: with a fourth,
  dead server registered, it is shown as unavailable and none of its tools
  reach the model. The system prompt says it's down, and the flow still runs
  on the other three.
- `checks_catch_wrong_order_and_a_note_that_changed`,
  `server_list_is_parsed_and_validated`, `repositories_are_normalized`,
  `mentions_matches_whole_names_only`, `helpers_behave`.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000 and pick one of the example requests, or
write your own.

Or with curl:

```bash
curl -s localhost:3000/api/servers | jq '.[] | {name, connected, tools: [.tools[].qualified]}'

curl -s -X POST localhost:3000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Compare the top 3 Rust HTTP client crates: crates.io stats and GitHub activity. Save a table to http-clients.md and read it back."}' \
  | jq '.calls[] | {round, server, tool, arguments}, .checks'

cat notes/http-clients.md

# Each server on its own, with any MCP client:
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://localhost:3000/mcp/crates (or /github, /notes)
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `MCP_SERVERS` | no | `crates=…/mcp/crates,github=…/mcp/github,notes=…/mcp/notes` on `http://127.0.0.1:$PORT`. Comma-separated `name=url`; names are lowercase letters, digits and `-`. Replaces the default list |
| `NOTES_DIR` | no | `notes`, in the working directory: where the `notes` server keeps files |
| `GITHUB_TOKEN` | no | unset: unauthenticated GitHub API, 60 requests/hour per IP |
| `PORT` | no | `3000` |

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4020`.
Its MCP servers are at `http://<VDS host>:4020/mcp/crates`, `/mcp/github` and
`/mcp/notes`.

The systemd unit's working directory is `~/apps/lesson-20`, so notes go to
`~/apps/lesson-20/notes/`. At most 100 notes of up to 64 KB each are kept, and
file names can't name a directory. Without `GITHUB_TOKEN`, GitHub allows 60
requests an hour, which is about ten long runs.

## Conclusion

_(to fill in after recording the demo)_

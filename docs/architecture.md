# Architecture

How the repository is put together: what a lesson is, how the lessons relate
to each other, and how code gets from a commit to a running service. The
short version of the conventions is in [`conventions.md`](./conventions.md),
the list of lessons is in [`lessons.md`](./lessons.md), and the HTTP and MCP
interfaces are in [`api.md`](./api.md).

## Repository layout

```
AI-Advent-Challenge/
├── README.md               course overview and the table of contents
├── AGENTS.md               instructions for AI coding agents (read first)
├── DEPLOYMENT.md           the CI/CD pipeline and the VDS layout
├── docs/                   project documentation (this folder)
│   ├── architecture.md     how the repo and the lessons are built
│   ├── conventions.md      code, README and naming conventions
│   ├── lessons.md          every lesson: kind, port, topic, what it reuses
│   └── api.md              HTTP and MCP interfaces, data schemas
├── .github/workflows/
│   ├── ci-cd.yml           build, test and deploy
│   └── ai-review.yml       AI review of every pull request (lesson 32)
├── 01. Rust LLM Chat CLI/  one folder per lesson ...
├── ...
├── 31. Developer Assistant/
├── ...
└── 34. Project File Assistant/
```

There is no Cargo workspace. Every lesson folder is an **independent Cargo
crate** with its own `Cargo.toml` and `src/main.rs`.
A lesson never depends on another lesson's crate. When a lesson builds on an
earlier one, the code is copied into the new folder and changed there, so
every finished lesson stays exactly as it was recorded.

## Lesson folders

A lesson folder is named `NN. Title`: a two-digit, zero-padded number, a dot,
a space and a title in English, e.g. `20. MCP Orchestration`. The number is
the course day. Day 27 has no folder.

Two shapes of lesson exist:

| Shape | Lessons | Stack | Deployed |
|---|---|---|---|
| CLI | 01, 02 | blocking `reqwest`, a stdin/stdout loop | no |
| Web app | 03 and later | `axum` + `tokio`, async `reqwest`, one HTML page | yes, port `4000 + NN` |

A web-app lesson has:

- `src/main.rs` (and sometimes more modules such as `index.rs`, `rerank.rs`,
  `task.rs`, `git.rs`, `mcp.rs`): one state struct held in axum `State`
  (`AppState` or `Agent`) that owns a `reqwest::Client` and has a
  `call_llm`/`chat`-shaped async helper;
- `static/index.html`: the whole UI with inline CSS and JS, built into the
  binary with `include_str!`, so a deploy ships only the binary;
- `README.md` following the lesson template (see `conventions.md`);
- tests inside the crate (`#[cfg(test)] mod tests`) that run without network
  or API keys, using fake HTTP servers on `127.0.0.1:0`.

## Course arcs

The lessons are grouped in arcs. Each arc builds on the code of the one
before it:

1. **LLM basics (01–05)**: a chat loop, response control, reasoning
   approaches, model comparison, temperature.
2. **Agents and context (06–12)**: the `Agent` struct, persistence, token
   counting, compression, context strategies, a layered memory model and
   personalization.
3. **Task state and guardrails (13–15)**: a task finite state machine,
   invariants, gated transitions.
4. **MCP (16–20)**: an MCP client on the official Rust SDK `rmcp`, own MCP
   servers with tools, a scheduler, tool composition, and orchestration of
   several MCP servers.
5. **RAG (21–25)**: document indexing with two chunking strategies, a first
   RAG query, reranking and filtering, citations with verified quotes, and a
   RAG chat with task memory.
6. **Local models (26, 28–30)**: Ollama on the Mac, local RAG, optimizing a
   3B model, and a private LLM service on the VDS behind a gateway.
7. **Developer assistant (31)**: RAG over this repository's docs plus an MCP
   server over its git checkout, behind a `/help` command.
8. **Code review (32)**: lesson 31's RAG and MCP turned on pull requests: a
   GitHub Action reviews every PR's diff against the repo's docs and code and
   posts the review as a PR comment.
9. **Assistants with write access (33–34)**: an MCP server that the
   assistant can change, not only read: a support CRM with ticket notes and
   statuses (33), and a project's files with search, edits and a diff of
   every change (34).

## LLM access

Most web lessons talk to an OpenAI-compatible chat completions API
(`POST {OPENAI_BASE_URL}/chat/completions`) with `OPENAI_API_KEY`. The
default base URL is `https://api.deepseek.com` and the default model
`deepseek-v4-flash`. Lesson 04 uses its own `WEAK_*`/`MEDIUM_*`/`STRONG_*`
variables, and lessons 26–30 talk to Ollama (`OLLAMA_HOST`).

## Retrieval (RAG)

Lessons 21–25, 28, 29, 31 and 32 share one index design:

- the corpus is Markdown (split by headings) and Rust (split by top-level
  items);
- **structural chunking**: one chunk per section, short sections joined up to
  300 characters, long ones split at paragraphs below 1800 characters;
- the **local embedder** is a hashed TF-IDF with 1024 dimensions, computed
  in-process with no network: words are lowercased and cut to their first 5
  characters, hashed with FNV-1a into signed buckets, log-scaled and weighted
  by IDF learned from the index's own chunks;
- a **heuristic reranker** rescores the top 20 by
  `0.3 · cosine + 0.5 · keyword coverage + 0.2 · heading match`, and a
  relevance threshold of `0.30` decides what reaches the prompt.

## MCP

MCP servers in this repo use `rmcp` with the Streamable HTTP transport. A
lesson's MCP servers are mounted on the same axum router as its page, under
`/mcp/...`, so they deploy as one binary on one port. Clients (the agents in
lessons 17–20 and 31–34) connect to them over HTTP by URL, the same way they
would connect to a remote server, perform `initialize`, read `tools/list`,
and pass the tools to the LLM as OpenAI function definitions.

## Build and deploy

Nothing is built on a developer machine. GitHub Actions does everything:

```
push to master ─▶ discover ─▶ build-test (every lesson) ─▶ deploy (newest web lesson)
                                cargo fmt/clippy (non-blocking)   static musl binary
                                cargo build --release             scp to the VDS
                                cargo test                        systemd --user unit
```

The deployed binary is a static `x86_64-unknown-linux-musl` build (`reqwest`
uses `rustls`, so there is no OpenSSL). On the VDS each lesson lives in
`~/apps/lesson-NN/` with its binary `app` and a `.env`, and runs as the user
service `ai-advent-lesson-NN` on port `4000 + NN`. Only the highest-numbered
lesson deploys on a normal push; older ones are redeployed with a manual
`workflow_dispatch` run. Full details are in `DEPLOYMENT.md`.

A second workflow, `ai-review.yml`, runs on pull requests rather than
pushes: it builds lesson 32 from the PR's base branch and posts an AI review
of the PR as a comment. It is advisory and deploys nothing.

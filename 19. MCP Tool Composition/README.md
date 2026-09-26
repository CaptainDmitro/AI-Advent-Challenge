# 19. MCP Tool Composition

## Задание

🔥 День 19. Композиция MCP-инструментов

Создайте несколько MCP-инструментов, например:

👉 search
👉 summarize
👉 saveToFile

Реализуйте пайплайн:

👉 первый инструмент получает данные
👉 второй — обрабатывает
👉 третий — сохраняет результат

Проверьте:

👉 автоматическое выполнение цепочки
👉 корректность передачи данных между инструментами

Результат:

Автоматический пайплайн из нескольких MCP-инструментов

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

Lessons 17 and 18 each had tools that did their job on their own. This lesson
has three tools on one MCP server that are built to be **chained**. Each one
takes the previous one's output and produces the next one's input:

| Step | Tool | What it does | Output |
|---|---|---|---|
| gets the data | `search(query, limit?)` | Searches public GitHub repositories, most-starred first | artifact `search-N`: JSON search results |
| processes it | `summarize(source_id, focus?, language?)` | Reads `search-N`, computes stats, and has the LLM write an overview, then builds a Markdown report | artifact `summary-N`: the report |
| saves the result | `save_to_file(source_id, filename?)` | Writes `summary-N` to `output/<filename>`, reads the file back, and confirms the bytes match | the file, plus its lineage |

```
                 ┌───────────────────── one binary, one port ─────────────────────┐
 browser ─/api/pipeline─▶ fixed pipeline ─┐                                        │
 browser ─/api/ask──────▶ Agent + LLM ────┤ tools/call over Streamable HTTP        │
                 │                        ▼                                        │
                 │   /mcp  PipelineServer                                          │
                 │     search ───────▶ api.github.com                              │
                 │        │ search-1 (JSON, checksum)                              │
                 │        ▼                                                        │
                 │     summarize ────▶ LLM (overview) + stats + sources            │
                 │        │ summary-2 (Markdown, checksum, derived_from search-1)  │
                 │        ▼                                                        │
                 │     save_to_file ─▶ output/report.md ─▶ read back, compare      │
                 │                                                                 │
                 │   chain_checks: re-verify every handoff after the run           │
                 └─────────────────────────────────────────────────────────────────┘
```

### Passing data between tools: artifacts, by id

Every tool stores what it produces on the server as an **artifact**. An
artifact has an id, its content, an FNV-1a checksum of that content, and
`derived_from`, the artifact it was computed from. The tool returns the
`artifact_id`, and the next tool takes it as `source_id`.

So the data itself never goes through the model. The LLM never has to copy a
large JSON blob from one tool call into the next, where it could shorten,
rewrite or invent parts of it. Only the id moves between calls. The next tool
reads the exact bytes the previous one stored, and the checksums let you
prove that.

Each tool also returns `structuredContent`: the artifact id, its checksum and
size, and the data itself (the repositories or the report). That is what the
pipeline and the checks below use.

### Running the chain automatically: two ways

**A. Fixed pipeline** (`POST /api/pipeline`). No LLM decides anything. The
code connects to `/mcp` like any MCP client would, calls `tools/list`, and
then goes through a three-stage table:

```rust
const PIPELINE: [Stage; 3] = [
    Stage { tool: "search",       arguments: search_arguments },    // from the request
    Stage { tool: "summarize",    arguments: summarize_arguments }, // source_id = previous.artifact_id
    Stage { tool: "save_to_file", arguments: save_arguments },      // source_id = previous.artifact_id
];
```

Each stage's arguments are built from the previous stage's
`structuredContent`. The first failure stops the chain, so nothing gets
saved from half a run.

**B. Agent** (`POST /api/ask`). The same three tools are given to the LLM as
functions, as in lesson 17. The system prompt describes the order and says to
pass ids through unchanged. The model then chains the tools by itself from a
request like "find the most popular Rust MCP SDKs, summarize them and save it
to rust-mcp.md". Each tool's result text also names the next step
(`Next step: summarize with source_id "search-1"`), which makes the chain easy
for the model to follow.

### Checking that the data arrived intact

After every run, in both modes, `chain_checks` walks back from the saved file
to the summary to the search. It checks each handoff using only the tool
calls' outputs and the disk, not what the tools say about themselves:

1. **search → summarize**: `summarize` read the artifact that `search`
   produced, with the same checksum and item count.
2. **Summary is grounded in the search results**: every repository from the
   search appears in the report. The report also links to no repository that
   the search didn't return, which catches an overview that invents projects.
3. **summarize → save_to_file**: the saved artifact has the summary's
   checksum and byte count.
4. **File on disk**: the file is read again, separately from the tool's own
   read-back, and its checksum is compared with the summary's.

The UI shows each step's arguments and result and what it handed on
(`↓ artifact search-1 · fnv1a64:… → source_id of summarize`). It also shows
the checks, and the saved file as read from disk.

### Verification

`cargo test` needs no outside network. It uses a fake GitHub, a fake LLM, and
the real app with its `/mcp` endpoint, on `127.0.0.1`:

- `tools_are_registered_with_described_parameters`: exactly three tools,
  every parameter described, and the required arguments are `query` /
  `source_id`.
- `pipeline_chains_the_three_tools_and_the_data_arrives_intact`: the calls run
  in the order search → summarize → save_to_file. Each step's `source_id` is
  the previous step's `artifact_id`. The LLM inside `summarize` received
  exactly the repositories that `search` returned. The file on disk equals the
  summary, and all four checks pass.
- `pipeline_stops_at_the_first_failed_step`: when the LLM fails,
  `save_to_file` never runs, no file is written, and the run reports which
  step failed.
- `agent_chains_the_tools_by_itself`: a scripted model reads each id from the
  previous tool result and chains all three tools. The checks pass the same
  way.
- `tool_arguments_are_validated`: unknown artifacts, the wrong artifact kind,
  and path-traversal filenames (`../escape.md`, `a/b.md`, `.hidden.md`) are
  all rejected as tool errors.
- `checks_catch_a_file_that_differs_from_the_summary`,
  `grounding_flags_invented_links_and_dropped_results`: the checks really do
  fail when the data is tampered with.
- `filenames_stay_inside_the_output_directory`, `checksum_is_stable_and_sensitive`,
  `lineage_follows_derived_from`.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000, where you can run **A · Fixed pipeline** or
**B · Agent chains the tools**.

Or with curl:

```bash
curl -s -X POST localhost:3000/api/pipeline \
  -H 'Content-Type: application/json' \
  -d '{"query": "mcp server language:rust", "limit": 5, "filename": "rust-mcp.md"}' \
  | jq '.calls[] | {step, name, arguments}, .checks'

curl -s -X POST localhost:3000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Find top Go LLM agent frameworks, summarize them, save to go-agents.md"}' | jq

cat output/rust-mcp.md

# Or chain the tools by hand with any MCP client:
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://localhost:3000/mcp
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `OUTPUT_DIR` | no | `output`, in the working directory: where `save_to_file` writes |
| `MCP_SERVER_URL` | no | `http://127.0.0.1:$PORT/mcp` (this process's own server) |
| `GITHUB_TOKEN` | no | unset: unauthenticated search, 10 requests/minute per IP |
| `PORT` | no | `3000` |

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4019`,
and its MCP endpoint is at `http://<VDS host>:4019/mcp`.

The systemd unit's working directory is `~/apps/lesson-19`, so saved files go
to `~/apps/lesson-19/output/`. At most 100 files are kept there, and
filenames can't name a directory. Artifacts live only in memory (the last
50), so a restart clears them, but saved files stay.

## Conclusion

_(to fill in after recording the demo)_

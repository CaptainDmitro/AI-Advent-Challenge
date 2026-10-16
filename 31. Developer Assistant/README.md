# 31. Developer Assistant

## Задание

🔥 День 31. Ассистент разработчика

Используйте RAG для подключения к документации вашего проекта:

👉 README
👉 project/docs
👉 схемы данных или API-описания

Через MCP подключите ассистента к проекту:

👉 минимум — получение текущей git-ветки
👉 по желанию — список файлов или diff

Реализуйте команду:

👉 /help — ассистент отвечает на вопросы о проекте
👉 использует документацию и контекст проекта

Минимальный результат:

👉 README + папка docs в RAG
👉 MCP — хотя бы git branch
👉 /help отвечает на вопросы о структуре проекта

Результат:

Ассистент разработчика, понимающий проект

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

A `/help` command that answers questions about **this repository**. It reads
the docs through RAG, and it reads the git checkout itself through an MCP
server.

```
/help <question>
   │
   ├─▶ MCP  project server ─ git_branch ───────────────────────────▶ "Current branch: feature/x (HEAD a1b2c3d)"
   │        (/mcp/project)    always first, before the model runs
   │
   ├─▶ RAG  README.md · AGENTS.md · DEPLOYMENT.md · docs/** · NN. */README.md
   │        structural chunks ─▶ top-20 ─▶ heuristic rerank ─▶ ≥ 0.30 ─▶ up to 5 fragments [1]..[5]
   │
   └─▶ LLM  system: branch + fragments + rules        ┌──────────────────────────────────────┐
            tools:  the MCP server's 6 tools  ◀──────▶│ git_status · git_log · git_diff       │
            ≤ 8 rounds of tool calls                   │ list_files · read_file · git_branch  │
                                                       └──────────────────────────────────────┘
            ─▶ answer with [n] citations, sources, every MCP call
```

### RAG over the project's documentation

The corpus is the project's own documentation, read from the git checkout:

| What | Files |
|---|---|
| README | `README.md`, the course overview and table of contents |
| Project docs | `docs/architecture.md`, `docs/conventions.md`, `docs/lessons.md`, `docs/api.md` (new in this lesson, see below) |
| Agent and deploy docs | `AGENTS.md`, `DEPLOYMENT.md` |
| Lesson docs | every `NN. Title/README.md` |
| API and data schemas | `docs/api.md`: HTTP and MCP endpoints of every lesson, this lesson's reply JSON, the chunk schema, the chat-completions shapes |

The root `docs/` folder is new. It holds the project documentation that used
to be spread over READMEs or not written down at all: how the repo is laid
out and why, the conventions, a table of every lesson (crate, kind, port,
what it builds on), and the API and data schemas.

The index is lesson 21's design: structural chunking and the local hashed
TF-IDF embedder, which needs no network or embeddings key. On top of it is
lesson 23's heuristic reranker with the 0.30 threshold. What's new is that
**the index follows the checkout**. Every question first computes the
checkout's signature (HEAD's commit plus a hash of `git status`). If the
signature changed, the docs are read again and re-indexed, which takes tens of
milliseconds. So after a `git checkout`, a `git pull` or an edit to a file in
`docs/`, the next answer already uses the new docs. When there is no checkout
yet, or none at all, the root docs and `docs/` that were compiled into the
binary are used instead.

The docs are in English and the index is lexical, so a question in Russian is
first rewritten into an English search query (one short LLM call). The search
then uses the original question and the rewrite together.

### MCP: the `project` server

`/mcp/project` is an MCP server (rmcp, Streamable HTTP, the same setup as
lesson 20) over the checkout. Every tool is read-only:

| Tool | What it does |
|---|---|
| `git_branch` | the current branch (or detached HEAD), HEAD's short hash, the upstream, every local branch |
| `git_status` | ahead/behind and every modified, added or untracked file |
| `git_log` | the last N commits, optionally only those that touched one path |
| `list_files` | tracked files (`git ls-files`), in a folder or matching a text, with a count per top-level folder |
| `read_file` | a file from the checkout, a range of lines at a time |
| `git_diff` | uncommitted changes, or with `base` what the branch changed since it left `base` (`base...HEAD`) |

The assistant connects to it as an MCP client over HTTP, the same way it
would connect to a remote server: `initialize`, then `tools/list`, then the
tools are passed to the LLM as functions. **`git_branch` is called on every
question before the model runs.** Its result goes into the system prompt and
into the reply's `branch` field, so the branch shown in the UI comes from MCP
and not from the model. After that the model calls tools when the docs don't
cover the question. For example "What changed on this branch?" leads to
`git_diff(base: "master")`, and "What's in docs?" leads to
`list_files(path: "docs")`. Any MCP client can use the server too:

```bash
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://<host>:4031/mcp/project
```

Safety: paths must be relative, have no `..`, and can't point into `.git`.
Symlinks are resolved, and the result must still be inside the checkout. A
revision can't start with `-` or contain `..`, so an argument can never
become a git option such as `--output=…`. `git` runs with no prompts, the C
locale and a 20-second timeout. `read_file` returns at most 400 lines or
40 KB, and `git_diff` at most 12 000 characters.

### The checkout

| Where it runs | `REPO_DIR` default | |
|---|---|---|
| `cargo run` in the lesson folder | `..` | the clone you're working in, so your branch and your edits |
| the VDS (`~/apps/lesson-31`) | `repo` | `git clone`d from GitHub on the first start, then **never pulled or reset by the app** |

On the VDS the clone runs in the background, so the page answers at once.
Until the clone is done, the assistant uses the docs built into the binary
and `git_branch` reports that the repository is still being cloned. The clone
goes into `repo.partial` and is renamed when it finishes, so an interrupted
clone never looks like a checkout. If `git` isn't installed, every tool says
so, and `/help` still answers from the built-in docs.

### `/help`

| Input | What happens |
|---|---|
| `/help` | an overview: the commands, the current branch (from MCP), the index, the tools, example questions. No LLM call |
| `/help <question>` | `git_branch` → search → the model, with tools → an answer with `[n]` citations |
| plain text | the same as `/help <text>` |
| `/anything-else` | `Unknown command …` |

The system prompt tells the model to: base every fact on the fragments or a
tool result; cite a fragment as `[n]` and name a tool when a fact comes from
one; say "the docs don't cover it" rather than invent files, commands or
ports; copy paths and names exactly; and answer in the question's language.

The page shows the branch badge (from MCP) in the header. Next to the chat it
shows the project panel (checkout, branches, MCP URL and tools) and the
index (built from the checkout or built-in, its files). For the last answer
it shows every MCP call with its arguments and result (the automatic
`git_branch` marked `auto`), the sources with cited ones highlighted, and the
retrieval candidates against the threshold.

### Verification

`cargo test` needs no network and no API key. It needs `git`, which GitHub's
runners have. Each test builds a small git repository in a temp folder:
`master` with `README.md`, `AGENTS.md`, `docs/architecture.md` and one lesson,
and a checked-out `feature/help-command` branch with one more commit adding
`docs/help.md`. The real app runs on a random port, against a scripted LLM.

- `project_mcp_server_reads_the_checkout`: over a real MCP connection, the
  server lists exactly the 6 described tools. `git_branch` says
  `feature/help-command` and lists both branches. `list_files` lists `docs/`
  and groups the root by folder. `read_file` reads a doc and refuses `../`,
  `.git/config`, an absolute path, a folder and a missing file. `git_log`
  lists both commits, or one for `docs/help.md`. `git_diff(base: master)`
  shows only `docs/help.md`, and `--output=…` or `master..HEAD` as a base is
  refused. After an edit, `git_status` shows ` M README.md` and `git_diff`
  shows the line. An unknown tool is refused before it reaches the server.
- `help_alone_shows_the_commands_and_the_branch_without_the_llm`: `/help`
  lists the commands, the branch from MCP (`auto` call), 5 documents from
  the checkout and 6 tools, with no LLM request. `/deploy` is an unknown
  command.
- `help_answers_from_the_docs_and_the_mcp_tools`: a question about where
  lessons live. The calls are `git_branch` (auto), then the model's
  `list_files(docs)`. `docs/architecture.md` is the top candidate and source
  `[1]`, and it is cited. The model's system prompt carries the branch and
  the fragment text, its request carries the 6 tools, and the tool result
  went back to it.
- `the_index_follows_the_checkout`: a question about a "zanzibar flag" finds
  nothing. An uncommitted `docs/zanzibar.md` is indexed by the next question
  (6 documents, a new signature) and becomes source `[1]`. After
  `git checkout master` the branch is `master`, `docs/help.md` is gone
  (4 documents), and nothing passes the threshold.
- `a_russian_question_is_searched_in_english_too`: the first LLM call is the
  rewrite. The search query is the question plus the rewrite, and it finds
  `docs/architecture.md`.
- `without_a_checkout_it_says_so_and_uses_the_built_in_docs`: `git_branch`
  fails and the system prompt says so. The index is the built-in one. A tool
  the server doesn't have comes back to the model as `Error: …`.
- `status_reports_the_repo_the_index_and_the_mcp_server`, and in the modules:
  path and revision validation, `load_repo` (docs, lesson READMEs, nothing
  else), every file in `docs/` is in the built-in corpus, retrieval of a
  known section, and the reranker.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000 and type `/help`, or click an example. Run
from the lesson folder inside a clone, the checkout is the clone itself:
switch branches or edit `docs/` and ask again.

From the terminal (the MCP server runs on a private port for one question):

```bash
OPENAI_API_KEY=sk-... cargo run -- help How do I add a new lesson?
```

```bash
OPENAI_API_KEY=sk-... cargo run -- help
```

With curl:

```bash
curl -s -X POST localhost:3000/api/help -H 'Content-Type: application/json' \
  -d '{"message": "/help What changed on this branch compared to master?"}' \
  | jq '{branch: .branch.branch, answer, sources: [.sources[] | {n, source, section, cited}], calls: [.calls[] | {tool, arguments}]}'
```

```bash
curl -s localhost:3000/api/status | jq '{branch: .branch.branch, index: .index | {origin, documents, chunks}, tools: .mcp.tools}'
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `REPO_DIR` | no | `..` when that is a checkout of this repo (`cargo run` in the lesson folder), otherwise `repo` in the working directory |
| `REPO_URL` | no | `https://github.com/CaptainDmitro/AI-Advent-Challenge.git`: cloned into `REPO_DIR` when it isn't a checkout |
| `REPO_CLONE` | no | on. `off` never clones: without a checkout, the built-in docs only |
| `MCP_URL` | no | `http://127.0.0.1:$PORT/mcp/project`: the MCP server the assistant connects to |
| `PORT` | no | `3000` |

The model has to support tool calling (DeepSeek and OpenAI do).

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4031`
and the MCP server at `http://<VDS host>:4031/mcp/project`, with the
`OPENAI_*` secrets the pipeline already supplies.

On first start the service clones the repo into `~/apps/lesson-31/repo/`.
This needs `git` on the VDS. A redeploy replaces only the binary and `.env`,
so the checkout stays, along with whatever branch it is on. To show the
assistant following the project:

```bash
ssh <user>@<vds-host> 'cd apps/lesson-31/repo && git pull -q && git checkout -b demo/help-31'
```

The next `/help` reports `demo/help-31`. Edit or add a file under `docs/` in
that checkout, and the next answer uses it. `git checkout master` switches
back.

## Conclusion

_(to fill in after recording the demo)_

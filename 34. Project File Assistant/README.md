# 34. Project File Assistant

## Задание

🔥 День 34. Ассистент для работы с файлами проекта

Сделайте ассистента, который активно работает с файлами, а не просто отвечает текстом.

Через MCP или доступный вам tooling ассистент должен уметь:

👉 читать файлы проекта
👉 искать информацию по нескольким файлам
👉 анализировать содержимое
👉 создавать или изменять файлы

Примеры сценариев (реализуйте минимум 2):

👉 найти все места, где используется определённый компонент / API
👉 обновить документацию на основе изменений в коде
👉 сгенерировать новый файл (README / ADR / changelog)
👉 проверить соответствие файлов правилам или инвариантам
👉 подготовить diff или список изменений

Важно:

👉 ассистент сам инициирует работу с файлами
👉 вы задаёте задачу на уровне цели, а не “открой файл X”

Минимальный результат:

👉 ассистент работает минимум с 2–3 файлами
👉 изменения сохраняются или выводятся как diff
👉 результат можно воспроизвести повторно

Результат:

AI-ассистент, который выполняет реальные операции с файлами проекта

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

An assistant that works on a project's files. You give it a goal («обнови
документацию по коду», «проверь правила», «напиши ADR»), not a file name. It
opens the project itself, searches and reads the files the goal depends on,
and edits them through an MCP server. Every change is written to disk and
returned as a unified diff, and the demo project can be reset to run again.

```
goal: «Обнови docs/api.md по коду»            ☐ dry run
   │
   ├─▶ MCP  files server (/mcp/files) ─ list_files ─────────────▶ "13 files in the project: app/routes/notes.py (71 lines) …"
   │        always first, before the model runs (auto)
   │
   ├─▶ LLM  system: the file list + rules                 ┌────────────────────────────────────────────────┐
   │        tools:  the files server's 6 tools  ◀────────▶│ read:  list_files · read_file · search_files   │
   │        ≤ 16 rounds of tool calls                     │        get_diff                                │
   │                                                      │ write: write_file · edit_file  (answer = diff) │
   │                                                      └────────────────────────────────────────────────┘
   │
   └─▶ diff  the project before the task vs. after it, read from disk
            ─▶ changes [{path, status, +added, −removed, diff}] + one patch
            ─▶ saved (or, in a dry run, shown and rolled back)
```

### The demo project

`sample/` is **Notely API**, a small FastAPI service (13 files: Python code,
docs, an ADR, a changelog and `PROJECT_RULES.md`). It is compiled into the
binary and copied to `workspace/` on the first start; the assistant edits
that copy, and **reset project** (`POST /api/reset`) puts `sample/` back. It
was written to have real work in it:

| Where | What's wrong |
|---|---|
| `app/routes/notes.py` | `delete_note` is the only notes endpoint without `Depends(require_auth)` (rule 1) |
| `docs/api.md` | lags behind the code: no `DELETE /notes/{note_id}`, no `POST /notes/{note_id}/share`, and `GET /notes` doesn't mention shared notes (rule 2) |
| `app/config.py` | `SECRET_KEY = "dev-secret-123"`, a secret in code (rule 3) |
| `app/routes/users.py` | `login` still calls the deprecated `legacy_hash` (rule 4) |
| `tests/` | there is `test_notes.py`, but no `test_users.py` (rule 5) |
| `app/storage.py` | `print(` instead of logging (rule 6) |
| `CHANGELOG.md` | no `## Unreleased` section for delete and share (rule 7) |

`PROJECT_DIR=/path/to/any/project` points the assistant at a real folder
instead. Then the baseline for "changes" is the folder as it was when the
service started, and reset is off: a real project is never rolled back
wholesale.

### Scenarios

The page has one button per scenario. Each one is a goal; which files to open
is the assistant's decision.

| Goal | What the assistant does | Files |
|---|---|---|
| «Найди все места, где используется `require_auth`, и скажи, какие эндпоинты остались без авторизации» | `search_files` over the project, reads `notes.py`, lists every use as `path:line` and finds `delete_note` without it | reads 2-6 |
| «Обнови docs/api.md по коду» | searches for the `@router` decorators, reads both route modules and the docs, adds the missing endpoints with `edit_file`, fixes `GET /notes` | reads 3, changes 1 |
| «Проверь проект на соответствие правилам из PROJECT_RULES.md» | reads the rules, checks each one with searches and reads, and reports passed / violated with `path:line` evidence. Changes nothing | reads 6+ |
| «Сгенерируй ADR 0002 … в формате существующего ADR» | reads `docs/adr/0001-use-fastapi.md` and `storage.py`, writes `docs/adr/0002-….md` in the same format | creates 1 |
| «Добавь в CHANGELOG.md раздел Unreleased …» | compares the code, the docs and the changelog, and adds the section | changes 1 |
| «Убери вызовы legacy_hash и подготовь diff» | finds the calls, edits `users.py` (and `auth.py` if it decides to), and the diff is the result | changes 1-2 |

### MCP: the `files` server

`/mcp/files` is an MCP server (rmcp, Streamable HTTP, the same setup as
lessons 31 and 33) over the project folder:

| Tool | Kind | What it does |
|---|---|---|
| `list_files` | read | the text files with their line counts, optionally under one folder |
| `read_file` | read | a file with line numbers, up to 400 lines per call (`start_line` / `end_line`) |
| `search_files` | read | a text or a regex across every file (or a folder, or an extension): `path:line: text` and matches per file |
| `get_diff` | read | every change since the baseline, as a unified diff with a list of files |
| `write_file` | write | create a file or replace its content; answers with the diff |
| `edit_file` | write | replace an exact piece of text (once, or `replace_all`); answers with the diff |

The assistant connects to it over HTTP like to any remote server
(`initialize`, `tools/list`, the tools go to the LLM as functions) and
**calls `list_files` itself before the model runs**, so the model starts from
the real file tree. The rest is the model's choice.

Guards, all checked on the server, so they hold for any MCP client:

- Paths are relative and `/`-separated. No `..`, no absolute paths, nothing
  hidden (`.env`, `.git`), nothing under `target`, `node_modules`, `venv` or
  `__pycache__`, and only text extensions (`.md`, `.py`, `.rs`, `.toml`,
  `.json`, …). Symlinks are not listed or followed out of the project.
- Files are at most 256 KB and must be UTF-8. Writes go through
  write-then-rename. There is no delete tool.
- `edit_file`'s `old_text` must occur exactly once (or `replace_all`), so an
  edit can't land in the wrong place. A miss says to re-read the file and copy
  the text without line numbers.
- A bad path, an ambiguous edit or an invalid regex is a tool result with
  `isError: true` that the model reads and corrects, not a protocol error.

```bash
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://<host>:4034/mcp/files
```

### Changes: saved and shown as a diff

The reply's `changes` don't come from what the model says it did. The
assistant takes a snapshot of every text file before the task and another
after it, and diffs the two (the `similar` crate, `git diff` format). So
`changes`, `diff` and `summary` are exactly what is on disk now, even when
the run failed halfway.

- **apply** (default): the changes stay. The file tree marks them `M` / `A`,
  **since baseline** shows everything changed since `sample/`, and
  `GET /api/diff.patch` downloads it as one patch (`git apply` / `patch -p1`).
- **dry run**: the model works the same way and is told it's a preview; after
  the task the snapshot is restored. The reply has the diff and
  `reverted: true`, and the project is left as it was.

Tasks run one at a time (a lock), so "what this task changed" is never mixed
with another task's writes.

### Reproducible

- `sample/` is compiled in, and reset brings it back byte for byte, so every
  demo run starts from the same 13 files.
- A dry run leaves no trace.
- The diff is computed from the files, so running the same goal twice from a
  reset gives a comparable result: a diff you can read, save and apply.

### Verification

`cargo test` needs no network and no API key. The real app runs on a random
port over a temp copy of the demo project, against a scripted LLM, and
everything goes over real HTTP and a real MCP connection.

- `files_mcp_server_reads_searches_and_writes`: the server lists exactly the 6
  tools, each with a description. `list_files` gives 13 files, and 2 under
  `app/routes`. `read_file` returns numbered lines and says when it is
  truncated. `search_files("require_auth")` finds 13 matches in 6 files, and
  10 in 3 under `app` (case-insensitive). The regex
  `@router\.(get|post|put|delete)` over `.py` finds the 9 endpoints. An invalid
  regex and an empty pattern are refused. `edit_file` refuses `Auth:
  required.` (5 occurrences) and a line-numbered `old_text`, then adds the
  `DELETE` section (+4 −0, on disk at once). `write_file` creates an ADR and
  refuses `../evil.md`, `.env`, `app/run.exe` and `/tmp/x.md`. `get_diff`
  shows both files, and one with `path`.
- `finding_usages_reads_and_changes_nothing`: `list_files` (auto), then the
  model's `search_files`. The prompt has the file tree and the 6 tools, the
  search result goes back to the model, and `changes` is empty.
- `docs_are_updated_from_the_code_and_saved`: search, read, a failed edit
  whose error goes back to the model, a corrected edit. The reply's diff is
  `docs/api.md` (+4 −0). The file on disk, `/api/file`, `/api/files` (`M`),
  `/api/diff` and `/api/diff.patch` all agree. Reset restores exactly
  `docs/api.md`, and the project equals `sample/` again.
- `a_dry_run_returns_the_diff_and_rolls_it_back`: the model reads ADR 0001 and
  writes ADR 0002. The reply has the `added` diff (`@@ -0,0 +1,3 @@`) and
  `reverted: true`, the file isn't on disk, and the prompt said DRY RUN.
- `status_files_and_validation`: status, the file tree, a file, a refused
  `../` path, an empty task, and an LLM that's down (an error string,
  `list_files` still called, nothing changed).
- In the modules: path normalization and every refused path, the demo
  project's creation, reopening and reset, a real folder's baseline and
  restore (with non-text files left alone), symlinks out of the project, the
  sample's planted problems, and unified diffs (modified, added, unchanged,
  deleted, the summary and the patch).

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000 and click a scenario. The first start
creates `workspace/` in the lesson folder from `sample/`.

With curl:

```bash
curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"message": "Обнови docs/api.md по коду", "dry_run": false}' \
  | jq '{answer, summary, calls: [.calls[] | {tool, auto, is_error}]}'
```

```bash
curl -s localhost:3000/api/diff.patch
```

```bash
curl -s -X POST localhost:3000/api/reset
```

On your own project (no reset there; review the diff before you keep it):

```bash
PROJECT_DIR=~/code/my-project OPENAI_API_KEY=sk-... cargo run
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `PROJECT_DIR` | no | unset: the demo project. Set: any existing folder, without reset |
| `WORKSPACE_DIR` | no | `workspace`: where the demo project is copied (relative to the working directory) |
| `MCP_URL` | no | `http://127.0.0.1:$PORT/mcp/files`: the MCP server the assistant connects to |
| `PORT` | no | `3000` |

The model has to support tool calling (DeepSeek and OpenAI do).

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4034`
and the MCP server at `http://<VDS host>:4034/mcp/files`, with the `OPENAI_*`
secrets the pipeline already supplies.

The demo project lives in `~/apps/lesson-34/workspace/`, created on the first
start. A redeploy replaces only the binary and `.env`, so the assistant's
changes stay until **reset project** is clicked (or the folder is deleted).

## Conclusion

_(to fill in after recording the demo)_

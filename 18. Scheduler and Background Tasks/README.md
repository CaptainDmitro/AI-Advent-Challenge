# 18. Scheduler and Background Tasks

## Задание

🔥 День 18. Планировщик и фоновые задачи

Сделайте MCP-инструмент с отложенным или периодическим выполнением.

Пример:

👉 reminder
👉 периодический сбор данных
👉 регулярный summary

Инструмент должен:

👉 сохранять данные (JSON / SQLite)
👉 выполняться по расписанию
👉 возвращать агрегированный результат

Результат:

Агент, который работает 24/7 и периодически выдаёт сводку

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

Lesson 17 had an MCP tool that ran once, when it was called. This lesson adds
MCP tools whose work **keeps going after the call returns**. A tool call only
*schedules* a job. A background scheduler inside the same process then runs
it on time, again and again, and saves everything to a JSON file. The agent
also has a job of its own, the digest, so it **reports every N minutes without
anyone asking**.

The lesson covers all three examples from the assignment:

| Example | Job kind | How it's scheduled |
|---|---|---|
| периодический сбор данных | `collect_commits`: fetch the newest commits of a GitHub repo and keep the ones not seen before | periodic, `watch_repo(owner, repo, every_minutes)` |
| reminder | `reminder`: fires once, then is shown in the UI and in the next digest | deferred, `remind_me(text, in_minutes)` |
| регулярный summary | `digest`: the agent calls `get_summary` over MCP and the LLM writes the digest | periodic, built in, created at startup |

```
                 ┌──────────── one binary, one port ─────────────┐
browser ──/api/ask──▶ Agent ──tools/call──▶ /mcp  SchedulerServer   │
                 │                         watch_repo  remind_me   │
                 │                         list_jobs   cancel_job  │
                 │                         get_summary ◀──┐        │
                 │                              │ write   │ read   │
                 │                              ▼         │        │
                 │                   scheduler-data.json (Store)   │
                 │                              ▲ due jobs│        │
                 │  Scheduler loop (every TICK_SECONDS) ──┘        │
                 │   collect_commits ──▶ api.github.com            │
                 │   reminder        ──▶ saved as "fired"          │
                 │   digest ──▶ Agent ──tools/call get_summary──▶ /mcp
                 │                  └──▶ LLM writes the digest     │
                 └────────────────────────────────────────────────┘
```

### The MCP tools

All five are registered with `rmcp` macros, the same way as in lesson 17.
Every argument has a description in the input schema, and every failure is
returned as a tool-level error (`isError: true`) with a readable message.

- `watch_repo(owner, repo, every_minutes?)` creates a periodic
  `collect_commits` job whose first run is on the next tick. Calling it again
  for the same repo changes the interval instead of adding a second job.
- `remind_me(text, in_minutes)` creates a one-shot `reminder` job.
- `list_jobs(include_inactive?)` returns each job's id, what it does, its
  schedule, the next run time, and the result of its last run.
- `cancel_job(job_id)` stops a job. The built-in digest job refuses to be
  cancelled, so the agent keeps reporting.
- `get_summary(hours?)` returns the **aggregated result** over the last N
  hours:
  - for each watched repo, the commits collected in the window, a
    commit count per author, and how many collections ran and failed, with
    the last error;
  - the reminders that fired;
  - the number of job runs and failures, and what's scheduled next.

  It returns `content` as compact text for the LLM and `structuredContent`
  as JSON with the same data.

Limits keep a public, always-on instance under GitHub's unauthenticated
limit of 60 requests/hour:

- at most 5 watched repos;
- a minimum interval of `MIN_INTERVAL_MINUTES`, 10 by default;
- at most 20 pending reminders.

### Saving the data: JSON

Everything lives in one file, `scheduler-data.json`:

- the jobs, with their schedule, next run, run count and last result;
- the collected commits, de-duplicated by SHA, up to 200 per repo;
- fired reminders, the run log, and the digests.

Every change saves the whole file before it returns, using write-then-rename
so a crash can't leave half a file behind. A restart or a redeploy therefore
loses nothing. On startup the loop's first tick runs right away, so anything
that came due while the process was down runs immediately. Several missed runs
of a periodic job collapse into one.

### Running on a schedule

`spawn_scheduler_loop` is a `tokio::time::interval` that wakes up every
`TICK_SECONDS`. Each time it takes the jobs whose `next_run <= now`, oldest
first, and runs them. It then records a run entry and reschedules the job:
`now + every` for periodic jobs, while a one-shot job is marked done. A
`tokio::sync::Mutex` makes sure only one tick runs at a time, so the
background loop and the UI's **Run now** button can never run the same job
twice.

### The 24/7 agent and its digest

At startup `ensure_digest_job` makes sure a `digest` job exists, with its
settings taken from `DIGEST_EVERY_MINUTES` and `DIGEST_WINDOW_HOURS`. When it
comes due, the agent:

1. connects to `/mcp` over Streamable HTTP, like any other MCP client would;
2. calls `get_summary` itself. This part is deterministic, so no LLM decides
   whether to call it;
3. hands the aggregated text to the LLM with a digest prompt ("use only this
   data, mention commits by short SHA, list fired reminders and failed
   collections");
4. saves the digest with the summary it was written from, so the UI can show
   both.

If nothing is watched and no reminder fired, the digest says so without
calling the LLM. An idle instance therefore costs nothing to run.

The chat box on the page is the lesson 17 agent loop pointed at these tools.
You can say "watch tokio-rs/axum every 10 minutes", "напомни через минуту…",
"what's scheduled?" or "stop watching axum", and the LLM picks the tool calls.

### Verification

`cargo test` needs no outside network. Everything runs in-process or on
`127.0.0.1`, with a fake GitHub and a fake LLM:

- `tools_are_registered_with_described_parameters`: all five tools are
  registered, every property has a description, and `watch_repo` requires
  `owner` and `repo`.
- `watch_repo_is_saved_to_json_and_survives_a_restart`: the job is in the
  file, and a fresh `Store::open` on that file sees it. A second
  `watch_repo` updates the job instead of adding one.
- `tool_arguments_and_limits_are_enforced`: the minimum interval, the
  watched-repo cap, bad names, reminder range, and that the digest job can't
  be cancelled.
- `scheduler_runs_due_collections_and_keeps_only_new_commits`: nothing
  runs before the job is due. When due, the job runs, saves 2 commits, and is
  rescheduled one interval later. The next period saves 0 new commits,
  because of the SHA de-duplication.
- `reminder_fires_once_at_its_time`: the reminder fires exactly once, at
  the right tick.
- `summary_aggregates_only_the_window`: per-author counts, run and failure
  counts, and the last error. Commits, runs and reminders outside the window
  are left out.
- `digest_job_aggregates_over_mcp_and_the_llm_writes_it`: end to end. The
  scheduler collects, then the digest job goes agent → real `/mcp` →
  `get_summary` → LLM. The saved digest contains data that only existed in
  the collected commits.
- `quiet_digest_skips_the_llm` and
  `agent_schedules_a_job_through_the_mcp_tool`: the LLM picks
  `watch_repo`, and the job appears in the store.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000. For a quick demo, shorten the schedule:

```bash
OPENAI_API_KEY=sk-... TICK_SECONDS=5 DIGEST_EVERY_MINUTES=2 MIN_INTERVAL_MINUTES=1 cargo run
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Watch tokio-rs/axum every 10 minutes and remind me in 1 minute to check it"}' | jq

curl -s localhost:3000/api/state | jq '.jobs, .digests[0]'

# Or use the tools directly with any MCP client:
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://localhost:3000/mcp
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `TICK_SECONDS` | no | `15`: how often the scheduler checks for due jobs |
| `DIGEST_EVERY_MINUTES` | no | `60` |
| `DIGEST_WINDOW_HOURS` | no | `24`: the window each digest summarizes |
| `DIGEST_LANGUAGE` | no | `English` |
| `MIN_INTERVAL_MINUTES` | no | `10`: smallest `every_minutes` that `watch_repo` accepts |
| `DATA_FILE` | no | `scheduler-data.json`, in the working directory |
| `MCP_SERVER_URL` | no | `http://127.0.0.1:$PORT/mcp` (this process's own server) |
| `GITHUB_TOKEN` | no | unset: unauthenticated GitHub API, 60 requests/hour per IP |
| `PORT` | no | `3000` |

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4018`,
and its MCP endpoint is at `http://<VDS host>:4018/mcp`.

The systemd unit's working directory is `~/apps/lesson-18`, so the data file
is `~/apps/lesson-18/scheduler-data.json`. The deploy only replaces `app` and
`.env`, so jobs, collected commits and digests survive redeploys. The pipeline
passes none of the optional variables above, so the deployed instance uses the
defaults: a digest every hour, a 10-minute minimum watch interval.

## Conclusion

_(to fill in after recording the demo)_

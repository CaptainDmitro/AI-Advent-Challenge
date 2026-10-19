# 35. Deploy Doctor

## Задание

🔥 День 35. Реальная задача

Придумайте реальную задачу, которую вы действительно хотите автоматизировать.

Примеры:

👉 выложить приложение в стор
👉 подготовить релиз
👉 завести инфраструктуру
👉 автоматизировать рабочий процесс

Сделайте приложение или сервис, который:

👉 использует AI
👉 работает в реальном окружении
👉 может быть не полностью рабочим (ключи / доступы)

Важно:

👉 опишите, какую задачу вы решаете
👉 покажите, как AI участвует в процессе

Результат:

Реальное AI-приложение или пайплайн под конкретную задачу

Формат:

Видео + Код

При сдаче укажите:

👉 какую задачу вы решали

## Demo

_(video to be added after recording)_

## What this is

**The task:** when this repository's CI/CD run fails, or deploys something
that doesn't work, find out why and say what to do about it.

The manual way to do that is in `DEPLOYMENT.md` under "Debugging a failed
deploy": open the run and find the failed step, download the log (which
needs authentication even on this public repo), SSH to the VDS, run
`systemctl --user status`, read `journalctl`, `cat -A` the `.env`, `curl`
the port from the box itself, and remember which failures are deliberate
(lesson 04's deploy, the non-blocking fmt/clippy steps). Deploy Doctor does
exactly that after every CI/CD run, with an LLM choosing the steps, and
reports the result as a GitHub issue:

```
push to master ─▶ ci-cd.yml (build, test, deploy) ─▶ completed (success or failure)
                                                        │ workflow_run
                                                        ▼
.github/workflows/deploy-doctor.yml
   │  cargo build --release (this lesson)
   │  ssh key from VDS_SSH_KEY, ssh-keyscan
   │  deploy-doctor diagnose --run-id <the CI/CD run>
   ▼
 1. run       GitHub API: the run, its jobs and their steps
 2. subjects  jobs grouped by lesson folder (`deploy (35. Deploy Doctor)` → lesson 35):
              a lesson whose job failed or that was deployed, or `pipeline` if `discover` failed
              lessons whose jobs all succeeded: "green", their open issues get closed
 3. agent     per subject: system = rules + DEPLOYMENT.md, user = the run and the failed steps
              ≤ 8 rounds of read-only tool calls ─▶ submit_verdict
              job_log · read_doc · service_status · service_logs · inspect_env · probe
 4. check     status/stage/evidence/doc_reference validated; a rejected verdict goes back
              to the model as the tool's error; "healthy" with a failed job becomes "broken"
 5. issues    broken: a new issue, or a comment on the open one; known_issue: one issue, once;
              healthy: comment "recovered" and close; + the Actions job summary
```

### How the AI takes part

The model doesn't summarize a log someone picked for it. It runs the
investigation. It gets the run and the subject's failed steps, then decides
which logs to grep, whether to look at the server, and when it has enough.
These are its tools, all read-only:

| Tool | What it does | Where |
|---|---|---|
| `job_log(job_id, grep?, tail?)` | a job's log without timestamps: the matching lines with 2 lines of context, or the last N lines. Only jobs of this run | GitHub API (`actions: read`) |
| `read_doc(name)` | `AGENTS.md` (compiled in) or the lesson README's Configuration and Deploy sections at the run's commit | binary / GitHub API |
| `service_status()` | `systemctl --user show` + `status` of `ai-advent-lesson-NN`, and the binary's size and date | VDS |
| `service_logs(lines?)` | the last 10-200 lines of the service's journal | VDS |
| `inspect_env()` | the `.env` without its secrets: which names are set, `EMPTY` or `MISSING`, CRLF endings, whitespace, duplicates. Only `PORT`, `OPENAI_BASE_URL` and `OPENAI_MODEL` show values | VDS |
| `probe(path?)` | `curl` to `127.0.0.1:40NN` from the VDS itself: status code and time | VDS |
| `submit_verdict(...)` | the answer, checked before it is accepted | — |

`DEPLOYMENT.md` is pinned into the system prompt. It holds the deploy
layout, the secrets table and the known failures, so the model can say
"lesson 04's deploy failed, and that is deliberate (DEPLOYMENT.md, Secrets &
variables)" instead of reporting a bug. The VDS tools are offered only for a
lesson deployed in this run.

The verdict:

```json
{ "status": "broken", "failed_stage": "deploy",
  "root_cause": "OPENAI_API_KEY is empty in ~/apps/lesson-35/.env, so the app exits on start. ...",
  "evidence": ["service_logs: Error: OPENAI_API_KEY environment variable is not set.",
               "inspect_env: OPENAI_API_KEY: EMPTY"],
  "doc_reference": null,
  "suggested_fix": "Add the OPENAI_API_KEY repository secret and re-run CI/CD.",
  "commands_to_run": ["systemctl --user status ai-advent-lesson-35"] }
```

| `status` | When | The issue |
|---|---|---|
| `healthy` | the jobs succeeded, the service is active, the probe answers 2xx/3xx | the open one gets "✅ Recovered" and is closed |
| `broken` | anything that needs a fix | created, or commented on if one is open |
| `known_issue` | a failure that `DEPLOYMENT.md`/`AGENTS.md` documents, with `doc_reference` | created once, then left alone |
| `unknown` | no AI verdict for a lesson whose jobs succeeded | nothing |

### Read-only by construction

The doctor diagnoses and never repairs. That isn't a line in the prompt.
There is simply no tool that writes anything:

- **The model never writes a command.** Every VDS probe is built in
  `vds.rs` from the subject's lesson number (1-99, from the job name), so
  the service, port and folder are fixed. The only model text that reaches
  a shell is a probe path, and it must match `/[A-Za-z0-9/._-]*` without
  `..`.
- **`.env` is parsed on the doctor's side.** `inspect_env` returns names
  and lengths, never values (except the three that aren't secret).
- **Everything is redacted before the model sees it**: the doctor's own
  `OPENAI_API_KEY` and `GITHUB_TOKEN` by value, `NAME=value` where the name
  contains KEY/TOKEN/SECRET/PASSWORD, `Bearer …`, and `sk-…` keys. Results
  are clipped to 8 000 characters.
- **Logs only of this run**: `job_log` refuses a job id that isn't one of
  the run's jobs.
- Tool results are data: the prompt says so, and a log line that says
  "ignore your instructions" is just quoted in `evidence`.
- The workflow's permissions are `actions: read`, `issues: write`,
  `contents: read`. The SSH key is the deploy key, used only for the four
  probes.

### Checks on the verdict

- `status` must be `healthy`, `broken` or `known_issue` (`Known issue` and
  `known-issue` are accepted). An unknown `failed_stage` is replaced with
  the one the run points at.
- `broken` and `known_issue` need a `root_cause` and at least one
  `evidence` quote. `known_issue` needs a `doc_reference`.
- A rejected verdict goes back to the model as the tool's error ("A
  `known_issue` needs `doc_reference` … Fix it and call submit_verdict
  again."), and the model gets another round.
- **"Healthy" with a failed job is overridden** to `broken`, with a note
  in the issue.
- Evidence and commands are trimmed to 8 items of 400 characters.
- If the model fails (no key, API down, 8 rounds without a valid verdict),
  the subject still gets a **fallback verdict from the run's data**: the
  failed job › step names and a link to the log, so a broken build is
  never silent. The job then exits 1, so the doctor's own failure shows too.

### The issue

~~~
<!-- deploy-doctor:lesson-35 -->
### 🔴 Broken — lesson 35 (`35. Deploy Doctor`)
**Run** [9001](…) · `abcdef1` · Add lesson 35 · stage `deploy`
**Root cause.** OPENAI_API_KEY is empty in ~/apps/lesson-35/.env, so the app exits on start. …
**Evidence** - service_logs: Error: … - inspect_env: OPENAI_API_KEY: EMPTY
**Suggested fix.** Add the OPENAI_API_KEY repository secret and re-run CI/CD.
```bash
systemctl --user status ai-advent-lesson-35
```
▸ How this was diagnosed: 7 tool calls in 4 rounds  (each call, its arguments and result)
Read-only diagnosis by Deploy Doctor with `deepseek-v4-flash` in 12.3 s. …
~~~

The title is `Deploy Doctor: lesson 35 — <the root cause's first sentence>`
and the label is `deploy-doctor`. The hidden first line is how the next run
finds the issue again (by body, so it works even if the label was never
applied). Text from the model is sanitized so it can't forge that marker or
break out of a code block.

### The page

`GET /` lists the recent CI/CD runs and the doctor's issues. Click a run to
diagnose it on the spot: the page shows every subject's verdict, the tool
calls with their arguments, results and timings, and the Markdown the
Action would post. **The page never posts.** On the VDS it runs the probes
locally (`VDS_MODE=local`), and without a `GITHUB_TOKEN` it can't read job
logs: the model gets that as the tool's error and works with the rest.

### Verification

`cargo test` needs no network and no keys. GitHub, the LLM and the VDS are
fakes: a GitHub API on `127.0.0.1:0` with runs, jobs, logs, contents and
issues kept in memory; a scripted OpenAI-compatible LLM; and a `Vds::Fake`
that answers by a piece of the script and records every script it got.

- `a_failed_deploy_is_investigated_and_reported_as_one_issue`: run 9001,
  where `deploy (35. Deploy Doctor)` failed at "Verify service is up". The
  model calls `service_status` and `probe`, then `service_logs`,
  `inspect_env` and `job_log` (grep `error`), submits a `known_issue`
  without `doc_reference` (rejected), then a `broken` verdict (accepted):
  7 calls in 4 rounds. The journal's `Bearer sk-…` and the log's GitHub
  token reach the model as `***`, and no request contains either. The
  scripts all start with `export XDG_RUNTIME_DIR` and use port 4035 and
  `ai-advent-lesson-35`. The system prompt has DEPLOYMENT.md, and the
  user prompt has the failed step. Issue #3 is created with the marker,
  the title from the root cause's first sentence and the `deploy-doctor`
  label. Lesson 34's open issue is closed because its jobs all succeeded,
  and an unrelated issue is untouched. A second run of the same diagnosis
  comments on #3 instead of opening another issue.
- `a_healthy_deploy_closes_the_open_issue`: active service, `200` from the
  probe, a `healthy` verdict, so lesson 35's issue gets "✅ Recovered in
  run" and is closed.
- `without_the_llm_a_build_failure_is_still_reported`: the LLM is
  unreachable. `build-test (33. Support Assistant)` failed at `cargo test`,
  and the issue is still created, with stage `test`, the failed step as
  evidence and the job's link.
- `the_page_diagnoses_without_posting`: `POST /api/diagnose` (run id as a
  string) returns the verdict and touches no issue. A missing `run_id` is
  an error string, and `/api/status`, `/api/runs`, `/api/issues` (only
  marked issues) and the page work.
- In the modules: job names → subjects and green lessons, log timestamps
  and BOM, grep with context and gaps, the tail; every verdict rule and the
  fallback; probe paths, lesson bounds, `.env` description (hidden values,
  EMPTY, MISSING, CRLF, duplicates); redaction; issue titles, markers and
  sanitizing; URL encoding; the command line.

## Run

The page, over this repo's runs (anonymous GitHub API works for a public
repo; with `GITHUB_TOKEN` the model can also read job logs):

```bash
OPENAI_API_KEY=sk-... VDS_MODE=off cargo run
```

Then open http://localhost:3000.

Diagnose a run from your machine without posting anything, with the probes
over SSH:

```bash
OPENAI_API_KEY=sk-... GITHUB_TOKEN=ghp_... VDS_HOST=1.2.3.4 VDS_USER=deploy VDS_SSH_KEY_PATH=~/.ssh/deploy_key cargo run -- diagnose --run-id 12345678 --dry-run
```

With curl:

```bash
curl -s -X POST localhost:3000/api/diagnose -H 'Content-Type: application/json' \
  -d '{"run_id": 12345678}' | jq '.diagnoses[] | {key: .subject.key, status: .verdict.status, root_cause: .verdict.root_cause, calls: [.calls[].tool]}'
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | no | — without it every verdict is the fallback from the run's data |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` (it has to support tool calling) |
| `REPORT_LANGUAGE` | no | `English`: the language of `root_cause` and `suggested_fix` |
| `GITHUB_TOKEN` | to read logs and post | — (Actions passes the job's token). Without it runs are read anonymously, `job_log` is unavailable and nothing is posted |
| `GITHUB_REPOSITORY` | no | `CaptainDmitro/AI-Advent-Challenge`; Actions sets it |
| `GITHUB_API_URL` | no | `https://api.github.com`; Actions sets it |
| `VDS_MODE` | no | `ssh` when `VDS_HOST` is set, otherwise `local`; `off` disables the server probes |
| `VDS_HOST`, `VDS_USER`, `VDS_SSH_KEY_PATH` | for `ssh` | — |
| `PORT` | no | `3000` |

## Deploy

**The GitHub Action** is `.github/workflows/deploy-doctor.yml`. Like every
workflow file in this repo, it is added by hand in GitHub's web editor (see
`AGENTS.md`). It needs no new secrets: it reuses `VDS_SSH_KEY`, `VDS_HOST`,
`VDS_USER` and the `OPENAI_*` settings the deploy already has, plus the
job's own `GITHUB_TOKEN`. It runs after every CI/CD run (`workflow_run`,
which only fires for workflow files on `master`), or by hand with a run id.

```yaml
name: Deploy Doctor
on:
  workflow_run:
    workflows: ["CI/CD"]
    types: [completed]
  workflow_dispatch:
    inputs:
      run_id:
        description: 'CI/CD run id to diagnose'
        required: true
        type: string
permissions:
  actions: read
  issues: write
  contents: read
concurrency:
  group: deploy-doctor-${{ github.event.workflow_run.id || inputs.run_id }}
jobs:
  diagnose:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable

      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: "35. Deploy Doctor"

      - name: cargo build --release
        working-directory: "35. Deploy Doctor"
        run: cargo build --release

      - name: Configure SSH
        env:
          SSH_KEY: ${{ secrets.VDS_SSH_KEY }}
          VDS_HOST: ${{ vars.VDS_HOST }}
        run: |
          set -euo pipefail
          : "${SSH_KEY:?VDS_SSH_KEY secret is empty}"
          : "${VDS_HOST:?VDS_HOST repository variable is empty}"
          mkdir -p ~/.ssh
          printf '%s\n' "$SSH_KEY" > ~/.ssh/doctor_key
          chmod 600 ~/.ssh/doctor_key
          ssh-keyscan -H "$VDS_HOST" >> ~/.ssh/known_hosts

      - name: Diagnose
        working-directory: "35. Deploy Doctor"
        env:
          RUN_ID: ${{ github.event.workflow_run.id || inputs.run_id }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          VDS_HOST: ${{ vars.VDS_HOST }}
          VDS_USER: ${{ vars.VDS_USER }}
          VDS_SSH_KEY_PATH: /home/runner/.ssh/doctor_key
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_BASE_URL: ${{ vars.OPENAI_BASE_URL }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL }}
        run: ./target/release/deploy-doctor diagnose --run-id "$RUN_ID"
```

The doctor is built from `master`, so a run that breaks lesson 35 is
diagnosed by the version that was there before it. Note that the doctor's
own `OPENAI_API_KEY` comes from the same secret as the lessons', so
deleting that secret leaves only the fallback verdict. For a demo of a
broken runtime, stop the service on the VDS instead
(`systemctl --user stop ai-advent-lesson-35`) and dispatch the doctor with
the last run's id.

**The page** deploys like every lesson. It is the highest-numbered web
lesson, so `ci-cd.yml` ships it on every push to `master`, to
`http://<VDS host>:4035`. It checks the services on its own box
(`VDS_MODE=local`). The VDS gets no `GITHUB_TOKEN`, so the page reads runs
anonymously (60 requests an hour), can't read job logs, and never posts.

## Conclusion

_(to fill in after recording the demo)_

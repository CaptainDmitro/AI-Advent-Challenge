# Agent instructions for AI-Advent-Challenge

This file is read by AI coding agents (Claude Code, and any other tool that
honors `AGENTS.md`) before working in this repo. Keep it in sync with reality
as the repo evolves — it exists to save a fresh session from re-deriving
context the hard way.

## Environment constraints — read this first

- **No local Rust toolchain, Docker, or (likely) `git`** exist on the
  machine these sessions typically run from. Do not attempt to run
  `cargo build`/`cargo test`/`docker` locally, and do not try to install
  them. The only place code actually compiles is inside GitHub Actions.
- **All repository reads/writes go through a GitHub MCP connector**, not a
  local clone. There is nothing to `git clone` on disk by default.
- **Editing files under `.github/workflows/` gets rejected** by Claude
  Code's own auto-mode safety classifier, independent of GitHub
  permissions — this is not a bug to route around. When a workflow file
  needs to change: prepare the new content (a diff or the full file), hand
  it to the user, and ask them to paste it into GitHub's web editor
  themselves. Don't retry the same write expecting a different result.
- Because there's no local build, **verification happens by pushing and
  watching GitHub Actions** (`gh`-less: poll the public REST API,
  e.g. `https://api.github.com/repos/CaptainDmitro/AI-Advent-Challenge/actions/runs`,
  or read the run in a browser tool — full log *downloads* need
  authentication even on this public repo, but run/job/step status does
  not).

## Repo shape

One folder per course lesson, named `NN. Title` (two-digit, zero-padded,
e.g. `06. First Agent`). Each folder is an **independent Cargo crate** —
there is no workspace. Lessons come in two shapes:

- **CLI lessons** (01, 02): blocking `reqwest`, a stdin/stdout chat loop,
  no `axum`, nothing to deploy.
- **Web-app lessons** (03, 04, 05, 06, and presumably every lesson after):
  `axum` + `tokio`, a single struct held in `State` (e.g. `AppState` or
  `Agent`) that owns a `reqwest::Client` and exposes a `call_llm`-shaped
  async helper, a `static/index.html` with inline CSS/JS embedded into the
  binary at compile time via `include_str!` (not served from disk at
  runtime — deploy only ever needs to ship the compiled binary), and
  errors surfaced as plain display strings in JSON responses rather than
  HTTP error statuses, so one failed call never breaks the rest of the UI.

Every lesson's `README.md` follows the same template: `## Задание` (the
original assignment text, in Russian), `## Demo` (a video, added by the
human after recording), `## What this is`, `## Run`, `## Configuration`
(env var table), and `## Conclusion` (filled in after the demo — leave as a
placeholder when scaffolding a new lesson).

## Environment variables are *not* uniform across lessons

Most web-app lessons use `OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`OPENAI_MODEL` / `PORT`, but this is a convention, not a guarantee —
**lesson 04 uses `WEAK_*`/`MEDIUM_*`/`STRONG_*` instead**, with no defaults
on several of them. Before assuming a lesson's env var shape (e.g. when
wiring it into the deploy pipeline), read that lesson's own README
Configuration table.

## CI/CD pipeline

`.github/workflows/ci-cd.yml` runs on every push to `master` and on manual
`workflow_dispatch`. See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full
architecture, the secrets/variables inventory, and the deploy-selection
logic (short version: only the highest-numbered lesson folder deploys by
default; redeploying anything else needs a manual `workflow_dispatch` run
with a `lessons` input).

Known, deliberate state — not bugs to "fix" on sight:

- **Lesson 04's deploy fails** and is left that way on purpose: it needs
  `MEDIUM_*`/`STRONG_*` secrets the pipeline doesn't currently supply.
  Don't silently patch this without checking with the user first — it was
  an explicit decision to defer.
- **`cargo fmt --check` and `cargo clippy` are non-blocking** in
  `build-test` (`continue-on-error: true`). This is because lessons 02–05
  predate the pipeline and were never run through rustfmt; making them
  blocking fails already-finished, already-recorded lessons on style
  alone. `cargo build`/`cargo test` remain the real gate. New lessons
  should still be written cleanly, but don't "fix" this back to blocking
  without accounting for the old lessons.

## Adding a new lesson

1. Create `NN. Title/` (next number, zero-padded) with `Cargo.toml`,
   `src/main.rs`, and (if it's a web app) `static/index.html`, following
   the shape described above.
2. Match the README template. Copy an existing lesson's README structure
   and adapt the `## Задание` block to the new assignment text.
3. Add a row to the root `README.md` table of contents.
4. Nothing needs to be registered in the CI workflow — `discover` finds
   every `Cargo.toml` automatically, and a web-app lesson becomes the new
   deploy default the moment it's the highest-numbered folder.
5. Push. Watch the Actions run. Fix anything `build`/`test` catch (fmt/
   clippy warnings are visible in the log but won't block).

## Where to look for more

- [`README.md`](./README.md) — the human/course-facing overview and lesson
  index.
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — deploy architecture, secrets
  inventory, and the pipeline's selection logic in full.

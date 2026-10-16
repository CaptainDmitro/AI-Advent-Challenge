# 32. AI Code Review

## Задание

🔥 День 32. Автоматизация ревью кода

Настройте пайплайн, в котором ассистент анализирует новый PR.

Ассистент должен:

👉 получать diff и изменённые файлы
👉 использовать RAG (документация + код)
👉 выдавать текст ревью

Минимальный результат:

👉 пайплайн (например GitHub Action) запускается на PR
👉 ассистент получает diff
👉 возвращает:

- потенциальные баги
- архитектурные проблемы
- рекомендации

Результат:

Автоматическое AI-ревью кода

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

An AI reviewer for this repository's pull requests. A GitHub Action runs it
on every PR. It reads the diff, retrieves related docs and code from the
repository (RAG), lets the model read more files over MCP, and posts one
review comment with **potential bugs**, **architecture issues** and
**recommendations**. Every new push updates the same comment. The same
binary also serves a web page that reviews any open PR on demand, showing
every step.

```
pull_request (opened / synchronize / reopened / ready_for_review)
   │
   ▼
.github/workflows/ai-review.yml
   │  checkout base → reviewer/   (the reviewer is built from the base branch)
   │  checkout head → pr/         (fetch-depth 0: the diff needs the base commit)
   │  cargo build --release
   │  ai-code-review review --repo-dir pr --pr N --post
   ▼
 1. PR        GitHub API: title, description, base/head SHAs
 2. diff      git diff base...head  ─▶ files, hunks, line numbers (both sides)
              lock files, binaries, deleted files: listed, not reviewed
 3. files     the head itself (Actions), or a temporary git worktree at the head (VDS)
 4. RAG       index of the whole repo at the head: every .md, .rs, .toml, .yml, .html
              structural chunks + hashed TF-IDF (lessons 21-31), each chunk knows its lines
              a query per hunk: the names it adds/removes + the item it is in + the lesson
              top-20 ─▶ rerank ─▶ ≥ 0.25 ─▶ ≤ 4 docs + ≤ 6 code, ≤ 2 per file,
              never a chunk the diff already shows; docs/conventions.md always pinned as [1]
 5. MCP       lesson 31's `project` server over the head, on a private port
              read_file · list_files · git_log · git_diff · git_status · git_branch
 6. LLM       system: rules + fragments [n] + tools   user: PR + numbered diff
              ≤ 6 rounds of tool calls ─▶ JSON {summary, findings[]}
 7. check     the file must be in the diff, the line must be one the diff shows,
              [n] must exist, duplicates merged ─▶ dropped findings listed
 8. comment   Markdown with a hidden marker ─▶ PATCH the earlier comment, or POST a new one
              + the Actions job summary
```

### The diff

`git diff base...head` is what the PR changed since its branch left the base,
the same diff GitHub shows. It runs with rename detection, 5 lines of context,
and fixed `a/`/`b/` prefixes and path quoting, whatever the git config says.
`diff.rs` parses it into files (added, modified, deleted, renamed), hunks and
lines with their old and new line numbers. Paths with spaces like
`31. Developer Assistant/src/main.rs` are handled too: git appends a tab to
those `---`/`+++` lines. The model sees each file with the new-side line
number in front of every added and context line:

```
### 07. Retry Lesson/src/main.rs (modified, +1 -1)
@@ -3,11 +3,11 @@ mod retry;
    3   use retry::{MAX_RETRIES, retry_delay};
    …
      -     for attempt in 0..MAX_RETRIES {
    8 +     for attempt in 0..=MAX_RETRIES {
```

Lock files, binaries, media and deleted files are listed in the prompt and
the comment but not sent. A file's patch is clipped at 12 000 characters.
Files are packed into batches of up to 40 000 characters, and each batch is
one model conversation. Almost every PR fits in one. After 4 batches the
remaining files are named as not reviewed instead of being silently dropped.

### RAG: documentation + code

Lesson 31 indexed only the docs. Here the corpus is **the whole repository at
the PR's head**: every README, `AGENTS.md`, `DEPLOYMENT.md`, `docs/`, every
lesson's Rust code, `Cargo.toml`s, pages and the CI workflows. Hidden folders
(except `.github`), `target/` and symlinks are skipped. That's about 150 files,
1.8 MB of them Rust, indexed in well under a second. The index uses the same
design as lessons 21-31 (structural chunks, the local hashed TF-IDF embedder,
the heuristic reranker). New in this lesson: every chunk knows whether it is
`doc` or `code` and which lines it spans.

A question becomes a query in lesson 31. Here **every hunk** does: the
identifiers it adds or removes (`MAX_RETRIES`, `retry_delay`, …, without
keywords and std names), the item git names in the hunk header, and the
lesson's title. Each chunk keeps its best score over all queries. Chunks
that overlap a hunk are dropped, since the diff already shows them. The
context is then:

| Kind | Up to | What it's for |
|---|---|---|
| pinned | all of `docs/conventions.md` | the project's rules, given whatever the diff is: an architecture finding has to cite one |
| docs | 4 fragments ≥ 0.25 | the lesson's README, the architecture and API docs the change touches |
| code | 6 fragments ≥ 0.25 | definitions and other callers of what the diff uses, and similar code elsewhere (duplication) |

At most 2 fragments come from one file. The model cites fragments as `[n]` in
each finding's `sources`, and the comment lists them with their lines and
scores.

### MCP

The diff and the fragments are often enough. When they aren't, the model
can read the PR's head itself: lesson 31's `project` MCP server runs over
the head on a private `127.0.0.1` port for the length of the review. The
reviewer connects to it as an MCP client over Streamable HTTP (`initialize`,
`tools/list`), and the 6 read-only tools become functions for the model.
The usual call is `read_file` of a definition or of the rest of a changed
file. After 6 rounds the model has to answer.

### The review and its checks

The model answers with JSON:

```json
{"summary": "...", "findings": [{"category": "bug", "severity": "high",
  "file": "07. Retry Lesson/src/main.rs", "line": 8, "title": "...",
  "details": "...", "suggestion": "...", "sources": [1, 4]}]}
```

The system prompt says what each category means. **bug** covers wrong
behaviour, panics on input, off-by-one errors, error handling, races,
security and broken tests. **architecture** covers breaking a documented
convention (cite it), duplicating code that already exists (cite where),
and wrong layering. **recommendation** covers naming, tests and docs to
add, and simpler ways to do the same thing. The prompt asks for no praise
and no rustfmt nits, and for an empty list when nothing is wrong. The PR
description is passed as the author's text, not as instructions. A reply
that isn't JSON gets one more chance.

Nothing from the model is posted unchecked:

- a finding whose file isn't in the diff, or with no title, is **dropped**
  (the page lists dropped findings with the reason);
- a line the diff doesn't show is cleared and the finding stays at the file
  level, marked `line_outside_diff`;
- `sources` that aren't fragment numbers are removed;
- `./x`, `b/x` and `x:42` are matched to the diff's paths; unknown
  categories become recommendations; duplicates are merged;
- findings are sorted bug → architecture → recommendation, then high →
  medium → low.

### The comment

```
<!-- ai-code-review -->
## 🤖 AI code review
<summary>
| 🐞 Potential bugs | 🏛 Architecture | 💡 Recommendations |
### 🐞 Potential bugs
1. 🔴 **Off-by-one: MAX_RETRIES + 1 attempts** — [`07. Retry Lesson/src/main.rs:8`](…/blob/<head>/…#L8)
   details · **Suggestion:** … · Context: [1] `docs/conventions.md` › Conventions › Code
### 🏛 Architecture …   ### 💡 Recommendations …
▸ 📚 Context: fragments with lines, scores, cited · MCP calls · files and what was skipped
Advisory review by `<model>` of `<base>...<head>` …
```

Links point to the head commit, so they stay right after later pushes. The
hidden first line is the marker: the next run finds that comment and edits
it (`PATCH`) instead of adding another. A failed review (LLM down, bad
config) is posted the same way with the reason, so the PR shows why there
is no review, and the job fails. Findings never fail the job, and the check
is advisory.

### Safety

- **The reviewer comes from the base branch**, not from the PR, so a PR
  can't change the code that reviews it. A PR that breaks lesson 32 still
  gets reviewed.
- `pull_request`, never `pull_request_target`. PRs from forks get no
  secrets, so the job skips them. Drafts wait for `ready_for_review`.
- `permissions: contents: read, pull-requests: write`, and the token is
  the job's own `GITHUB_TOKEN`.
- Nothing from the PR is executed: it is diffed, read and indexed. Every
  MCP tool is read-only, with lesson 31's path and revision checks.
  Revisions from the page or the API go through the same check, so
  `--output=…` can't become a git option.
- `concurrency` cancels a running review when a newer push arrives.

### The page

`GET /` lists the repository's open PRs (from the GitHub API). Click one, or
enter two revisions of the checkout (`master...HEAD`, `HEAD~3...HEAD`), to
run the same pipeline. The page shows the findings by category with links
and `[n]` citations, the pipeline, the context fragments (pinned/doc/code,
score, lines, which file they were found for, cited), every retrieval
candidate and why it was left out, the queries, the MCP calls, the files,
dropped findings, and the exact Markdown the Action posts. **The page never
posts.** On the VDS a PR's commits are fetched into the service's own clone
(`refs/review/pr-N`), and the review runs in a temporary worktree, so the
clone's branch is never touched.

### Verification

`cargo test` needs no network and no keys, only `git`. The tests build a
small repository: `master` with `docs/conventions.md` ("retries go through
`retry_delay`; don't write a second backoff") and a lesson with
`src/retry.rs` and `src/main.rs`. Branch `feature/faster-retries` changes
the retry loop to `0..=MAX_RETRIES` (line 8) and adds a second backoff in
`src/jitter.rs`. A scripted LLM and a fake GitHub API run on `127.0.0.1:0`.

- `a_pr_is_reviewed_with_rag_and_mcp_and_the_comment_is_posted_once`: PR #7,
  with the checkout on `master`, so a worktree is used and then removed. The
  diff has the 2 files, +7 −1. The fragments are `docs/conventions.md` as
  pinned `[1]`, `src/retry.rs` (code, with `fn retry_delay`) and the lesson
  README (doc), and nothing of `main.rs` that the diff already shows. The
  index has the 7 files of the head. The model's `read_file` went to the
  real MCP server, and its result went back to the model. The system prompt
  carries the fragments, and the user prompt carries the PR and the numbered
  diff (`    8 +     for attempt in 0..=MAX_RETRIES {`). Of the model's
  findings, the bug on line 8 survives with `sources [1]` (99 removed), the
  architecture finding on line 999 is moved to the file level, and the one
  on a file outside the PR and the one with no title are dropped. The
  comment has the marker, the link to `…/blob/<head>/07.%20Retry%20Lesson/src/main.rs#L8`
  and the MCP call. It is `created` with the token. A second run `updated`
  the same comment, so there is still 1 comment.
- `the_page_reviews_two_revisions_without_posting`: `POST /api/review`
  `master...HEAD` with the head checked out (no worktree), so findings come
  back and nothing is posted. Identical revisions give "No changes" without
  calling the model. A missing branch and `--output=/tmp/x` are error
  strings. `/api/pulls`, `/api/status` and the page work.
- `a_failed_review_is_posted_as_a_failure_and_cleans_up`: with the LLM
  unreachable, the retrieval still ran, the failure comment is posted, and
  the worktree is removed.
- In the modules: the diff parser (renames, new, deleted and binary files,
  a removed line that starts with `--`, paths with spaces), the numbered and
  clipped rendering, identifiers and queries, JSON in fences and prose,
  every validation rule, the comment's sections, links and escaping, the
  corpus (docs, code, workflows; not `target/`, `.git` or dotfiles), chunk
  line ranges, the reranker, CLI arguments, path and revision checks.

## Run

The page, over the clone you're in (`..` when run from the lesson folder):

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000. Open PRs come from the GitHub API
(unauthenticated works for a public repo; `GITHUB_TOKEN` raises the rate
limit).

Review your branch before you push it:

```bash
OPENAI_API_KEY=sk-... cargo run -- review --base master
```

A PR, exactly like the Action does (add `--post` and a `GITHUB_TOKEN` to post):

```bash
OPENAI_API_KEY=sk-... cargo run -- review --pr 12 --out review.md
```

With curl:

```bash
curl -s -X POST localhost:3000/api/review -H 'Content-Type: application/json' \
  -d '{"pr": 12}' | jq '{summary, findings: [.findings[] | {category, severity, file, line, title}], fragments: [.fragments[] | {n, source, kind, score}], calls: [.calls[] | .tool]}'
```

```bash
curl -s -X POST localhost:3000/api/review -H 'Content-Type: application/json' \
  -d '{"base": "master", "head": "HEAD"}' | jq -r .markdown
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` (it has to support tool calling) |
| `REVIEW_LANGUAGE` | no | `English`: the language of the review text |
| `GITHUB_TOKEN` | to post | — (Actions passes the job's token). Without it PRs are read anonymously and nothing can be posted |
| `GITHUB_REPOSITORY` | no | from `REPO_URL` (`CaptainDmitro/AI-Advent-Challenge`); Actions sets it |
| `GITHUB_API_URL` | no | `https://api.github.com`; Actions sets it |
| `REPO_DIR` | no | `..` when that is a checkout of this repo, otherwise `repo` in the working directory; `--repo-dir` overrides it |
| `REPO_URL` | no | `https://github.com/CaptainDmitro/AI-Advent-Challenge.git`: cloned into `REPO_DIR` for the page when it isn't a checkout |
| `REPO_CLONE` | no | on. `off` never clones |
| `PORT` | no | `3000` |

## Deploy

**The GitHub Action** is `.github/workflows/ai-review.yml`. Like every
workflow file in this repo, it is added by hand in GitHub's web editor (see
`AGENTS.md`). It needs no new secrets: it uses `OPENAI_API_KEY` and the
optional `OPENAI_BASE_URL`/`OPENAI_MODEL` that the deploy already has, plus
the job's own `GITHUB_TOKEN`. A run takes about 2-4 minutes, most of it the
cached `cargo build`.

```yaml
name: AI code review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

permissions:
  contents: read
  pull-requests: write

concurrency:
  group: ai-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository && !github.event.pull_request.draft
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Check out the reviewer (base branch)
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          path: reviewer
      - name: Check out the pull request (head)
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          path: pr
          fetch-depth: 0
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: reviewer/32. AI Code Review
          key: ai-review
      - name: Build the reviewer
        working-directory: reviewer/32. AI Code Review
        run: cargo build --release
      - name: Review the pull request
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_BASE_URL: ${{ vars.OPENAI_BASE_URL }}
          OPENAI_MODEL: ${{ vars.OPENAI_MODEL }}
          GITHUB_TOKEN: ${{ github.token }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: |
          "reviewer/32. AI Code Review/target/release/ai-code-review" review \
            --repo-dir pr --pr "$PR_NUMBER" --post
```

**The page** deploys like every lesson. It is the highest-numbered web
lesson, so `ci-cd.yml` ships it on every push to `master`, to
`http://<VDS host>:4032`. On its first start it clones the repo into
`~/apps/lesson-32/repo/` (this needs `git` on the VDS, as for lesson 31). A
PR's commits are fetched into that clone when the PR is reviewed. The VDS
gets no `GITHUB_TOKEN`, so the page reads PRs anonymously (60 requests an
hour) and never posts.

## Conclusion

_(to fill in after recording the demo)_

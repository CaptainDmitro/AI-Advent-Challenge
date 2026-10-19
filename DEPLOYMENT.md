# Deployment

How `.github/workflows/ci-cd.yml` gets a lesson from a `git push` to a
running service, and everything you need to reconstruct or debug it
without archaeology.

## Architecture

```
push to master (or manual "Run workflow")
        |
        v
  discover  --------------------------> figures out which lesson folders
        |                                exist, which are web apps (depend
        |                                on axum), and which one(s) to
        |                                deploy this run
        v
  build-test (matrix: every lesson) ---> cargo build/test/clippy/fmt
        |
        v
  deploy (matrix: selected lesson(s)) -> cross-compile to a static musl
        |                                binary, scp it to the VDS, write
        |                                a systemd --user unit + env file,
        v                                restart the service
  http://<VDS_HOST>:<4000 + lesson number>
```

`build-test` always runs for every lesson (cheap, catches regressions
repo-wide). `deploy` only runs for lessons that (a) depend on `axum` and
(b) were selected — see "Which lessons deploy" below.

## Why a static musl binary instead of Docker

The binary is cross-compiled to `x86_64-unknown-linux-musl`, which is
fully statically linked — no glibc version dependency, no OpenSSL (the
lessons use `reqwest` with `rustls`), and no need to install Docker on the
VDS at all. The VDS stays as bare as the constraint that started this
whole pipeline (no unnecessary tooling anywhere).

## Where things live on the VDS

Each lesson `NN` gets:

```
~/apps/lesson-NN/app                                  the binary
~/apps/lesson-NN/.env                                 PORT, and whatever secrets that lesson needs
~/.config/systemd/user/ai-advent-lesson-NN.service     systemd unit
```

Port is always `4000 + NN` (lesson 6 → `4006`). Service name is always
`ai-advent-lesson-NN`. Both are derived mechanically from the lesson
folder's leading number in the workflow — nothing to configure per lesson.

Services run as **user-level systemd units** (`systemctl --user`), not
system-wide, so the deploy SSH key never needs `sudo`. This requires one
manual one-time step per VDS that isn't repeatable from CI:

```bash
sudo loginctl enable-linger <ssh-user>
```

Without this, a service stops the moment the SSH session that started it
closes. This setting is tied to the VDS's OS installation, not to any
particular lesson or deploy — it persists across reboots and every future
redeploy on the *same* box. You only need to run it again if the VDS is
rebuilt or reimaged from scratch, or if you ever provision a second VDS.
If services on an existing, previously-working VDS suddenly start dying
right after each deploy, that's a sign linger got disabled somehow (or
you're on a new box) — re-run the command above to confirm.

## Secrets & variables (GitHub → Settings → Secrets and variables → Actions)

Names and purpose only — values are never in the repo or in chat history.

| Name | Type | Purpose |
|---|---|---|
| `VDS_SSH_KEY` | Secret | Private half of a dedicated deploy-only SSH keypair. Public half is in the VDS's `authorized_keys`. |
| `VDS_HOST` | Variable | The VDS's IP or hostname. Not secret — needed unmasked in logs for debugging. |
| `VDS_USER` | Variable | SSH login username on the VDS. |
| `OPENAI_API_KEY` | Secret | API key written into deployed lessons' `.env` under `OPENAI_API_KEY`. Used by lessons that follow the `OPENAI_*` convention. |
| `OPENAI_BASE_URL` | Variable | Optional. Only written into `.env` if non-empty — an empty value would override the app's own built-in default with an empty string, which is worse than not setting it. Points at whatever OpenAI-compatible endpoint you're actually using (e.g. DeepSeek's API). |
| `OPENAI_MODEL` | Variable | Optional, same non-empty rule as above. |

**Lesson 04 is not covered by this table.** It needs `MEDIUM_API_KEY`,
`MEDIUM_BASE_URL`, `STRONG_API_KEY`, `STRONG_BASE_URL` (required, no
defaults) and optionally `WEAK_*` — none of these are wired into the
deploy step yet. This is why its deploy job fails; it's deliberately left
that way until someone decides how per-lesson env conventions should be
generalized.

## Which lessons deploy

The `discover` job computes the deploy list:

- **Default** (a normal `git push`, or a manual run with the `lessons`
  input left blank): only the **single highest-numbered lesson folder**,
  if it depends on `axum`. If the latest lesson is a CLI, nothing deploys
  that run — there's nothing to deploy.
- **Manual override**: go to Actions → CI/CD → "Run workflow" and fill in
  the `lessons` text input:
  - `06` or `03,06` — deploy exactly those lesson numbers (non-web
    numbers are silently skipped, not an error).
  - `all` — deploy every lesson that depends on `axum`.

This means old lessons stay frozen at whatever they last deployed as —
pushing an update to lesson 6 today won't touch lessons 3 or 5's running
instances. To push a change to an older lesson, use the manual override.

## AI review of pull requests (`ai-review.yml`)

A second workflow, separate from the deploy pipeline above:

```
pull_request opened / synchronize / reopened / ready_for_review
        |
        v
  review  -----> checkout base -> reviewer/   (the reviewer comes from the base branch)
                 checkout head -> pr/         (fetch-depth 0)
                 cargo build --release in reviewer/32. AI Code Review
                 ai-code-review review --repo-dir pr --pr N --post
                        -> one PR comment, edited in place on every push
```

- **Permissions**: `contents: read`, `pull-requests: write`, with the job's
  own `GITHUB_TOKEN`. No new secrets: it reuses `OPENAI_API_KEY`,
  `OPENAI_BASE_URL` and `OPENAI_MODEL` from the table above.
- **Skipped** for PRs from forks (they get no secrets) and for drafts.
  `concurrency` cancels a running review when a newer push arrives.
- **Advisory**: findings never fail the job. It fails only when the review
  can't run (LLM error, missing key), and then it posts a comment saying why.
- **The reviewer is built from the base branch**, so a PR that changes or
  breaks lesson 32 is still reviewed by the version on `master`.
- Like every workflow here, the file is changed by hand in GitHub's web
  editor; its full text is in `32. AI Code Review/README.md`.

## AI diagnosis of every run (`deploy-doctor.yml`)

A third workflow runs when CI/CD completes, whether it succeeded or failed:

```
CI/CD completed (workflow_run), or "Run workflow" with a run_id
        |
        v
  diagnose  ---> cargo build --release in 35. Deploy Doctor (from master)
                 ssh key from VDS_SSH_KEY + ssh-keyscan VDS_HOST
                 deploy-doctor diagnose --run-id N
                   per failed or deployed lesson: an LLM with read-only tools
                   (job logs; systemctl/journalctl/.env/curl on the VDS)
                        -> one issue per lesson, labeled deploy-doctor
```

- **Permissions**: `actions: read` (job logs), `issues: write`,
  `contents: read`, with the job's own `GITHUB_TOKEN`. No new secrets: it
  reuses `VDS_SSH_KEY`, `VDS_HOST`, `VDS_USER` and the `OPENAI_*` settings
  from the table above.
- **Read-only**: the probes are built by the doctor from the lesson number,
  never by the model, and `.env` values never reach the model. It runs
  exactly the commands listed in "Debugging a failed deploy" below.
- **Issues**: broken lessons get an issue (commented on by later runs while
  they stay broken), lesson 04's deliberate failure is recognized as a
  known issue, and an issue is closed when its lesson is healthy again.
- **Advisory**: what it finds never fails it. It fails only when it can't
  diagnose (no run, no model, can't post), and the issue is still posted
  from the run's own data.
- `workflow_run` only fires for workflow files on `master`. Like every
  workflow here, the file is changed by hand in GitHub's web editor; its
  full text is in `35. Deploy Doctor/README.md`.

## Debugging a failed deploy

- `GET`/`POST` to `http://<VDS_HOST>:<port>/` timing out or returning an
  empty reply from *your own machine* isn't necessarily real — sandboxed
  agent environments have been observed to flake on outbound requests to
  non-standard ports on arbitrary IPs while `nc -zv` still reports the TCP
  connection succeeding. Confirm from the VDS itself first
  (`curl localhost:<port>`, `systemctl --user status
  ai-advent-lesson-NN`, `journalctl --user -u ai-advent-lesson-NN`) before
  concluding the deployment is actually broken.
- If a value that should be in `.env` shows up empty (check with
  `cat -A ~/apps/lesson-NN/.env` on the VDS), the most likely cause is
  that the GitHub secret/variable referenced in the workflow doesn't
  actually exist under that exact name — GitHub returns an empty string
  for a missing secret rather than erroring, which is silent and easy to
  miss.
- Full Actions log downloads need authentication even on this public
  repo; run/job/step-level status (including which step failed) is
  available unauthenticated via
  `https://api.github.com/repos/CaptainDmitro/AI-Advent-Challenge/actions/runs/<id>/jobs`.

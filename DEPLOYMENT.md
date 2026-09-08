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
closes.

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

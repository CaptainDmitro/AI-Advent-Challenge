# Conventions

Rules every lesson follows. When in doubt, copy the newest lesson and change
it, rather than inventing a new pattern.

## Adding a new lesson

1. Create `NN. Title/` with the next number, zero-padded (`31. Developer
   Assistant`). The title is in English.
2. Add `Cargo.toml` (edition 2024; the crate `name` is a short kebab-case
   name, usually the title, e.g. `developer-assistant`; it is also the
   binary name the deploy step copies), `src/main.rs` and, for a web app, `static/index.html`.
3. Write the `README.md` from the template below.
4. Add a row to the table of contents in the root `README.md`.
5. Push to `master`. The pipeline finds the new `Cargo.toml` on its own: no
   workflow change is needed, and a web-app lesson becomes the deploy
   default as soon as it is the highest-numbered folder.

## Lesson README template

Every lesson README has these sections, in this order:

| Section | Content |
|---|---|
| `# NN. Title` | the folder name |
| `## Задание` | the original assignment text, in Russian, copied as is |
| `## Demo` | a video link, added by the author after recording |
| `## What this is` | what was built and how it works, with diagrams in code blocks |
| `## Run` | how to run it locally (`cargo run`, curl examples) |
| `## Configuration` | a table of environment variables: name, required, default |
| `## Deploy` | (web lessons) port and anything that lives on the VDS |
| `## Conclusion` | filled in after the demo; a placeholder until then |

## Code

- **Rust 2024 edition**, formatted with `rustfmt`. `cargo fmt --check` and
  `cargo clippy -- -D warnings` run in CI but do not block (older lessons
  predate them); new code should still pass both.
- **`reqwest` with `rustls`** and `default-features = false`, so the musl
  build is static.
- **Errors are strings for the UI.** Web lessons return errors as a display
  string inside a normal JSON response (HTTP 200), so one failed call never
  breaks the page. Lesson 30 is the deliberate exception: as a service, it
  uses real HTTP statuses.
- **The page is compiled in**: `include_str!("../static/index.html")`.
- **Files are written with write-then-rename**, so a reader never sees half a
  file (sessions, indexes, notes).
- **Paths from the outside are validated**: no `..`, no absolute paths, a
  file-name allowlist where it applies.
- **Tests need no network and no keys.** External APIs (the LLM, crates.io,
  GitHub, Ollama) are replaced by small axum routers on `127.0.0.1:0` that
  return fixed or scripted answers. Tests drive the real app over real HTTP.
- **Section banners** split long files:
  `// ----` lines around a one-line title.

## Environment variables

- LLM: `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` (default
  `https://api.deepseek.com`), `OPENAI_MODEL` (default `deepseek-v4-flash`).
- `PORT`: default `3000` locally; the deploy writes `4000 + NN`.
- Anything else is per lesson and listed in that lesson's Configuration
  table. Lesson 04 is the known exception to the `OPENAI_*` shape.
- An empty variable counts as unset (`env_non_empty`).

## Git

- One branch, `master`. Every push to `master` runs CI and deploys the newest
  web lesson.
- Commit messages: a short summary line (`Add lesson 31: Developer
  Assistant`, `Lesson 29: add the demo video`), a blank line, and a body that
  explains what and why.
- Files under `.github/workflows/` are changed by hand in GitHub's web editor,
  not by AI agents.

## Ports and services

| Lesson | Port | systemd user service | Folder on the VDS |
|---|---|---|---|
| `NN` | `4000 + NN` | `ai-advent-lesson-NN` | `~/apps/lesson-NN/` |

For example lesson 25 runs on port 4025 as `ai-advent-lesson-25`, and lesson
31 on port 4031 as `ai-advent-lesson-31`.

# 22. First RAG Query

## Задание

🔥 День 22. Первый RAG-запрос

Реализуйте функцию:

👉 вопрос → поиск релевантных чанков → объединение с вопросом → запрос к LLM

Сравните:

👉 ответ модели без RAG
👉 ответ модели с RAG

Усиление:

👉 составьте мини-набор из 10 контрольных вопросов по вашей базе
👉 для каждого вопроса зафиксируйте:

- ожидание (что должно быть в ответе)
- какие источники должны быть использованы (если применимо)

Результат:

Агент с двумя режимами (с RAG / без RAG) + 10 контрольных вопросов и сравнение качества

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

An agent that answers questions about **this repository** in two modes,
over the document index from [lesson 21](../21.%20Document%20Indexing):

```
                    ┌─────────────── without RAG ───────────────┐
 question ──────────┤                                           ├──▶ LLM ──▶ answer
                    └─▶ search the index ─▶ top-k chunks ─▶ merge ┘
                        (embed the question,  [1] DEPLOYMENT.md — …
                         cosine similarity)   [2] AGENTS.md — …
                                              Question: …
```

### The base

The same built-in corpus as lesson 21 plus lesson 21's own README: every
lesson's README from 01 to 21, the root `README.md`, `AGENTS.md`,
`DEPLOYMENT.md`, and the source of lessons 06, 16 and 20. That is 27 files,
compiled into the binary with `include_str!`. On start they are cut with
lesson 21's **structural** strategy, which lesson 21 found keeps a chunk on
one topic. Each Markdown section or top-level Rust item becomes one chunk,
sections under 300 characters are joined, and sections over 1800 are split
at paragraphs. The chunks are then embedded. By default this uses the local
hashed TF-IDF embedder, which needs no network, so the index is built in
memory in milliseconds.

### The pipeline

`Agent::ask(question, mode, k)` in [`src/main.rs`](./src/main.rs):

1. **Search.** RAG mode only. The question is embedded the same way as the
   chunks, and the `k` closest chunks by cosine similarity are kept. The
   default is 5.
2. **Merge.** `build_messages` puts the chunks into the user message as
   numbered fragments, each with its file and section, followed by
   `Question: …`. The system prompt gains one rule: answer only from the
   fragments, cite them as `[n]`, and say so if they don't cover the
   question.
3. **Ask.** One `/chat/completions` call.

Both modes get the same base system prompt. It names the repository and
asks the model to say "I don't know" instead of guessing. Both run at
`temperature: 0`. The only difference between the two answers is therefore
the retrieved context. The page shows the exact messages sent, so you can
see the merge.

### The 10 control questions

Each question records what a correct answer must say and which document
section it comes from. The expected facts are also written as checks: each
fact passes if the answer contains any of its variants, ignoring case. No
variant appears in the question itself, so an answer that only restates the
question scores nothing.

| # | Question | Expected answer | Sources |
|---|---|---|---|
| 01 | Which port and which systemd service name does a deployed lesson get on the VDS? | Port 4000 + the lesson number (lesson 6 → 4006); the service is `ai-advent-lesson-NN`, a user-level systemd unit. | `DEPLOYMENT.md` › Where things live on the VDS |
| 02 | Why is each lesson deployed as a static musl binary instead of a Docker container? | `x86_64-unknown-linux-musl` is fully statically linked: no glibc version dependency, no OpenSSL (reqwest uses rustls), and no Docker needed on the VDS. | `DEPLOYMENT.md` › Why a static musl binary instead of Docker |
| 03 | What one-time command must be run on the VDS so the lesson services keep running after the SSH session closes? | `sudo loginctl enable-linger <ssh-user>`, once per VDS. Without it, user services stop when the SSH session that started them closes. | `DEPLOYMENT.md` › Where things live on the VDS |
| 04 | Why does the deploy job of lesson 04 fail? | Lesson 04 needs `MEDIUM_*` and `STRONG_*` (and optionally `WEAK_*`) variables that the pipeline doesn't supply. It is left failing on purpose. | `DEPLOYMENT.md` › Secrets & variables<br>`AGENTS.md` › CI/CD pipeline |
| 05 | Which lessons get deployed on a normal push to master, and how do you redeploy an older lesson? | Only the highest-numbered lesson folder, and only if it depends on axum. An older lesson is redeployed with a manual "Run workflow" (`workflow_dispatch`) and the `lessons` input, e.g. `06`, `03,06` or `all`. | `DEPLOYMENT.md` › Which lessons deploy |
| 06 | Do cargo fmt and cargo clippy block the build-test job in CI, and why? | No. They run with `continue-on-error`, because lessons 02–05 predate the pipeline and were never run through rustfmt. `cargo build` and `cargo test` are the real gate. | `AGENTS.md` › CI/CD pipeline |
| 07 | How does the local embedder in the document indexing lesson turn text into a vector? | A hashed TF-IDF vector of 1024 dimensions. Words are lowercased and cut to their first 5 characters, then stems and stem pairs are hashed into signed buckets. The vector is log-scaled, weighted by IDF and normalized. | `21. Document Indexing/README.md` › Embeddings |
| 08 | Which two chunking strategies does the document indexing lesson compare, and what sizes do they use by default? | Fixed windows of 1000 characters with 150 overlap, and structural chunks: one per section, joined under 300 characters and split over 1800. | `21. Document Indexing/README.md` › Chunking: two strategies |
| 09 | In the MCP orchestration lesson, how are tools from several MCP servers named, and how is a tool call routed to the right server? | Every tool is shown as `<server>__<tool>` (e.g. `crates__crate_info`). The router splits the name on `__`, finds the connected server and checks its `tools/list` before sending `tools/call`. | `20. MCP Orchestration/README.md` › Choosing the tool and routing the call |
| 10 | In the invariant guardrails lesson, what are the two layers of enforcement of an invariant? | A deterministic check of `forbidden_terms` in the user's message that refuses before the model is called, and every invariant injected as a system message into every request, telling the model to check and refuse. | `14. Invariant Guardrails/README.md` › Two layers of enforcement |

For every question the comparison reports:

- **facts without RAG / with RAG**: how many of the expected facts each
  answer states;
- **source retrieved**: whether a chunk from the expected file is among the
  `k` retrieved chunks. This checks retrieval on its own, separately from
  the model;
- **cited**: whether the RAG answer cites (`[n]`) a chunk from the expected
  file. This shows the answer actually came from the right place.

The page has a free-form question box and the table of control questions.
**Run all 10** fills the table row by row, and clicking a row shows both
answers side by side. Each answer has its facts marked ✓/✗, the retrieved
chunks (the expected ones outlined), and the prompt that was sent. The same
comparison runs from the command line with `cargo run -- eval`, which prints
a Markdown table and every answer.

### Verification

`cargo test` needs no network or API key:

- `control_questions_are_grounded_in_the_corpus`: every expected source file
  and section heading exists, every expected fact is in its source, and no
  fact is given away by the question.
- `retrieval_finds_the_expected_source_for_every_control_question`: with the
  local embedder and `k = 5`, all 10 questions retrieve a chunk from their
  expected file.
- `rag_merges_chunks_with_the_question_and_beats_plain_on_the_control_set`:
  this test uses a fake model that knows nothing about the repo. It echoes
  the context back when there is one and says "I don't know" otherwise. The
  test checks the exact prompt: the RAG rules in the system prompt, every
  retrieved chunk under its `[n] file — section` header, and the question
  last. It also checks temperature 0 and that without RAG the score is 0
  while with RAG it is ≥ 70% and every answer cites its expected source.
- `http_api_answers_in_both_modes_and_reports_errors`: `/api/config`,
  `/api/cases`, `/api/ask` with a control question, a free question, one
  mode, an empty question, an unknown case, and a model returning 404 (the
  error shows on that answer and the retrieved chunks are still returned).
- `citations_are_parsed`, plus the index's own tests: every lesson's README
  is in the corpus, chunk ids are unique, and the musl question finds its
  section.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

The control questions in both modes, printed as a table:

```bash
OPENAI_API_KEY=sk-... cargo run -- eval
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"case_id": "q03"}' | jq '.answers[] | {mode, text, score: .score | {found, total, source_retrieved, cited}}'

curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "What does lesson 05 compare?", "mode": "rag", "k": 3}' | jq '.answers[0] | {text, sources: [.sources[] | {rank, score, source, section}]}'
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `EMBEDDER` | no | `local`: hashed TF-IDF in-process. `openai`: an OpenAI-compatible `/embeddings` endpoint |
| `EMBEDDING_API_KEY` | with `openai` | — |
| `EMBEDDING_BASE_URL` | no | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `PORT` | no | `3000` |

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4022`
with the `OPENAI_*` secrets the pipeline already supplies, and with the local
embedder.

## Conclusion

_(to fill in after recording the demo)_

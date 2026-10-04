# 23. Reranking and Filtering

## Задание

🔥 День 23. Реранкинг и фильтрация

Добавьте второй этап после поиска:

👉 reranker или фильтр релевантности (порог similarity / отдельная модель / heuristic)

Настройте:

👉 порог отсечения нерелевантных результатов
👉 топ-K до и после фильтрации

Сравните:

👉 качество без фильтра/rewriting
👉 качество с фильтром

Результат:

Улучшенный RAG: фильтрация/реранкинг + query rewrite + сравнение режимов

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

Lesson 22's RAG agent with a second stage after the vector search, and a
query rewrite before it. All three modes run side by side on the same
question:

```
 basic     question ──────────────▶ top-5 by cosine ─────────────────────────────────────────▶ LLM
 filtered  question ──────────────▶ top-20 by cosine ─▶ rerank ─▶ score ≥ 0.30 ─▶ top-5 ──────▶ LLM
 improved  question ─▶ rewrite ───▶ top-20 by cosine ─▶ rerank ─▶ score ≥ 0.30 ─▶ top-5 ──────▶ LLM
                                                                      │
                                                     nothing left ────┴─▶ "the documents don't
                                                                          cover this" (no LLM call)
```

The base and the index are lesson 22's: 27 files of this repository (every
lesson README up to 21, `AGENTS.md`, `DEPLOYMENT.md`, three lessons' source),
structural chunks, and the local hashed TF-IDF embedder. Lesson 22's own
README is left out, because it lists the control questions next to their
answers.

### Stage 1: the vector search, wider

Filtered and improved modes take **top-K before = 20** chunks by cosine
similarity, four times what goes into the prompt. Lesson 22 showed that the
right section is often in the index's top 20 but not its top 5. For q04 it is
at cosine rank 16, for q08 at 12.

### Stage 2: rerank and filter

`src/rerank.rs` scores every candidate again, in-process, from three signals,
each 0..1:

| Signal | What it measures | Weight |
|---|---|---|
| cosine | the vector search's similarity, divided by 0.5 and capped at 1 (with the local embedder even the best chunk rarely goes above 0.5) | 0.3 |
| coverage | the share of the query's terms found in the chunk, weighted by IDF. Terms are the same 5-character stems the embedder uses. Stems found in over a quarter of all chunks ("lesson", "the") are ignored. | 0.5 |
| heading | the same share, in the chunk's file name and section heading only | 0.2 |

Cosine similarity of hashed vectors is fuzzy: a chunk can be close because it
shares many common words. Coverage asks whether the chunk contains the rare
words the question is actually about, like "linger", "musl" or "forbidden".
The heading signal favours the section that is named after the topic.

Then the filter: candidates are sorted by score. Any candidate under the
**threshold (0.30)** is dropped, and of the rest at most **top-K after = 5**
go into the prompt. This means the prompt can hold fewer than 5 chunks, or
none. When none are left, the agent doesn't call the model. It answers that
the documents don't cover the question, in Russian if the question is in
Russian.

There is also an **LLM reranker** (`reranker: llm` on the page, `eval llm`
on the command line). One extra call shows the model the question and a
700-character preview of every candidate. The model answers
`{"scores": [...]}`, 0–10 per fragment. Its scores, divided by 10, replace the
heuristic's, and the same threshold applies (≥ 3/10 to stay). If that call
fails or returns something that isn't one score per fragment, the heuristic's
scores are used, and the answer carries a note saying so.

### The query rewrite

In improved mode the LLM first turns the question into a search query. The
prompt asks for an English query, because the documents are in English. It
asks to keep every name, number and identifier, and to name the lesson when
it's clear which one is meant. It also lists the 27 documents, so "the
document indexing lesson" can become "21. Document Indexing". The search and
the reranker use the rewrite. The answer is still given to the original
question.

### Choosing the weights and the threshold

There is no Rust toolchain on the machine this lesson was written on, so the
calibration ran on a Python port of lesson 22's chunker and embedder. Its
output matches lesson 22's own test, and the Rust tests below check the same
numbers. A grid search over weights and thresholds on the control set picked
0.3 / 0.5 / 0.2 and 0.30. That threshold falls in the gap between:

- the **weakest relevant section kept**: 0.31 (r02 after its rewrite);
- the **strongest out-of-scope candidate**: 0.27 (n03 after its rewrite).

The gap is narrow, and it belongs to this embedder. With `EMBEDDER=openai`
cosine similarities sit higher (often 0.3–0.6), so the threshold needs to be
set again. The page has a slider for it.

### The control set

Lesson 22's 10 questions, plus 8 that probe what the second stage is for:

- **r01–r03**: lesson 22 questions asked in Russian. Against English
  documents the local embedder finds nothing without a rewrite.
- **v01**: "why no docker?", a vague question.
- **n01–n04**: out of scope (the capital of Australia, Diesel migrations, the
  weather in Moscow, Kubernetes). The correct answer says the documents don't
  cover them. For these the fact check is that refusal, and the prompt should
  get 0 chunks.

For every answer the comparison reports: facts stated (as in lesson 22),
whether the expected **section** is in the prompt, **precision** (the share
of prompt chunks from an expected file), and the prompt's **chunk and token
count**.

### What the stages do, measured without an LLM

`cargo test` runs all 18 questions through all three modes against a fake
model. The fake rewrites with a fixed table (what a good rewriter would
answer) and "answers" by echoing the prompt's context back. This is a reader
that trusts whatever it's given, so the score measures retrieval alone:

| Mode | Facts stated | Expected section in prompt | Out of scope, said so | Chunks in prompts |
|---|---|---|---|---|
| basic (lesson 22) | 60% | 8/14 | 0/4 | 90 |
| filtered | 83% | 10/14 | 4/4 | 46 |
| rewrite + filtered | **100%** | **13/14** | **4/4** | 56 |

Per question:

| # | basic | filtered | rewrite + filtered |
|---|---|---|---|
| q01–q02, q05, q07, q09–q10 | all facts · 5 chunks | all facts · 5 chunks (reordered) | same |
| q03 | 1/1 · 5 chunks | 1/1 · **1 chunk** | 1/1 · 1 chunk |
| q04 | 2/2 · 5 chunks | 2/2 · **1 chunk** | 2/2 · 1 chunk |
| q06 | 2/2 · 5 chunks | 2/2 · 4 chunks | 2/2 · 4 chunks |
| q08 | 4/5 · 5 chunks | **5/5** · 5 chunks (the section moves from cosine rank 12 into the top 5) | 5/5 |
| r01–r03 | 0 facts · 5 unrelated chunks | 0 facts · 0 chunks: "not covered" | **all facts** (1–5 chunks) |
| v01 | 2/2 · 5 chunks | 2/2 · 5 chunks | 2/2 · 5 chunks |
| n01–n04 | 5 unrelated chunks, repeated as an answer | **0 chunks, "not covered"**, no LLM call | same |

The filter alone shrinks the context and makes out-of-scope questions safe.
Without a rewrite, though, it also throws away everything for Russian
questions, which were noise in basic mode anyway. The rewrite then brings
those questions back.

With a real model, run `cargo run -- eval` (heuristic) or
`cargo run -- eval llm` (LLM reranker) to get the same table from real
answers.

### Verification

`cargo test` needs no network or API key:

- `filtering_and_rewriting_beat_basic_rag_on_the_control_set`: the run above.
  It asserts that facts go basic < filtered < improved with improved ≥ 95%,
  and that improved has more expected sections, higher precision, and fewer
  chunks and tokens than basic. Every in-scope question retrieves its file in
  improved mode. Every out-of-scope question gets 0 chunks, no model call and
  a "not covered" answer in filtered and improved, while basic repeats its
  5 unrelated chunks. Filtered keeps nothing for the Russian questions and
  improved rewrites them. The kept candidates are exactly the prompt's
  fragments, renumbered, all at or above the threshold. The prompt has the
  same rules in every mode.
- `llm_reranker_scores_every_candidate_and_its_scores_decide`: a fake
  reranker scores DEPLOYMENT.md 9 and anything else 1. Only DEPLOYMENT.md
  chunks are kept. When the reranker can't be reached, the heuristic decides
  and the note says so.
- `the_heuristic_prefers_the_chunk_that_names_what_is_asked`: common stems
  are ignored. The `enable-linger` section wins on coverage.
- `select_orders_by_score_and_applies_threshold_then_top_k`,
  `settings_are_clamped`, `citations_rewrites_and_rerank_scores_are_parsed`.
- `control_questions_are_grounded_in_the_corpus`: every expected section and
  fact is in its source file, no fact is given away by its question, and
  out-of-scope questions check for a refusal.
- `http_api_compares_the_modes_and_reports_errors`: config, cases, all three
  modes in order, the filter keeping fewer chunks than basic, a Russian
  question rewritten, clamped settings, a free question, the "not covered"
  answer, an empty question, and an unknown case.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000. Pick a control question or type one. The
three answers appear side by side. The filtered and improved cards list every
candidate with its cosine rank, coverage, heading match, score (the red mark
is the threshold) and why it was dropped. Change top-K before, top-K after,
the threshold or the reranker, then press **Run all** to see how the totals
move.

The control set in all three modes, as a table:

```bash
OPENAI_API_KEY=sk-... cargo run -- eval
```

```bash
OPENAI_API_KEY=sk-... cargo run -- eval llm
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"case_id": "r02"}' | jq '.answers[] | {mode, query, kept: .score.kept, facts: "\(.score.found)/\(.score.total)", text}'

curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "Why is lesson 04 not deployed?", "modes": ["filtered"], "k_before": 30, "k_after": 3, "threshold": 0.4, "reranker": "llm"}' \
  | jq '.answers[0].candidates[] | {rank, kept, score, llm, coverage, source, section}'
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `EMBEDDER` | no | `local`: hashed TF-IDF in-process. `openai`: an OpenAI-compatible `/embeddings` endpoint (the threshold needs recalibrating) |
| `EMBEDDING_API_KEY` | with `openai` | — |
| `EMBEDDING_BASE_URL` | no | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `PORT` | no | `3000` |

Top-K before (default 20, up to 40), top-K after (default 5, up to 10), the
threshold (default 0.30) and the reranker (`heuristic` or `llm`) are set per
request, on the page or in the `/api/ask` body.

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4023`
with the `OPENAI_*` secrets the pipeline already supplies, and with the local
embedder.

## Conclusion

_(to fill in after recording the demo)_

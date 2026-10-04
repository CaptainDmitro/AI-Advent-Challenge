# 24. Citations and Sources

## Задание

🔥 День 24. Цитаты, источники и анти-галлюцинации

Доработайте RAG так, чтобы модель обязательно возвращала:

👉 ответ
👉 список источников (source + section/chunk_id)
👉 цитаты (фрагменты из найденных чанков)

Проверьте на 10 вопросах:

👉 есть ли источники в каждом ответе
👉 есть ли цитаты в каждом ответе
👉 совпадает ли смысл ответа с цитатами

Усиление:

👉 добавьте правило: если релевантность ниже порога — ассистент обязан сказать “не знаю” и попросить уточнение

Результат:

Ответы с обязательными источниками и цитатами + режим “не знаю” при слабом контексте

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

Lesson 23's filtered RAG (same index, same heuristic reranker, same 0.30
threshold), with an answer that has to prove itself:

```
question ─▶ top-20 by cosine ─▶ rerank ─▶ best score < 0.30? ──yes──▶ "I don't know" + please clarify   (no LLM call)
                                              │ no
                                              ▼
                              top-5 ─▶ LLM: {answer, sources, quotes}
                                              │
                         check every quote word for word in its chunk ──fails──▶ retry once with the errors
                                              │                                        │ fails again
                                              │ ok                                     ▼
                                              ▼                         "I don't know" + please clarify
                     LLM judge: does the answer mean what its quotes say?
```

### The answer format

The model has to reply with one JSON object (`response_format: json_object`):

```json
{
  "status": "answered",
  "answer": "Port 4000 + the lesson number [1]; the unit is ai-advent-lesson-NN [1].",
  "sources": [1],
  "quotes": [{ "fragment": 1, "text": "Port is always `4000 + NN` (lesson 6 → `4006`)." }]
}
```

or, when the fragments don't answer the question,
`{"status": "unknown", "clarification": "..."}`.

The server resolves each fragment number to the chunk in the prompt, so the
API returns:

- `text`: the answer with `[n]` markers;
- `sources`: `n`, `source` (file), `section` (heading path), `chunk_id`
  (`<file>#<ordinal>`) and the reranker's score;
- `quotes`: `n`, `chunk_id`, `source`, `section` and the quoted `text`.

### Anti-hallucination checks

`check_reply` accepts a reply only when all of these hold:

| Check | Rejected when |
|---|---|
| it's the JSON object | there's no `{...}` or it doesn't parse |
| sources | none, or a number that isn't a fragment of the prompt |
| quotes | none, a quote under 12 characters, or a quote from a fragment that isn't in `sources` |
| **word for word** | the quote isn't in its chunk. Case, whitespace, Markdown backticks, asterisks and table pipes, and typographic quotes are ignored; `...` may skip text, with the parts in order |
| every source is quoted | a source has no quote |
| citations | the answer cites `[n]` that isn't in `sources` |

A rejected reply gets one retry. The model gets its reply back with the
list of problems. If the second reply fails too, the answer isn't shown:
the assistant says "I don't know for sure" and asks for a clarification.

### Does the meaning match the quotes?

Quotes copied word for word can still be attached to an answer that says
something else. Two checks look at that:

- **LLM judge** (on by default, `judge: false` turns it off): one call gets
  the question, the answer and its quotes, but not the chunks. It replies
  `supported`, `partial` or `unsupported`, and lists the statements no quote
  backs.
- **Deterministic**: `support` is the share of the answer's content terms
  (stems that are rare in the corpus) that occur in its quotes.
  `unsupported_terms` lists the numbers and identifiers of the answer
  (`4006`, `x86_64`, `enable-linger`) that no quote contains. Without the
  judge, an answer is grounded when support ≥ 0.6 and nothing is unquoted.

`grounding.grounded` is the result: the judge's `supported`, or the
deterministic rule when the judge is off.

### "I don't know" mode

There are three ways to get it, and each ends with a request to clarify. A
Russian question gets its answer in Russian.

| `unknown_reason` | When | Model called? |
|---|---|---|
| `low_relevance` | no reranked chunk reaches the threshold (0.30): the context is too weak to answer from | no |
| `model_unsure` | the model says the fragments don't contain the answer; its own clarifying question is shown | yes |
| `unverified` | neither reply could be backed by verbatim quotes | yes, twice |

For example:

> Не знаю: в документах нет достаточно релевантного фрагмента (лучшая
> релевантность 0.21, порог 0.30). Уточните, пожалуйста, вопрос: о каком
> уроке, файле или части пайплайна деплоя идёт речь?

The nearest three sections are returned as `hints`, so the user can see what
the documents do cover.

The threshold is lesson 23's calibration for the local embedder: on its
control set the weakest relevant section scores 0.31 and the strongest
out-of-scope candidate 0.27. The page has a slider. Moving it up turns an
answerable question into "I don't know".

### The control set

The task asks for 10 questions: they are lesson 22's ten, q01–q10, each with
its expected facts and expected section. Four out-of-scope questions, n01–n04
(the capital of Australia, Diesel migrations, the weather in Moscow in
Russian, Kubernetes), check the "I don't know" mode. For every answer the
check reports:

- **sources**: present, and each resolves to a prompt chunk;
- **quotes**: present, and each is word for word in its chunk;
- **meaning**: the judge's verdict, `support` and unquoted identifiers;
- **facts**: expected facts stated, and how many of them a quote also contains;
- **status**: answered in scope, "I don't know" out of scope.

### Verification

`cargo test` needs no network or API key. The tests run against a fake model
that plays several roles:

- `every_answer_has_sources_and_verified_quotes_or_says_it_does_not_know`:
  all 14 control questions through the real index and reranker. The fake
  quotes lines of fragments [1] and [2] and answers with just those quotes.
  Every one of q01–q10 is answered on the first attempt with sources and
  quotes. Each source and quote resolves to its prompt chunk by number,
  `chunk_id`, file and section. Every quote is found word for word, the judge
  says `supported`, and nothing is unquoted. Each of n01–n04 gets
  `low_relevance`, a best score under 0.30, a clarification request and
  3 hints, without a model call. The model sees exactly 20 requests: an
  answer and a judge call for each in-scope question.
- `a_rejected_reply_is_retried_and_an_unverifiable_one_becomes_dont_know`:
  - a model that forgets its quotes is told so and fixes them on the retry;
  - a model that makes a quote up is rejected twice. The answer becomes
    `unverified` "I don't know", without its invented content and without a
    judge call.
  - a model that says `unknown` gets `model_unsure` with its own
    clarification;
  - with the judge off, the deterministic check decides.
- `replies_are_checked_against_their_fragments`: every rejection rule, on
  real chunks.
- `quotes_are_matched_word_for_word`: normalization, ellipses, paraphrases and
  parts out of order are rejected. Also citations and identifiers.
- `http_api_returns_answers_with_sources_and_quotes`: the page, config,
  cases, an answered question with `chunk_id`/`source`/`section`, a Russian
  "Не знаю", a 0.99 threshold turning q03 into "I don't know", clamped
  settings, and errors.
- `dont_know_judgements_and_settings`, `select_orders_by_score_and_applies_threshold_then_top_k`,
  `control_questions_are_grounded_in_the_corpus`.

With a real model, `cargo run -- eval` prints the same checks for all 14
questions, then every answer with its sources, quotes, rejected attempts and
the judge's objections.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000. Click a control question or type one. The
answer card shows the status, the answer with clickable `[n]`, the checks
(sources, quotes, word for word, judge, terms in quotes, facts), the sources
table with `chunk_id`, the quotes, and any rejected replies. The candidates
are shown against the threshold, together with the prompt and raw replies.
**Run all** fills the control table and its totals.

The control set as a table:

```bash
OPENAI_API_KEY=sk-... cargo run -- eval
```

```bash
OPENAI_API_KEY=sk-... cargo run -- eval nojudge
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"case_id": "q01"}' | jq '.answer | {status, text, sources, quotes, grounding}'

curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"question": "Как настроить Kubernetes?"}' | jq '.answer | {status, unknown_reason, text, hints}'
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
threshold (default 0.30) and the judge (default on) are set per request, on
the page or in the `/api/ask` body.

The model has to support `response_format: {"type": "json_object"}`
(DeepSeek and OpenAI do).

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4024`
with the `OPENAI_*` secrets the pipeline already supplies, and with the local
embedder.

## Conclusion

_(to fill in after recording the demo)_

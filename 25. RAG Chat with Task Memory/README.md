# 25. RAG Chat with Task Memory

## Задание

🔥 День 25. Мини-чат с RAG + памятью (production-like)

Реализуйте мини-чат (CLI/веб), который:

👉 хранит историю диалога
👉 при каждом новом вопросе ищет контекст в базе через RAG
👉 отвечает с учётом найденной информации
👉 всегда выводит источники

Усиление:

👉 добавьте “память задачи” (task state):

- что пользователь уже уточнил
- какие ограничения/термины зафиксированы
- что является целью диалога

Проверьте:

👉 на 2 длинных сценариях по 10–15 сообщений
👉 что ассистент не теряет цель и продолжает выдавать ответы с источниками

Результат:

Мини-чат с RAG + источниками + памятью задачи

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

A chat, in the browser and in the terminal, on top of lesson 24's RAG: the
same index of this repo's docs, the same heuristic reranker and 0.30
relevance gate, and the same answer format whose quotes are checked word for
word. What's new is the conversation around it.

```
message ─▶ task state update (1 LLM call) ──▶ goal / clarified / constraints / terms   (merged by rules)
                      │
                      └─▶ self-contained search query ─▶ top-20 ─▶ rerank ─▶ ≥ 0.30?
                                                                              │ no ─▶ "I don't know" + nearest sections
                                                                              │       + "how does this relate to our goal?"
                                                                              ▼ yes
              system: rules + TASK STATE  ·  last 6 turns  ·  fragments + message ─▶ LLM ─▶ answer, sources, quotes
                                                                                           (verified, 1 retry)
```

### Dialogue history

A session is the task state plus a list of turns. Each turn stores the user's
message, the search query, the answer, its sources (`chunk_id`, file, section,
score) and quotes, or the "I don't know" with the nearest sections, and what
the turn changed in the task state. Sessions are kept in memory and written to
`sessions/<id>.json` after every turn. The file is written to a temporary file
first and then renamed, so a restart (or a redeploy) continues the
conversation. The page remembers its session id in `localStorage`. The
terminal chat uses a session named `cli` by default.

The last **6 turns** go into every answer request as real `user`/`assistant`
messages, with the `[n]` markers stripped, because they pointed at fragments
of an earlier prompt. Anything older reaches the model only through the task
state. So by message 12, message 1 is no longer sent, and the goal it set is
still there.

### Task state

```json
{
  "goal": "Добавить в репозиторий новый урок и задеплоить его на VDS через существующий пайплайн",
  "clarified": ["Урок — веб-приложение на axum", "Номер урока — 25"],
  "constraints": ["Без Docker на VDS", "У деплой-ключа нет sudo"],
  "terms": [{ "term": "урок", "meaning": "папка вида `NN. Title` с отдельным Cargo-крейтом" }]
}
```

One LLM call per message (`STATE_PROMPT`) gets the current state, the last 3
turns and the new message. It returns only **what this message changes**: the
goal, the new clarifications, constraints and terms, the items the user has
withdrawn, and the message rewritten as a self-contained English search query.
"Which port will **it** get?" becomes "Which port and systemd service name
will the deployed lesson 25 get on the VDS?". The model proposes; the merge
is done in code (`TaskState::apply`, `src/task.rs`), by rules the model can't
break:

| Rule | So that |
|---|---|
| The goal is set by the first message that has one, and changes only when the update says `goal_changed: true` | an aside, an off-topic question or a reworded goal can't replace it |
| An empty or missing field changes nothing | a reply that forgets to repeat the state doesn't erase it |
| Lists only grow; a repeat (any case, spacing, end punctuation) isn't added again | the state doesn't lose an early constraint or fill with duplicates |
| An item leaves only by an explicit `remove`, matched by its text | "not X, but Y" replaces X in one turn |
| A term with a new meaning replaces the old one | the latest definition wins |
| At most 20 items per list, the oldest goes | the prompt stays bounded |
| A failed or unparsable update leaves the state as it was, and the message itself is searched | one bad reply never breaks a turn |

The state goes into the system prompt of every answer as a `TASK STATE` block,
with rules to answer in the light of the goal, read "it" or "my lesson"
through the clarifications, never suggest anything that breaks a constraint
(and say so when the documents conflict with one), use terms as defined, and
on a recap restate the state. Facts taken from the state need no quote. Facts
taken from the documents do.

The page shows the state live, with what the last turn added marked **new**.
Each answer shows chips such as `+ constraint: Без Docker на VDS`.

### Sources, always

Every turn ends with its sources:

- **answered**: the fragments the answer cites, each with `chunk_id`, file,
  section and score. They are checked as in lesson 24: at least one source,
  a quote from each, every quote word for word in its chunk, no citation
  outside the sources, and one retry with the list of problems.
- **"I don't know"**: when nothing reaches the threshold, the model isn't
  called for an answer. The turn shows the nearest three sections with the
  best score. The clarification request names the goal, for example: «Как это
  связано с нашей целью — «Добавить в репозиторий новый урок и задеплоить
  его…»? Уточните, пожалуйста, или вернёмся к ней.» An aside doesn't derail
  the conversation, and the state isn't touched by it.

### The two scenarios

`src/scenarios.rs` holds two dialogues of 12 messages each. Both follow the
same pattern: the goal is set in message 1, clarifications, constraints and a
term come in along the way, follow-ups only make sense with what came before,
there is one off-topic aside, and the last message asks for a recap.

| | **deploy** (Russian) | **memory** (English) |
|---|---|---|
| goal | add a new lesson and deploy it to the VDS | choose a context-management approach for a long support chatbot |
| clarified | an axum web app, number 25 | must remember customer preferences across sessions |
| constraints | no Docker on the VDS, no sudo for the deploy key | at most one extra LLM call per turn |
| term | «урок» = an `NN. Title` folder with its own crate | "memory" = only what's injected into the prompt |
| needs the state | "На каком порту и под каким именем сервиса **он** поднимется?" → 4025, `ai-advent-lesson-25` | "Given **my constraint**, which approach fits best?" → sticky facts |
| aside | the weather in Moscow tomorrow | the capital of Australia |
| recap | «Подведи итог: какая у нас цель…» | "Sum up: what is my goal…" |

Each message carries what a good turn looks like: the search query a good
state update would write, the notes it adds to the state (with keywords to
look for), the files its sources should come from, and the facts the answer
should state. Each turn is checked for:

- **status**: answered in scope, "I don't know" for the aside;
- **sources shown**: verified sources and quotes, or the nearest sections;
- **expected file**: a source comes from the file that has the answer;
- **goal kept**: after the turn the goal is set and still about the scenario;
- **noted in state**: this message's clarification, constraint or term is in
  the state;
- **facts**: e.g. `4025`, `enable-linger`, `workflow_dispatch`, `sticky`.

### Verification

`cargo test` needs no network or API key. A fake model plays both roles. As
the state updater it returns each scenario message's notes and reference
query. As the answerer it quotes fragments [1] and [2] exactly.

- `both_long_scenarios_keep_the_goal_and_answer_with_sources`: both
  scenarios, 24 messages through the real index and reranker. Every in-scope
  message is answered on the first attempt with verified quotes, and a source
  comes from the expected file. Both asides get `low_relevance` with 3 nearest
  sections and a clarification that names the goal, and no answer call is
  made. After every turn the goal is exactly the one message 1 set, and every
  note survives to the end. The answer request carries the goal in its system
  prompt and at most 6 earlier turns. By the last turn, message 1 is no longer
  in the request.
- `failures_are_retried_or_end_in_dont_know_and_a_broken_update_keeps_the_state`:
  a reply without quotes is fixed on the retry. An invented quote twice ends in
  `unverified` "I don't know" that names the goal. The model's own "unknown"
  is passed on. A state update that isn't JSON leaves the state untouched, and
  the raw message is searched and still answered.
- `sessions_survive_a_restart_and_ids_are_checked`: a session written by one
  store is read back by a new one from its file. Ids that could escape the
  directory are refused.
- `http_api_keeps_sessions_and_returns_sources_and_state`: create, chat by
  scenario step, chat by message, read back, a new session when no id is
  given, reset, and errors.
- `prompts_carry_the_state_and_only_the_recent_dialogue`, and in `task.rs`:
  the goal is set once and changes only on request, lists grow without repeats
  and shrink only by removal, the caps, and lenient parsing.

With a real model, `cargo run -- eval` runs both scenarios and prints a table
per scenario, the totals, the final task state and every answer with its
sources.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000 and chat, or click **▶ Деплой нового урока**
or **▶ Choosing a memory strategy**. Each scenario runs in a new session
message by message and fills a table of checks with totals. The right column
shows the task state and the last turn's insides: the candidates against the
threshold, the state update's raw reply, and the full answer request.

In the terminal:

```bash
OPENAI_API_KEY=sk-... cargo run -- chat
```

`/state` prints the task state, `/history` the dialogue, `/reset` clears both,
and `/quit` exits. `cargo run -- chat my-session` continues a named session.

The scenarios, checked:

```bash
OPENAI_API_KEY=sk-... cargo run -- eval
```

```bash
OPENAI_API_KEY=sk-... cargo run -- eval deploy
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/chat -H 'Content-Type: application/json' \
  -d '{"message": "Хочу задеплоить новый урок на VDS, без Docker. С чего начать?"}' \
  | jq '{session_id, text: .turn.text, sources: .turn.sources, state}'
```

| Endpoint | |
|---|---|
| `POST /api/chat` | `{session_id?, message}` or `{session_id?, scenario, step}`; returns the turn, the state, a `check` for scenario steps, and the `trace` |
| `POST /api/session` | a new session |
| `GET /api/session/{id}` | its state and turns |
| `POST /api/session/{id}/reset` | clears both, keeps the id |
| `GET /api/scenarios`, `GET /api/config` | |

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `CHAT_DIR` | no | `sessions`: where session files go, relative to the working directory. `off` keeps sessions in memory |
| `EMBEDDER` | no | `local`: hashed TF-IDF in-process. `openai`: an OpenAI-compatible `/embeddings` endpoint (the threshold needs recalibrating) |
| `EMBEDDING_API_KEY` | with `openai` | — |
| `EMBEDDING_BASE_URL` | no | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `PORT` | no | `3000` |

Top-K before (default 20) and after (default 5) and the threshold (default
0.30) can be set per request in the `/api/chat` body; the page has a slider
for the threshold. The model has to support
`response_format: {"type": "json_object"}` (DeepSeek and OpenAI do).

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4025`
with the `OPENAI_*` secrets the pipeline already supplies. Sessions are written
to `~/apps/lesson-25/sessions/`, the unit's working directory, so they survive
a redeploy.

## Conclusion

_(to fill in after recording the demo)_

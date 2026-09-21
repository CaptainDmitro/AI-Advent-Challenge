# 11. Agent Memory Model

## Задание

🔥 День 11. Модель памяти агента

Опишите и реализуйте модель памяти для агента

Разделите информацию минимум на 3 типа:

👉 краткосрочная (текущий диалог)
👉 рабочая (данные текущей задачи)
👉 долговременная (профиль, решения, знания)

Сделайте так, чтобы:

👉 разные типы памяти хранились отдельно
👉 вы явно выбирали, что и куда сохраняется

Проверьте:

👉 какие данные попадают в каждый слой
👉 как это влияет на ответы агента

Результат:

Агент с явной моделью памяти (memory layers)

Формат:

Видео + Код / Текст

## Demo

https://github.com/user-attachments/assets/d13275c4-9fc4-4b0d-9608-ce33dc67c603

## What this is

Three independent memory layers, each with its own struct, its own file on disk, and its own write rule - not one blob of "context" split by convention, but three things that are actually stored, gated, and injected separately in `src/main.rs`:

- **Short-term memory** (`ShortTermMemory`, `short_term.json`) - the live, ordered transcript of the current conversation. Its write rule is unconditional: every user/assistant turn is appended here regardless of what else is going on. This is also the *only* layer `/api/reset` ever touches - resetting the conversation doesn't make the agent forget who you are or what task it's mid-way through.
- **Working memory** (`WorkingMemory`, `working_memory.json`) - a scratchpad scoped to exactly one task (`task: Option<String>` + a `facts` map). It only exists while a task is active, is only ever written to while a task is active, and is thrown away (or selectively promoted - see below) the moment the task finishes.
- **Long-term memory** (`LongTermMemory`, `long_term.json`) - durable, cross-task, cross-session knowledge about the user: identity, standing preferences, decisions explicitly meant to stick. Nothing here is ever cleared by `/api/reset` or `/api/task/finish`; only an explicit `/api/memory/forget` call touches it.

### Why three separate files, not three fields in one JSON document

Lessons 07-10 all persisted everything to a single `state.json`. Here each layer gets its own file (`SHORT_TERM_FILE` / `WORKING_MEMORY_FILE` / `LONG_TERM_FILE`), loaded and saved independently (`load_or_default` / `save_json`, generic over the three structs). This isn't just aesthetic: a corrupted or hand-edited `working_memory.json` falls back to an empty working memory without touching `long_term.json` at all, which is exactly the property you'd want from memory systems that are conceptually unrelated - a bug in the task scratchpad should never be able to wipe out what the agent durably knows about you.

### Explicitly choosing what goes where

This is the actual point of the exercise, and it's implemented as **three narrowly-scoped LLM prompts, each with a single allowed destination**, rather than one classifier call that sorts everything at once:

1. `WORKING_MEMORY_PROMPT` runs only while a task is active. It's given the task, the current working facts, and the latest turn, and is explicitly told to extract only what's needed to finish *this* task - and explicitly told to leave out anything about the user as a person, "that belongs to a different memory system."
2. `LONG_TERM_MEMORY_PROMPT` runs on every turn, task or no task. It's told the opposite: only extract something if it would still be true and relevant in a *completely unrelated* task later - never a task-specific parameter.
3. `PROMOTION_PROMPT` runs exactly once, when a task finishes with promotion requested (`POST /api/task/finish {"promote": true}`, the default). It's the one explicit bridge between working and long-term: given the task's working facts (about to be discarded) and the current long-term facts, it decides which working facts are durable enough to survive and merges only those into long-term before working memory is cleared.

Because each prompt can only write to one destination, "where does this fact go" is never an implicit side effect of a general-purpose extractor - it's a property of *which prompt ran*, which is a property of *which layer's lifecycle event just happened* (a turn during an active task, a turn at all, or a task finishing).

On top of the automatic extractors, `POST /api/memory/remember {"layer", "key", "value"}` writes directly into `working` or `long_term`, bypassing the LLM entirely - the clearest possible demonstration that the destination is a deliberate choice, not something only a prompt gets to decide. (Short-term is deliberately excluded from `remember` - it's the raw dialogue, not a key-value store, and the API says so if you try.)

### How memory shapes what gets sent

`build_context` (pure, unit-tested) assembles one request as: system prompt → long-term block (only if non-empty) → working-memory block (only if a task is active) → the full short-term dialogue, unmodified. Unlike lesson 10, short-term is never trimmed by a sliding window here - the point of this lesson is the *layering*, not context compression, so keeping the dialogue itself simple avoids conflating the two concerns.

### Testing what lands where, and what it changes

Scenario used while building this: start a fresh agent, don't start a task yet, and open with *"I'm vegetarian and I prefer short, direct answers."* The long-term extractor picks this up (durable, identity-shaped) - `long_term_facts` now has `diet: vegetarian` and `communication_style: short and direct` - while working memory stays empty since no task is active yet.

Then `POST /api/task/start {"name": "Plan a birthday dinner"}` and continue: *"Budget is $150, six guests, Saturday evening."* These land in `working_facts` (`budget: $150`, `guests: 6`, `day: Saturday evening`) - the working-memory extractor was explicitly told to ignore anything about the user as a person, so it never touches the vegetarian/style facts, and the long-term extractor (still running every turn) correctly declines to treat a one-off budget number as durable.

Ask *"what have we decided so far?"* mid-task: the reply correctly reflects both layers - it respects the short, direct style from long-term *and* recites the budget/guest count from working memory, because `build_context` puts both blocks in front of the model every turn a task is active.

Finish the task (`POST /api/task/finish {"promote": true}`) and start an unrelated one - `POST /api/task/start {"name": "Plan a work meeting"}` - then ask *"what's my budget?"* The agent correctly has no idea; `working_facts` was cleared when the party task finished, and the budget was never durable enough for the promotion step to carry it into long-term. Ask *"any dietary notes for me?"* in the same new task, though, and the vegetarian fact is still there, still shaping the reply, still short and direct - because long-term memory was never task-scoped to begin with.

Turning working memory off entirely (never calling `/api/task/start`) reproduces something close to a plain chat agent with a small persistent profile: every turn still gets the long-term block, but nothing task-specific ever gets extracted or injected, and `/api/memory` shows `working_task: null` throughout.

### Token cost per layer

Every `/api/chat` and `/api/memory` response includes a `memory` / `report` object breaking `sent_context_tokens` down by source: `system_prompt_tokens`, `long_term_tokens`, `working_memory_tokens`, `short_term_tokens`, plus `actual` (the API's own `usage`, when available) and `estimated_cost_usd` when price env vars are set. Long-term and working blocks are typically tiny (a handful of short lines) next to the raw dialogue - the cost of this model isn't the injected memory, it's the two-to-three extra LLM calls per turn needed to keep it current (one working-memory update while a task is active, one long-term update always, one promotion call - but only once, when a task finishes).

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `SHORT_TERM_FILE` | no | `short_term.json` |
| `WORKING_MEMORY_FILE` | no | `working_memory.json` |
| `LONG_TERM_FILE` | no | `long_term.json` |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

Each file holds exactly one memory layer and is loaded/saved independently, so a restart resumes the conversation, the active task and its facts, and the durable profile all separately - and a problem reading any one of them only resets that layer.

## Deploy

Deployed automatically by GitHub Actions on every push to `master` (this becomes the highest-numbered lesson, so it becomes the new default deploy target - see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). Live at `http://<VDS host>:4011`.

## Conclusion

_(to fill in after recording the demo)_

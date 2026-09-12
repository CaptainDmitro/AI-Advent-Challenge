# 10. Context Management Strategies

## Задание

🔥 День 10. Управление контекстом: разные стратегии (без summary)

Реализуйте в агенте 3 разных стратегии управления контекстом (минимум), и переключатель между ними:

👉 Стратегия 1: Sliding Window

- храните только последние N сообщений
- всё остальное отбрасывайте

👉 Стратегия 2: Sticky Facts / Key-Value Memory

- введите отдельный блок "facts" (ключ-значение), который хранит важные данные из диалога

    (например: цель, ограничения, предпочтения, решения, договорённости)

- обновляйте facts после каждого сообщения пользователя
- в запрос отправляйте: facts + последние N сообщений

👉 Стратегия 3: Branching (ветки диалога)

- сохраните checkpoint в диалоге
- создайте 2 ветки от одного места
- продолжите диалог в каждой ветке независимо
- переключайтесь между ветками

Протестируйте на одном и том же сценарии (например: "собираем ТЗ 10-15 сообщений"):

👉 прогоните сценарий на каждой стратегии
👉 сравните ответы и поведение агента

Сравните:

👉 качество ответа
👉 стабильность (не теряет ли важные детали)
👉 расход токенов
👉 удобство для пользователя

Результат:

Агент с 3 стратегиями управления контекстом (Sliding Window / Facts / Branching) + сравнение результатов

Формат:

Видео + Код

## Demo

_(to be added after recording the demo)_

## What this is

[Lesson 09](../09.%20Context%20Compression)'s `Agent`, rebuilt around three independent, switchable context-management strategies - none of which ever produce an LLM-written summary of the conversation:

- **Sliding Window** - the system prompt plus only the last `KEEP_LAST_N_MESSAGES` raw messages. Everything older is simply dropped from the request; it isn't folded into anything, it's gone.
- **Sticky Facts / Key-Value Memory** - a small `facts` map (goal, constraints, preferences, decisions, agreements - whatever a later turn might need) that gets re-extracted by a dedicated LLM call *after every user message*, independent of which strategy is currently active. When this strategy is selected, the request is system prompt + one synthetic system message rendering the current facts + the last `KEEP_LAST_N_MESSAGES` raw messages.
- **Branching** - the active branch's entire raw history, sent unmodified. Its context-management story isn't per-turn trimming at all: you save a **checkpoint** at some point in the conversation, **fork** two (or more) branches from it, and continue each independently - so exploring "what if the client wants mobile instead of web" never pollutes the branch where you kept going with web, and switching back doesn't lose anything either side said.

A strategy switch (or an explicit `strategy` field on `/api/chat`) governs only *what gets sent this turn* - the raw, complete history of whichever branch is active is always kept in full and persisted to `state.json`, so switching strategies mid-conversation never discards anything, and neither does switching branches.

### Why sticky facts run on every turn, unconditionally

Lesson 09's summary only got refreshed once enough messages piled up past its window. Facts here are re-extracted after *every* user message regardless of the active strategy (`Agent::update_facts` in `src/main.rs`), so switching to Sticky Facts mid-conversation sees a memory that's already current rather than empty. It's a single extra LLM call per turn (`FACTS_SYSTEM_PROMPT` - given the current facts as JSON plus the latest turn, return the complete updated JSON object), best-effort like lesson 09's compaction: any failure just leaves the prior facts in place and gets retried on the next turn without ever blocking the chat reply.

### Why branching is a fork of message lists, not a summary

A `Branch` is just an id, a name, and its own `Vec<Message>` (the system prompt plus everything said on that branch). `create_checkpoint` records how many visible messages a branch held at a moment in time; `fork_branch` copies exactly that many messages (and no more) into a brand-new branch, which then evolves completely independently - pushing to one branch's history never touches another's. Forking never switches the active branch automatically, so creating two forks from the same checkpoint back-to-back doesn't have the second overwrite the view of the first; `switch_branch` is a separate, explicit step.

### Comparing the three on the same scenario

Run the same "gathering a spec" conversation (10-15 messages, gradually adding requirements, changing your mind partway through, mentioning a constraint early that only matters again at message 12) under each strategy, resetting between runs:

1. **Sliding Window** - once the conversation passes `KEEP_LAST_N_MESSAGES`, anything mentioned earlier is gone from what the model sees. Cheapest per turn by far, but a question that depends on an early detail (a constraint mentioned at message 2, asked about again at message 14) will typically get an "I don't have that information" - a real, visible failure of *this specific* mechanism, not a bug.
2. **Sticky Facts** - the same early constraint, if the extractor judged it worth keeping, survives as a compact fact and answers correctly even past the window - at the cost of one extra LLM call per turn and a small, fixed facts block added to every request regardless of how long the conversation gets.
3. **Branching** - nothing is ever dropped within a branch, so quality on a single branch is identical to sending the full history every time (same trade-off lesson 08 already had: it grows unbounded). The distinct win shows up when the scenario branches for real - e.g. requirements diverge into "web app" vs. "mobile app" partway through: checkpoint there, fork both, and each branch's answers stay consistent with *only* what was actually decided on that branch, with no cross-contamination and no re-typing.

### Comparing token cost

Every `/api/chat`, `/api/reset`, and `/api/history` response carries, in `tokens`:

- `uncompacted_context_tokens` - what this turn would cost with no strategy applied at all: the full active-branch history including the message you just sent. The common baseline all three are compared against.
- `sent_context_tokens` - what was actually sent this turn, under whichever strategy is active.
- `tokens_saved_this_turn` - the difference. Always `0` for Branching, since it sends the full history by construction - the saving there is architectural (smaller per-branch histories from not merging directions), not a per-turn number.
- `raw_history_tokens_after` - the active branch's complete, ever-growing transcript, regardless of strategy.
- `facts_count` / `facts_tokens` - how much the sticky facts memory currently holds, and roughly how many tokens that would cost if spliced in.
- `percent_of_limit` - `sent_context_tokens` (or the API's real `prompt_tokens`, when available) against `CONTEXT_LIMIT_TOKENS`, since that's what determines whether the *next* call risks rejection.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Comparing convenience for the user

1. **Sliding Window** - zero setup, nothing to manage; the trade-off only shows up as silently forgotten details, which the user has to notice on their own.
2. **Sticky Facts** - the facts panel makes the agent's memory visible and auditable (you can see exactly what it thinks it knows), but a wrong or dropped fact is a subtler failure mode than Sliding Window's - the model *looks* confident either way.
3. **Branching** - the most manual of the three: the user has to remember to checkpoint before a decision point and explicitly fork/switch. In exchange, it's the only strategy where you can go back and continue an abandoned direction without re-explaining anything.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `STATE_FILE` | no | `state.json` |
| `KEEP_LAST_N_MESSAGES` | no | `6` |
| `DEFAULT_STRATEGY` | no | `sliding_window` (`sliding_window` \| `sticky_facts` \| `branching`) |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

`STATE_FILE` holds every branch, checkpoint, the sticky facts map, and the current default strategy in one JSON document, so a restart resumes exactly where the conversation left off - including which branch was active. `DEFAULT_STRATEGY` only sets the starting point; the UI's strategy switch (and `/api/chat`'s `strategy` field) can always override it per turn or persistently via `/api/strategy`.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4010`.

## Conclusion

_(to fill in after recording the demo)_

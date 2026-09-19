# 13. Task State Machine

## Задание

🔥 День 13. Состояние задачи (Task State Machine)

Реализуйте состояние задачи как конечный автомат:

👉 этап задачи
👉 текущий шаг
👉 ожидаемое действие

Пример состояний:

👉 planning → execution → validation → done

Проверьте:

👉 паузу на любом этапе
👉 продолжение без повторных объяснений

Результат:

Агент с формализованным состоянием задачи

Формат:

Видео + Код / Текст

## Demo

_(added by the human after recording)_

## What this is

Lessons 11-12 gave working memory a task, but only as a bare `Option<String>` name - "a task is active" or it isn't, nothing in between. This lesson formalizes that into an actual finite state machine: a task now has a **stage** (`planning` → `execution` → `validation` → `done`), a free-text **step** describing progress within that stage, and a free-text **expected action** naming what the FSM is currently waiting on. Everything else from lesson 12 - the personalization profile, short-term dialogue, long-term memory - is unchanged.

### The state machine

```rust
enum TaskStage { Planning, Execution, Validation, Done }

struct TaskFsm {
    name: String,
    stage: TaskStage,
    step: String,
    expected_action: String,
    paused: bool,
    history: Vec<TaskEvent>,   // full audit trail: every start/advance/pause/resume
}
```

Stage transitions are validated against an explicit table, not a free-for-all:

```
planning   -> execution
execution  -> validation
validation -> execution (rework after a failed check) | done
done       -> (terminal - nothing advances out of it)
```

`Agent::advance_task` refuses any transition not on this table - `planning` can't jump straight to `done`, and once a task reaches `done` it can't be moved anywhere. The one backward edge (`validation` → `execution`) exists on purpose: a real validation step can fail and send work back for rework, so the FSM isn't a straight line.

Just as important: **nothing about stage transitions is ever inferred by the LLM.** The profile in lesson 12 is only ever written by an explicit API call, never guessed at by a model - the task FSM follows the same rule. `POST /api/task/advance`, `POST /api/task/step`, `POST /api/task/pause`/`resume`, and `POST /api/task/finish` are the *only* ways the state changes. This is what "formalized" means here: the state machine's behavior is 100% deterministic and testable, independent of what any model happens to say.

### Pause, at any stage

`paused` is a flag orthogonal to `stage`, not a stage of its own - a task can be paused while still in `planning`, mid-`execution`, mid-`validation`, or even after reaching `done` (before it's finished). Pausing freezes every state-mutating action:

- `advance_task`, `update_task_step`, and `finish_task` all refuse to run while paused, with a clear error telling the caller to resume first.
- The working-memory extractor (the LLM call that keeps task-scoped facts up to date after each chat turn) is skipped entirely while paused - nothing about the task changes on a turn where it's frozen.
- Explicit `remember()` calls into working memory are refused too, for the same reason.

Resuming clears the flag and changes nothing else - stage, step, expected action, and every fact captured so far come back exactly as they were.

### Continuation without repeated explanations

The whole point of formalizing this instead of just remembering "a task is active" is that **the full state rides along on every single chat turn**, not just the task's name. `build_context` (extended from lesson 12) injects a system block with the current stage, step, and expected action on every request - and while paused, it explicitly tells the model not to assume progress happened and not to push the task forward on its own. Concretely:

1. Start a task, advance it into `execution`, and set a step ("writing the auth middleware") and an expected action ("waiting for the user to review the diff").
2. Pause it. Have an unrelated conversation, or just close the tab.
3. Resume it later (a fresh process restart works too, since working memory is persisted to disk) and say "continue."

The agent picks up exactly where it left off - stage, step, and expected action are all still in context - without the user restating any of it. That's the deliverable: not just persistence (lesson 07 already had that), but a formal state the agent can reason about and pick back up mid-flight.

### Token cost

`working_memory_tokens` in `MemoryReport` now covers the whole FSM block (stage, step, expected action, plus any working facts), the same way it covered a bare task name in lesson 12 - formalizing the state didn't meaningfully change its cost, since it's still just a handful of short lines.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

Or drive it directly via curl:

```bash
# Start a task (planning stage)
curl -X POST localhost:3000/api/task/start \
  -H 'Content-Type: application/json' \
  -d '{"name": "Refactor the auth module", "step": "list the call sites", "expected_action": "waiting for a go-ahead to start"}'

# Move to execution
curl -X POST localhost:3000/api/task/advance \
  -H 'Content-Type: application/json' \
  -d '{"stage": "execution", "step": "rewrite the token refresh path", "expected_action": "waiting on a code review"}'

# Pause / resume
curl -X POST localhost:3000/api/task/pause
curl -X POST localhost:3000/api/task/resume

# Validation, then done, then finish
curl -X POST localhost:3000/api/task/advance -H 'Content-Type: application/json' -d '{"stage": "validation", "step": "run the test suite"}'
curl -X POST localhost:3000/api/task/advance -H 'Content-Type: application/json' -d '{"stage": "done"}'
curl -X POST localhost:3000/api/task/finish -H 'Content-Type: application/json' -d '{"promote": true}'

curl localhost:3000/api/memory   # full inspection, including task.history
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `PROFILE_FILE` | no | `profile.json` |
| `SHORT_TERM_FILE` | no | `short_term.json` |
| `WORKING_MEMORY_FILE` | no | `working_memory.json` |
| `LONG_TERM_FILE` | no | `long_term.json` |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

`WORKING_MEMORY_FILE` now persists the task's full FSM state (stage, step, expected action, pause flag, and its whole history), not just a name - a restart resumes a task exactly where it was paused or left off.

## Deploy

Deployed automatically by GitHub Actions on every push to `master` (this becomes the highest-numbered lesson, so it becomes the new default deploy target - see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). Live at `http://<VDS host>:4013`.

## Conclusion

_(to fill in after recording the demo)_

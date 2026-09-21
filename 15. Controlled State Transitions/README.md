# 15. Controlled State Transitions

## Задание

🔥 День 15. Контролируемые переходы состояний

Реализуйте явные переходы между состояниями задачи.

Сделайте так, чтобы:

👉 у задачи были допустимые состояния
👉 были разрешённые переходы между ними
👉 ассистент не мог "перепрыгнуть" этап

Пример:

👉 нельзя делать реализацию до утверждённого плана
👉 нельзя делать финал без валидации

Проверьте:

👉 попытки перейти в недопустимое состояние
👉 реакцию ассистента
👉 корректность продолжения после паузы

Результат:

Ассистент с контролируемым жизненным циклом задачи

Формат:

Видео + Код / Текст

## Demo

https://github.com/user-attachments/assets/941b86c0-4b04-4f46-af8c-d3e801529c57

## What this is

Lesson 13 already gave a task an explicit finite state machine - `planning -> execution -> validation -> done`, with `Agent::advance_task` refusing any edge not on that structural table. That answered "what are the valid states and edges" but not the two concrete examples this assignment names: nothing in lesson 13 actually required a plan to be *approved* before execution started, or a validation to have *passed* before the task could be marked done - `advance_task` would walk the whole pipeline the moment each structural edge existed. This lesson adds exactly those two missing checks as explicit, named **gates**, on top of the same table, and - the part the assignment calls out separately - makes sure the assistant can't be talked into skipping them in chat either. Everything else from lesson 14 (invariants, personalization, memory) is unchanged.

### Two gates, two dedicated endpoints

```rust
struct TaskFsm {
    // ...stage, step, expected_action, paused, history - unchanged...
    plan_approved: bool,     // flipped only by POST /api/task/approve_plan
    validated: bool,         // flipped only by POST /api/task/validate
    validation_notes: String,
}
```

Neither flag can be set by `advance_task` itself, by a chat message, or by anything the LLM infers - the same "state only ever changes through an explicit call" rule the FSM's `stage` has followed since lesson 13 now also covers these two gates:

- `POST /api/task/approve_plan` - the only way `plan_approved` becomes `true`. Only valid while the task is still in `planning`; calling it after execution has already started would be misleading, so it's rejected rather than silently no-op'd.
- `POST /api/task/validate` (`{"passed": bool, "notes": "..."}`) - the only way `validated` is set, in either direction. Only valid while the task is in `validation`, for the same reason - a check can't be "recorded" from a stage where it couldn't have actually run.

`Agent::advance_task` now validates every transition two ways: structurally, against the same table lesson 13 built (`TaskStage::can_advance_to` - `planning` still can't jump straight to `done`), and then, for the two edges this assignment names, against `stage_gate_error`:

```
planning   -> execution   requires plan_approved == true
validation -> done        requires validated == true
```

Try to advance past either edge without the gate and `advance_task` returns a plain-Rust error naming exactly which endpoint to call first - never a silent success, never something the model has to decide. A rework loop (`validation -> execution`, unchanged from lesson 13) resets `validated` back to `false`: the code under review is about to change, so a stale "already validated" can't carry over to the next pass.

### The assistant can't skip a stage in chat either

A hard API check stops the *state* from skipping ahead, but the assignment specifically asks to verify the *assistant's* reaction to a skip attempt - a user can still type "let's just start coding" while the plan sits unapproved, and a plain chat model would often just... start coding. Two layers stop that, mirroring lesson 14's invariant enforcement exactly:

1. **Deterministic.** `find_blocked_stage_request` scans the raw message for a small, fixed set of phrases ("start coding", "let's implement", "write the code", ...) against the active task's current gate state, and - if the task is in `planning` with an unapproved plan, or in any stage before `done` with `validated: false` - blocks the turn in plain Rust before the model is ever called, exactly like `find_violated_invariant` does for a forbidden term. `MemoryReport.blocked_by_stage_gate` is set to `"plan_not_approved"` or `"not_validated"`, and `actual` (real API usage) stays `None` since no call was made.
2. **Model-reasoned.** `format_working_block` now renders a `GATES` block alongside the FSM state on every single turn: "Plan approved: yes/no", "Validation passed: yes/not yet", and an explicit instruction not to write, describe, or act as though implementation/finishing happened while its gate reads "no" - refuse instead, name the missing gate, name the exact endpoint that satisfies it. This is the semantic backstop for phrasing the fixed list doesn't catch (paraphrases, other languages, indirect asks), the same way the invariants block backs up `forbidden_terms`.

Either way the turn still lands in short-term history (the user's ask and the refusal both), so the conversation reads naturally and the same refusal applies again if the user repeats the request - and, just like an invariant, this can't be argued around in chat; only the matching API call changes it.

### Pause still freezes everything, gates included

`find_blocked_stage_request` explicitly returns `None` while the task is paused - a paused task's next message isn't a stage-skip attempt, it's a message the FSM shouldn't be reading progress into at all (unchanged rule from lesson 13). `approve_plan` and `record_validation` both refuse while paused, the same way `advance_task` and `update_task_step` already did, so a gate can never be flipped out from under a task the user thinks is frozen. Resuming changes nothing about `plan_approved` or `validated` - they come back exactly as they were, same as stage/step/expected action always have.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

Or drive it directly via curl:

```bash
# Start a task (planning stage, plan not yet approved)
curl -X POST localhost:3000/api/task/start \
  -H 'Content-Type: application/json' \
  -d '{"name": "Refactor the auth module", "step": "list the call sites"}'

# This is refused - deterministically, without ever calling the model
curl -X POST localhost:3000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Great, let'"'"'s start coding this now."}'

# advance_task rejects the same jump at the API level too
curl -X POST localhost:3000/api/task/advance \
  -H 'Content-Type: application/json' \
  -d '{"stage": "execution"}'
# -> 400 "Cannot start execution: the plan has not been approved yet. ..."

# Approve the plan - the only call that can unlock execution
curl -X POST localhost:3000/api/task/approve_plan

# Now this succeeds
curl -X POST localhost:3000/api/task/advance \
  -H 'Content-Type: application/json' \
  -d '{"stage": "execution", "step": "rewrite the token refresh path"}'

curl -X POST localhost:3000/api/task/advance \
  -H 'Content-Type: application/json' \
  -d '{"stage": "validation", "step": "run the test suite"}'

# "done" is refused until a validation outcome is recorded
curl -X POST localhost:3000/api/task/advance -H 'Content-Type: application/json' -d '{"stage": "done"}'
# -> 400 "Cannot mark the task done: validation has not passed yet. ..."

curl -X POST localhost:3000/api/task/validate \
  -H 'Content-Type: application/json' \
  -d '{"passed": true, "notes": "all tests green"}'

curl -X POST localhost:3000/api/task/advance -H 'Content-Type: application/json' -d '{"stage": "done"}'
curl -X POST localhost:3000/api/task/finish -H 'Content-Type: application/json' -d '{"promote": true}'

# Pause / resume - gates and stage survive exactly as they were
curl -X POST localhost:3000/api/task/pause
curl -X POST localhost:3000/api/task/resume

curl localhost:3000/api/memory   # full inspection, including plan_approved/validated/history
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
| `INVARIANTS_FILE` | no | `invariants.json` |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

`WORKING_MEMORY_FILE` now also persists `plan_approved`, `validated`, and `validation_notes` alongside the rest of the task FSM - a restart resumes a task with both gates exactly where they were left.

## Deploy

Deployed automatically by GitHub Actions on every push to `master` (this becomes the highest-numbered lesson, so it becomes the new default deploy target - see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). Live at `http://<VDS host>:4015`.

## Conclusion

_(to fill in after recording the demo)_

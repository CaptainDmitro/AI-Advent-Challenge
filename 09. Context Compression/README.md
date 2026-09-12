# 09. Context Compression

## Задание

🔥 День 9. Управление контекстом: сжатие истории

Реализуйте механизм управления контекстом:
👉 храните последние N сообщений "как есть"
👉 остальное заменяйте summary (например каждые 10 сообщений)
👉 храните summary отдельно и подставляйте его в запрос вместо полной истории

Сравните:
👉 качество ответов без сжатия
👉 качество ответов со сжатием
👉 расход токенов до/после

Результат:
Агент, который работает с компрессией истории и экономит токены

Формат:
Видео + Код

## Demo

_(to be added after recording the demo)_

## What this is

[Lesson 08](../08.%20Token%20Counting)'s `Agent`, now able to send a *compacted* context instead of the full conversation history — and reporting, on every turn, exactly how many tokens that saved.

The raw, uncompressed conversation is never touched: every message ever exchanged still lives in `history.json`, exactly as in lessons 07–08. What's new is a second, much smaller piece of state, persisted separately in `summary.json`:

- **`summary`** — a running natural-language condensation of everything older than the last `keep_last_n` messages, produced by a dedicated LLM call whenever enough new material has piled up.
- **`summarized_through`** — how many of the raw (non-system) messages are currently folded into that summary.

When compaction is switched on for a turn, the request sent to the model is `[system prompt, summary-note (if any), last N raw messages]` instead of the entire history. The summary is *substituted in*, not merged into or mixed with the raw messages it replaces — so the older messages are represented exactly once, either as raw text (if within the last-N window) or as summary (if older), never both.

### Keeping the last N as-is, folding the rest every 10 messages

`build_context` (in `src/main.rs`) does the substitution: system prompt, then an optional synthetic system message carrying the summary, then a slice of the last `KEEP_LAST_N_MESSAGES` raw messages. `plan_compaction` is the pure decision function behind it — given how many messages exist, how many are already folded, and the configured window/threshold, it decides whether a compaction pass is due. By default a pass fires once 10 raw messages have piled up past the last-N window (`SUMMARIZE_EVERY_N_MESSAGES=10`, matching the assignment's "every 10 messages"), folding them into the summary via one extra call to the same LLM (with its own narrow instruction: preserve facts/names/decisions, drop small talk, output only the updated summary — see `SUMMARY_SYSTEM_PROMPT`). That call is best-effort: if it fails, the summary simply stays as it was and the next turn tries again — it never blocks or breaks the actual chat turn it rode in on.

Both mechanisms are visible and controllable from the UI: a **Compaction On/Off** toggle next to the model picker lets you A/B the *same* conversation with and without compression, and a **Compress now** button (`POST /api/compact`) forces an immediate fold without waiting for the 10-message threshold — useful for demonstrating the effect on demand rather than by typing for a while.

### Comparing quality with vs. without compression

Toggle **Compaction** off and the agent behaves exactly like lesson 08 — the model sees the entire conversation, verbatim, every turn. Toggle it on and the model sees only the system prompt, the running summary, and your last few messages. Ask something that depends on a detail from early in a long conversation (a name, a number, a decision made 15 messages ago) under both settings:

- **Off** — the model has the literal text and answers however well it always did; nothing about *this* mechanism can degrade it.
- **On** — the answer is only as good as the summary. If the fact survived condensation (the summarizer is instructed to preserve exactly this kind of thing), the answer is usually just as good. If it was dropped as "small talk" incorrectly, or the summary drifted after several fold cycles, the answer degrades — this is the real, honest trade-off of compaction, not a hidden one.

### Comparing token cost before/after

Every `/api/chat`, `/api/reset`, and `/api/history` response carries both numbers for that turn:

- `uncompacted_context_tokens` — what would have been sent with compaction off (the full history including the message you just sent).
- `sent_context_tokens` — what was actually sent this turn.
- `tokens_saved_this_turn` — the difference; zero whenever compaction is off, since the two are then identical by construction.

The stats bar under the chat renders all three, plus `raw_history_tokens_after` (the complete, ever-growing transcript kept on disk) and `percent_of_limit` — computed against `sent_context_tokens`/the API's own `prompt_tokens`, not the raw history, since that's what actually determines whether the *next* call risks the provider rejecting it. This is the clearest way to see the mechanism work: with compaction off, `percent_of_limit` climbs every turn just like lesson 08; with it on, it stays roughly flat once the summary + last-N window stabilizes, however long the conversation runs.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Comparing short / long / compressed dialogs

1. **Short dialog, compaction off** — send 2–3 messages with the toggle off. `sent_context_tokens` and `uncompacted_context_tokens` will be identical every turn — nothing to compress yet.
2. **Long dialog, compaction off** — keep going for 15–20 turns. Watch `raw_history_tokens_after` (and `sent_context_tokens`, which tracks it exactly in this mode) climb without bound, exactly like lesson 08.
3. **Same long dialog, compaction on** — flip the toggle (or start fresh) and keep going past `KEEP_LAST_N_MESSAGES + SUMMARIZE_EVERY_N_MESSAGES` messages. Watch the **Running summary** panel pick up its first fold, `sent_context_tokens` drop back down near `system + summary + last N`, and `tokens_saved_this_turn` turn positive — while `raw_history_tokens_after` keeps climbing regardless, since the full transcript is still being kept.
4. **Quality check** — ask a question that depends on something said early on, once with the toggle on and once off (use **Reset** between runs, or just compare against the off-mode answer you already have). See "Comparing quality" above for what to expect.
5. **Compress now** — click it any time there's unfolded material past the last-N window, to see a fold happen immediately rather than waiting for 10 more messages.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `HISTORY_FILE` | no | `history.json` |
| `SUMMARY_FILE` | no | `summary.json` |
| `KEEP_LAST_N_MESSAGES` | no | `6` |
| `SUMMARIZE_EVERY_N_MESSAGES` | no | `10` |
| `COMPACTION_ENABLED` | no | `true` (any value except `false`/`0`) |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

`COMPACTION_ENABLED` only sets the *default* used when a request doesn't specify one — the UI's toggle (and `/api/chat`'s `compaction` field) can always override it per turn. `CONTEXT_LIMIT_TOKENS` is only a reference value for the progress indicator, same caveat as lesson 08. `PRICE_PER_1M_INPUT_TOKENS`/`PRICE_PER_1M_OUTPUT_TOKENS` are USD per 1,000,000 tokens.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4009`.

## Conclusion

_(to fill in after recording the demo)_

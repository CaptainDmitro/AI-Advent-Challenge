# 08. Token Counting

## Задание

🔥 День 8. Работа с токенами

Добавьте в код агента подсчёт токенов:
👉 для текущего запроса
👉 для всей истории диалога
👉 для ответа модели

Сравните:
👉 короткий диалог
👉 длинный диалог
👉 диалог, который превышает лимит модели

Покажите:
👉 как растёт стоимость/токены по мере диалога
👉 что ломается при переполнении

Результат:
Код, который считает токены и показывает, как они влияют на поведение агента

Формат:
Видео + Код

## Demo

_(to be added after recording the demo)_

## What this is

[Lesson 07](../07.%20Context%20Persistence)'s `Agent`, now reporting token usage on every turn instead of just chatting silently. Two numbers are tracked, and both are shown, clearly labeled:

- **Estimated** — a dependency-free local heuristic (`estimate_tokens` in `src/main.rs`) computed *before* the request goes out: letter/digit runs at ~4 characters per token, one token per punctuation character, plus a small per-message overhead for chat-format structure. It's deliberately approximate — DeepSeek doesn't publish a tokenizer, so anything claiming precision here would be lying. It's what lets the UI show "this message ≈ N tokens" without waiting on the network.
- **Actual** — most OpenAI-compatible chat APIs, DeepSeek included, return a `usage: {prompt_tokens, completion_tokens, total_tokens}` object on the completion response. That's deserialized and used as the authoritative number for the turn that just happened, falling back to the estimate only if a response omits it.

`/api/chat`'s response gains a `tokens` object carrying both: the pre-call estimate for this request and for the history it was appended to, the actual (or estimated, if actual is missing) usage for the turn, a running `history_tokens_after`, and `percent_of_limit` against a configurable reference context size. `/api/reset` and `/api/history` return the same shape so the UI can show a running total on load and after a reset. The chat page renders a token caption under every bubble and a stats bar with the running history total, last-turn tokens, and (if you've configured pricing) a per-turn cost — plus a banner once history crosses 80%/100% of the reference limit.

`Message` itself is unchanged (`{role, content}`), so lesson 07's persisted `history.json` files keep working with no migration — token/cost figures are recomputed from history content on the fly rather than stored per message.

### Why a heuristic instead of a real tokenizer library

DeepSeek's tokenizer isn't public, so a BPE library like `tiktoken-rs` would just be running someone else's (OpenAI's) vocabulary against DeepSeek's actual API — a fancier-looking approximation, not a correct one. The heuristic here is honest about being approximate, needs no new dependency, and keeps the `x86_64-unknown-linux-musl` cross-compile in the [deploy pipeline](../DEPLOYMENT.md) exactly as simple as lesson 07 left it. Where precision actually matters, the API's own `usage` field is used instead.

### Why overflow isn't artificially blocked

`CONTEXT_LIMIT_TOKENS` is a configurable *reference* number for the progress indicator — not a claim about the real limit of the `deepseek-v4-flash`/`deepseek-v4-pro` aliases this course uses, which isn't published anywhere I can verify. Rather than guess wrong and simulate a fake failure, this agent lets the real backend be the source of truth: once history is large enough, the provider itself rejects the request, and that raw error surfaces through the same error path lesson 06 already built (`"API error (400): ..."`, shown directly in the chat bubble, with a 200 from *this* server so one failed call never breaks the rest of the UI). The warning banner just narrates the run-up to that moment.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Comparing short / long / over-limit dialogs

1. **Short dialog** — send 2–3 short messages. Watch the stats bar: history tokens stay small, and (if pricing is configured) the turn cost stays near-zero.
2. **Long dialog** — keep the conversation going for 15–20 turns, or paste a long block of text as one message. Watch `history_tokens_after` and the cost climb turn over turn, and the caption under each reply compare the estimate to the API's actual usage.
3. **Over-limit dialog** — keep going, or send one very large message (e.g. a large pasted document repeated a few times). Once history is big enough, the warning banner lights up around 80%, then red past 100% of `CONTEXT_LIMIT_TOKENS` — and eventually the real API rejects the request outright, which shows up as a plain `API error (...)` reply in the chat, with the last token report attached below it. That's the actual breakage, not a simulated one.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `HISTORY_FILE` | no | `history.json` |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

`CONTEXT_LIMIT_TOKENS` is only a reference value for the progress indicator and warning banner — see above. `PRICE_PER_1M_INPUT_TOKENS`/`PRICE_PER_1M_OUTPUT_TOKENS` are USD per 1,000,000 tokens; set both to your actual provider rate to see cost figures, or leave them unset (the default) to hide cost entirely rather than show a number sourced from a guess.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4008`.

## Conclusion

_(to fill in after recording the demo)_

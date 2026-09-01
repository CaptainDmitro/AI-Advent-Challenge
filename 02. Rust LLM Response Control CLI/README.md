# 02. Rust LLM Response Control CLI

## Задание

Отправьте один и тот же запрос, но:
👉 добавьте явное описание формата ответа
👉 добавьте ограничение на длину ответа
👉 добавьте условие завершения ответа (stop sequence или явную инструкцию)

Сравните ответы:
👉 без ограничений
👉 с ограничениями

Результат:
Один и тот же запрос с разным уровнем контроля ответа через API

## Demo

https://github.com/user-attachments/assets/f21c0599-93c1-4138-81a4-0a04cf9de107

## What this is

An interactive CLI chat, same as [lesson 01](../01.%20Rust%20LLM%20Chat%20CLI), extended with live commands to control how the API generates its response: an explicit format instruction, a `max_tokens` limit, and a `stop` sequence. This lets you ask a question, tweak the controls, and resend the exact same question to compare the unrestricted vs. restricted response.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

## Configuration

Set via environment variables:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | no | `gpt-4o-mini` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server (e.g. a local or self-hosted one).

## Commands

Any line starting with `:` is treated as a command instead of being sent to the LLM.

| Command | Effect |
|---|---|
| `:set format <text>` / `:set format none` | Sets/clears an explicit response-format instruction (sent as a system message) |
| `:set max_tokens <n>` / `:set max_tokens none` | Sets/clears the API's `max_tokens` limit |
| `:set stop <sequence>` / `:set stop none` | Sets/clears the API's `stop` sequence |
| `:show` | Prints the currently active format / max_tokens / stop |
| `:resend` | Resends the last question with the current settings, without adding it to the conversation history again |
| `:reset` | Clears format, max_tokens, and stop |
| `:help` | Lists the commands |

### Example: comparing restricted vs. unrestricted

```
> What is a black hole?
[full, unrestricted answer]

> :set format Respond with exactly one sentence.
> :set max_tokens 20
> :set stop .
> :resend
[short answer, cut off by max_tokens/stop]

> :reset
```

`:resend` is sent as a standalone request (not appended to the running conversation), so the comparison reflects only the current settings rather than accumulated conversation context.

# 26. Local LLM

## Задание

🔥 День 26. Запуск локальной LLM

Установите и запустите любую локальную LLM (Ollama, LM Studio, llama.cpp, Mistral, Qwen, Phi и т.п.)

Проверьте:

👉 модель запускается локально
👉 к ней можно обратиться через CLI или HTTP API
👉 модель отвечает на простой запрос

Сделайте:

👉 минимум 3 запроса разной сложности

Результат:

Локальная LLM запущена и отвечает на запросы

Формат:

Видео + Код

## Demo

_(video to be added)_

## What this is

Every lesson so far has called a hosted model. This one runs the model on
the machine itself: [Ollama](https://ollama.com) serving **Qwen 2.5 3B**
(`qwen2.5:3b`, 1.9 GB, Q4_K_M). It needs no API key and nothing leaves
`localhost`. The same model is reached three ways:

```
                       ┌──────────────── ollama serve (localhost:11434) ────────────────┐
ollama run …    ──CLI──▶                                                                 │
curl /api/chat  ──HTTP─▶  native API: streaming NDJSON, load / eval timings in ns        ├─▶ qwen2.5:3b
curl /v1/chat/… ──HTTP─▶  OpenAI-compatible: the shape lessons 01–25 already speak       │   (Metal on Apple Silicon)
cargo run       ──HTTP─▶  /api/version, /api/tags, /api/chat (stream)                    │
                       └─────────────────────────────────────────────────────────────────┘
```

- **`requests.sh`**: the assignment without Rust. Server version and model
  list, then one request per level of difficulty, each through a different
  door. The easy one goes through `ollama run`, the medium one through
  `/api/chat`, and the hard one through `/v1/chat/completions`.
- **`cargo run`**: a Rust CLI on Ollama's native API. It checks that the
  server is up and the model is pulled, then sends the same three requests at
  temperature 0, streams each answer token by token, checks it for the
  expected answer, and prints the server's counters: prompt and completion
  tokens, tokens/s, time to first token, total time and model load time.

### The three requests

| Level | Prompt | Checked for |
|---|---|---|
| easy | Какая столица Франции? Ответь одним словом. | `Париж` |
| medium | Объясни в трёх предложениях, чем TCP отличается от UDP, и назови по одному протоколу или сервису поверх каждого | `TCP`, `UDP` |
| hard | Two trains, 300 km apart, leave at 9:00 (60 km/h) and 10:00 (90 km/h) toward each other. When do they meet? Step by step, time on the last line | `11:36` |

The hard one has a single right answer. By 10:00 the first train has covered
60 km, 240 km remain, they close at 150 km/h, so 1.6 h later: **11:36**. A 3B
model usually gets it wrong. In one run it forgot the one-hour head start and
answered 12:00, in another it divided by the second train's speed alone and
answered 13:00. The check is a substring match, so the demo shows ✓ or ✗ for
each answer and a summary table at the end.

### Measured on an M-series MacBook, 16 GB

| Level | Answer | Tokens | Speed |
|---|---|---|---|
| easy | Париж ✓ | 3 | ~1.5 s with `ollama run`, model already loaded |
| medium | correct, HTTP over TCP and VoIP over UDP ✓ | 190 | 45.5 tok/s, 4.4 s |
| hard | 12:00 ✗ (correct: 11:36) | 278 | not reported: `/v1` has no timings |

The first request after `ollama serve` starts also pays for loading the
weights into memory (a few seconds). Ollama keeps the model loaded for 5
minutes, so later requests show `model load 0.01s`.

### Verification

`cargo test` needs neither Ollama nor the network. A fake Ollama (a raw
`TcpListener` on a random port) returns canned responses and records what it
was sent.

- `reads_version_and_installed_models`: `/api/version` and `/api/tags` are
  parsed. `qwen2.5` matches `qwen2.5:latest`, and a model that isn't pulled
  doesn't match.
- `streams_a_reply_token_by_token_and_reads_the_counters`: an NDJSON stream
  arrives as separate tokens and is joined into the text. The nanosecond
  counters become tokens/s, load and total time. The request is checked to
  carry the model, `stream: true`, `temperature: 0` and the message.
- `reports_errors_before_during_and_instead_of_the_stream`: a 404 with
  Ollama's `{"error": …}` body, an error line inside the stream, a stream cut
  off before `done`, and a server that isn't running each end in a readable
  error, never a panic.
- `checks_answers_for_expected_keywords`, `normalizes_ollama_host_like_ollama_does`.

## Run

Install and start Ollama, then pull the model (macOS):

```bash
brew install ollama
```

```bash
brew services start ollama
```

```bash
ollama pull qwen2.5:3b
```

On Linux: `curl -fsSL https://ollama.com/install.sh | sh`, then the same
`ollama pull`.

Check it answers:

```bash
ollama run qwen2.5:3b "Привет! Кто ты?"
```

```bash
curl -s localhost:11434/api/chat -d '{"model":"qwen2.5:3b","stream":false,"messages":[{"role":"user","content":"Привет! Кто ты?"}]}'
```

The three requests, without Rust:

```bash
./requests.sh
```

The Rust CLI:

```bash
cargo run
```

```bash
cargo run -- ask "Напиши хайку про Rust"
```

```bash
cargo run -- chat
```

`chat` keeps the dialogue history and prints the counters after every
answer. `/reset` clears the history, and `/quit` exits.

Any other pulled model works too:

```bash
OLLAMA_MODEL=llama3.2 cargo run
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OLLAMA_HOST` | no | `http://localhost:11434`. Same variable as Ollama's own CLI, with or without `http://` |
| `OLLAMA_MODEL` | no | `qwen2.5:3b` |

No API key: the model runs locally.

## Deploy

None. This is a CLI lesson (no `axum`), so the pipeline builds and tests it
but doesn't deploy it (see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). That is
on purpose: the point is a model on your own machine, and the VDS has no
Ollama.

## Conclusion

_(to fill in after recording the demo)_

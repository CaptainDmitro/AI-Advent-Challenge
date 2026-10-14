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

https://github.com/user-attachments/assets/166b688c-c3f8-4be2-a3e5-f8e108adb99a

## What this is

Lesson 06's chat, but the model runs on this machine:
[Ollama](https://ollama.com) serving **Qwen 2.5 3B** (`qwen2.5:3b`, 1.9 GB,
Q4_K_M). It needs no API key, and nothing leaves `localhost`.

```
browser ── /api/chat ──▶ axum (this crate, :3000) ── POST /api/chat ──▶ ollama serve (:11434) ──▶ qwen2.5:3b
        ◀─ reply + stats ─                       ◀── message + counters ─                         (Metal on Apple Silicon)
```

- **One conversation**, held in memory: a system prompt plus every turn. The
  whole dialogue goes to Ollama on each message. A failed turn is removed
  from the history, so a retry isn't sent twice. **Reset** clears the history.
- **Ollama status** at the top of the page: version, host and the number of
  pulled models (`/api/version`, `/api/tags`). It turns red with the reason
  when Ollama isn't running. ↻ checks again.
- **Model picker** filled with the models Ollama actually has. The default
  model is listed even if it isn't pulled, marked `(not pulled)`, so the chat
  shows Ollama's "try pulling it first" instead of silently using another.
  **Temperature** goes from 0 to 2.
- **Stats under every answer**, from Ollama's native counters: prompt →
  completion tokens, tokens/s, total time and model load time. The first
  message after a while shows the load (≈1 s for this model). After that it's
  `0.00 s`, because Ollama keeps the model in memory for 5 minutes.
- **Three preset requests** of growing difficulty, one click each. Each
  answer is checked for its one expected result (✓ / ✗):

| Level | Prompt | Checked for |
|---|---|---|
| easy | Какая столица Франции? Ответь одним словом. | `Париж` |
| medium | Объясни в трёх предложениях, чем TCP отличается от UDP, и назови по одному протоколу или сервису поверх каждого | `TCP`, `UDP` |
| hard | Two trains, 300 km apart, leave at 9:00 (60 km/h) and 10:00 (90 km/h) toward each other. When do they meet? Step by step, time on the last line | `11:36` |

The hard one has a single right answer. By 10:00 the first train has covered
60 km, 240 km remain, they close at 150 km/h, so 1.6 h later: **11:36**. The
3B model gets it wrong in a different way almost every time: 13:00, 12:00,
or the right 1.6 h added to the wrong start time (03:36). Easy and medium
come back right.

The server talks to Ollama's **native** `/api/chat` with `stream: false`, not
the OpenAI-compatible `/v1`, because only the native endpoint reports load and
generation time separately. Errors (Ollama not running, model not pulled) come
back as a plain string in the JSON, so the page shows them in the chat and
keeps working.

### Without the browser: CLI and HTTP

`requests.sh` asks the same three questions straight from the terminal, each
through a different door. The easy one goes through `ollama run`, the medium
one through the native `/api/chat`, and the hard one through the
OpenAI-compatible `/v1/chat/completions`. It needs only `curl` and `python3`.

Measured on an M-series MacBook, 16 GB:

| Level | Answer | Tokens | Speed |
|---|---|---|---|
| easy | Париж ✓ | 4 | 57 tok/s, 1.4 s, of which 1.05 s model load |
| medium | correct, HTTP over TCP and VoIP over UDP ✓ | 190 | 45.5 tok/s, 4.4 s |
| hard | 1.6 h right, but 03:36 ✗ (correct: 11:36) | 295 | 44 tok/s, 7.0 s |

### Verification

`cargo test` needs neither Ollama nor the network. A fake Ollama (a raw
`TcpListener` on a random port) returns canned responses and records what it
was sent. The tests call the axum handlers directly.

- `status_lists_the_installed_models`: `/api/version` and `/api/tags` become
  the status the page shows.
- `status_says_when_ollama_is_not_running`: a closed port gives a readable
  "Can't reach Ollama" instead of an error status.
- `chat_keeps_the_history_and_returns_the_counters`: the second request
  carries `system, user, assistant, user`, with the first answer in it. The
  model can be switched per message, and the temperature is clamped to 0–2.
  `stream` is `false`. The nanosecond counters become tokens/s, load and total
  time.
- `a_failed_turn_is_reported_and_leaves_no_trace`: Ollama's 404 "not found"
  reaches the page as text. An empty message never reaches Ollama. Neither one
  stays in the history.
- `reset_keeps_only_the_system_prompt`, `normalizes_ollama_host_like_ollama_does`.

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

Start the chat and open http://localhost:3000:

```bash
cargo run
```

Any other pulled model works too, as the default or from the picker:

```bash
OLLAMA_MODEL=llama3.2 cargo run
```

The same thing over HTTP:

```bash
curl -s localhost:3000/api/chat -H 'Content-Type: application/json' \
  -d '{"message": "Какая столица Франции?", "temperature": 0}'
```

| Endpoint | |
|---|---|
| `GET /api/status` | `{host, default_model, version, models, error}` |
| `POST /api/chat` | `{message, model?, temperature?}` → `{model, reply, stats}` or `{model, error}` |
| `POST /api/reset` | clears the conversation |

The three requests from the terminal, without the web app:

```bash
./requests.sh
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OLLAMA_HOST` | no | `http://localhost:11434`. Same variable as Ollama's own CLI, with or without `http://` |
| `OLLAMA_MODEL` | no | `qwen2.5:3b`: the default in the picker |
| `PORT` | no | `3000` |

No API key: the model runs locally.

## Deploy

The pipeline deploys it to `http://<VDS host>:4026` on push to `master`, as the
newest web-app lesson (see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). The VDS
has no GPU and no Ollama. The deployed chat uses **the Ollama on your Mac**
through a reverse SSH tunnel:

```
browser ──▶ VDS :4026 (this app) ──▶ VDS 127.0.0.1:11434 ══ ssh -R ══▶ Mac 127.0.0.1:11434 (ollama serve) ──▶ qwen2.5:3b
```

The Mac connects out to the VDS, so NAT and home routers don't matter. On the
VDS the forwarded port is bound to `127.0.0.1` only, so Ollama (which has no
auth) is never reachable from the internet, only through the app. The app's
default `OLLAMA_HOST` is already `http://localhost:11434`, so neither the app
nor the pipeline needs any extra config.

On the Mac, with Ollama running:

```bash
./tunnel.sh <user>@<vds-host> -i ~/path/to/key
```

Anything after the host goes to `ssh` as is. It reconnects by itself after sleep or a network change. Check from the VDS:

```bash
ssh -i ~/path/to/key <user>@<vds-host> curl -s localhost:11434/api/version
```

While the tunnel is down, the page shows a red "Can't reach Ollama" status.
↻ turns it green once the tunnel is back, with no redeploy needed. Keep the
Mac awake while it serves (`caffeinate -i ./tunnel.sh …`). Anyone who opens
the page uses your Mac's model and shares the one in-memory conversation, as
in lesson 06.

## Conclusion

_(to fill in after recording the demo)_

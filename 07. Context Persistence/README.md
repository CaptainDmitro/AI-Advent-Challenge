# 07. Context Persistence

## Задание

🔥 День 7. Сохранение контекста

Добавьте агенту сохранение контекста:
👉 храните историю диалога (messages) в JSON или SQLite
👉 при перезапуске агента загружайте историю обратно
👉 продолжайте диалог так, как будто агент не выключался

Проверьте на практике:
👉 начните диалог
👉 перезапустите приложение
👉 продолжите диалог и убедитесь, что агент помнит прошлые сообщения

Результат:
Агент, который сохраняет и восстанавливает контекст между запусками

## Demo

_(to be added after recording the demo)_

## What this is

[Lesson 06](../06.%20First%20Agent)'s `Agent`, with its conversation history now backed by a JSON file instead of living only in memory:

- on startup, `Agent::new` reads `history.json` (path configurable via `HISTORY_FILE`) and restores the full conversation instead of seeding just the system prompt; a missing, empty, or corrupted file falls back to a fresh conversation rather than failing startup,
- after every completed turn (and on `/api/reset`), the agent writes its full history back to that file — a failed turn (network error, non-2xx response, unparseable body) is rolled back in memory exactly as in lesson 06, and since that rollback happens before any write, it never touches disk,
- a new `GET /api/history` endpoint exposes the current (non-system) messages, and the chat page fetches it on load — so reloading the browser after the server has restarted shows the same conversation instead of an empty chat window sitting on top of an agent that actually still remembers everything.

Everything else — the model toggle, temperature/max_tokens/stop advanced controls, error handling, model allow-list — is unchanged from lesson 06.

### Why JSON over SQLite

The assignment allows either. JSON was chosen because `serde_json` is already a dependency here, the data is just a `Vec<Message>` with no querying needs beyond "load it all, save it all," and — concretely for this repo — it avoids pulling a C dependency (`rusqlite`'s bundled SQLite) into the `x86_64-unknown-linux-musl` cross-compile the [deploy pipeline](../DEPLOYMENT.md) already has to do.

### Why the history file survives a real deploy, not just a local restart

The deploy pipeline's systemd unit sets `WorkingDirectory` to the lesson's app directory on the VDS, and a redeploy only ever overwrites the binary and `.env` there — it never clears the directory. So a relative `history.json` path persists across both a `systemctl --user restart` and a full redeploy, with no changes needed to the workflow or `DEPLOYMENT.md`.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000. Send a few messages, stop the process (Ctrl+C), and run it again — reload the page and the conversation is still there, both in the UI and in what gets sent to the model.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `HISTORY_FILE` | no | `history.json` |
| `PORT` | no | `3000` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server. `OPENAI_MODEL` only matters as the fallback for `/api/chat` requests that omit `model` or send one the server doesn't recognize; it does not affect the chat page's Flash/Pro toggle, which always starts on Flash. Must be `deepseek-v4-flash` or `deepseek-v4-pro`; anything else falls back to `deepseek-v4-flash`. `HISTORY_FILE` is a path relative to the working directory unless given as absolute.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4007`.

## Conclusion

_(to fill in after recording the demo)_

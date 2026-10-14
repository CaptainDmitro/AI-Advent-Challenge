# 30. Private LLM Service

## Задание

🔥 День 30. Локальная LLM как приватный сервис

Разверните локальную LLM как сервис:

👉 на VPS или домашнем сервере
👉 с HTTP API
👉 с возможностью чата

Проверьте:

👉 доступ к модели по сети
👉 стабильность при нескольких запросах
👉 базовые ограничения (rate limit / max context)

Результат:

Приватный AI-сервис на базе локальной LLM

Формат:

Видео + Код

## Demo

https://github.com/user-attachments/assets/b2b284f3-ff4d-4051-b456-ee7a00ccb892

## What this is

Since [lesson 26](../26.%20Local%20LLM), Qwen 2.5 3B has run in Ollama on my
Mac. A reverse SSH tunnel makes it reachable on the VDS. But that Ollama is
bound to the VDS's `127.0.0.1` on purpose, because it has no auth, no limits
and no queue. Lessons 26–29 just put a page in front of it. This lesson makes
it a **service**: a gateway on the VDS that is the only public way in, with
keys, limits and a queue, and an HTTP API any OpenAI client can talk to.

```
any client ─┐                    VDS :4030 (this crate)                              Mac
curl        ├─ HTTP ──▶ auth ▶ rate limit ▶ validate + max context ▶ queue ──▶ 127.0.0.1:11434 ══ ssh -R ══▶ ollama serve ─▶ qwen2.5:3b
this page  ─┘  Bearer key  401     429           400 / 413                503        (tunnel, VDS-local only)
```

Every refusal happens **before the model is touched**, and the cheapest checks
come first. Ollama only sees requests that passed all of them, and never more
than `MAX_CONCURRENT` at once.

### HTTP API

OpenAI-compatible, so the official SDKs, `curl`, or any "OpenAI-compatible
endpoint" setting in a tool works with `base_url = http://<vds>:4030/v1`.

| Endpoint | Key | |
|---|---|---|
| `POST /v1/chat/completions` | yes | `{messages, model?, temperature?, max_tokens?}` → OpenAI `chat.completion` (`choices`, `usage`) plus a `service` block: estimated prompt tokens, effective `max_tokens`, `num_ctx`, queue/upstream/load ms, tok/s, requests left this minute |
| `GET /v1/models` | yes | the models pulled on the Mac, plus the default |
| `GET /api/stats` | yes | limits, running/queued now, responses by status, latency avg/p50/p95/max over the last 200, tokens generated, the caller's remaining requests |
| `GET /health` | no | the service is up, and whether it can reach Ollama (version or the error). Nothing about keys or usage |
| `GET /` | no | the chat page; it asks for a key and then uses the same API |

```bash
curl -s http://<vds-host>:4030/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Привет! Кто ты?"}],"max_tokens":100}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://<vds-host>:4030/v1", api_key=KEY)
print(client.chat.completions.create(model="qwen2.5:3b",
      messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

Errors are OpenAI-shaped too, `{"error": {"message", "type", "code"}}`, and
come with **real HTTP statuses**. This is a deliberate break from the
repo-wide "errors as 200 + string" convention: for a service, the status *is*
the API. Clients retry on 429/503, honour `Retry-After`, and don't retry
401/413.

| Status | `code` | When |
|---|---|---|
| 401 | `invalid_api_key` | no `Authorization: Bearer`, or a key that isn't in the list (compared in constant time) |
| 429 | `rate_limit_exceeded` | over `RATE_LIMIT_PER_MINUTE` for this key. `Retry-After` says when the oldest request leaves the window |
| 400 / 422 | `empty_messages`, `invalid_role`, `stream_unsupported`, `invalid_max_tokens`, `invalid_body` | malformed request |
| 413 | `context_length_exceeded` | estimated prompt + `max_tokens` > `MAX_CONTEXT_TOKENS`. The message has the numbers |
| 413 | `invalid_body` | the body is over `MAX_BODY_BYTES`. Refused before parsing |
| 503 | `server_busy` | all slots generating and the queue full. `Retry-After: 2` |
| 404 | `model_not_found` | Ollama doesn't have that model |
| 502 / 504 | `ollama_unreachable`, `ollama_error`, `ollama_timeout` | the Mac or the tunnel is down, or Ollama failed |

### The limits

- **Rate limit:** a sliding 60 s window per key (timestamps, not a fixed
  minute, so there's no double burst at the minute boundary). It's checked
  right after auth, before the body is even looked at, so **every**
  authenticated request counts, junk included. A client hammering with broken
  requests is cut off as cheaply as one hammering with real ones. Each key has
  its own budget.
- **Max context:** the service has no tokenizer, so it estimates. Qwen's
  tokenizer is byte-level BPE, at ≈4 bytes per token in English and fewer
  characters per token in Russian (2 bytes per Cyrillic letter). So
  `UTF-8 bytes / 4 + 4 per message` overestimates rather than underestimates.
  If the prompt plus `max_tokens` doesn't fit, the request gets 413 with the
  arithmetic. The same limit goes to Ollama as `num_ctx`. Without it, Ollama
  would load its default window and **silently drop the start** of a prompt
  that doesn't fit. Now that can't happen: the gateway refuses first. The
  page's context meter uses the exact same estimate, so it turns red where
  the 413 starts.
- **Max answer:** `max_tokens` is clamped to `MAX_OUTPUT_TOKENS` (Ollama's
  `num_predict`). An answer cut by it comes back with `finish_reason:
  "length"`.
- **Concurrency and queue:** a semaphore lets `MAX_CONCURRENT` requests
  generate at once, and up to `MAX_QUEUE` more wait in FIFO order. One more
  gets **503 at once**, instead of piling up until every client times out.
  This is what keeps the service stable under several requests. The Mac does
  a bounded amount of work, waiting is bounded, and overload is an immediate,
  retryable answer. A client that disconnects while queued gives its place
  back (a drop guard), and so does a failed call.
- **Body size:** `MAX_BODY_BYTES`, enforced by axum before the JSON is parsed.

### Private

- Ollama stays on the VDS's loopback (the tunnel binds `127.0.0.1` only). The
  gateway is the only public way in, and it needs a key.
- Keys come from `API_KEYS` or `api_keys.txt` (see Configuration). The deploy
  pipeline doesn't have to know them. On first start the service generates
  `sk-local-<48 hex>` from `/dev/urandom` into `api_keys.txt` (mode 600), and
  logs only the file's path, never the key.
- The conversation isn't stored on the server. The page keeps the history
  in the tab and sends it with every message, like any OpenAI client. So
  several people (or tabs) chat independently, unlike lessons 06/26's one
  shared in-memory conversation. Nothing leaves the VDS and the Mac.
- `/api/stats` shows keys only as `sk-local-xxx…`.

### The page

- **Chat:** a model picker filled from `/v1/models`, temperature, Markdown
  replies. Under every answer: tokens, tok/s, time, queue and load ms, and
  requests left this minute. The **context meter** shows the estimated size
  of the next request against the limit. A refused turn leaves no trace in
  the history, so a retry isn't sent twice.
- **Service panel** (polled every 2 s): generation slots and queue places
  filling up live, the limits, the caller's remaining requests, latency
  avg/p50/p95, and the count of every response status so far.
- **Checks**, one click each, through the same public API:
  - **Parallel N:** N short requests at once (`max_tokens: 16`). The log
    shows each one's status, time and queue wait. With the defaults (2 + 4),
    6 all get answered in pairs, and 8 gives 6 × 200 + 2 × 503.
  - **Rate limit:** sends `{"messages": []}` (cheap, never reaches the model)
    until the 429, and shows `Retry-After`.
  - **Context:** a prompt just over `MAX_CONTEXT_TOKENS` → 413 in a few ms.
  - **No key:** the same request without a key and with a guessed one → 401.

### From the terminal

`check.sh` does the same checks from any machine, so it also proves network
access. It needs only `curl` and `python3`:

```bash
bash check.sh http://<vds-host>:4030 "$KEY" 8
```

It checks `/health`, 401 without/with a wrong key, one real answer with its
usage, 413 on a ~26 KB prompt, 8 parallel requests (each one's status, time
and queue wait, then a summary), invalid requests until 429 with
`Retry-After`, and the service's stats at the end. Run it last in the demo:
it spends the key's rate budget for the current minute.

### Verification

`cargo test` needs neither Ollama nor the network. A fake Ollama (an axum
router on a random port) echoes the last message after a configurable delay,
records every request, and tracks how many it served at the same time. The
tests drive the real service over real HTTP.

- `everything_but_the_page_and_health_needs_a_key`: 401 with
  `WWW-Authenticate: Bearer` without a key or with a wrong one, on every API
  route. A refused request never reaches Ollama. `/health` and `/` are public,
  and `/v1/models` lists Ollama's models.
- `chat_completion_is_openai_shaped_and_carries_the_limits_to_ollama`: the
  response shape (`choices`, `usage`, `finish_reason`, the `service` block,
  `x-ratelimit-*` headers). Ollama receives `num_ctx = MAX_CONTEXT_TOKENS`,
  `num_predict` = the clamped `max_tokens`, the temperature only when given,
  and the whole conversation. `/api/stats` counts it all.
- `bad_requests_are_refused_before_ollama`: every 400/422 case, broken JSON,
  413 for ≈5004 + 512 tokens over 4096 (with the numbers in the message). A
  prompt that fits alone but not with `max_tokens: 512`, and fits with 100.
  413 for a body over the size limit. Only the one valid request reached
  Ollama.
- `rate_limit_is_per_key_and_says_when_to_retry`: junk counts toward the
  limit, the 4th of 3 gets 429 with `Retry-After` in 1–60 s, and another key
  is unaffected. `rate_window_slides` checks the window itself with explicit
  timestamps.
- `parallel_requests_share_the_slots_and_wait_in_the_queue`: 6 at once with
  2 slots + 4 queue all get answered, each with its own reply. Ollama never
  sees more than 2 at a time. The later ones report their queue wait, and
  the counters return to 0.
- `a_full_queue_answers_503_at_once`: 1 slot + 2 queue, 5 at once → 3 × 200,
  2 × 503 with `Retry-After`. Ollama saw 3, and the next request after that
  goes through.
- `ollama_problems_come_back_as_readable_errors`: tunnel down → 502 "Can't
  reach Ollama" while `/health` still answers (`reachable: false`). An
  unknown model → 404 with Ollama's message, and the slot is given back.
- `keys_come_from_the_env_or_a_generated_file`, `small_helpers` (token
  estimate, constant-time compare, key label, host, latency percentiles).

## Run

Locally, against the Mac's own Ollama (see [lesson 26](../26.%20Local%20LLM)
for installing it):

```bash
ollama pull qwen2.5:3b
```

```bash
cargo run
```

The first start creates `api_keys.txt` in the current directory. Open
http://localhost:3000 and paste the key from it:

```bash
cat api_keys.txt
```

Or pick the keys yourself, and tighter limits to see them trip quickly:

```bash
API_KEYS=sk-local-demo RATE_LIMIT_PER_MINUTE=5 MAX_CONTEXT_TOKENS=2048 MAX_CONCURRENT=1 MAX_QUEUE=2 cargo run
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `API_KEYS` | no | comma-separated keys. If unset, the keys in `API_KEYS_FILE` |
| `API_KEYS_FILE` | no | `api_keys.txt`, one key per line (`#` comments allowed). Created with one random key if missing |
| `RATE_LIMIT_PER_MINUTE` | no | `20` requests per key per sliding minute |
| `MAX_CONTEXT_TOKENS` | no | `4096`: estimated prompt + `max_tokens`, also sent as Ollama's `num_ctx` |
| `MAX_OUTPUT_TOKENS` | no | `512`: cap on `max_tokens` (`num_predict`) |
| `MAX_CONCURRENT` | no | `2` requests generating at once |
| `MAX_QUEUE` | no | `4` requests waiting; one more gets 503 |
| `MAX_BODY_BYTES` | no | `262144` (256 KB) |
| `OLLAMA_HOST` | no | `http://localhost:11434`, with or without `http://` |
| `OLLAMA_MODEL` | no | `qwen2.5:3b`: used when a request has no `model` |
| `PORT` | no | `3000` |

Every number is at least 1.

## Deploy

The pipeline deploys it to `http://<VDS host>:4030` on push to `master`, as
the newest web-app lesson (see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs
with the defaults above, because the pipeline writes only `PORT` (and the
unused `OPENAI_*`) into `.env`. The model is the Mac's, through lesson 26's
tunnel:

```bash
"../26. Local LLM/tunnel.sh" <user>@<vds-host> -i ~/path/to/key
```

The key is generated on the VDS on first start, in the service's working
directory. A redeploy replaces only the binary and `.env`, so the key
survives:

```bash
ssh -i ~/path/to/key <user>@<vds-host> cat apps/lesson-30/api_keys.txt
```

To add or revoke keys, edit that file (one per line) and restart the service:

```bash
ssh -i ~/path/to/key <user>@<vds-host> 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user restart ai-advent-lesson-30'
```

While the tunnel is down, the service itself keeps answering: `/health` says
the model is unreachable, and chat requests get 502 with the reason. It
recovers by itself once the tunnel is back. Because the gateway sends
`num_ctx: 4096`, sharing the Mac's Ollama with an older lesson that uses
another `num_ctx` (lesson 28's 8192) makes Ollama reload the model when they
alternate. Expect a ~1 s `load_ms` on the first request after a switch.

## Conclusion

_(to fill in after recording the demo)_

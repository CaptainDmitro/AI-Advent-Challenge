# 06. First Agent

## Задание

🔥 День 6. Первый агент

Реализуйте простого агента, который:
👉 принимает запрос пользователя
👉 отправляет его в LLM через API
👉 получает ответ
👉 выводит результат в вашем интерфейсе

(простой чат, CLI или web, запросы через HTTP-клиент)

Важно:
👉 агент должен быть отдельной сущностью, а не просто один вызов API
👉 логика запроса и ответа должна быть инкапсулирована в агенте

Результат:
Агент принимает запрос и корректно вызывает LLM через API

## What this is

A small web chat backed by an `Agent` struct (`src/main.rs`) that owns the conversation history and the LLM call itself — the HTTP handlers are thin wrappers that just call `agent.respond(message)`. Unlike a single inline API call, the agent:

- keeps the running conversation (system + user + assistant turns) internally, so follow-up messages carry context,
- is the only thing that knows how to talk to the LLM API — building the request, sending it, and parsing the response all live inside `Agent::respond`,
- can be reset independently of the HTTP layer via `Agent::reset`.

This is also the first lesson wired into the repo's CI/CD pipeline (`.github/workflows/ci-cd.yml`): every push to `master` builds, tests, and lints this crate, then cross-compiles it to a static `x86_64-unknown-linux-musl` binary and deploys it to a systemd user service on the project's VDS.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | no | `deepseek-v4-pro` |
| `PORT` | no | `3000` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4006`.

## Conclusion

_(to fill in after recording the demo)_

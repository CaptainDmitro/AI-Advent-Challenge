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

## Demo

https://github.com/user-attachments/assets/ff01b8fe-45d3-452b-ab5b-195f6d3d5110

## What this is

A small web chat backed by an `Agent` struct (`src/main.rs`) that owns the conversation history and the LLM call itself — the HTTP handlers are thin wrappers that just call `agent.respond(message, options)`. Unlike a single inline API call, the agent:

- keeps the running conversation (system + user + assistant turns) internally, so follow-up messages carry context,
- is the only thing that knows how to talk to the LLM API — building the request, sending it, and parsing the response all live inside `Agent::respond`,
- validates the requested model against an allow-list (`deepseek-v4-flash` / `deepseek-v4-pro`) rather than forwarding arbitrary client input to the upstream API,
- can be reset independently of the HTTP layer via `Agent::reset`.

The chat UI has a model toggle (Flash/Pro — both behind the same endpoint and API key) next to the composer, plus a collapsed "Advanced" section for `temperature`, `max_tokens`, and up to 4 `stop` sequences. All of these are optional: leaving a control untouched omits it from the request entirely rather than sending an explicit default, matching how [lesson 02](../02.%20Rust%20LLM%20Response%20Control%20CLI)'s `:set`/`:reset` commands behaved.

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
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `PORT` | no | `3000` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server. `OPENAI_MODEL` only sets the *default* selection shown in the UI at startup — it must be `deepseek-v4-flash` or `deepseek-v4-pro`; anything else falls back to `deepseek-v4-flash`. The model actually used for any given message can be changed at any time via the Flash/Pro toggle.

## Deploy

Deployed automatically by GitHub Actions on every push to `master`. Live at `http://<VDS host>:4006`.

## Conclusion

_(to fill in after recording the demo)_

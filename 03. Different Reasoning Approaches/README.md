# 03. Different Reasoning Approaches

## Задание

🔥 День 3. Разные способы рассуждения

Возьмите одну задачу
(логическую, алгоритмическую или аналитическую)

Решите её через API четырьмя способами:
👉 получите прямой ответ без дополнительных инструкций
👉 добавьте в промпт инструкцию: «решай пошагово»
👉 попросите модель сначала составить промпт для решения задачи,
а затем используйте его
👉 создайте в промпте группу экспертов
(например: аналитик, инженер, критик)
и получите решение от каждого

Сравните:
👉 отличаются ли ответы
👉 какой способ дал наиболее точный результат

Результат:
Несколько решений одной задачи и их сравнение

Формат:
Видео + Код

## Demo

https://github.com/user-attachments/assets/29884017-4b7d-4775-b1e1-ce1e07f28f69

## What this is

A small web app: one prompt box, a Run button, and four output boxes showing the same task solved four different ways:

1. **Direct** — the prompt sent as-is
2. **Step by step** — the prompt with an added "решай пошагово" instruction
3. **Self-planned prompt** — the model is first asked to write a prompt for solving the task, then that generated prompt is sent to get the final answer (both are shown)
4. **Expert panel** — three independent API calls, one per persona (analyst / engineer / critic), each answering the task from their own point of view

All four run in parallel when you click Run, and each box fills in independently as its call finishes. Boxes 3 and 4 are split into labeled sub-sections (generated prompt / answer, and one per expert) for readability, and all output is rendered as Markdown.

A Rust ([axum](https://github.com/tokio-rs/axum)) server holds the API key and proxies the LLM calls; the frontend is a single dependency-free HTML/JS page (including its own small Markdown renderer, so there's no external JS dependency).

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

## Configuration

Set via environment variables:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | no | `gpt-4o-mini` |
| `PORT` | no | `3000` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server (e.g. a local or self-hosted one).

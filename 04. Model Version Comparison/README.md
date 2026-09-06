# 04. Model Version Comparison

## Задание

🔥 День 5. Версии моделей

Выполните один и тот же запрос:
👉 на слабой модели
👉 на средней модели
👉 на сильной модели

(например: из начала, середины и конца списка HuggingFace)

Замерьте:
👉 время ответа
👉 количество токенов
👉 стоимость (если модель платная)

Сравните:
👉 качество ответов
👉 скорость
👉 ресурсоёмкость

Результат:
Короткий вывод о различиях между моделями + ссылки

Формат:
Видео + Код

## Demo

_(coming soon)_

## What this is

A web app very similar to [lesson 03](../03.%20Different%20Reasoning%20Approaches), but instead of comparing reasoning strategies, it sends the exact same plain prompt to three model tiers at once and compares them:

- **Weak** — `qwen2.5:0.5b`, run locally
- **Medium** — `deepseek-v4-flash`
- **Strong** — `deepseek-v4-pro`

Each box shows the model's answer (rendered as Markdown) plus a stats row with:
- **Response time** — measured server-side around the call to that model
- **Token count** — parsed from the API's `usage` field (prompt + completion), shown as "—" if a server doesn't return it
- **Cost** — computed from token counts and per-tier `$/1M token` rates, only if you've configured them (see below); otherwise shown as "—" rather than guessing

All three requests run in parallel and each box fills in independently as its call finishes — since the tiers usually differ a lot in latency, you can visibly watch the weak/local model answer first.

## Run

```bash
cargo run
```

Then open http://localhost:3000.

## Configuration

Each tier is configured independently — same shape, no assumption that any two tiers share a provider (even though medium/strong might in practice):

| Variable | Required | Default |
|---|---|---|
| `WEAK_BASE_URL` | no | `http://localhost:11434/v1` (a local Ollama-style server) |
| `WEAK_API_KEY` | no | `ollama` (local servers usually ignore it) |
| `WEAK_MODEL` | no | `qwen2.5:0.5b` |
| `MEDIUM_BASE_URL` | yes | — |
| `MEDIUM_API_KEY` | yes | — |
| `MEDIUM_MODEL` | no | `deepseek-v4-flash` |
| `STRONG_BASE_URL` | yes | — |
| `STRONG_API_KEY` | yes | — |
| `STRONG_MODEL` | no | `deepseek-v4-pro` |
| `PORT` | no | `3000` |

Cost calculation is opt-in per tier (omit either and cost shows as "—" for that tier):

| Variable | Meaning |
|---|---|
| `MEDIUM_INPUT_COST_PER_1M` | USD per 1M prompt tokens for the medium model |
| `MEDIUM_OUTPUT_COST_PER_1M` | USD per 1M completion tokens for the medium model |
| `STRONG_INPUT_COST_PER_1M` | USD per 1M prompt tokens for the strong model |
| `STRONG_OUTPUT_COST_PER_1M` | USD per 1M completion tokens for the strong model |

Example:

```bash
MEDIUM_BASE_URL=https://api.deepseek.com \
MEDIUM_API_KEY=sk-... \
MEDIUM_INPUT_COST_PER_1M=0.27 \
MEDIUM_OUTPUT_COST_PER_1M=1.10 \
STRONG_BASE_URL=https://api.deepseek.com \
STRONG_API_KEY=sk-... \
cargo run
```

## Conclusion

_(to fill in after running a real comparison across the three models — quality, speed, and resource-cost differences, plus links to the models used)_

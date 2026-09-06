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
- **Cost** — computed from token counts and hardcoded `$/1M token` rates for the medium and strong models (see below); the weak model is local/free, so its cost always shows "—"

All three requests run in parallel and each box fills in independently as its call finishes — since the tiers usually differ a lot in latency, you can visibly watch the weak/local model answer first.

Once all three answers are in, a fourth box below automates the lesson's "compare quality/speed/cost" step: it sends the original task plus all three answers (each with its own time/tokens/cost) to the **strong** model, asking it to judge the differences in quality against the speed/cost trade-offs, and prints its verdict.

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

### Cost rates

Cost is computed from hardcoded per-1M-token rates in [src/main.rs](src/main.rs), approximating DeepSeek's [published pricing](https://api-docs.deepseek.com/quick_start/pricing) (off-peak, cache-miss) as a single flat number per direction:

| Model | Input $/1M | Output $/1M |
|---|---|---|
| `deepseek-v4-flash` (medium) | 0.22 | 0.66 |
| `deepseek-v4-pro` (strong) | 0.66 | 1.98 |

DeepSeek's actual billing varies up to 6x on top of these depending on cache hits and peak/off-peak hours — this tool doesn't model that, so treat the cost figures as ballpark estimates, not exact charges. To change the rates, edit the constants at the top of `src/main.rs` and rebuild.

Example run:

```bash
MEDIUM_BASE_URL=https://api.deepseek.com \
MEDIUM_API_KEY=sk-... \
STRONG_BASE_URL=https://api.deepseek.com \
STRONG_API_KEY=sk-... \
cargo run
```

## Conclusion

_(to fill in after running a real comparison across the three models — quality, speed, and resource-cost differences, plus links to the models used)_

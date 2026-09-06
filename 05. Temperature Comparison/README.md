# 05. Temperature Comparison

## Задание

🔥 День 4. Температура

Выполните один и тот же запрос с параметрами:
👉 temperature = 0
👉 temperature = 0.7
👉 temperature = 1.2

Сравните ответы по:
👉 точности
👉 креативности
👉 разнообразию

Сформулируйте:
👉 для каких задач лучше подходит каждая настройка

Результат:
Примеры ответов с разной температурой и выводы по их использованию

Формат:
Видео + Код

## Demo

_(coming soon)_

## What this is

A web app very similar to [lesson 03](../03.%20Different%20Reasoning%20Approaches) and [lesson 04](../04.%20Model%20Version%20Comparison), but this time only one model (`deepseek-v4-pro`) is used, and the variable being compared is the API's `temperature` parameter instead of the model or the prompting strategy.

Three boxes, each with its own slider (defaulting to 0 / 0.7 / 1.2, matching the task), let you set that box's temperature. Clicking Run sends the same prompt to each box **three times** at its temperature — nine calls total — because a single sample can't show diversity: you need to see the same prompt+temperature produce identical output at `temperature=0` and varied output at higher temperatures across repeated runs.

Once all nine runs are in, a fourth box automates the lesson's "state which tasks each setting suits" step: it sends the original task plus all three temperatures' sampled answers to the model (at `temperature=0`, for a focused judgment) and asks it to compare accuracy/creativity/diversity and conclude which setting fits which kind of task.

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

## Conclusion

_(to fill in after running a real comparison — example answers at each temperature, and which tasks each setting suits)_

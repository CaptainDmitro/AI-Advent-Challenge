# 12. Personalization

## Задание

🔥 День 12. Персонализация ассистента

Добавьте персонализацию поверх модели памяти:

👉 создайте профиль пользователя
👉 опишите предпочтения (стиль, формат, ограничения)
👉 подключите профиль к каждому запросу

Проверьте:

👉 ответы для разных профилей
👉 что ассистент учитывает автоматически

Результат:

Персонализированный агент, адаптированный под пользователя

Формат:

Видео + Код

## Demo

https://github.com/user-attachments/assets/89f1a781-ee5e-4d8d-a1c3-ed9a2e858c4e

## What this is

Lesson 11 built three memory layers - short-term, working, long-term - each with its own file and its own write rule. This lesson adds a fourth thing that sits *beside* those layers rather than inside them: a **personalization profile** (`UserProfile`, `profile.json`) that describes *how* the agent should answer, not *what* it knows.

The line between "profile" and "long-term memory" is deliberately the write path, not the content:

- **Long-term memory** (lesson 11, unchanged) is written by an LLM extractor that watches the conversation and infers durable facts - the agent *guesses* what's worth remembering.
- **The profile** is only ever written by an explicit `PUT /api/profile` call. Nothing an LLM infers ever lands here. If the user doesn't set style, tone, format, language, or constraints, the profile is empty and the agent behaves exactly like lesson 11's plain agent - personalization is opt-in, not a hidden inference the user can't see or override.

This mirrors the exercise's own framing: style/format/constraints are things a user should be able to *guarantee*, not things left to a model's judgment about what "seems" durable.

### The profile shape

```rust
struct UserProfile {
    style: Option<String>,        // e.g. "concise, no fluff" / "detailed, step-by-step"
    tone: Option<String>,         // e.g. "direct and professional" / "warm and encouraging"
    format: Option<String>,       // e.g. "short bullet points" / "plain prose with analogies"
    language: Option<String>,     // e.g. "English" / "Russian"
    constraints: Vec<String>,     // hard rules, e.g. "Never use emojis"
}
```

`GET /api/profile` reads it, `PUT /api/profile` replaces it wholesale (posting `{"style": "concise"}` clears every other field rather than merging - explicit over implicit, same philosophy as lesson 11's `remember`), `DELETE /api/profile` clears it back to empty.

### Connected to every request

`build_context` (pure, unit-tested, extended from lesson 11) now assembles: system prompt → **profile block** (if set) → long-term block (if non-empty) → working-memory block (if a task is active) → the full short-term dialogue. The profile is checked first and is the only block with no gate at all besides "is it empty" - unlike working memory, it doesn't depend on a task being active, and unlike long-term, it never needs an LLM call to populate. Every `/api/chat` turn goes through this same function, so there's no code path that "forgets" to apply the profile.

The long-term extractor prompt (`LONG_TERM_MEMORY_PROMPT`) was also tightened: it's now explicitly told *not* to extract style/tone/format/language preferences, since those have a dedicated home. Without that change, a user saying "please be more concise" could get captured twice - once correctly in the profile (if they set it explicitly) and once redundantly as an inferred long-term fact - which would silently duplicate instructions in the context and make it unclear which layer is actually responsible for the behavior.

### Testing what changes per profile, and what happens automatically

Two presets ship in the UI for a fast A/B:

- **Concise expert** - style: "concise, no fluff"; tone: "direct and professional"; format: "short bullet points"; constraints: "Never use emojis", "assume advanced technical background".
- **Friendly beginner** - style: "detailed, step-by-step"; tone: "warm and encouraging"; format: "plain prose with simple analogies"; constraints: "avoid jargon", "never assume prior expertise".

Scenario used while building this: with no profile set, ask *"How does a hash map work?"* - a normal, medium-length explanation with a bit of jargon (buckets, load factor). Apply the **concise expert** preset and ask the exact same question in a fresh conversation (`POST /api/reset` first, since short-term is the only thing that needs clearing) - the reply comes back as a handful of terse bullet points, no analogies, no hedging. Switch to **friendly beginner** and ask again - the reply turns into a walked-through explanation with an analogy (a hash map as "a wall of labeled mailboxes") and no unexplained jargon. Same question, same model, same temperature - the only thing that changed between the three runs is which JSON file `build_context` read from.

What the agent accounts for *automatically*, without the user restating anything mid-conversation:

- The profile block is present on **every** turn once set, including the very first message of a brand-new conversation after `/api/reset` - there's no "warm-up" turn needed, because the profile isn't learned, it's just read.
- Long-term facts stack on top of the profile independently. Set the concise-expert profile, then mention *"I'm allergic to shellfish"* in one conversation and later ask an unrelated food question in a different task - the shellfish fact still surfaces (long-term, inferred, task-independent) while the reply is still terse bullet points (profile, explicit, always-on). The two layers are additive and don't interfere with each other's gate.
- Clearing the profile (`DELETE /api/profile`) immediately reverts every subsequent reply to the plain lesson-11 behavior, with long-term and working memory completely untouched - personalization is fully reversible without losing what the agent has learned about the user.

### Token cost of personalization

`MemoryReport` (extended from lesson 11) now reports `profile_tokens` alongside `system_prompt_tokens` / `long_term_tokens` / `working_memory_tokens` / `short_term_tokens`, plus a `profile_is_set` flag. In practice a filled-out profile costs on the order of 30-60 tokens per request - one of the cheapest levers in this whole memory system, since (unlike long-term or working memory) it never needs an extra LLM call to keep it current; it's a static block until the user explicitly changes it.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000.

Set a profile directly via curl instead of the UI, if you prefer:

```bash
curl -X PUT localhost:3000/api/profile \
  -H 'Content-Type: application/json' \
  -d '{"style": "concise, no fluff", "tone": "direct", "format": "bullet points", "constraints": ["Never use emojis"]}'

curl localhost:3000/api/profile      # inspect
curl -X DELETE localhost:3000/api/profile   # clear back to empty
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `PROFILE_FILE` | no | `profile.json` |
| `SHORT_TERM_FILE` | no | `short_term.json` |
| `WORKING_MEMORY_FILE` | no | `working_memory.json` |
| `LONG_TERM_FILE` | no | `long_term.json` |
| `CONTEXT_LIMIT_TOKENS` | no | `64000` |
| `PRICE_PER_1M_INPUT_TOKENS` | no | unset (cost hidden) |
| `PRICE_PER_1M_OUTPUT_TOKENS` | no | unset (cost hidden) |
| `PORT` | no | `3000` |

The profile lives in its own file, loaded/saved independently from the three memory layers - a restart resumes the profile, the conversation, the active task, and the durable facts all separately, and a problem reading any one of them only resets that layer.

## Deploy

Deployed automatically by GitHub Actions on every push to `master` (this becomes the highest-numbered lesson, so it becomes the new default deploy target - see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). Live at `http://<VDS host>:4012`.

## Conclusion

_(to fill in after recording the demo)_

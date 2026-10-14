# 29. Local LLM Optimization

## Задание

🔥 День 29. Оптимизация локальной LLM

Оптимизируйте локальную модель под свою задачу:

👉 настройте параметры (temperature, max tokens, context window)
👉 попробуйте квантование (если доступно)
👉 измените prompt-шаблон под конкретный кейс

Сравните:

👉 качество ответов до оптимизации
👉 качество после
👉 скорость и потребление ресурсов

Результат:

Оптимизированная локальная LLM под конкретную задачу

Формат:

Видео + Код

## Demo

https://github.com/user-attachments/assets/62dee509-e1a8-426b-874e-7f1706a3ba2f

## What this is

The task is the one [lesson 28](../28.%20Local%20RAG) gave the local model:
answer a question about this repo from five retrieved fragments, and cite
them as `[n]`. Lesson 28 measured where `qwen2.5:3b` falls short at it. It
stated 83% of the expected facts and lost the rest by paraphrasing
(`forbidden_terms` became "a keyword pre-check"). It cited a fragment in
**0 of 60** answers. This lesson tunes the same local model for that task
and measures every step, on lesson 22's 10 control questions.

Retrieval doesn't change. [`src/main.rs`](./src/main.rs) uses lesson 28's
[`index.rs`](../28.%20Local%20RAG/src/index.rs) directly (`#[path]`, not a
copy), and every profile gets the same top-5 chunks. Only what goes to Ollama
changes, **one thing at a time**:

| Profile | Model | Prompt | temperature | seed | num_ctx | num_predict |
|---|---|---|---|---|---|---|
| `baseline`: lesson 28 as it was | `qwen2.5:3b` (Q4_K_M) | lesson 28's | not sent (Ollama's 0.8) | – | 8192 | no cap |
| `params`: tuned parameters | same | same | **0** | **42** | **4096** | **300** |
| `optimized`: + tuned prompt | same | **tuned** | 0 | 42 | 4096 | 300 |
| `q8`: optimized on Q8_0 | **`qwen2.5:3b-instruct-q8_0`** | tuned | 0 | 42 | 4096 | 300 |
| `q3`: optimized on Q3_K_M | **`qwen2.5:3b-instruct-q3_K_M`** | tuned | 0 | 42 | 4096 | 300 |

### What each change is for

- **temperature 0 + seed.** The task has one right answer, and creativity
  only costs facts. In lesson 28 at 0.7, 3 of 10 questions changed their fact
  score between runs. Greedy decoding with a fixed seed gives the same answer
  every run, so a result can be reproduced.
- **num_ctx 8192 → 4096.** The context window is memory: Ollama allocates the
  KV cache for the whole window when it loads the model. Measured on this
  Mac, `qwen2.5:3b` takes **2.26 GB** at 8192 and **2.09 GB** at 4096
  (`ollama ps`), so 173 MB less. The prompt still fits: five chunks of at
  most 1800 characters plus the rules come to ≈2.5k tokens at most. The page
  shows the longest prompt next to `num_ctx`, so this can be checked, not
  assumed. 2048 would cut long prompts, and Ollama would silently drop the
  start of them.
- **num_predict 300.** The max tokens of the answer. Correct answers are
  40–120 tokens long. The cap only stops a runaway answer (the model copying
  a whole fragment), and the page counts how many answers it cut
  (`done_reason: "length"`).
- **The prompt template.** Lesson 28's prompt is general: one paragraph of
  rules in the system prompt, the fragments and the question below. A 3B
  model reads the fragments well but ignores the rules. The tuned template
  is written for this model and this task:
  1. **short numbered rules** instead of a paragraph,
  2. **an exact output shape**: 1–5 bullets of one sentence each, every one
     ending in `[n]`,
  3. **"copy names, numbers, commands and identifiers exactly"**, against
     the paraphrasing that cost lesson 28 its facts,
  4. **one worked example** (a user turn with made-up fragments and an
     assistant turn answering in the exact format). A small model copies a
     format it was shown much better than one it was told about. The example
     is about an invented `docs/app.md`, and a test checks that it contains
     none of the 10 questions' expected facts, so it can't score by leaking
     answers,
  5. **the format reminder repeated after the question**, because the last
     thing a small model reads is what it follows.
- **Quantization.** The same 3.1B model at three precisions. Q4_K_M is
  Ollama's default (`qwen2.5:3b`, 1.93 GB file). Q8_0 is nearly lossless and
  twice the size. Q3_K_M is smaller and lighter. The question is whether
  8-bit buys back quality the task needs, and whether 3-bit still does the
  job.

A quick check of the template on this Mac, before the full run. q01 with the
DEPLOYMENT.md and AGENTS.md sections as fragments:

```
baseline:   A deployed lesson gets a binary at `~/apps/lesson-NN/app`, … The port is always
            `4000 + NN` and the service name is always `ai-advent-lesson-NN`.
optimized:  - A deployed lesson gets … a systemd service named `ai-advent-lesson-NN` on the VDS [3].
            - The service runs as user-level systemd units and the port is `4000 + NN` [3].
```

Same facts, now with the source. The first version of the template, without
"one sentence each, only what the question asks", made the model copy whole
fragments into a bullet. That rule is there because of it.

### How it's measured

Every profile runs over the same 10 control questions (1–5 runs each). Before
a profile's questions, **every model is unloaded and the profile's model is
cold-loaded with its own `num_ctx`**, so the load time and memory belong to
that profile alone.

| | What | How |
|---|---|---|
| **Quality** | expected facts stated | each fact passes if the answer contains any of its variants (lesson 22's checks) |
| | answers citing the expected source | the answer cites `[n]`, and fragment `n` is from the expected file |
| | answers with any `[n]` citation | does the model follow the format at all |
| | citations of fragments that don't exist | `[7]` when there are five: a made-up source |
| | answers cut by `num_predict` | `done_reason: "length"`. A cap that cuts correct answers is too low |
| **Speed** | time per answer | wall-clock time of the call |
| | generation tok/s, prefill | Ollama's own nanosecond counters |
| | longest prompt / `num_ctx` | proof that the smaller window still fits the prompt |
| | cold load | wall-clock time of loading the model from nothing with its `num_ctx` |
| **Resources** | in memory, on the GPU, loaded context | `/api/ps`: weights + KV cache + buffers (`size`, `size_vram`, `context_length`) |
| | model file, quantization | `/api/tags` |
| **Stability** | failed calls, one answer across runs, fact score changed between runs | as in lesson 28 |

The page has the profiles side by side (the model of each can be switched to
any pulled one, e.g. `qwen2.5:3b-instruct-q5_K_M`), a question box, and
**Run all 10 on ticked profiles**. The **before → after** table fills in as it
runs and marks the best value in each row. Clicking a question shows every
profile's answer next to the others, with the facts ✓/✗, made-up citations
struck through, and the exact request (options and messages).

The same from the terminal, as Markdown tables plus every answer:

```bash
cargo run -- eval --runs 3
```

`--profiles baseline,optimized` runs only those.

### Results

Measured on the deployed app (`http://<VDS host>:4029`): retrieval on the
VDS, generation on an M-series MacBook (16 GB) through the tunnel, Ollama
0.40.3. All 10 control questions, 3 runs each, every profile cold-loaded
alone first: 150 answers, no failed calls.

| | | `baseline` | `params` | `optimized` | `q8` (Q8_0) | `q3` (Q3_K_M) |
|---|---|---|---|---|---|---|
| **Quality** | expected facts stated | 75% | **83%** | 79% | 81% | 25% |
| | answers citing the expected source | 1/30 | 0/30 | **30/30** | **30/30** | 27/30 |
| | citations of fragments that don't exist | 0 | 0 | 0 | 0 | 0 |
| | answers cut by `num_predict` | 0/30 | 0/30 | 0/30 | 0/30 | 12/30 |
| **Speed** | time per answer | 3.23 s | 3.35 s | 3.24 s | 4.17 s | 7.95 s |
| | generation tok/s | 41.9 | 41.7 | 37.1 | 25.7 | 28.7 |
| | prefill | 1.10 s | 1.21 s | 1.54 s | 1.49 s | 1.65 s |
| | answer length, tokens | 83 | 83 | **57** | 64 | 173 |
| | longest prompt / `num_ctx` | 2066 / 8192 | 2066 / 4096 | 2245 / 4096 | 2245 / 4096 | 2245 / 4096 |
| | cold load | 2.44 s | 0.63 s | 0.64 s | 3.17 s | 1.91 s |
| **Resources** | in memory (`ollama ps`, all on the GPU) | 2.26 GB | **2.09 GB** | **2.09 GB** | 3.44 GB | 1.75 GB |
| | model file | 1.93 GB | 1.93 GB | 1.93 GB | 3.29 GB | 1.59 GB |
| **Stability** | questions with one answer across 3 runs | 0/10 | 6/10 | 9/10 | **10/10** | 2/10 |
| | questions whose fact score changed between runs | 3/10 | 0/10 | 0/10 | 0/10 | 2/10 |

Per question: facts found in each of the 3 runs, and how many runs cite the
expected source:

| # | `baseline` | `params` | `optimized` | `q8` | `q3` |
|---|---|---|---|---|---|
| q01 | 2 2 2 / 2 · 0 | 2 2 2 / 2 · 0 | 2 2 2 / 2 · 3 | 2 2 2 / 2 · 3 | 1 1 1 / 2 · 3 |
| q02 | 3 0 2 / 3 · 0 | 3 3 3 / 3 · 0 | 3 3 3 / 3 · 3 | 3 3 3 / 3 · 3 | 0 0 0 / 3 · 3 |
| q03 | 1 1 1 / 1 · 0 | 1 1 1 / 1 · 0 | 1 1 1 / 1 · 3 | 1 1 1 / 1 · 3 | 0 0 0 / 1 · 3 |
| q04 | 2 2 2 / 2 · 0 | 2 2 2 / 2 · 0 | 2 2 2 / 2 · 3 | 2 2 2 / 2 · 3 | 0 0 0 / 2 · 3 |
| q05 | 2 2 2 / 3 · 0 | 2 2 2 / 3 · 0 | 2 2 2 / 3 · 3 | 2 2 2 / 3 · 3 | 3 2 2 / 3 · 3 |
| q06 | 1 1 1 / 2 · 1 | 1 1 1 / 2 · 0 | 2 2 2 / 2 · 3 | 2 2 2 / 2 · 3 | 1 1 1 / 2 · 3 |
| q07 | 4 4 4 / 4 · 0 | 4 4 4 / 4 · 0 | 4 4 4 / 4 · 3 | 4 4 4 / 4 · 3 | 0 0 0 / 4 · 3 |
| q08 | 4 4 4 / 5 · 0 | 4 4 4 / 5 · 0 | 3 3 3 / 5 · 3 | 4 4 4 / 5 · 3 | 3 4 4 / 5 · 0 |
| q09 | 2 1 2 / 3 · 0 | 3 3 3 / 3 · 0 | 1 1 1 / 3 · 3 | 0 0 0 / 3 · 3 | 0 0 0 / 3 · 3 |
| q10 | 2 1 1 / 3 · 0 | 1 1 1 / 3 · 0 | 1 1 1 / 3 · 3 | 2 2 2 / 3 · 3 | 0 0 0 / 3 · 3 |

What this shows:

- **Parameters fix stability, and that alone raises quality.** At
  Ollama's default 0.8 the baseline never gave the same text twice, and on 3
  questions its fact score jumped between runs (q02: 3, 0, 2 of 3). Temperature 0
  with a seed removes the bad samples: 75% → 83%, no drift. The 4096 window
  saves 170 MB, and the longest prompt (2245 tokens) still fits with room
  to spare. Nothing ran into the 300-token cap. Speed doesn't change.
- **The prompt template fixes the format.** Citations go from 0–1 of 30 to
  **30 of 30**, all pointing at the expected source, and none at a fragment
  that doesn't exist. Answers are 30% shorter, and q06 now gives the reason
  (`rustfmt`) instead of only "non-blocking". The price is 4 points of facts:
  one short sentence per bullet drops detail that a long answer had by
  accident. On q09 the bullets keep the rule (`<server>{SEP}<tool>`) but no
  longer the example (`crates__crate_info`) that has the `__`, and on q08 one
  of the sizes is gone. Time per answer is the same, because the
  worked example adds ≈180 tokens of prefill and the shorter answer wins them
  back.
- **Q8_0 buys almost nothing on this task.** It has the same quality (81%,
  30/30) and the most stable answers (identical on all 10 questions), but
  1.6× the memory (3.44 GB), 30% slower generation and the slowest cold load.
  Q4_K_M already keeps everything this task needs.
- **Q3_K_M breaks.** At 3 bits the model can't tell the worked example from
  the real question. On q02 it answered the example's question about
  `~/.cache/app` instead, and on q07 it repeated one sentence until
  `num_predict` cut it off (12 of 30 answers). It saves 340 MB, but facts drop
  to 25% and it is 2.5× slower, because the answers are longer. A few-shot
  prompt and heavy quantization don't go together.

### Verification

`cargo test` needs neither Ollama nor the network. A fake Ollama (an axum
router on a random port) answers `/api/chat`, `/api/ps`, `/api/tags` and
`/api/version`. It echoes the fragments back, keeps track of what is loaded
with which `num_ctx`, and reports a memory size that grows with the context.

- `profiles_change_one_thing_at_a_time`: the baseline sends only
  `num_ctx: 8192`, like lesson 28. `params` changes only the options,
  `optimized` only the prompt, and `q8`/`q3` only the model.
- `each_profile_sends_its_own_prompt_and_options`: what Ollama actually
  receives per profile. For lesson 28's prompt: `system, user`, and no
  `temperature`. For the tuned one: `system, user, assistant, user`, with
  the example answer, the real fragments from `[1]`, and the format reminder
  after the question. The quantized profiles send exactly the optimized
  request.
- `the_worked_example_gives_nothing_away`: no expected fact of any control
  question is in the example, and the example follows its own format.
- `warm_up_starts_cold_and_reports_memory`: an already loaded model is
  unloaded first, the model is loaded with the profile's `num_ctx`, and
  `/api/ps` and `/api/tags` become the memory, context and file size. The
  baseline's 8192 window takes more memory than 4096.
- `scores_and_summary_count_citations_truncation_and_stability`: `[7]` out of
  five fragments is a made-up citation, `done_reason: "length"` counts as cut,
  three runs at temperature 0 give one answer, and a model that isn't pulled
  gives Ollama's 404 as text.
- `http_api_runs_profiles_and_reports_errors`: every endpoint, a model
  override, unknown profile or question, an empty question, and Ollama down
  (readable errors, the retrieved chunks still shown).
- `control_questions_are_grounded_in_the_corpus`, `citations_are_parsed`,
  `normalizes_ollama_host_and_model_names_like_ollama_does`, plus lesson 28's
  index tests.

## Run

Ollama with the three quantizations of the model (macOS):

```bash
brew services start ollama
```

```bash
ollama pull qwen2.5:3b
```

```bash
ollama pull qwen2.5:3b-instruct-q8_0
```

```bash
ollama pull qwen2.5:3b-instruct-q3_K_M
```

Then open http://localhost:3000:

```bash
cargo run
```

A profile whose model isn't pulled shows "not pulled" with the `ollama pull`
command, and its answers show Ollama's error. The other profiles still run.

Over HTTP:

```bash
curl -s localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"case_id": "q06", "profile": "optimized", "runs": 2}' | jq '{model, options, answer: .runs[0].text, score: .runs[0].score}'
```

| Endpoint | |
|---|---|
| `GET /api/config` | the profiles with all their settings, the index, `top_k` |
| `GET /api/status` | Ollama's version, the pulled models (size, quantization) and what's in memory |
| `GET /api/cases` | the 10 control questions |
| `POST /api/ask` | `{question \| case_id, profile, model?, runs?}` → retrieved `sources`, the exact `messages` and `options`, the `runs` with their scores and counters, `stability` |
| `POST /api/warmup` | `{profile, model?}`: unloads every model, cold-loads this one with the profile's `num_ctx` → `{load_ms, loaded, disk_bytes, quantization}` |

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OLLAMA_HOST` | no | `http://localhost:11434`, with or without `http://` |
| `EMBEDDER` | no | `local`: lesson 28's hashed TF-IDF in-process. `ollama`: `nomic-embed-text` from the same Ollama |
| `OLLAMA_EMBED_MODEL` | with `ollama` | `nomic-embed-text` |
| `PORT` | no | `3000` |

The profiles are in the code (`PROFILES`), so every run of every profile
sends exactly what this README says. No API key: nothing leaves the machine.

## Deploy

The pipeline deploys it to `http://<VDS host>:4029` on push to `master`, as the
newest web-app lesson (see [`../DEPLOYMENT.md`](../DEPLOYMENT.md)). The
retrieval runs on the VDS inside the binary, and generation uses **the Ollama
on your Mac** through lesson 26's reverse SSH tunnel:

```bash
"../26. Local LLM/tunnel.sh" <user>@<vds-host> -i ~/path/to/key
```

The memory and load numbers are then the Mac's, which is where the model
runs. A warm-up unloads every model on that Ollama. While the tunnel is down,
the status turns red and the answers show "Can't reach Ollama". ↻ turns it
green once the tunnel is back.

## Conclusion

The optimized local LLM for this task is the `optimized` profile on the default
Q4_K_M quantization: temperature 0 + seed, `num_ctx` 4096, `num_predict` 300,
and the tuned prompt template. Against lesson 28's setup on the same model:

- **Quality:** citations go from 0–1 of 30 to 30 of 30, and facts from 75% to 79%.
- **Stability:** near-identical answers across runs, against a different text every time.
- **Resources:** 170 MB less memory at the same speed (≈3.2 s per answer).

Each change fixed one thing. Parameters fixed stability. The prompt fixed
the format, which lesson 28 thought would need a bigger model. Quantization
showed where the floor is: 8-bit adds memory, not quality, and 3-bit falls
apart on the same prompt. What's left is a 4-point fact gap against the
long-answer `params` profile. It comes from answers that are shorter, not
wrong.

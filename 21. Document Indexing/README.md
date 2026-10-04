# 21. Document Indexing

## Задание

🔥 День 21. Индексация документов

Возьмите набор документов:

👉 README / статьи / код / pdf → текст
👉 минимум 20–30 страниц текста суммарно (или эквивалент в коде)

Реализуйте пайплайн индексации:

👉 разбиение на чанки (chunking)
👉 генерация эмбеддингов
👉 сохранение индекса (FAISS / SQLite / JSON)

Усиление:

👉 добавьте метаданные к каждому чанку (source, title/file, section, chunk_id)
👉 сделайте минимум 2 стратегии chunking и сравните их:

- по фиксированному размеру
- по структуре (заголовки/разделы/файлы)

Результат:

Локальный индекс документов с эмбеддингами + метаданные + сравнение 2 стратегий chunking

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

An indexing pipeline over **this repository itself**:

```
 documents ──▶ outline ──▶ chunking ──────────────▶ embeddings ──▶ index/<strategy>.json
 (26 files)    sections    ├─ fixed: 1000-char windows, 150 overlap      ├─ chunks + metadata
               per file    └─ structural: one chunk per section          └─ vectors (+ IDF)
                                                                                  │
                         comparison: size stats · structure checks · retrieval on 20 known questions
```

### The documents

The corpus is built into the binary at compile time with `include_str!`. It
has every lesson's `README.md`, the root `README.md`, `AGENTS.md` and
`DEPLOYMENT.md`, which are Markdown split by headings. It also has the source
of lessons 06, 16 and 20, which is Rust split by top-level items. That is 26
files and about 260 000 characters, roughly **145 pages** at 1800 characters a
page. CI checks out the whole repo, so the sibling folders are there when it
compiles, and the deployed binary carries the corpus inside it.

`DOCS_DIR=/some/folder` indexes your own `.md`, `.rs` and `.txt` files instead.

### Chunking: two strategies

Both strategies start from the same **outline** of each file:

- **Markdown**: one section per heading, named by its path of headings, e.g.
  `20. MCP Orchestration › What this is › Verification`. A `#` inside a
  fenced code block, such as a shell comment, is not a heading.
- **Rust**: one section per top-level block (`fn checksum`, `impl Registry`,
  `struct AppState`, `mod tests`). rustfmt puts a blank line between
  top-level items and indents what's inside them, so a block starts at a
  non-indented line after a blank line. Doc comments and attributes stay with
  their item. A `// ---- banner ----` is named by its text.

| | **fixed** | **structural** |
|---|---|---|
| How it cuts | windows of `fixed_size` chars (1000), each sharing `overlap` chars (150) with the previous one. It backs off to the last space so no word is cut in half | one chunk per section. Sections under `min_chars` (300), such as a title line, are joined with the next ones. Sections over `max_chars` (1800) are split at paragraphs: for code, between the methods of an `impl` or the tests of `mod tests` |
| Knows about structure | no. The section is looked up afterwards: where the window starts, and every section it touches | yes. A chunk is a section, and a split piece of code is named `impl Agent › fn respond` |
| Overlap | yes, so text is indexed about 1.16× | none, so text is indexed exactly once |

### Metadata on every chunk

```json
{
  "chunk_id": "structural:DEPLOYMENT.md#001",
  "source":   "DEPLOYMENT.md",
  "title":    "Deployment",
  "kind":     "markdown",
  "section":  "Deployment › Why a static musl binary instead of Docker",
  "sections": ["Deployment › Why a static musl binary instead of Docker"],
  "ordinal":  1,
  "start": 1163, "end": 1549,
  "chars": 384, "tokens": 97,
  "text": "## Why a static musl binary instead of Docker\n\nThe binary is …",
  "embedding": [ …1024 floats… ]
}
```

`sections` lists every section the chunk has text from. A clean chunk has
exactly one. A fixed window usually has two or three, and the page draws
those chunks striped. `start`/`end` are byte offsets in the source file, so
a chunk can always be traced back to where it came from.

### Embeddings

- **`local`** (the default) is a hashed TF-IDF vector with 1024 dimensions,
  computed in-process with no network and no model download. Words are
  lowercased and cut to their first 5 characters, so `chunk`, `chunks` and
  `chunking`, or `сообщения` and `сообщений`, land together. `snake_case`
  identifiers also stay whole. Each stem and each adjacent pair of stems is
  hashed into a signed bucket (the hashing trick) and log-scaled. The vector
  is then weighted by per-bucket IDF learned from that index's chunks and
  made unit-length. The IDF table is saved with the index, because a query
  has to be embedded the same way. This is lexical similarity, not semantics:
  it finds `musl` when you ask about `musl`, but not a synonym. That is enough
  to compare the chunking strategies, and the lesson's deploy only has a
  DeepSeek key, and DeepSeek has no embeddings endpoint.
- **`openai`** works with any OpenAI-compatible `/embeddings` endpoint, such as
  OpenAI `text-embedding-3-small` or a local Ollama `nomic-embed-text`. Inputs
  go in batches of 64, results are put back in input order by their `index`,
  and every vector is normalized.

The embedder's id is stored in the index. Searching an index that was built
with a different embedder is refused with "rebuild the index", instead of
silently comparing vectors from two different spaces.

### The index on disk

`index/fixed.json` and `index/structural.json` each hold `meta` and
`chunks`. `meta` has the strategy, its parameters, the embedder, the
dimensions, the creation time, a checksum of the corpus, the counts, the
chunking and embedding times, and the IDF for the local embedder. `chunks` is
the list above, with vectors. Files are written with write-then-rename. On
startup the index is loaded from disk if it matches the corpus and the
embedder. Otherwise it is rebuilt, keeping the last chunking parameters.

### Comparing the strategies

**Structure**: chunk count, size (average, median, min, max, standard
deviation), ≈ tokens, and redundancy (characters indexed ÷ corpus
characters). It also measures what share of chunks are **inside one
section**, **start at a section**, and **end at a paragraph** rather than
mid-sentence.

**Retrieval**: 20 questions with a known answer section, such as *"Why
compile a static musl binary instead of using Docker?"* →
`DEPLOYMENT.md › Why a static musl binary…`. They cover the deploy docs,
lesson READMEs, a Russian `Задание`, and two pieces of code (`impl Agent`,
`fn inspect`). For each strategy it reports:

- **hit@1 / hit@3 / MRR@10**: is a chunk from the answer section ranked first,
  in the top 3, and how high on average;
- **focus**: what share of that relevant chunk is actually the answer section.
  The rest is text from neighbouring sections that a RAG prompt would carry
  as noise;
- **coverage@3**: what share of the answer section the top 3 chunks contain,
  i.e. whether the whole answer comes back or only a piece of it.

A fixed window that straddles two sections counts as a hit for both, so
fixed chunking can score well on hit@k and still score low on focus. The
comparison shows both numbers next to each other.

The page shows the corpus, the parameters with a **Rebuild** button, the
comparison table with the better value of each metric highlighted, and the
per-question ranks. It also has a **chunk map**: for any file, a bar of its
sections next to where each strategy cut it, and clicking a chunk shows its
metadata, text and the start of its vector. Finally, a search box queries
both indexes side by side.

### Verification

`cargo test` needs no network:

- `builtin_corpus_is_big_enough`: 26 documents, Markdown and Rust, at least 30
  pages.
- `markdown_outline_follows_headings_and_skips_code_fences`,
  `rust_outline_finds_top_level_items`: heading paths, preamble, `#` in code
  fences, doc comments and attributes attached to their item, banners and
  `impl<T> … for …`.
- `fixed_windows_cover_the_text_with_overlap_and_respect_utf8`: Cyrillic text,
  windows ≤ size, overlap or exact adjacency, no text lost.
- `pack_respects_the_limit_and_paragraphs`,
  `structural_chunks_follow_sections`,
  `long_code_sections_split_between_methods_and_keep_their_names`.
- `both_strategies_cover_every_document_within_their_limits`: on every corpus
  file, every non-blank character is in some chunk, chunks fit their limits,
  every chunk has a section, and structural chunks never overlap.
- `local_embedder_is_deterministic_and_ranks_related_text_higher`: unit
  vectors, the same output twice, and stems matching Russian word forms.
- `indexes_carry_metadata_and_round_trip_through_json`: unique `chunk_id`s,
  `text` equal to the source slice, and the index read back from disk equal
  to the one that was saved.
- `eval_cases_point_at_real_sections`: every question's expected section
  exists, so a README edit can't silently break the benchmark.
- `comparison_shows_how_the_strategies_differ`: fixed redundancy > 1.05 and
  structural ≤ 1. Structural chunks start at sections, stay inside them and
  end at paragraphs more often, and have higher focus. Both strategies
  retrieve (hit@3 > 0.3).
- `remote_embedder_batches_and_keeps_the_order`: against a fake
  `/embeddings`, 70 texts are sent as 64 + 6 with the bearer key and model,
  answers that arrive reversed are put back in order, and an HTTP error is
  reported.
- `http_api_builds_compares_and_searches`, `params_are_validated`,
  `load_dir_reads_supported_files_only`.

## Run

```bash
cargo run
```

Then open http://localhost:3000. The index is built on first start (the local
embedder needs no network) and saved to `index/`.

Only rebuild and print the comparison:

```bash
cargo run -- index
```

With real embeddings:

```bash
EMBEDDER=openai EMBEDDING_API_KEY=sk-... cargo run
EMBEDDER=openai EMBEDDING_BASE_URL=http://localhost:11434/v1 EMBEDDING_MODEL=nomic-embed-text EMBEDDING_API_KEY=ollama cargo run
```

Or with curl:

```bash
curl -s -X POST localhost:3000/api/index -H 'Content-Type: application/json' \
  -d '{"fixed_size": 800, "overlap": 100}' | jq '.strategies[] | {strategy, stats, eval: (.eval | del(.rows))}'

curl -s -X POST localhost:3000/api/search -H 'Content-Type: application/json' \
  -d '{"query": "which port does a deployed lesson get", "k": 3}' \
  | jq '.results[] | {strategy, hits: [.hits[] | {score, chunk_id, section}]}'

jq '.chunks[0] | del(.embedding)' index/structural.json
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `EMBEDDER` | no | `local`: hashed TF-IDF in-process. `openai`: an OpenAI-compatible `/embeddings` endpoint |
| `EMBEDDING_API_KEY` | with `openai` | falls back to `OPENAI_API_KEY` |
| `EMBEDDING_BASE_URL` | no | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `DOCS_DIR` | no | unset: the built-in corpus (this repo's docs and code). Otherwise every `.md`/`.rs`/`.txt` under that folder, up to 500 files of up to 1 MB |
| `INDEX_DIR` | no | `index`, in the working directory |
| `PORT` | no | `3000` |

No LLM key is needed: nothing in this lesson calls a chat model.

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4021`
with the local embedder: the pipeline only supplies `OPENAI_*`, and this
lesson doesn't read them. The index is saved to `~/apps/lesson-21/index/` and
rebuilt on start whenever the corpus inside the new binary has changed.

## Conclusion

_(to fill in after recording the demo)_

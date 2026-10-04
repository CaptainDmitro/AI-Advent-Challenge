//! The document index from lesson 21, cut down to what a RAG query needs: the
//! same built-in corpus, the structural chunking strategy (one chunk per
//! Markdown section or top-level Rust item, which lesson 21 showed keeps chunks
//! focused on one topic) and the same two embedders.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

/// Structural chunking: the largest chunk; a longer section is split at
/// paragraphs.
const MAX_CHARS: usize = 1800;
/// Structural chunking: a shorter section is joined with the ones after it.
const MIN_CHARS: usize = 300;
/// Dimensions of the local hashed embedding.
const LOCAL_DIMS: usize = 1024;
/// Local stems: the first few characters of a word, so "chunk", "chunks" and
/// "chunking", or "сообщения" and "сообщений", become the same feature.
const STEM_CHARS: usize = 5;
/// How many texts go into one `/embeddings` request.
const EMBED_BATCH: usize = 64;
/// The section of whatever comes before a file's first heading or item.
const PREAMBLE: &str = "(top of file)";
const PATH_SEP: &str = " › ";

// ---------------------------------------------------------------------------
// The corpus: this repo's own docs and code, built into the binary.
// ---------------------------------------------------------------------------

macro_rules! builtin_corpus {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../../", $path)))),*]
    };
}

/// Every earlier lesson's README and the repo-level docs (Markdown, split by
/// headings), plus the source of three lessons (Rust, split by top-level
/// items). CI checks out the whole repo, so the sibling folders exist at
/// compile time and the deployed binary carries the corpus inside it.
pub const BUILTIN: &[(&str, &str)] = builtin_corpus![
    "README.md",
    "AGENTS.md",
    "DEPLOYMENT.md",
    "01. Rust LLM Chat CLI/README.md",
    "02. Rust LLM Response Control CLI/README.md",
    "03. Different Reasoning Approaches/README.md",
    "04. Model Version Comparison/README.md",
    "05. Temperature Comparison/README.md",
    "06. First Agent/README.md",
    "07. Context Persistence/README.md",
    "08. Token Counting/README.md",
    "09. Context Compression/README.md",
    "10. Context Management Strategies/README.md",
    "11. Agent Memory Model/README.md",
    "12. Personalization/README.md",
    "13. Task State Machine/README.md",
    "14. Invariant Guardrails/README.md",
    "15. Controlled State Transitions/README.md",
    "16. MCP Connection/README.md",
    "17. First MCP Tool/README.md",
    "18. Scheduler and Background Tasks/README.md",
    "19. MCP Tool Composition/README.md",
    "20. MCP Orchestration/README.md",
    "21. Document Indexing/README.md",
    "06. First Agent/src/main.rs",
    "16. MCP Connection/src/main.rs",
    "20. MCP Orchestration/src/main.rs",
];

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum DocKind {
    Markdown,
    Rust,
    Text,
}

impl DocKind {
    fn of(path: &str) -> Option<DocKind> {
        let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "md" | "markdown" => Some(DocKind::Markdown),
            "rs" => Some(DocKind::Rust),
            "txt" => Some(DocKind::Text),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Document {
    /// Path relative to the corpus root, e.g. `20. MCP Orchestration/README.md`.
    source: String,
    /// The first `# heading` for Markdown, the path otherwise.
    title: String,
    kind: DocKind,
    text: String,
}

fn document(source: &str, text: &str) -> Option<Document> {
    let kind = DocKind::of(source)?;
    let text = text.replace("\r\n", "\n");
    let title = match kind {
        DocKind::Markdown => markdown_title(&text),
        _ => None,
    }
    .unwrap_or_else(|| source.to_string());
    Some(Document {
        source: source.to_string(),
        title,
        kind,
        text,
    })
}

pub fn builtin_documents() -> Vec<Document> {
    BUILTIN
        .iter()
        .filter_map(|(path, text)| document(path, text))
        .collect()
}

// ---------------------------------------------------------------------------
// Structure: where a document's sections are.
// ---------------------------------------------------------------------------

/// One section of a document: byte range and its heading path, e.g.
/// `20. MCP Orchestration › What this is › Verification`, or for code the
/// item it declares, e.g. `impl Registry`.
#[derive(Serialize, Clone, Debug, PartialEq)]
struct Span {
    start: usize,
    end: usize,
    path: String,
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn shorten(text: &str, max: usize) -> String {
    let text = text.trim();
    if char_len(text) <= max {
        return text.to_string();
    }
    let mut short: String = text.chars().take(max).collect();
    short.push('…');
    short
}

/// Every line with the byte offset it starts at, without its `\n`.
fn lines_with_offsets(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut offset = 0;
    text.split_inclusive('\n').map(move |line| {
        let start = offset;
        offset += line.len();
        (start, line.trim_end_matches('\n'))
    })
}

fn is_fence(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("```") || line.starts_with("~~~")
}

/// `## Title` → `(2, "Title")`: one to six `#`, then a space.
fn heading(line: &str) -> Option<(usize, String)> {
    let level = line.bytes().take_while(|&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &line[level..];
    if !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    (!title.is_empty()).then(|| (level, title.to_string()))
}

fn markdown_title(text: &str) -> Option<String> {
    let mut in_fence = false;
    for line in text.lines() {
        if is_fence(line) {
            in_fence = !in_fence;
        } else if !in_fence && let Some((1, title)) = heading(line) {
            return Some(title);
        }
    }
    None
}

/// Spans from `(start, path)` pairs: each one runs to the next start. Text
/// before the first start becomes the preamble, unless it's only whitespace.
fn spans_from_starts(text: &str, mut starts: Vec<(usize, String)>) -> Vec<Span> {
    if starts.first().is_none_or(|(start, _)| *start > 0) {
        starts.insert(0, (0, PREAMBLE.to_string()));
    }
    let mut spans: Vec<Span> = (0..starts.len())
        .map(|i| Span {
            start: starts[i].0,
            end: starts.get(i + 1).map_or(text.len(), |(next, _)| *next),
            path: starts[i].1.clone(),
        })
        .collect();
    spans.retain(|s| s.path != PREAMBLE || !text[s.start..s.end].trim().is_empty());
    spans
}

/// Sections of a Markdown file: one per heading, named by the path of
/// headings above it. `#` lines inside code fences (shell comments in a
/// ```bash block) are not headings.
fn markdown_outline(text: &str) -> Vec<Span> {
    let mut starts = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut in_fence = false;
    for (offset, line) in lines_with_offsets(text) {
        if is_fence(line) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some((level, title)) = heading(line) {
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, title));
            let path: Vec<&str> = stack.iter().map(|(_, t)| t.as_str()).collect();
            starts.push((offset, path.join(PATH_SEP)));
        }
    }
    spans_from_starts(text, starts)
}

/// Prefixes that don't change what an item is.
const ITEM_PREFIXES: [&str; 6] = [
    "pub(crate) ",
    "pub(super) ",
    "pub ",
    "async ",
    "unsafe ",
    "const fn ",
];
const ITEM_KEYWORDS: [&str; 10] = [
    "fn",
    "struct",
    "enum",
    "trait",
    "mod",
    "const",
    "static",
    "type",
    "use",
    "macro_rules!",
];

/// What one line of Rust declares: `fn checksum`, `struct AppState`,
/// `impl ServerHandler for CratesServer`, or `None` if it isn't an item.
fn item_label(line: &str) -> Option<String> {
    let mut rest = line.trim();
    while let Some(prefix) = ITEM_PREFIXES.iter().find(|p| rest.starts_with(**p)) {
        if *prefix == "const fn " {
            return named("fn", &rest[prefix.len()..]);
        }
        rest = &rest[prefix.len()..];
    }
    if let Some(tail) = rest.strip_prefix("impl")
        && (tail.starts_with(' ') || tail.starts_with('<'))
    {
        let head = rest.split('{').next().unwrap_or(rest);
        return Some(shorten(head, 70));
    }
    let (keyword, tail) = rest.split_once(' ')?;
    if !ITEM_KEYWORDS.contains(&keyword) {
        return None;
    }
    if keyword == "use" {
        return Some("use …".to_string());
    }
    named(keyword, tail)
}

fn named(keyword: &str, tail: &str) -> Option<String> {
    let name: String = tail
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then(|| format!("{keyword} {name}"))
}

/// What a top-level block of code is about: the first item it declares, or
/// for a comment banner, the comment.
fn rust_block_label<'a>(lines: impl Iterator<Item = &'a str>) -> String {
    let mut comment = None;
    for line in lines {
        let line = line.trim();
        if let Some(item) = item_label(line) {
            return item;
        }
        if comment.is_none() && line.starts_with("//") {
            let text = line
                .trim_start_matches('/')
                .trim_start_matches('!')
                .trim()
                .trim_matches('-')
                .trim();
            if !text.is_empty() {
                comment = Some(format!("// {}", shorten(text, 60)));
            }
        }
    }
    comment.unwrap_or_else(|| "(code)".to_string())
}

/// Sections of a Rust file: one per top-level block. rustfmt'd code puts a
/// blank line between top-level items and indents everything inside them, so
/// a block starts at a non-indented line right after a blank line. Its doc
/// comments and attributes come with it.
fn rust_outline(text: &str) -> Vec<Span> {
    let lines: Vec<(usize, &str)> = lines_with_offsets(text).collect();
    let opens = |i: usize| {
        let first = lines[i].1.chars().next();
        first.is_some_and(|c| !c.is_whitespace() && !matches!(c, '}' | ')' | ']'))
            && (i == 0 || lines[i - 1].1.trim().is_empty())
    };
    let openers: Vec<usize> = (0..lines.len()).filter(|&i| opens(i)).collect();
    let starts = openers
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let until = openers.get(k + 1).copied().unwrap_or(lines.len());
            let label = rust_block_label(lines[i..until].iter().map(|(_, line)| *line));
            (lines[i].0, label)
        })
        .collect();
    spans_from_starts(text, starts)
}

fn outline(doc: &Document) -> Vec<Span> {
    match doc.kind {
        DocKind::Markdown => markdown_outline(&doc.text),
        DocKind::Rust => rust_outline(&doc.text),
        DocKind::Text => spans_from_starts(&doc.text, Vec::new()),
    }
}

/// A chunk before it gets text and an embedding.
#[derive(Debug, Clone, PartialEq)]
struct Draft {
    start: usize,
    end: usize,
    section: String,
    sections: Vec<String>,
}

/// Windows of `size` characters, each starting `size - overlap` after the
/// previous one. A window ends at the last whitespace in its final fifth, so
/// words aren't cut in half, and an overlap starts at a word. Byte ranges.
fn fixed_windows(text: &str, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte = |i: usize| chars.get(i).map_or(text.len(), |(b, _)| *b);
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = (start + size).min(n);
        if end < n {
            let floor = (end - size / 5).max(start + 1);
            if let Some(ws) = (floor..end).rev().find(|&i| chars[i].1.is_whitespace()) {
                end = ws;
            }
        }
        out.push((byte(start), byte(end)));
        if end >= n {
            break;
        }
        let mut next = end.saturating_sub(overlap).max(start + 1);
        if overlap > 0 {
            let ceiling = (next + size / 5).min(end);
            if let Some(i) = (next..ceiling).find(|&i| chars[i - 1].1.is_whitespace()) {
                next = i;
            }
        }
        start = next;
    }
    out
}

/// Where a new paragraph starts inside `[start, end)`: after a blank line,
/// and in Markdown never inside a code fence.
fn paragraph_starts(text: &str, start: usize, end: usize, kind: DocKind) -> Vec<usize> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut prev_blank = false;
    for (offset, line) in lines_with_offsets(&text[start..end]) {
        let blank = line.trim().is_empty();
        if offset > 0 && prev_blank && !blank && !in_fence {
            out.push(start + offset);
        }
        if kind == DocKind::Markdown && is_fence(line) {
            in_fence = !in_fence;
        }
        prev_blank = blank;
    }
    out
}

/// Splits `[start, end)` at `cuts` and packs the pieces greedily into chunks
/// of at most `max` characters. A single piece longer than that (a huge code
/// block) is cut into fixed windows as a last resort.
fn pack(text: &str, start: usize, end: usize, cuts: &[usize], max: usize) -> Vec<(usize, usize)> {
    let mut bounds = vec![start];
    bounds.extend(cuts.iter().copied().filter(|&c| c > start && c < end));
    bounds.push(end);
    let mut out = Vec::new();
    let (mut cur_start, mut cur_end) = (start, start);
    for pair in bounds.windows(2) {
        let (piece_start, piece_end) = (pair[0], pair[1]);
        if char_len(&text[cur_start..piece_end]) <= max {
            cur_end = piece_end;
            continue;
        }
        if cur_end > cur_start {
            out.push((cur_start, cur_end));
        }
        if char_len(&text[piece_start..piece_end]) > max {
            let piece = &text[piece_start..piece_end];
            out.extend(
                fixed_windows(piece, max, 0)
                    .into_iter()
                    .map(|(a, b)| (piece_start + a, piece_start + b)),
            );
            cur_start = piece_end;
        } else {
            cur_start = piece_start;
        }
        cur_end = piece_end;
    }
    if cur_end > cur_start {
        out.push((cur_start, cur_end));
    }
    out
}

/// The sections a byte range actually has text from.
fn sections_in(text: &str, spans: &[Span], start: usize, end: usize) -> Vec<String> {
    spans
        .iter()
        .filter(|sp| {
            let (a, b) = (sp.start.max(start), sp.end.min(end));
            a < b && !text[a..b].trim().is_empty()
        })
        .map(|sp| sp.path.clone())
        .collect()
}

/// Structural chunking: one chunk per section (Markdown heading or top-level code
/// item). Sections shorter than `min_chars` are joined with the ones after
/// them while the result still fits; sections longer than `max_chars` are
/// split at paragraph boundaries (for code: between the methods of an
/// `impl`, the tests of `mod tests`), each piece keeping its section.
fn chunk_structural(doc: &Document, spans: &[Span]) -> Vec<Draft> {
    let text = &doc.text;
    let mut drafts = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        let mut j = i + 1;
        while j < spans.len()
            && char_len(text[spans[i].start..spans[j - 1].end].trim()) < MIN_CHARS
            && char_len(&text[spans[i].start..spans[j].end]) <= MAX_CHARS
        {
            j += 1;
        }
        let (start, end) = (spans[i].start, spans[j - 1].end);
        if char_len(&text[start..end]) <= MAX_CHARS {
            let group = &spans[i..j];
            let section = group
                .iter()
                .max_by_key(|sp| char_len(&text[sp.start..sp.end]))
                .map(|sp| sp.path.clone())
                .unwrap_or_default();
            drafts.push(Draft {
                start,
                end,
                section,
                sections: sections_in(text, group, start, end),
            });
        } else {
            // Joining never makes a group too long, so this is one section.
            let span = &spans[i];
            let cuts = paragraph_starts(text, span.start, span.end, doc.kind);
            for (s, e) in pack(text, span.start, span.end, &cuts, MAX_CHARS) {
                if text[s..e].trim().is_empty() {
                    continue;
                }
                let section = match doc.kind {
                    DocKind::Rust => text[s..e]
                        .lines()
                        .find_map(item_label)
                        .filter(|sub| *sub != span.path)
                        .map_or_else(
                            || span.path.clone(),
                            |sub| format!("{}{PATH_SEP}{sub}", span.path),
                        ),
                    _ => span.path.clone(),
                };
                drafts.push(Draft {
                    start: s,
                    end: e,
                    sections: vec![section.clone()],
                    section,
                });
            }
        }
        i = j;
    }
    drafts
}

// ---------------------------------------------------------------------------
// Embeddings: a local hashed TF-IDF by default, any OpenAI-compatible
// `/embeddings` endpoint when configured.
// ---------------------------------------------------------------------------

/// FNV-1a: a stable hash, so the same word lands in the same bucket in every
/// build and every run.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Lower-cased stems: words (letters, digits, `_`) cut to their first
/// `STEM_CHARS` characters. A `snake_case` name is kept whole, as an exact
/// identifier, and also gives the stems of its parts.
fn stems(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let word = word.trim_matches('_').to_lowercase();
        if word.contains('_') {
            out.push(word.clone());
        }
        for part in word.split('_') {
            let len = char_len(part);
            if len >= 2 || (len == 1 && part.chars().all(|c| c.is_ascii_digit())) {
                out.push(part.chars().take(STEM_CHARS).collect());
            }
        }
    }
    out
}

/// Term frequencies of stems and adjacent stem pairs, hashed into `dims`
/// buckets with a sign (the hashing trick) and log-scaled.
fn hashed_tf(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0f32; dims];
    let mut add = |feature: &str, weight: f32| {
        let hash = fnv1a(feature.as_bytes());
        let bucket = (hash % dims as u64) as usize;
        let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
        v[bucket] += sign * weight;
    };
    let stems = stems(text);
    for (i, stem) in stems.iter().enumerate() {
        add(stem.as_str(), 1.0);
        if i > 0 {
            add(&format!("{} {stem}", stems[i - 1]), 0.5);
        }
    }
    for x in v.iter_mut() {
        *x = x.signum() * x.abs().ln_1p();
    }
    v
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn weigh(mut v: Vec<f32>, idf: &[f32]) -> Vec<f32> {
    for (x, w) in v.iter_mut().zip(idf) {
        *x *= w;
    }
    normalize(v)
}

/// Cosine similarity of two unit vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The local embedding of a set of chunks: hashed term frequencies weighted
/// by IDF over these same chunks (so it's learned per index and stored with
/// it), then unit-length.
fn local_fit(texts: &[&str], dims: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let raw: Vec<Vec<f32>> = texts.iter().map(|t| hashed_tf(t, dims)).collect();
    let n = raw.len() as f32;
    let mut df = vec![0f32; dims];
    for v in &raw {
        for (d, x) in df.iter_mut().zip(v) {
            if *x != 0.0 {
                *d += 1.0;
            }
        }
    }
    let idf: Vec<f32> = df
        .iter()
        .map(|d| ((n + 1.0) / (d + 1.0)).ln() + 1.0)
        .collect();
    let vectors = raw.into_iter().map(|v| weigh(v, &idf)).collect();
    (vectors, idf)
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Deserialize)]
pub struct ErrorDetail {
    pub message: String,
}

pub struct RemoteEmbedder {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
}

impl RemoteEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            let response = self
                .http
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&json!({ "model": self.model, "input": batch }))
                .send()
                .await
                .map_err(|e| format!("Embedding request failed: {e}"))?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                let detail = serde_json::from_str::<ErrorResponse>(&body)
                    .map(|e| e.error.message)
                    .unwrap_or(body);
                return Err(format!("Embedding API error ({status}): {detail}"));
            }
            let mut data = serde_json::from_str::<EmbeddingResponse>(&body)
                .map_err(|e| format!("Failed to parse the embedding response: {e}"))?
                .data;
            if data.len() != batch.len() {
                return Err(format!(
                    "Asked for {} embeddings, got {}.",
                    batch.len(),
                    data.len()
                ));
            }
            // The API may answer in any order; `index` says which input is which.
            data.sort_by_key(|item| item.index);
            out.extend(data.into_iter().map(|item| normalize(item.embedding)));
        }
        Ok(out)
    }
}

pub enum Embedder {
    Local { dims: usize },
    Remote(RemoteEmbedder),
}

impl Embedder {
    /// Stored with every index: a query has to be embedded the same way.
    pub fn id(&self) -> String {
        match self {
            Embedder::Local { dims } => format!("local:hashed-tfidf-{dims}"),
            Embedder::Remote(r) => format!("openai:{}", r.model),
        }
    }

    /// Chunk vectors, plus the IDF table for the local embedder (empty for
    /// a remote one).
    async fn embed_chunks(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        match self {
            Embedder::Local { dims } => Ok(local_fit(texts, *dims)),
            Embedder::Remote(r) => Ok((r.embed(texts).await?, Vec::new())),
        }
    }

    /// Query vectors before an index's IDF is applied (see `in_space`).
    async fn embed_queries(&self, queries: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            Embedder::Local { dims } => Ok(queries.iter().map(|q| hashed_tf(q, *dims)).collect()),
            Embedder::Remote(r) => r.embed(queries).await,
        }
    }
}

// ---------------------------------------------------------------------------
// The index: structural chunks with metadata and embeddings, in memory.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
pub struct Chunk {
    /// `<source>#<ordinal>`, unique within the index.
    pub chunk_id: String,
    pub source: String,
    pub title: String,
    /// The section this chunk belongs to (heading path or code item).
    pub section: String,
    pub tokens: usize,
    pub text: String,
    #[serde(skip)]
    embedding: Vec<f32>,
}

/// One retrieved chunk: its rank and cosine similarity to the query.
#[derive(Serialize, Clone, Debug)]
pub struct Hit {
    pub rank: usize,
    pub score: f32,
    #[serde(flatten)]
    pub chunk: Chunk,
}

pub struct Index {
    pub embedder: Embedder,
    pub documents: usize,
    pub chunks: Vec<Chunk>,
    /// The local embedder's per-bucket IDF; queries are weighted with it.
    idf: Vec<f32>,
}

/// ≈ tokens: about 4 characters per token for English and code, about 2 for
/// Cyrillic.
pub fn estimate_tokens(text: &str) -> usize {
    let (ascii, other) = text.chars().fold((0usize, 0usize), |(a, o), c| {
        if c.is_ascii() { (a + 1, o) } else { (a, o + 1) }
    });
    ascii.div_ceil(4) + other.div_ceil(2)
}

impl Index {
    /// The indexing pipeline: outline → structural chunks → embeddings.
    pub async fn build(docs: Vec<Document>, embedder: Embedder) -> Result<Index, String> {
        let mut chunks = Vec::new();
        for doc in &docs {
            let spans = outline(doc);
            for (ordinal, d) in chunk_structural(doc, &spans).into_iter().enumerate() {
                let text = doc.text[d.start..d.end].to_string();
                chunks.push(Chunk {
                    chunk_id: format!("{}#{ordinal:03}", doc.source),
                    source: doc.source.clone(),
                    title: doc.title.clone(),
                    section: d.section,
                    tokens: estimate_tokens(&text),
                    text,
                    embedding: Vec::new(),
                });
            }
        }
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let (vectors, idf) = embedder.embed_chunks(&texts).await?;
        if vectors.len() != chunks.len() {
            return Err(format!(
                "{} chunks but {} embeddings.",
                chunks.len(),
                vectors.len()
            ));
        }
        for (chunk, vector) in chunks.iter_mut().zip(vectors) {
            chunk.embedding = vector;
        }
        Ok(Index {
            embedder,
            documents: docs.len(),
            chunks,
            idf,
        })
    }

    /// The `k` chunks closest to `query` by cosine similarity, best first.
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<Hit>, String> {
        let raw = self
            .embedder
            .embed_queries(&[query])
            .await?
            .into_iter()
            .next()
            .ok_or("The embedder returned nothing.")?;
        let query = if self.idf.is_empty() {
            raw
        } else {
            weigh(raw, &self.idf)
        };
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot(&c.embedding, &query)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (i, score))| Hit {
                rank: rank + 1,
                score: (score * 1000.0).round() / 1000.0,
                chunk: self.chunks[i].clone(),
            })
            .collect())
    }
}

pub fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// `EMBEDDER=openai` uses an OpenAI-compatible `/embeddings` endpoint;
/// anything else (the default) the local one, which needs no network.
pub fn embedder_from_env() -> Result<Embedder, String> {
    let kind = env_non_empty("EMBEDDER").unwrap_or_else(|| "local".to_string());
    match kind.trim() {
        "local" => Ok(Embedder::Local { dims: LOCAL_DIMS }),
        "openai" => {
            let api_key = env_non_empty("EMBEDDING_API_KEY")
                .ok_or("EMBEDDER=openai needs EMBEDDING_API_KEY.")?;
            let base = env_non_empty("EMBEDDING_BASE_URL")
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let model = env_non_empty("EMBEDDING_MODEL")
                .unwrap_or_else(|| "text-embedding-3-small".to_string());
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|e| format!("Can't build the HTTP client: {e}"))?;
            Ok(Embedder::Remote(RemoteEmbedder {
                http,
                endpoint: format!("{}/embeddings", base.trim_end_matches('/')),
                api_key,
                model,
            }))
        }
        other => Err(format!("EMBEDDER must be local or openai, not {other:?}.")),
    }
}

#[cfg(test)]
pub fn local() -> Embedder {
    Embedder::Local { dims: LOCAL_DIMS }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_corpus_has_every_earlier_lesson() {
        let docs = builtin_documents();
        assert_eq!(docs.len(), BUILTIN.len());
        assert!(docs.iter().all(|d| !d.text.trim().is_empty()));
        for n in 1..=21 {
            let prefix = format!("{n:02}. ");
            assert!(
                docs.iter()
                    .any(|d| d.source.starts_with(&prefix) && d.source.ends_with("README.md")),
                "lesson {n:02} README is missing"
            );
        }
    }

    #[tokio::test]
    async fn index_has_metadata_and_finds_the_right_section() {
        let index = Index::build(builtin_documents(), local()).await.unwrap();
        assert_eq!(index.documents, BUILTIN.len());
        assert!(index.chunks.len() > 100, "{} chunks", index.chunks.len());
        let mut ids: Vec<&str> = index.chunks.iter().map(|c| c.chunk_id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), index.chunks.len(), "chunk ids are unique");
        assert!(index.chunks.iter().all(|c| !c.section.is_empty()));
        assert!(index.chunks.iter().all(|c| char_len(&c.text) <= MAX_CHARS));

        let hits = index.search("static musl binary instead of Docker", 3).await.unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits.iter().map(|h| h.rank).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(
            hits.iter().any(|h| h.chunk.source == "DEPLOYMENT.md"
                && h.chunk.section.ends_with("Why a static musl binary instead of Docker")),
            "{hits:?}"
        );
    }
}

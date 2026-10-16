//! The project index, from lessons 21-25 and 31: structural chunking (one
//! chunk per Markdown section or top-level Rust item) and the local hashed
//! TF-IDF embedder. What's new is the corpus: not only the documentation but
//! the whole repository as it is at the head of the change under review,
//! docs and code, and every chunk knows its line range, so the chunks the
//! diff already shows can be left out of the context.

use std::path::Path;

use serde::{Deserialize, Serialize};

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
/// The section of whatever comes before a file's first heading or item.
const PREAMBLE: &str = "(top of file)";
const PATH_SEP: &str = " › ";
/// Limits on what is read from a checkout.
const MAX_FILE_BYTES: u64 = 512 * 1024;
const MAX_FILES: usize = 2000;
/// Folders that are never indexed: build output and dependencies.
const SKIP_DIRS: [&str; 2] = ["target", "node_modules"];
/// The one hidden folder that is indexed: the CI workflows.
const HIDDEN_ALLOWED: &str = ".github";

// ---------------------------------------------------------------------------
// The corpus: the repository's documentation and code.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum DocKind {
    Markdown,
    Rust,
    /// Anything else worth searching (TOML, YAML, HTML): split at blank lines.
    Text,
}

impl DocKind {
    fn of(path: &str) -> Option<DocKind> {
        let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "md" | "markdown" => Some(DocKind::Markdown),
            "rs" => Some(DocKind::Rust),
            "toml" | "yml" | "yaml" | "html" | "txt" => Some(DocKind::Text),
            _ => None,
        }
    }

    /// Documentation or code: the review takes some of each.
    fn class(self) -> &'static str {
        match self {
            DocKind::Markdown => "doc",
            _ => "code",
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

/// A document from its path and text, or `None` for a file type that isn't
/// indexed.
pub fn document(source: &str, text: &str) -> Option<Document> {
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

/// Every indexable file under `rel`, as `/`-separated paths relative to the
/// checkout, sorted. Hidden entries are skipped (except `.github` at the
/// root), and so are `target/` and `node_modules/`.
fn walk(root: &Path, rel: &str, out: &mut Vec<String>) {
    let dir = if rel.is_empty() {
        root.to_path_buf()
    } else {
        root.join(rel)
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<(String, bool)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            // Not following symlinks: a link can't pull in files from
            // outside the checkout.
            let kind = e.file_type().ok()?;
            (kind.is_dir() || kind.is_file()).then_some((name, kind.is_dir()))
        })
        .collect();
    entries.sort();
    for (name, is_dir) in entries {
        let path = if rel.is_empty() {
            name.clone()
        } else {
            format!("{rel}/{name}")
        };
        let hidden = name.starts_with('.') && path != HIDDEN_ALLOWED;
        if hidden || out.len() >= MAX_FILES {
            continue;
        }
        if is_dir {
            if !SKIP_DIRS.contains(&name.as_str()) {
                walk(root, &path, out);
            }
        } else if DocKind::of(&path).is_some() {
            out.push(path);
        }
    }
}

/// The corpus of a checkout: every Markdown, Rust, TOML, YAML, HTML and text
/// file in it (README, AGENTS.md, docs/, every lesson's README and code, the
/// CI workflows). Files over 512 KB are skipped.
pub fn load_corpus(root: &Path) -> Result<Vec<Document>, String> {
    if !root.is_dir() {
        return Err(format!("{} is not a directory.", root.display()));
    }
    let mut files = Vec::new();
    walk(root, "", &mut files);
    let mut docs = Vec::new();
    for rel in files {
        let path = root.join(&rel);
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&path)
            && !text.trim().is_empty()
            && let Some(doc) = document(&rel, &text)
        {
            docs.push(doc);
        }
    }
    if docs.is_empty() {
        return Err(format!("Nothing to index in {}.", root.display()));
    }
    Ok(docs)
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
pub fn stems(text: &str) -> Vec<String> {
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

pub enum Embedder {
    Local { dims: usize },
}

impl Embedder {
    /// Stored with every index: a query has to be embedded the same way.
    pub fn id(&self) -> String {
        match self {
            Embedder::Local { dims } => format!("local:hashed-tfidf-{dims}"),
        }
    }

    /// Chunk vectors, plus the IDF table they were weighted with.
    async fn embed_chunks(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        match self {
            Embedder::Local { dims } => Ok(local_fit(texts, *dims)),
        }
    }

    /// Query vectors before an index's IDF is applied.
    async fn embed_queries(&self, queries: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            Embedder::Local { dims } => Ok(queries.iter().map(|q| hashed_tf(q, *dims)).collect()),
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
    /// `doc` (Markdown) or `code` (everything else).
    pub kind: String,
    /// 1-based, inclusive: where the chunk is in its file.
    pub start_line: usize,
    pub end_line: usize,
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
                let start_line = doc.text[..d.start].matches('\n').count() + 1;
                let end_line = start_line + text.trim_end_matches('\n').matches('\n').count();
                chunks.push(Chunk {
                    chunk_id: format!("{}#{ordinal:03}", doc.source),
                    source: doc.source.clone(),
                    title: doc.title.clone(),
                    section: d.section,
                    kind: doc.kind.class().to_string(),
                    start_line,
                    end_line,
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

/// The local embedder: no network, no model download. (Lesson 21's remote
/// `/embeddings` option is left out: the deploy has no embeddings key.)
pub fn local() -> Embedder {
    Embedder::Local { dims: LOCAL_DIMS }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_corpus_is_the_docs_the_code_and_the_workflows() {
        let root = std::env::temp_dir().join(format!("lesson32-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let files = [
            ("README.md", "# Project\n\nThe overview."),
            ("docs/guide.md", "# Guide\n\n## Setup\n\nRun it."),
            ("docs/.hidden.md", "# Hidden"),
            ("docs/image.png", "not text"),
            ("07. Some Lesson/README.md", "# 07. Some Lesson\n\nWhat it is."),
            ("07. Some Lesson/Cargo.toml", "[package]\nname = \"some-lesson\"\n"),
            ("07. Some Lesson/src/main.rs", "fn main() {}\n"),
            ("07. Some Lesson/target/debug/build.rs", "fn generated() {}\n"),
            (".github/workflows/ci.yml", "name: CI\non: push\n"),
            (".git/config", "[core]"),
            (".env", "SECRET=1"),
        ];
        for (path, text) in files {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        let docs = load_corpus(&root).unwrap();
        let sources: Vec<&str> = docs.iter().map(|d| d.source.as_str()).collect();
        assert_eq!(
            sources,
            [
                ".github/workflows/ci.yml",
                "07. Some Lesson/Cargo.toml",
                "07. Some Lesson/README.md",
                "07. Some Lesson/src/main.rs",
                "README.md",
                "docs/guide.md",
            ]
        );
        assert_eq!(docs[5].title, "Guide");
        assert!(load_corpus(&root.join("nope")).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    const CODE: &str = "use std::time::Duration;\n\n\
        /// How long to wait before the next attempt.\n\
        pub fn retry_delay(attempt: u32) -> Duration {\n    \
            Duration::from_millis(200 * 2u64.pow(attempt))\n}\n\n\
        pub const MAX_RETRIES: u32 = 5;\n";

    #[tokio::test]
    async fn chunks_know_their_kind_and_lines_and_code_is_found() {
        let docs = vec![
            document("lesson/src/retry.rs", CODE).unwrap(),
            document(
                "docs/retries.md",
                "# Retries\n\nFailed LLM calls are retried with exponential backoff.\n",
            )
            .unwrap(),
            document("docs/other.md", "# Other\n\nUnrelated text about ports.\n").unwrap(),
        ];
        assert!(document("notes.bin", "x").is_none());
        let index = Index::build(docs, local()).await.unwrap();
        assert_eq!(index.documents, 3);
        let code: Vec<&Chunk> = index.chunks.iter().filter(|c| c.kind == "code").collect();
        assert!(!code.is_empty());
        for c in &index.chunks {
            assert!(c.start_line >= 1 && c.end_line >= c.start_line, "{c:?}");
            assert!(!c.section.is_empty());
        }
        let retry = index
            .chunks
            .iter()
            .find(|c| c.text.contains("fn retry_delay"))
            .unwrap();
        assert_eq!(retry.source, "lesson/src/retry.rs");
        let first = CODE
            .lines()
            .position(|l| !l.is_empty() && retry.text.starts_with(l))
            .unwrap()
            + 1;
        assert_eq!(retry.start_line, first);
        assert!(index.chunks.iter().any(|c| c.kind == "doc"));

        let hits = index.search("retry_delay attempt backoff", 2).await.unwrap();
        assert_eq!(hits[0].chunk.source, "lesson/src/retry.rs", "{hits:?}");
    }
}

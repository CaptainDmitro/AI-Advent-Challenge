use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{Query, State},
    response::Html,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::RwLock;

/// A printed page of plain text is roughly 1800 characters (30 lines × 60).
const CHARS_PER_PAGE: f64 = 1800.0;
const DEFAULT_FIXED_SIZE: usize = 1000;
const DEFAULT_OVERLAP: usize = 150;
const DEFAULT_MAX_CHARS: usize = 1800;
const DEFAULT_MIN_CHARS: usize = 300;
/// Dimensions of the local hashed embedding.
const LOCAL_DIMS: usize = 1024;
/// Local stems: the first few characters of a word, so "chunk", "chunks" and
/// "chunking", or "сообщения" and "сообщений", become the same feature.
const STEM_CHARS: usize = 5;
/// How many texts go into one `/embeddings` request.
const EMBED_BATCH: usize = 64;
const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 20;
/// The rank cut-off for MRR in the retrieval check.
const EVAL_DEPTH: usize = 10;
const MAX_QUERY_CHARS: usize = 500;
/// Limits for `DOCS_DIR`, so a wrong path can't make the indexer read a disk.
const MAX_FILES: usize = 500;
const MAX_FILE_BYTES: u64 = 1024 * 1024;
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

/// Every lesson's README and the repo-level docs (Markdown, split by
/// headings), plus the source of three lessons (Rust, split by top-level
/// items). CI checks out the whole repo, so the sibling folders exist at
/// compile time and the deployed binary carries the corpus inside it.
const BUILTIN: &[(&str, &str)] = builtin_corpus![
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
struct Document {
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

fn builtin_documents() -> Vec<Document> {
    BUILTIN
        .iter()
        .filter_map(|(path, text)| document(path, text))
        .collect()
}

/// Every `.md`, `.rs` and `.txt` file under `root`, sorted by path. Hidden
/// folders and `target` are skipped; files that aren't UTF-8 are skipped.
fn load_dir(root: &Path) -> Result<Vec<Document>, String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)
        .map_err(|e| format!("Can't read {}: {e}", root.display()))?;
    files.sort();
    let docs: Vec<Document> = files
        .iter()
        .filter_map(|rel| {
            let text = std::fs::read_to_string(root.join(rel)).ok()?;
            document(rel, &text)
        })
        .collect();
    if docs.is_empty() {
        return Err(format!(
            "No .md, .rs or .txt files under {}",
            root.display()
        ));
    }
    Ok(docs)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(root, &path, out)?;
        } else if file_type.is_file()
            && DocKind::of(&name).is_some()
            && out.len() < MAX_FILES
            && entry.metadata()?.len() <= MAX_FILE_BYTES
        {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect();
            out.push(parts.join("/"));
        }
    }
    Ok(())
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

// ---------------------------------------------------------------------------
// Chunking: two strategies over the same documents.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Strategy {
    Fixed,
    Structural,
}

const STRATEGIES: [Strategy; 2] = [Strategy::Fixed, Strategy::Structural];

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::Fixed => "fixed",
            Strategy::Structural => "structural",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(default)]
struct ChunkParams {
    /// Fixed: window size, in characters.
    fixed_size: usize,
    /// Fixed: characters shared by neighbouring windows.
    overlap: usize,
    /// Structural: the largest chunk; a longer section is split at paragraphs.
    max_chars: usize,
    /// Structural: a shorter section is joined with the ones after it.
    min_chars: usize,
}

impl Default for ChunkParams {
    fn default() -> Self {
        ChunkParams {
            fixed_size: DEFAULT_FIXED_SIZE,
            overlap: DEFAULT_OVERLAP,
            max_chars: DEFAULT_MAX_CHARS,
            min_chars: DEFAULT_MIN_CHARS,
        }
    }
}

impl ChunkParams {
    fn validate(&self) -> Result<(), String> {
        if !(200..=4000).contains(&self.fixed_size) {
            return Err("fixed_size must be between 200 and 4000 characters.".to_string());
        }
        if self.overlap > self.fixed_size / 2 {
            return Err("overlap can be at most half of fixed_size.".to_string());
        }
        if !(300..=6000).contains(&self.max_chars) {
            return Err("max_chars must be between 300 and 6000 characters.".to_string());
        }
        if self.min_chars > self.max_chars / 2 {
            return Err("min_chars can be at most half of max_chars.".to_string());
        }
        Ok(())
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

/// Strategy 1: windows of a fixed size with overlap, blind to structure. The
/// section metadata is only looked up afterwards: where the window starts,
/// and every section it touches.
fn chunk_fixed(doc: &Document, spans: &[Span], params: &ChunkParams) -> Vec<Draft> {
    fixed_windows(&doc.text, params.fixed_size, params.overlap)
        .into_iter()
        .filter(|&(s, e)| !doc.text[s..e].trim().is_empty())
        .map(|(start, end)| {
            let sections = sections_in(&doc.text, spans, start, end);
            let section = sections
                .first()
                .cloned()
                .unwrap_or_else(|| PREAMBLE.to_string());
            Draft {
                start,
                end,
                section,
                sections,
            }
        })
        .collect()
}

/// Strategy 2: one chunk per section (Markdown heading or top-level code
/// item). Sections shorter than `min_chars` are joined with the ones after
/// them while the result still fits; sections longer than `max_chars` are
/// split at paragraph boundaries (for code: between the methods of an
/// `impl`, the tests of `mod tests`), each piece keeping its section.
fn chunk_structural(doc: &Document, spans: &[Span], params: &ChunkParams) -> Vec<Draft> {
    let text = &doc.text;
    let mut drafts = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        let mut j = i + 1;
        while j < spans.len()
            && char_len(text[spans[i].start..spans[j - 1].end].trim()) < params.min_chars
            && char_len(&text[spans[i].start..spans[j].end]) <= params.max_chars
        {
            j += 1;
        }
        let (start, end) = (spans[i].start, spans[j - 1].end);
        if char_len(&text[start..end]) <= params.max_chars {
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
            for (s, e) in pack(text, span.start, span.end, &cuts, params.max_chars) {
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

fn checksum(text: &str) -> String {
    format!("fnv1a64:{:016x}", fnv1a(text.as_bytes()))
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
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

struct RemoteEmbedder {
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

enum Embedder {
    Local { dims: usize },
    Remote(RemoteEmbedder),
}

impl Embedder {
    /// Stored with every index: a query has to be embedded the same way.
    fn id(&self) -> String {
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
// The index: chunks + metadata + embeddings, saved as JSON.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Chunk {
    /// `<strategy>:<source>#<ordinal>`, unique within the index.
    chunk_id: String,
    source: String,
    title: String,
    kind: DocKind,
    /// The section this chunk belongs to (heading path or code item).
    section: String,
    /// Every section the chunk has text from: one for a clean chunk.
    sections: Vec<String>,
    /// Position of the chunk within its document.
    ordinal: usize,
    /// Byte range in the document.
    start: usize,
    end: usize,
    chars: usize,
    tokens: usize,
    text: String,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct IndexMeta {
    strategy: Strategy,
    params: ChunkParams,
    embedder: String,
    dims: usize,
    created_at: u64,
    corpus_checksum: String,
    documents: usize,
    chunks: usize,
    chunk_ms: f64,
    embed_ms: f64,
    /// The local embedder's per-bucket IDF; queries are weighted with it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    idf: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Index {
    meta: IndexMeta,
    chunks: Vec<Chunk>,
}

impl Index {
    /// A query vector in this index's space.
    fn in_space(&self, raw: &[f32]) -> Vec<f32> {
        if self.meta.idf.is_empty() {
            raw.to_vec()
        } else {
            weigh(raw.to_vec(), &self.meta.idf)
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot(&c.embedding, query)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }
}

/// ≈ tokens: about 4 characters per token for English and code, about 2 for
/// Cyrillic.
fn estimate_tokens(text: &str) -> usize {
    let (ascii, other) = text.chars().fold((0usize, 0usize), |(a, o), c| {
        if c.is_ascii() { (a + 1, o) } else { (a, o + 1) }
    });
    ascii.div_ceil(4) + other.div_ceil(2)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn elapsed_ms(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

struct Corpus {
    /// Where the documents came from: the built-in repo docs or `DOCS_DIR`.
    origin: String,
    docs: Vec<Document>,
    /// `outlines[i]` are the sections of `docs[i]`.
    outlines: Vec<Vec<Span>>,
    checksum: String,
}

impl Corpus {
    fn new(origin: String, docs: Vec<Document>) -> Corpus {
        let outlines = docs.iter().map(outline).collect();
        let mut all = String::new();
        for doc in &docs {
            all.push_str(&doc.source);
            all.push('\0');
            all.push_str(&doc.text);
            all.push('\0');
        }
        Corpus {
            origin,
            docs,
            outlines,
            checksum: checksum(&all),
        }
    }

    fn find(&self, source: &str) -> Option<(&Document, &[Span])> {
        let i = self.docs.iter().position(|d| d.source == source)?;
        Some((&self.docs[i], self.outlines[i].as_slice()))
    }

    fn chars(&self) -> usize {
        self.docs.iter().map(|d| char_len(&d.text)).sum()
    }
}

fn chunk_corpus(corpus: &Corpus, strategy: Strategy, params: &ChunkParams) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for (doc, spans) in corpus.docs.iter().zip(&corpus.outlines) {
        let drafts = match strategy {
            Strategy::Fixed => chunk_fixed(doc, spans, params),
            Strategy::Structural => chunk_structural(doc, spans, params),
        };
        for (ordinal, d) in drafts.into_iter().enumerate() {
            let text = doc.text[d.start..d.end].to_string();
            chunks.push(Chunk {
                chunk_id: format!("{}:{}#{ordinal:03}", strategy.name(), doc.source),
                source: doc.source.clone(),
                title: doc.title.clone(),
                kind: doc.kind,
                section: d.section,
                sections: d.sections,
                ordinal,
                start: d.start,
                end: d.end,
                chars: char_len(&text),
                tokens: estimate_tokens(&text),
                text,
                embedding: Vec::new(),
            });
        }
    }
    chunks
}

/// The whole pipeline for one strategy: chunk → embed → index.
async fn build_index(
    corpus: &Corpus,
    strategy: Strategy,
    params: &ChunkParams,
    embedder: &Embedder,
) -> Result<Index, String> {
    let started = Instant::now();
    let mut chunks = chunk_corpus(corpus, strategy, params);
    let chunk_ms = elapsed_ms(started);

    let started = Instant::now();
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
    let embed_ms = elapsed_ms(started);

    let dims = chunks.first().map_or(0, |c| c.embedding.len());
    if chunks.iter().any(|c| c.embedding.len() != dims) {
        return Err("The embeddings don't all have the same length.".to_string());
    }
    Ok(Index {
        meta: IndexMeta {
            strategy,
            params: *params,
            embedder: embedder.id(),
            dims,
            created_at: unix_now(),
            corpus_checksum: corpus.checksum.clone(),
            documents: corpus.docs.len(),
            chunks: chunks.len(),
            chunk_ms,
            embed_ms,
            idf,
        },
        chunks,
    })
}

fn index_path(dir: &Path, strategy: Strategy) -> PathBuf {
    dir.join(format!("{}.json", strategy.name()))
}

/// Writes `<dir>/<strategy>.json` (write-then-rename, so a crash never leaves
/// half an index) and returns its size in bytes.
fn save_index(dir: &Path, index: &Index) -> Result<u64, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Can't create {}: {e}", dir.display()))?;
    let path = index_path(dir, index.meta.strategy);
    let body = serde_json::to_vec(index).map_err(|e| format!("Can't serialize the index: {e}"))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &body)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .map_err(|e| format!("Can't write {}: {e}", path.display()))?;
    Ok(body.len() as u64)
}

fn load_index(dir: &Path, strategy: Strategy) -> Result<Option<Loaded>, String> {
    let path = index_path(dir, strategy);
    let body = match std::fs::read(&path) {
        Ok(body) => body,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("Can't read {}: {e}", path.display())),
    };
    let index: Index = serde_json::from_slice(&body)
        .map_err(|e| format!("{} is not a valid index: {e}", path.display()))?;
    if index.meta.strategy != strategy {
        return Err(format!("{} holds a different strategy.", path.display()));
    }
    Ok(Some(Loaded {
        file_bytes: body.len() as u64,
        index,
    }))
}

#[derive(Clone, Debug)]
struct Loaded {
    index: Index,
    file_bytes: u64,
}

/// Builds and saves both indexes.
async fn build_all(
    corpus: &Corpus,
    embedder: &Embedder,
    params: &ChunkParams,
    dir: &Path,
) -> Result<Vec<Loaded>, String> {
    let mut out = Vec::new();
    for strategy in STRATEGIES {
        let index = build_index(corpus, strategy, params, embedder).await?;
        let file_bytes = save_index(dir, &index)?;
        out.push(Loaded { index, file_bytes });
    }
    Ok(out)
}

/// Both indexes from disk, if both are there.
fn load_all(dir: &Path) -> Result<Option<Vec<Loaded>>, String> {
    let mut out = Vec::new();
    for strategy in STRATEGIES {
        match load_index(dir, strategy)? {
            Some(loaded) => out.push(loaded),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Comparing the two strategies.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug, Default)]
struct FileInfo {
    source: String,
    title: String,
    kind: String,
    chars: usize,
    pages: f64,
    sections: usize,
}

#[derive(Serialize, Clone, Debug, Default)]
struct CorpusInfo {
    origin: String,
    documents: usize,
    chars: usize,
    pages: f64,
    tokens: usize,
    checksum: String,
    files: Vec<FileInfo>,
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        round2(part as f64 / whole as f64)
    }
}

fn corpus_info(corpus: &Corpus) -> CorpusInfo {
    let files: Vec<FileInfo> = corpus
        .docs
        .iter()
        .zip(&corpus.outlines)
        .map(|(doc, spans)| {
            let chars = char_len(&doc.text);
            FileInfo {
                source: doc.source.clone(),
                title: doc.title.clone(),
                kind: format!("{:?}", doc.kind).to_lowercase(),
                chars,
                pages: round2(chars as f64 / CHARS_PER_PAGE),
                sections: spans.len(),
            }
        })
        .collect();
    let chars = corpus.chars();
    CorpusInfo {
        origin: corpus.origin.clone(),
        documents: corpus.docs.len(),
        chars,
        pages: round2(chars as f64 / CHARS_PER_PAGE),
        tokens: corpus.docs.iter().map(|d| estimate_tokens(&d.text)).sum(),
        checksum: corpus.checksum.clone(),
        files,
    }
}

#[derive(Serialize, Clone, Debug, Default)]
struct Stats {
    chunks: usize,
    avg_chars: f64,
    median_chars: usize,
    min_chars: usize,
    max_chars: usize,
    stddev_chars: f64,
    avg_tokens: f64,
    total_tokens: usize,
    /// Characters indexed ÷ characters in the corpus: overlap makes it > 1.
    redundancy: f64,
    /// Share of chunks with text from exactly one section.
    single_section: f64,
    /// Share of chunks that start where a section starts.
    starts_at_section: f64,
    /// Share of chunks that end at a blank line, a section's end or the file's end.
    ends_at_paragraph: f64,
    file_bytes: u64,
    chunk_ms: f64,
    embed_ms: f64,
}

/// Byte offset of the first non-whitespace character at or after `pos`.
fn skip_ws(text: &str, pos: usize) -> usize {
    let rest = &text[pos..];
    pos + (rest.len() - rest.trim_start().len())
}

fn ends_at_paragraph(text: &str, spans: &[Span], end: usize) -> bool {
    let before = &text[..end];
    let from = before.trim_end().len();
    let to = skip_ws(text, end);
    to == text.len() || text[from..to].contains("\n\n") || spans.iter().any(|s| s.start == end)
}

fn stats(corpus: &Corpus, loaded: &Loaded) -> Stats {
    let chunks = &loaded.index.chunks;
    let n = chunks.len();
    let mut sizes: Vec<usize> = chunks.iter().map(|c| c.chars).collect();
    sizes.sort_unstable();
    let total: usize = sizes.iter().sum();
    let avg = if n == 0 { 0.0 } else { total as f64 / n as f64 };
    let variance = if n == 0 {
        0.0
    } else {
        sizes.iter().map(|&s| (s as f64 - avg).powi(2)).sum::<f64>() / n as f64
    };
    let total_tokens: usize = chunks.iter().map(|c| c.tokens).sum();
    let (mut starts, mut ends) = (0, 0);
    for chunk in chunks {
        let Some((doc, spans)) = corpus.find(&chunk.source) else {
            continue;
        };
        let text = &doc.text;
        if chunk.end > text.len() {
            continue;
        }
        let first = skip_ws(text, chunk.start);
        if spans.iter().any(|s| skip_ws(text, s.start) == first) {
            starts += 1;
        }
        if ends_at_paragraph(text, spans, chunk.end) {
            ends += 1;
        }
    }
    let meta = &loaded.index.meta;
    Stats {
        chunks: n,
        avg_chars: round2(avg),
        median_chars: sizes.get(n / 2).copied().unwrap_or(0),
        min_chars: sizes.first().copied().unwrap_or(0),
        max_chars: sizes.last().copied().unwrap_or(0),
        stddev_chars: round2(variance.sqrt()),
        avg_tokens: if n == 0 {
            0.0
        } else {
            round2(total_tokens as f64 / n as f64)
        },
        total_tokens,
        redundancy: ratio(total, corpus.chars()),
        single_section: ratio(chunks.iter().filter(|c| c.sections.len() == 1).count(), n),
        starts_at_section: ratio(starts, n),
        ends_at_paragraph: ratio(ends, n),
        file_bytes: loaded.file_bytes,
        chunk_ms: meta.chunk_ms,
        embed_ms: meta.embed_ms,
    }
}

/// A question whose answer is in a known section of a known file.
struct EvalCase {
    question: &'static str,
    source: &'static str,
    /// Part of the section's heading path (or code item).
    section: &'static str,
}

const EVAL: &[EvalCase] = &[
    EvalCase {
        question: "What port and systemd service name does a deployed lesson get on the VDS?",
        source: "DEPLOYMENT.md",
        section: "Where things live on the VDS",
    },
    EvalCase {
        question: "Why compile a static musl binary instead of using Docker?",
        source: "DEPLOYMENT.md",
        section: "Why a static musl binary",
    },
    EvalCase {
        question: "How do I redeploy an older lesson from the Run workflow lessons input?",
        source: "DEPLOYMENT.md",
        section: "Which lessons deploy",
    },
    EvalCase {
        question: "A secret shows up empty in the .env file on the VDS, what is the likely cause?",
        source: "DEPLOYMENT.md",
        section: "Debugging a failed deploy",
    },
    EvalCase {
        question: "What happens when an agent tries to edit files under .github/workflows?",
        source: "AGENTS.md",
        section: "Environment constraints",
    },
    EvalCase {
        question: "Why are cargo fmt and clippy non-blocking in build-test?",
        source: "AGENTS.md",
        section: "CI/CD pipeline",
    },
    EvalCase {
        question: "Steps to add a new lesson folder to the repo",
        source: "AGENTS.md",
        section: "Adding a new lesson",
    },
    EvalCase {
        question: "Why store the conversation history as JSON rather than SQLite?",
        source: "07. Context Persistence/README.md",
        section: "Why JSON over SQLite",
    },
    EvalCase {
        question: "Why count tokens with a heuristic instead of a tokenizer library?",
        source: "08. Token Counting/README.md",
        section: "Why a heuristic instead of a real tokenizer",
    },
    EvalCase {
        question: "How does branching fork the message list at a checkpoint?",
        source: "10. Context Management Strategies/README.md",
        section: "Why branching is a fork",
    },
    EvalCase {
        question: "Why keep short-term, working and long-term memory in three separate files?",
        source: "11. Agent Memory Model/README.md",
        section: "Why three separate files",
    },
    EvalCase {
        question: "What does the agent do when a request conflicts with an invariant?",
        source: "14. Invariant Guardrails/README.md",
        section: "What happens at a conflict",
    },
    EvalCase {
        question: "Which endpoints approve the plan and record the validation gate?",
        source: "15. Controlled State Transitions/README.md",
        section: "Two gates, two dedicated endpoints",
    },
    EvalCase {
        question: "Why connect to the remote DeepWiki MCP server instead of a local one?",
        source: "16. MCP Connection/README.md",
        section: "Why a remote server",
    },
    EvalCase {
        question: "How is scheduler data saved to a JSON file with write-then-rename?",
        source: "18. Scheduler and Background Tasks/README.md",
        section: "Saving the data",
    },
    EvalCase {
        question: "How do chained tools pass data by artifact id with checksums?",
        source: "19. MCP Tool Composition/README.md",
        section: "Passing data between tools",
    },
    EvalCase {
        question: "How does the router refuse a tool called with the wrong server prefix?",
        source: "20. MCP Orchestration/README.md",
        section: "Choosing the tool and routing the call",
    },
    EvalCase {
        question: "храните последние N сообщений как есть, остальное заменяйте summary",
        source: "09. Context Compression/README.md",
        section: "Задание",
    },
    EvalCase {
        question: "Agent method that sends the conversation history to the LLM and returns the reply",
        source: "06. First Agent/src/main.rs",
        section: "impl Agent",
    },
    EvalCase {
        question: "open an MCP session over a transport, initialize and page through tools/list",
        source: "16. MCP Connection/src/main.rs",
        section: "fn inspect",
    },
];

fn is_relevant(chunk: &Chunk, case: &EvalCase) -> bool {
    chunk.source == case.source && chunk.sections.iter().any(|s| s.contains(case.section))
}

/// The byte ranges of the expected answer: every section of the case's file
/// whose path contains the case's section.
fn answer_spans<'a>(corpus: &'a Corpus, case: &EvalCase) -> Option<(&'a Document, Vec<&'a Span>)> {
    let (doc, spans) = corpus.find(case.source)?;
    let answer: Vec<&Span> = spans
        .iter()
        .filter(|s| s.path.contains(case.section))
        .collect();
    Some((doc, answer))
}

/// Characters of the answer inside the (sorted, possibly overlapping) byte
/// ranges, each counted once.
fn answer_chars(text: &str, answer: &[&Span], ranges: &[(usize, usize)]) -> usize {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in ranges {
        let n = merged.len();
        if n > 0 && s <= merged[n - 1].1 {
            merged[n - 1].1 = merged[n - 1].1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    let mut total = 0;
    for span in answer {
        for &(s, e) in &merged {
            let (a, b) = (span.start.max(s), span.end.min(e));
            if a < b {
                total += char_len(&text[a..b]);
            }
        }
    }
    total
}

#[derive(Serialize, Clone, Debug, Default)]
struct EvalRow {
    question: String,
    source: String,
    section: String,
    /// 1-based rank of the first relevant chunk within the top `EVAL_DEPTH`.
    rank: Option<usize>,
    top_chunk: String,
    top_section: String,
    top_score: f32,
    /// Share of the first relevant chunk that is the answer section: the
    /// rest is noise the LLM would get along with it.
    focus: Option<f64>,
    /// Share of the answer section that the top 3 chunks contain.
    coverage: f64,
}

#[derive(Serialize, Clone, Debug, Default)]
struct EvalSummary {
    cases: usize,
    hit_at_1: f64,
    hit_at_3: f64,
    mrr: f64,
    /// Mean `focus` over the cases with a relevant chunk in the top 10.
    focus: f64,
    /// Mean `coverage` over all cases.
    coverage_at_3: f64,
    rows: Vec<EvalRow>,
}

/// The cases whose file is in this corpus (all of them for the built-in one).
fn eval_cases(corpus: &Corpus) -> Vec<&'static EvalCase> {
    EVAL.iter()
        .filter(|case| corpus.find(case.source).is_some())
        .collect()
}

fn evaluate(corpus: &Corpus, index: &Index, cases: &[&EvalCase], raw: &[Vec<f32>]) -> EvalSummary {
    let mut rows = Vec::new();
    let (mut hit1, mut hit3, mut mrr) = (0usize, 0usize, 0f64);
    let (mut focus_sum, mut focused, mut coverage_sum) = (0f64, 0usize, 0f64);
    for (case, raw) in cases.iter().zip(raw) {
        let hits = index.search(&index.in_space(raw), EVAL_DEPTH);
        let rank = hits
            .iter()
            .position(|(i, _)| is_relevant(&index.chunks[*i], case))
            .map(|p| p + 1);
        if rank == Some(1) {
            hit1 += 1;
        }
        if rank.is_some_and(|r| r <= 3) {
            hit3 += 1;
        }
        if let Some(r) = rank {
            mrr += 1.0 / r as f64;
        }

        let (mut focus, mut coverage) = (None, 0.0);
        if let Some((doc, answer)) = answer_spans(corpus, case) {
            let chunk_range = |i: usize| (index.chunks[i].start, index.chunks[i].end);
            if let Some(r) = rank {
                let chunk = &index.chunks[hits[r - 1].0];
                let inside = answer_chars(&doc.text, &answer, &[chunk_range(hits[r - 1].0)]);
                focus = Some(ratio(inside, chunk.chars));
            }
            let mut top3: Vec<(usize, usize)> = hits
                .iter()
                .take(3)
                .filter(|(i, _)| index.chunks[*i].source == case.source)
                .map(|(i, _)| chunk_range(*i))
                .filter(|&(_, e)| e <= doc.text.len())
                .collect();
            top3.sort_unstable();
            let total: usize = answer
                .iter()
                .map(|s| char_len(&doc.text[s.start..s.end]))
                .sum();
            coverage = ratio(answer_chars(&doc.text, &answer, &top3), total);
        }
        if let Some(f) = focus {
            focus_sum += f;
            focused += 1;
        }
        coverage_sum += coverage;

        let top = hits.first().map(|(i, s)| (&index.chunks[*i], *s));
        rows.push(EvalRow {
            question: case.question.to_string(),
            source: case.source.to_string(),
            section: case.section.to_string(),
            rank,
            top_chunk: top.map(|(c, _)| c.chunk_id.clone()).unwrap_or_default(),
            top_section: top.map(|(c, _)| c.section.clone()).unwrap_or_default(),
            top_score: top.map_or(0.0, |(_, s)| (s * 1000.0).round() / 1000.0),
            focus,
            coverage,
        });
    }
    let n = cases.len();
    EvalSummary {
        cases: n,
        hit_at_1: ratio(hit1, n),
        hit_at_3: ratio(hit3, n),
        mrr: if n == 0 { 0.0 } else { round2(mrr / n as f64) },
        focus: if focused == 0 {
            0.0
        } else {
            round2(focus_sum / focused as f64)
        },
        coverage_at_3: if n == 0 {
            0.0
        } else {
            round2(coverage_sum / n as f64)
        },
        rows,
    }
}

#[derive(Serialize, Clone, Debug)]
struct StrategyReport {
    strategy: Strategy,
    params: ChunkParams,
    embedder: String,
    dims: usize,
    created_at: u64,
    /// The index was built from a different version of the corpus.
    stale: bool,
    stats: Stats,
    eval: Option<EvalSummary>,
}

#[derive(Serialize, Clone, Debug, Default)]
struct Comparison {
    corpus: CorpusInfo,
    strategies: Vec<StrategyReport>,
    error: Option<String>,
}

fn same_embedder(loaded: &[Loaded], embedder: &Embedder) -> Result<(), String> {
    let id = embedder.id();
    match loaded.iter().find(|l| l.index.meta.embedder != id) {
        Some(l) => Err(format!(
            "The {} index was built with {}, but the current embedder is {id}. Rebuild the index.",
            l.index.meta.strategy.name(),
            l.index.meta.embedder
        )),
        None => Ok(()),
    }
}

async fn compare(corpus: &Corpus, embedder: &Embedder, loaded: &[Loaded]) -> Comparison {
    let cases = eval_cases(corpus);
    let questions: Vec<&str> = cases.iter().map(|c| c.question).collect();
    let raw = match same_embedder(loaded, embedder) {
        Ok(()) if !questions.is_empty() => embedder.embed_queries(&questions).await,
        Ok(()) => Ok(Vec::new()),
        Err(e) => Err(e),
    };
    let (raw, error) = match raw {
        Ok(raw) => (Some(raw), None),
        Err(e) => (None, Some(e)),
    };
    let strategies = loaded
        .iter()
        .map(|l| StrategyReport {
            strategy: l.index.meta.strategy,
            params: l.index.meta.params,
            embedder: l.index.meta.embedder.clone(),
            dims: l.index.meta.dims,
            created_at: l.index.meta.created_at,
            stale: l.index.meta.corpus_checksum != corpus.checksum,
            stats: stats(corpus, l),
            eval: raw
                .as_ref()
                .filter(|_| !cases.is_empty())
                .map(|raw| evaluate(corpus, &l.index, &cases, raw)),
        })
        .collect();
    Comparison {
        corpus: corpus_info(corpus),
        strategies,
        error,
    }
}

fn print_report(cmp: &Comparison) {
    let c = &cmp.corpus;
    println!(
        "corpus: {} — {} documents, {} chars ≈ {} pages ≈ {} tokens",
        c.origin, c.documents, c.chars, c.pages, c.tokens
    );
    println!(
        "{:<12} {:>7} {:>8} {:>7} {:>6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "strategy",
        "chunks",
        "avg",
        "median",
        "max",
        "stddev",
        "redund",
        "1-sect",
        "starts",
        "ends",
        "hit@1",
        "hit@3",
        "mrr",
        "focus",
        "cover"
    );
    for s in &cmp.strategies {
        let st = &s.stats;
        let e = s.eval.clone().unwrap_or_default();
        println!(
            "{:<12} {:>7} {:>8} {:>7} {:>6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
            s.strategy.name(),
            st.chunks,
            st.avg_chars,
            st.median_chars,
            st.max_chars,
            st.stddev_chars,
            st.redundancy,
            st.single_section,
            st.starts_at_section,
            st.ends_at_paragraph,
            e.hit_at_1,
            e.hit_at_3,
            e.mrr,
            e.focus,
            e.coverage_at_3
        );
    }
    if let Some(error) = &cmp.error {
        println!("error: {error}");
    }
}

// ---------------------------------------------------------------------------
// HTTP API and page.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct Settings {
    embedder: String,
    index_dir: String,
    corpus_origin: String,
    defaults: ChunkParams,
}

struct Built {
    loaded: Vec<Loaded>,
    comparison: Comparison,
}

#[derive(Clone)]
struct AppState {
    corpus: Arc<Corpus>,
    embedder: Arc<Embedder>,
    settings: Arc<Settings>,
    index_dir: PathBuf,
    built: Arc<RwLock<Built>>,
}

async fn config(State(app): State<AppState>) -> Json<Settings> {
    Json((*app.settings).clone())
}

async fn corpus_api(State(app): State<AppState>) -> Json<CorpusInfo> {
    Json(corpus_info(&app.corpus))
}

async fn compare_api(State(app): State<AppState>) -> Json<Comparison> {
    Json(app.built.read().await.comparison.clone())
}

async fn index_api(
    State(app): State<AppState>,
    Json(params): Json<ChunkParams>,
) -> Json<Comparison> {
    let mut built = app.built.write().await;
    if let Err(e) = params.validate() {
        let mut cmp = built.comparison.clone();
        cmp.error = Some(e);
        return Json(cmp);
    }
    match build_all(&app.corpus, &app.embedder, &params, &app.index_dir).await {
        Ok(loaded) => {
            let cmp = compare(&app.corpus, &app.embedder, &loaded).await;
            built.loaded = loaded;
            built.comparison = cmp.clone();
            Json(cmp)
        }
        Err(e) => {
            let mut cmp = built.comparison.clone();
            cmp.error = Some(e);
            Json(cmp)
        }
    }
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    k: Option<usize>,
}

#[derive(Serialize, Clone, Debug)]
struct Hit {
    rank: usize,
    score: f32,
    chunk_id: String,
    source: String,
    title: String,
    section: String,
    sections: Vec<String>,
    chars: usize,
    tokens: usize,
    text: String,
}

#[derive(Serialize, Clone, Debug)]
struct StrategyHits {
    strategy: Strategy,
    hits: Vec<Hit>,
}

#[derive(Serialize, Clone, Debug, Default)]
struct SearchResponse {
    query: String,
    results: Vec<StrategyHits>,
    error: Option<String>,
}

async fn search_api(
    State(app): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let query = req.query.trim().to_string();
    let fail = |error: String| SearchResponse {
        query: query.clone(),
        results: Vec::new(),
        error: Some(error),
    };
    if query.is_empty() {
        return Json(fail("Type a query first.".to_string()));
    }
    if char_len(&query) > MAX_QUERY_CHARS {
        return Json(fail(format!(
            "Keep the query under {MAX_QUERY_CHARS} characters."
        )));
    }
    let k = req.k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let built = app.built.read().await;
    if built.loaded.is_empty() {
        return Json(fail("There is no index yet. Build it first.".to_string()));
    }
    if let Err(e) = same_embedder(&built.loaded, &app.embedder) {
        return Json(fail(e));
    }
    let raw = match app.embedder.embed_queries(&[query.as_str()]).await {
        Ok(mut raw) if !raw.is_empty() => raw.remove(0),
        Ok(_) => return Json(fail("The embedder returned nothing.".to_string())),
        Err(e) => return Json(fail(e)),
    };
    let results = built
        .loaded
        .iter()
        .map(|l| {
            let index = &l.index;
            let hits = index
                .search(&index.in_space(&raw), k)
                .into_iter()
                .enumerate()
                .map(|(rank, (i, score))| {
                    let c = &index.chunks[i];
                    Hit {
                        rank: rank + 1,
                        score: (score * 1000.0).round() / 1000.0,
                        chunk_id: c.chunk_id.clone(),
                        source: c.source.clone(),
                        title: c.title.clone(),
                        section: c.section.clone(),
                        sections: c.sections.clone(),
                        chars: c.chars,
                        tokens: c.tokens,
                        text: c.text.clone(),
                    }
                })
                .collect();
            StrategyHits {
                strategy: index.meta.strategy,
                hits,
            }
        })
        .collect();
    Json(SearchResponse {
        query,
        results,
        error: None,
    })
}

#[derive(Deserialize)]
struct SourceQuery {
    source: String,
}

#[derive(Serialize, Clone, Debug)]
struct MapChunk {
    chunk_id: String,
    start: usize,
    end: usize,
    chars: usize,
    section: String,
    sections: usize,
}

/// One document's sections and where each strategy cut it, for the page's
/// side-by-side chunk map.
async fn map_api(
    State(app): State<AppState>,
    Query(q): Query<SourceQuery>,
) -> Json<serde_json::Value> {
    let Some((doc, spans)) = app.corpus.find(&q.source) else {
        return Json(json!({ "error": format!("No document {:?}.", q.source) }));
    };
    let built = app.built.read().await;
    let strategies: Vec<serde_json::Value> = built
        .loaded
        .iter()
        .map(|l| {
            let chunks: Vec<MapChunk> = l
                .index
                .chunks
                .iter()
                .filter(|c| c.source == doc.source)
                .map(|c| MapChunk {
                    chunk_id: c.chunk_id.clone(),
                    start: c.start,
                    end: c.end,
                    chars: c.chars,
                    section: c.section.clone(),
                    sections: c.sections.len(),
                })
                .collect();
            json!({ "strategy": l.index.meta.strategy, "chunks": chunks })
        })
        .collect();
    Json(json!({
        "source": doc.source,
        "title": doc.title,
        "bytes": doc.text.len(),
        "sections": spans,
        "strategies": strategies,
    }))
}

#[derive(Deserialize)]
struct ChunkQuery {
    id: String,
}

/// One chunk with its metadata and the start of its embedding (the full
/// vector is in the index file).
async fn chunk_api(
    State(app): State<AppState>,
    Query(q): Query<ChunkQuery>,
) -> Json<serde_json::Value> {
    let built = app.built.read().await;
    let found = built
        .loaded
        .iter()
        .flat_map(|l| l.index.chunks.iter())
        .find(|c| c.chunk_id == q.id);
    Json(match found {
        Some(c) => {
            let nonzero = c.embedding.iter().filter(|x| **x != 0.0).count();
            let preview: Vec<f32> = c
                .embedding
                .iter()
                .take(12)
                .map(|x| (x * 10_000.0).round() / 10_000.0)
                .collect();
            json!({
                "chunk_id": c.chunk_id,
                "source": c.source,
                "title": c.title,
                "kind": c.kind,
                "section": c.section,
                "sections": c.sections,
                "ordinal": c.ordinal,
                "start": c.start,
                "end": c.end,
                "chars": c.chars,
                "tokens": c.tokens,
                "text": c.text,
                "dims": c.embedding.len(),
                "nonzero": nonzero,
                "embedding_preview": preview,
            })
        }
        None => json!({ "error": format!("No chunk {:?}.", q.id) }),
    })
}

async fn index_page() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_page))
        .route("/api/config", get(config))
        .route("/api/corpus", get(corpus_api))
        .route("/api/compare", get(compare_api))
        .route("/api/index", post(index_api))
        .route("/api/search", post(search_api))
        .route("/api/map", get(map_api))
        .route("/api/chunk", get(chunk_api))
        .with_state(state)
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// `EMBEDDER=openai` uses an OpenAI-compatible `/embeddings` endpoint;
/// anything else (the default) the local one, which needs no network.
fn embedder_from_env() -> Result<Embedder, String> {
    let kind = env_non_empty("EMBEDDER").unwrap_or_else(|| "local".to_string());
    match kind.trim() {
        "local" => Ok(Embedder::Local { dims: LOCAL_DIMS }),
        "openai" => {
            let api_key = env_non_empty("EMBEDDING_API_KEY")
                .or_else(|| env_non_empty("OPENAI_API_KEY"))
                .ok_or("EMBEDDER=openai needs EMBEDDING_API_KEY (or OPENAI_API_KEY).")?;
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

/// The index from disk if it's there and matches the corpus and embedder;
/// otherwise a fresh build.
async fn startup_index(corpus: &Corpus, embedder: &Embedder, dir: &Path) -> Built {
    let reusable = match load_all(dir) {
        Ok(Some(loaded))
            if same_embedder(&loaded, embedder).is_ok()
                && loaded
                    .iter()
                    .all(|l| l.index.meta.corpus_checksum == corpus.checksum) =>
        {
            Some(loaded)
        }
        Ok(_) => None,
        Err(e) => {
            eprintln!("warning: {e}; rebuilding");
            None
        }
    };
    let loaded = match reusable {
        Some(loaded) => {
            println!("loaded the index from {}", dir.display());
            Ok(loaded)
        }
        None => {
            let params = reusable_params(dir).unwrap_or_default();
            build_all(corpus, embedder, &params, dir).await
        }
    };
    match loaded {
        Ok(loaded) => {
            let comparison = compare(corpus, embedder, &loaded).await;
            Built { loaded, comparison }
        }
        Err(e) => Built {
            loaded: Vec::new(),
            comparison: Comparison {
                corpus: corpus_info(corpus),
                strategies: Vec::new(),
                error: Some(e),
            },
        },
    }
}

/// The chunking parameters of the last index on disk, so a rebuild after a
/// corpus change keeps them.
fn reusable_params(dir: &Path) -> Option<ChunkParams> {
    let loaded = load_index(dir, Strategy::Structural).ok()??;
    let params = loaded.index.meta.params;
    params.validate().ok().map(|_| params)
}

#[tokio::main]
async fn main() {
    let port = env_non_empty("PORT").unwrap_or_else(|| "3000".to_string());
    // Relative to the working directory: the lesson folder under `cargo run`,
    // `~/apps/lesson-21` on the VDS (the systemd unit's WorkingDirectory).
    let index_dir =
        PathBuf::from(env_non_empty("INDEX_DIR").unwrap_or_else(|| "index".to_string()));
    let embedder = embedder_from_env().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    let corpus = match env_non_empty("DOCS_DIR") {
        Some(dir) => {
            let docs = load_dir(Path::new(&dir)).unwrap_or_else(|e| {
                eprintln!("Error: DOCS_DIR: {e}");
                std::process::exit(1);
            });
            Corpus::new(dir, docs)
        }
        None => Corpus::new(
            "built-in: this repo's docs and code".to_string(),
            builtin_documents(),
        ),
    };

    // `app index`: rebuild both indexes, print the comparison, exit.
    if std::env::args().nth(1).as_deref() == Some("index") {
        let params = ChunkParams::default();
        match build_all(&corpus, &embedder, &params, &index_dir).await {
            Ok(loaded) => {
                print_report(&compare(&corpus, &embedder, &loaded).await);
                println!("saved to {}", index_dir.display());
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let built = startup_index(&corpus, &embedder, &index_dir).await;
    print_report(&built.comparison);
    let settings = Settings {
        embedder: embedder.id(),
        index_dir: index_dir.display().to_string(),
        corpus_origin: corpus.origin.clone(),
        defaults: ChunkParams::default(),
    };
    let state = AppState {
        corpus: Arc::new(corpus),
        embedder: Arc::new(embedder),
        settings: Arc::new(settings),
        index_dir,
        built: Arc::new(RwLock::new(built)),
    };

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app(state)).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lesson21-{}-{name}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn builtin() -> Corpus {
        Corpus::new("test".to_string(), builtin_documents())
    }

    fn local() -> Embedder {
        Embedder::Local { dims: LOCAL_DIMS }
    }

    /// Every non-whitespace byte of `text` is inside one of `ranges`.
    fn covers(text: &str, ranges: &[(usize, usize)]) -> bool {
        let mut covered = vec![false; text.len()];
        for &(s, e) in ranges {
            for c in covered.iter_mut().take(e).skip(s) {
                *c = true;
            }
        }
        text.char_indices()
            .all(|(i, c)| c.is_whitespace() || covered[i])
    }

    const SAMPLE_MD: &str = "# Title\n\nIntro.\n\n## Run\n\n```bash\n# not a heading\ncargo run\n```\n\n### Details\n\nMore text about the details.\n\n## Configuration\n\nA table of variables.\n";

    #[test]
    fn builtin_corpus_is_big_enough() {
        let corpus = builtin();
        let info = corpus_info(&corpus);
        assert_eq!(info.documents, BUILTIN.len());
        assert!(info.documents >= 20, "{} documents", info.documents);
        assert!(info.pages >= 30.0, "only {} pages", info.pages);
        let kinds: Vec<&str> = info.files.iter().map(|f| f.kind.as_str()).collect();
        assert!(kinds.contains(&"markdown") && kinds.contains(&"rust"));
        let lesson20 = corpus.find("20. MCP Orchestration/README.md").unwrap().0;
        assert_eq!(lesson20.title, "20. MCP Orchestration");
        assert_eq!(
            corpus.find("06. First Agent/src/main.rs").unwrap().0.kind,
            DocKind::Rust
        );
    }

    #[test]
    fn markdown_outline_follows_headings_and_skips_code_fences() {
        let spans = markdown_outline(SAMPLE_MD);
        let paths: Vec<&str> = spans.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "Title",
                "Title › Run",
                "Title › Run › Details",
                "Title › Configuration"
            ]
        );
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans.last().unwrap().end, SAMPLE_MD.len());
        for pair in spans.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert!(SAMPLE_MD[spans[1].start..spans[1].end].contains("# not a heading"));
        assert_eq!(markdown_title(SAMPLE_MD).as_deref(), Some("Title"));

        let with_preamble = markdown_outline("Some intro.\n\n# Head\n\nBody\n");
        assert_eq!(with_preamble[0].path, PREAMBLE);
        assert_eq!(with_preamble[1].path, "Head");
        assert!(heading("#[derive(Debug)]").is_none());
        assert!(heading("####### seven").is_none());
    }

    const SAMPLE_RS: &str = "use std::fmt;\nuse std::io;\n\n// -----\n// Helpers.\n// -----\n\n/// Adds.\n#[inline]\npub fn add(a: u32, b: u32) -> u32 {\n    a + b\n\n    // still inside\n}\n\nstruct Point {\n    x: i32,\n}\n\nimpl<T: fmt::Debug> Show for Wrapper<T> {\n    fn show(&self) {}\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn works() {}\n}\n";

    #[test]
    fn rust_outline_finds_top_level_items() {
        let spans = rust_outline(SAMPLE_RS);
        let paths: Vec<&str> = spans.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "use …",
                "// Helpers.",
                "fn add",
                "struct Point",
                "impl<T: fmt::Debug> Show for Wrapper<T>",
                "mod tests"
            ]
        );
        assert!(SAMPLE_RS[spans[2].start..spans[2].end].starts_with("/// Adds."));
        assert!(SAMPLE_RS[spans[2].start..spans[2].end].contains("// still inside"));
        assert_eq!(
            item_label("    pub async fn respond(&self)").as_deref(),
            Some("fn respond")
        );
        assert_eq!(item_label("let x = 1;"), None);
        assert_eq!(item_label("impl Agent {").as_deref(), Some("impl Agent"));

        let corpus = builtin();
        let (_, spans) = corpus.find("06. First Agent/src/main.rs").unwrap();
        assert!(spans.iter().any(|s| s.path == "impl Agent"), "{spans:?}");
        assert!(spans.iter().any(|s| s.path == "fn main"));
    }

    #[test]
    fn fixed_windows_cover_the_text_with_overlap_and_respect_utf8() {
        let text = "Привет, мир! Это проверка разбиения. ".repeat(40) + &"word ".repeat(200);
        for overlap in [0, 50] {
            let windows = fixed_windows(&text, 200, overlap);
            assert!(windows.len() > 5);
            assert_eq!(windows[0].0, 0);
            assert_eq!(windows.last().unwrap().1, text.len());
            for &(s, e) in &windows {
                // Slicing panics if a range isn't on char boundaries.
                let piece = &text[s..e];
                assert!(char_len(piece) <= 200);
                assert!(char_len(piece) >= 200 * 4 / 5 - 1 || e == text.len());
            }
            for pair in windows.windows(2) {
                if overlap == 0 {
                    assert_eq!(pair[0].1, pair[1].0);
                } else {
                    assert!(pair[1].0 < pair[0].1, "windows should overlap");
                    assert!(pair[1].0 > pair[0].0);
                }
            }
            assert!(covers(&text, &windows));
        }
        assert!(fixed_windows("", 100, 10).is_empty());
        assert_eq!(fixed_windows("short", 100, 10), vec![(0, 5)]);
    }

    #[test]
    fn pack_respects_the_limit_and_paragraphs() {
        let text = "aaaa\n\nbbbb\n\ncccc\n\n".to_string() + &"x".repeat(25);
        let cuts = paragraph_starts(&text, 0, text.len(), DocKind::Text);
        assert_eq!(cuts, vec![6, 12, 18]);
        let packed = pack(&text, 0, text.len(), &cuts, 12);
        assert_eq!(&text[packed[0].0..packed[0].1], "aaaa\n\nbbbb\n\n");
        for &(s, e) in &packed {
            assert!(char_len(&text[s..e]) <= 12);
        }
        assert!(covers(&text, &packed));

        let fenced = "intro\n\n```\na\n\nb\n```\n\nafter\n";
        let cuts = paragraph_starts(fenced, 0, fenced.len(), DocKind::Markdown);
        assert_eq!(cuts, vec![7, 21], "no cut inside the code fence");
    }

    #[test]
    fn both_strategies_cover_every_document_within_their_limits() {
        let corpus = builtin();
        let params = ChunkParams::default();
        for (doc, spans) in corpus.docs.iter().zip(&corpus.outlines) {
            let fixed = chunk_fixed(doc, spans, &params);
            let structural = chunk_structural(doc, spans, &params);
            for (name, drafts, limit) in [
                ("fixed", &fixed, params.fixed_size),
                ("structural", &structural, params.max_chars),
            ] {
                let ranges: Vec<(usize, usize)> = drafts.iter().map(|d| (d.start, d.end)).collect();
                assert!(
                    covers(&doc.text, &ranges),
                    "{name} misses text in {}",
                    doc.source
                );
                for d in drafts.iter() {
                    assert!(
                        char_len(&doc.text[d.start..d.end]) <= limit,
                        "{name} {}",
                        doc.source
                    );
                    assert!(
                        !d.section.is_empty() && !d.sections.is_empty(),
                        "{name} {}",
                        doc.source
                    );
                }
            }
            for pair in structural.windows(2) {
                assert!(
                    pair[0].end <= pair[1].start,
                    "structural chunks never overlap"
                );
            }
        }
    }

    #[test]
    fn structural_chunks_follow_sections() {
        let doc = document("sample.md", SAMPLE_MD).unwrap();
        let spans = outline(&doc);
        let params = ChunkParams {
            min_chars: 30,
            max_chars: 300,
            ..ChunkParams::default()
        };
        let drafts = chunk_structural(&doc, &spans, &params);
        // "# Title" + its one-line intro is shorter than min_chars, so it
        // joins "## Run"; the rest are a chunk each.
        let sections: Vec<Vec<String>> = drafts.iter().map(|d| d.sections.clone()).collect();
        assert_eq!(
            sections,
            vec![
                vec!["Title".to_string(), "Title › Run".to_string()],
                vec!["Title › Run › Details".to_string()],
                vec!["Title › Configuration".to_string()],
            ]
        );
        assert!(doc.text[drafts[2].start..drafts[2].end].starts_with("## Configuration"));
    }

    #[test]
    fn long_code_sections_split_between_methods_and_keep_their_names() {
        let mut code = String::from("impl Agent {\n");
        for i in 0..12 {
            code.push_str(&format!(
                "    /// Method {i}.\n    pub fn method_{i}(&self) -> usize {{\n{}    }}\n\n",
                "        let x = 1;\n".repeat(8)
            ));
        }
        code.push_str("}\n");
        let doc = document("agent.rs", &code).unwrap();
        let spans = outline(&doc);
        assert_eq!(spans.len(), 1);
        let params = ChunkParams {
            max_chars: 600,
            ..ChunkParams::default()
        };
        let drafts = chunk_structural(&doc, &spans, &params);
        assert!(drafts.len() > 2);
        assert_eq!(drafts[0].section, "impl Agent");
        assert!(
            drafts[1].section.starts_with("impl Agent › fn method_"),
            "{:?}",
            drafts[1]
        );
        for d in &drafts[1..] {
            assert!(
                doc.text[d.start..d.end]
                    .trim_start()
                    .starts_with("/// Method")
            );
        }
    }

    #[test]
    fn local_embedder_is_deterministic_and_ranks_related_text_higher() {
        let texts = [
            "Deploy the binary to the VDS over SSH and restart the systemd service.",
            "Count tokens with a heuristic instead of a tokenizer library.",
            "Храните последние сообщения как есть, остальное сжимайте в summary.",
        ];
        let (vectors, idf) = local_fit(&texts, LOCAL_DIMS);
        assert_eq!(idf.len(), LOCAL_DIMS);
        for v in &vectors {
            assert_eq!(v.len(), LOCAL_DIMS);
            assert!((dot(v, v) - 1.0).abs() < 1e-4);
        }
        assert_eq!(local_fit(&texts, LOCAL_DIMS).0, vectors);
        let query = weigh(
            hashed_tf("how is the service deployed to the VDS", LOCAL_DIMS),
            &idf,
        );
        let scores: Vec<f32> = vectors.iter().map(|v| dot(v, &query)).collect();
        assert!(scores[0] > scores[1] && scores[0] > scores[2], "{scores:?}");
        // Stems make Russian word forms meet: "сообщений" ~ "сообщения".
        let query = weigh(hashed_tf("последних сообщений", LOCAL_DIMS), &idf);
        let scores: Vec<f32> = vectors.iter().map(|v| dot(v, &query)).collect();
        assert!(scores[2] > scores[0] && scores[2] > scores[1], "{scores:?}");
        assert_eq!(
            stems("crate_info Chunking"),
            vec!["crate_info", "crate", "info", "chunk"]
        );
    }

    #[tokio::test]
    async fn indexes_carry_metadata_and_round_trip_through_json() {
        let corpus = builtin();
        let dir = temp_dir("roundtrip");
        let loaded = build_all(&corpus, &local(), &ChunkParams::default(), &dir)
            .await
            .unwrap();
        assert_eq!(loaded.len(), 2);
        for l in &loaded {
            let index = &l.index;
            assert_eq!(index.meta.chunks, index.chunks.len());
            assert_eq!(index.meta.dims, LOCAL_DIMS);
            assert_eq!(index.meta.idf.len(), LOCAL_DIMS);
            assert_eq!(index.meta.documents, corpus.docs.len());
            let mut ids = std::collections::HashSet::new();
            for c in &index.chunks {
                assert!(
                    ids.insert(c.chunk_id.clone()),
                    "duplicate id {}",
                    c.chunk_id
                );
                assert!(c.chunk_id.starts_with(index.meta.strategy.name()));
                assert!(!c.source.is_empty() && !c.title.is_empty() && !c.section.is_empty());
                let doc = corpus.find(&c.source).unwrap().0;
                assert_eq!(c.text, doc.text[c.start..c.end]);
                assert_eq!(c.embedding.len(), LOCAL_DIMS);
                // Unit length, or all zeros for a chunk without a single word.
                let norm = dot(&c.embedding, &c.embedding);
                assert!((norm - 1.0).abs() < 1e-3 || norm == 0.0, "{}", c.chunk_id);
            }
            let back = load_index(&dir, index.meta.strategy).unwrap().unwrap();
            assert_eq!(back.index, *index);
            assert_eq!(back.file_bytes, l.file_bytes);
        }
        assert!(load_all(&dir).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eval_cases_point_at_real_sections() {
        let corpus = builtin();
        assert_eq!(eval_cases(&corpus).len(), EVAL.len());
        for case in EVAL {
            let (_, spans) = corpus.find(case.source).unwrap();
            assert!(
                spans.iter().any(|s| s.path.contains(case.section)),
                "{} has no section {:?}",
                case.source,
                case.section
            );
        }
    }

    #[tokio::test]
    async fn comparison_shows_how_the_strategies_differ() {
        let corpus = builtin();
        let dir = temp_dir("compare");
        let loaded = build_all(&corpus, &local(), &ChunkParams::default(), &dir)
            .await
            .unwrap();
        let cmp = compare(&corpus, &local(), &loaded).await;
        assert!(cmp.error.is_none(), "{:?}", cmp.error);
        assert_eq!(cmp.strategies.len(), 2);
        let (fixed, structural) = (&cmp.strategies[0], &cmp.strategies[1]);
        assert_eq!(fixed.strategy, Strategy::Fixed);
        assert_eq!(structural.strategy, Strategy::Structural);
        print_report(&cmp);

        // Overlap makes the fixed index repeat text; sections never overlap.
        assert!(fixed.stats.redundancy > 1.05, "{:?}", fixed.stats);
        assert!(structural.stats.redundancy <= 1.0, "{:?}", structural.stats);
        // Structural chunks start where sections start and stay inside them.
        assert!(structural.stats.starts_at_section > fixed.stats.starts_at_section);
        assert!(structural.stats.single_section > fixed.stats.single_section);
        assert!(structural.stats.ends_at_paragraph > fixed.stats.ends_at_paragraph);
        assert!(fixed.stats.max_chars <= DEFAULT_FIXED_SIZE);
        assert!(structural.stats.max_chars <= DEFAULT_MAX_CHARS);

        for report in &cmp.strategies {
            let eval = report.eval.as_ref().unwrap();
            assert_eq!(eval.cases, EVAL.len());
            assert_eq!(eval.rows.len(), EVAL.len());
            for x in [
                eval.hit_at_1,
                eval.hit_at_3,
                eval.mrr,
                eval.focus,
                eval.coverage_at_3,
            ] {
                assert!((0.0..=1.0).contains(&x));
            }
            assert!(eval.hit_at_3 >= eval.hit_at_1);
            assert!(
                eval.hit_at_3 > 0.3,
                "{} retrieval is broken: {eval:?}",
                report.strategy.name()
            );
            assert!(!report.stale);
        }
        // A structural chunk is its section; a fixed window drags in the
        // neighbouring ones.
        let focus = |r: &StrategyReport| r.eval.as_ref().unwrap().focus;
        assert!(
            focus(structural) > focus(fixed),
            "{} vs {}",
            focus(structural),
            focus(fixed)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn params_are_validated() {
        assert!(ChunkParams::default().validate().is_ok());
        let bad = [
            ChunkParams {
                fixed_size: 50,
                ..ChunkParams::default()
            },
            ChunkParams {
                overlap: 600,
                ..ChunkParams::default()
            },
            ChunkParams {
                max_chars: 100,
                ..ChunkParams::default()
            },
            ChunkParams {
                min_chars: 1000,
                ..ChunkParams::default()
            },
        ];
        for p in bad {
            assert!(p.validate().is_err(), "{p:?}");
        }
        let partial: ChunkParams = serde_json::from_str(r#"{"fixed_size": 500}"#).unwrap();
        assert_eq!(partial.fixed_size, 500);
        assert_eq!(partial.overlap, DEFAULT_OVERLAP);
    }

    #[test]
    fn load_dir_reads_supported_files_only() {
        let dir = temp_dir("docs");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join("a.md"), "# A\r\n\r\ntext\r\n").unwrap();
        std::fs::write(dir.join("sub/b.rs"), "fn b() {}\n").unwrap();
        std::fs::write(dir.join("c.png"), [0u8, 1, 2]).unwrap();
        std::fs::write(dir.join(".git/d.md"), "# hidden").unwrap();
        let docs = load_dir(&dir).unwrap();
        let sources: Vec<&str> = docs.iter().map(|d| d.source.as_str()).collect();
        assert_eq!(sources, vec!["a.md", "sub/b.rs"]);
        assert_eq!(docs[0].title, "A");
        assert!(!docs[0].text.contains('\r'));
        assert!(load_dir(&dir.join("sub/missing")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn spawn(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[derive(Default)]
    struct FakeApi {
        requests: AtomicUsize,
        authorized: AtomicUsize,
    }

    /// A fake `/embeddings`: the vector is `[text length, 1, 0]`, and the
    /// items come back in reverse order, as the API is allowed to.
    async fn fake_embeddings(
        State(api): State<Arc<FakeApi>>,
        headers: axum::http::HeaderMap,
        Json(body): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        api.requests.fetch_add(1, Ordering::SeqCst);
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
        if auth == Some("Bearer test-key") && body["model"] == "test-model" {
            api.authorized.fetch_add(1, Ordering::SeqCst);
        }
        let inputs = body["input"].as_array().cloned().unwrap_or_default();
        let data: Vec<serde_json::Value> = inputs
            .iter()
            .enumerate()
            .rev()
            .map(|(i, t)| {
                let len = t.as_str().unwrap_or_default().len() as f32;
                json!({ "object": "embedding", "index": i, "embedding": [len, 1.0, 0.0] })
            })
            .collect();
        Json(json!({ "object": "list", "data": data, "model": body["model"] }))
    }

    #[tokio::test]
    async fn remote_embedder_batches_and_keeps_the_order() {
        let api = Arc::new(FakeApi::default());
        let base = spawn(
            Router::new()
                .route("/v1/embeddings", post(fake_embeddings))
                .with_state(api.clone()),
        )
        .await;
        let remote = RemoteEmbedder {
            http: reqwest::Client::new(),
            endpoint: format!("{base}/v1/embeddings"),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
        };
        let texts: Vec<String> = (0..70).map(|i| "x".repeat(i + 1)).collect();
        let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
        let vectors = remote.embed(&refs).await.unwrap();
        assert_eq!(api.requests.load(Ordering::SeqCst), 2, "70 texts = 64 + 6");
        assert_eq!(api.authorized.load(Ordering::SeqCst), 2);
        assert_eq!(vectors.len(), 70);
        for (i, v) in vectors.iter().enumerate() {
            let expected = normalize(vec![(i + 1) as f32, 1.0, 0.0]);
            assert!(
                (v[0] - expected[0]).abs() < 1e-6,
                "vector {i} is out of order"
            );
            assert!((dot(v, v) - 1.0).abs() < 1e-5);
        }

        let embedder = Embedder::Remote(remote);
        assert_eq!(embedder.id(), "openai:test-model");
        let corpus = Corpus::new(
            "test".to_string(),
            vec![document("a.md", "# A\n\nalpha beta\n\n## B\n\ngamma delta\n").unwrap()],
        );
        let index = build_index(
            &corpus,
            Strategy::Structural,
            &ChunkParams::default(),
            &embedder,
        )
        .await
        .unwrap();
        assert_eq!(index.meta.dims, 3);
        assert!(index.meta.idf.is_empty());

        let broken = RemoteEmbedder {
            http: reqwest::Client::new(),
            endpoint: format!("{base}/v1/nothing-here"),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
        };
        let err = broken.embed(&["a"]).await.unwrap_err();
        assert!(err.contains("404"), "{err}");
    }

    #[tokio::test]
    async fn http_api_builds_compares_and_searches() {
        let dir = temp_dir("api");
        let corpus = builtin();
        let embedder = local();
        let built = startup_index(&corpus, &embedder, &dir).await;
        assert_eq!(built.loaded.len(), 2);
        assert!(index_path(&dir, Strategy::Fixed).exists());
        assert!(index_path(&dir, Strategy::Structural).exists());
        let state = AppState {
            settings: Arc::new(Settings {
                embedder: embedder.id(),
                index_dir: dir.display().to_string(),
                corpus_origin: corpus.origin.clone(),
                defaults: ChunkParams::default(),
            }),
            corpus: Arc::new(corpus),
            embedder: Arc::new(embedder),
            index_dir: dir.clone(),
            built: Arc::new(RwLock::new(built)),
        };
        let base = spawn(app(state)).await;
        let http = reqwest::Client::new();

        let page = http.get(&base).send().await.unwrap().text().await.unwrap();
        assert!(page.contains("<title>Document Indexing</title>"));

        let cmp: serde_json::Value = http
            .post(format!("{base}/api/index"))
            .json(&json!({ "fixed_size": 600, "overlap": 100 }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(cmp["error"].is_null(), "{cmp}");
        assert_eq!(cmp["strategies"][0]["params"]["fixed_size"], 600);
        assert_eq!(cmp["strategies"][1]["strategy"], "structural");
        assert!(cmp["strategies"][0]["stats"]["max_chars"].as_u64().unwrap() <= 600);

        let bad: serde_json::Value = http
            .post(format!("{base}/api/index"))
            .json(&json!({ "fixed_size": 10 }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(bad["error"].as_str().unwrap().contains("fixed_size"));

        let found: serde_json::Value = http
            .post(format!("{base}/api/search"))
            .json(&json!({ "query": "static musl binary instead of Docker", "k": 3 }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(found["error"].is_null(), "{found}");
        let results = found["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        let mut deployment_hit = None;
        for r in results {
            let hits = r["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 3);
            let hit = hits.iter().find(|h| h["source"] == "DEPLOYMENT.md");
            assert!(hit.is_some(), "DEPLOYMENT.md should be in the top 3: {r}");
            deployment_hit = hit.map(|h| h["chunk_id"].as_str().unwrap().to_string());
        }
        let id = deployment_hit.unwrap();

        let chunk: serde_json::Value = http
            .get(format!("{base}/api/chunk"))
            .query(&[("id", id.as_str())])
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(chunk["chunk_id"], id.as_str());
        assert_eq!(chunk["source"], "DEPLOYMENT.md");
        assert_eq!(chunk["dims"], LOCAL_DIMS);
        assert_eq!(chunk["embedding_preview"].as_array().unwrap().len(), 12);

        let map: serde_json::Value = http
            .get(format!("{base}/api/map"))
            .query(&[("source", "DEPLOYMENT.md")])
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(map["sections"].as_array().unwrap().len() >= 6);
        assert_eq!(map["strategies"].as_array().unwrap().len(), 2);

        let empty: serde_json::Value = http
            .post(format!("{base}/api/search"))
            .json(&json!({ "query": "  " }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(empty["error"].is_string());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

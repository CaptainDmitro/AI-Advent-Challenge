//! The knowledge base index, from lessons 21-25 and 31: structural chunking
//! (one chunk per Markdown section) and the local hashed TF-IDF embedder.
//! The corpus is Notely's FAQ and documentation in `kb/`, compiled into the
//! binary, so the deploy ships nothing but the binary.

use serde::Serialize;

/// Structural chunking: the largest chunk; a longer section is split at
/// paragraphs.
const MAX_CHARS: usize = 1800;
/// Structural chunking: a shorter section is joined with the ones after it.
const MIN_CHARS: usize = 300;
/// Dimensions of the local hashed embedding.
const LOCAL_DIMS: usize = 1024;
/// Local stems: the first few characters of a word, so "lock", "locked" and
/// "locking" become the same feature.
const STEM_CHARS: usize = 5;
/// The section of whatever comes before a file's first heading.
const PREAMBLE: &str = "(top of file)";
const PATH_SEP: &str = " › ";

// ---------------------------------------------------------------------------
// The corpus: Notely's FAQ and documentation.
// ---------------------------------------------------------------------------

macro_rules! kb_corpus {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../", $path)))),*]
    };
}

/// Every file in `kb/`. A test checks that none is missing.
pub const KB: &[(&str, &str)] = kb_corpus![
    "kb/faq.md",
    "kb/sign-in.md",
    "kb/two-factor.md",
    "kb/sso.md",
    "kb/sync.md",
    "kb/plans-and-billing.md",
    "kb/account.md",
];

#[derive(Clone, Debug)]
pub struct Document {
    /// Path relative to the lesson folder, e.g. `kb/sign-in.md`.
    source: String,
    /// The first `# heading`, or the path.
    title: String,
    text: String,
}

fn document(source: &str, text: &str) -> Document {
    let text = text.replace("\r\n", "\n");
    let title = markdown_title(&text).unwrap_or_else(|| source.to_string());
    Document {
        source: source.to_string(),
        title,
        text,
    }
}

pub fn kb_documents() -> Vec<Document> {
    KB.iter().map(|(path, text)| document(path, text)).collect()
}

// ---------------------------------------------------------------------------
// Structure: where a document's sections are.
// ---------------------------------------------------------------------------

/// One section of a document: byte range and its heading path, e.g.
/// `Signing in to Notely › ACCOUNT_LOCKED: too many failed sign-in attempts`.
#[derive(Clone, Debug, PartialEq)]
struct Span {
    start: usize,
    end: usize,
    path: String,
}

fn char_len(text: &str) -> usize {
    text.chars().count()
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
/// headings above it. `#` lines inside code fences are not headings.
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

/// A chunk before it gets text and an embedding.
#[derive(Debug, Clone, PartialEq)]
struct Draft {
    start: usize,
    end: usize,
    section: String,
}

/// Windows of `size` characters, each starting `size - overlap` after the
/// previous one. A window ends at the last whitespace in its final fifth, so
/// words aren't cut in half. Byte ranges.
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
/// never inside a code fence.
fn paragraph_starts(text: &str, start: usize, end: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut prev_blank = false;
    for (offset, line) in lines_with_offsets(&text[start..end]) {
        let blank = line.trim().is_empty();
        if offset > 0 && prev_blank && !blank && !in_fence {
            out.push(start + offset);
        }
        if is_fence(line) {
            in_fence = !in_fence;
        }
        prev_blank = blank;
    }
    out
}

/// Splits `[start, end)` at `cuts` and packs the pieces greedily into chunks
/// of at most `max` characters. A single piece longer than that is cut into
/// fixed windows as a last resort.
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

/// Structural chunking: one chunk per section. Sections shorter than
/// `MIN_CHARS` are joined with the ones after them while the result still
/// fits; sections longer than `MAX_CHARS` are split at paragraphs, each piece
/// keeping its section.
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
            let section = spans[i..j]
                .iter()
                .max_by_key(|sp| char_len(&text[sp.start..sp.end]))
                .map(|sp| sp.path.clone())
                .unwrap_or_default();
            drafts.push(Draft {
                start,
                end,
                section,
            });
        } else {
            // Joining never makes a group too long, so this is one section.
            let span = &spans[i];
            let cuts = paragraph_starts(text, span.start, span.end);
            for (s, e) in pack(text, span.start, span.end, &cuts, MAX_CHARS) {
                if !text[s..e].trim().is_empty() {
                    drafts.push(Draft {
                        start: s,
                        end: e,
                        section: span.path.clone(),
                    });
                }
            }
        }
        i = j;
    }
    drafts
}

// ---------------------------------------------------------------------------
// Embeddings: the local hashed TF-IDF.
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
/// `STEM_CHARS` characters. A `SNAKE_CASE` name such as an error code is kept
/// whole, as an exact identifier, and also gives the stems of its parts.
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
/// by IDF over these same chunks, then unit-length.
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

/// The local embedder: no network, no model download, no key.
pub struct Embedder {
    dims: usize,
}

impl Embedder {
    pub fn id(&self) -> String {
        format!("local:hashed-tfidf-{}", self.dims)
    }
}

pub fn local() -> Embedder {
    Embedder { dims: LOCAL_DIMS }
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
    /// The heading path of the section this chunk belongs to.
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
    /// `(source, title)` of every document, in corpus order.
    pub titles: Vec<(String, String)>,
    pub chunks: Vec<Chunk>,
    /// The per-bucket IDF; queries are weighted with it.
    idf: Vec<f32>,
}

/// ≈ tokens: about 4 characters per token for English, about 2 for
/// Cyrillic.
pub fn estimate_tokens(text: &str) -> usize {
    let (ascii, other) = text.chars().fold((0usize, 0usize), |(a, o), c| {
        if c.is_ascii() { (a + 1, o) } else { (a, o + 1) }
    });
    ascii.div_ceil(4) + other.div_ceil(2)
}

impl Index {
    /// The indexing pipeline: outline → structural chunks → embeddings.
    pub fn build(docs: Vec<Document>, embedder: Embedder) -> Index {
        let mut chunks = Vec::new();
        for doc in &docs {
            let spans = markdown_outline(&doc.text);
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
        let (vectors, idf) = local_fit(&texts, embedder.dims);
        for (chunk, vector) in chunks.iter_mut().zip(vectors) {
            chunk.embedding = vector;
        }
        Index {
            embedder,
            documents: docs.len(),
            titles: docs
                .iter()
                .map(|d| (d.source.clone(), d.title.clone()))
                .collect(),
            chunks,
            idf,
        }
    }

    /// The `k` chunks closest to `query` by cosine similarity, best first.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let query = weigh(hashed_tf(query, self.embedder.dims), &self.idf);
        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot(&c.embedding, &query)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (i, score))| Hit {
                rank: rank + 1,
                score: (score * 1000.0).round() / 1000.0,
                chunk: self.chunks[i].clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_corpus_is_every_file_in_kb() {
        let docs = kb_documents();
        assert_eq!(docs.len(), KB.len());
        assert!(docs.iter().all(|d| !d.text.trim().is_empty()));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kb");
        let mut on_disk: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_str().map(|n| format!("kb/{n}")))
            .filter(|n| n.ends_with(".md"))
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = KB.iter().map(|(p, _)| p.to_string()).collect();
        listed.sort();
        assert_eq!(on_disk, listed, "a file in kb/ has to be listed in KB");
        assert_eq!(docs[1].title, "Signing in to Notely");
    }

    #[test]
    fn index_has_metadata_and_finds_the_right_section() {
        let index = Index::build(kb_documents(), local());
        assert_eq!(index.documents, KB.len());
        assert!(index.chunks.len() > 20, "{} chunks", index.chunks.len());
        let mut ids: Vec<&str> = index.chunks.iter().map(|c| c.chunk_id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), index.chunks.len(), "chunk ids are unique");
        assert!(index.chunks.iter().all(|c| !c.section.is_empty()));
        assert!(index.chunks.iter().all(|c| char_len(&c.text) <= MAX_CHARS));
        // Every sign-in error code has a section of its own.
        for code in ["ACCOUNT_LOCKED", "EMAIL_NOT_VERIFIED", "SSO_REQUIRED", "INVALID_2FA_CODE"] {
            assert!(
                index
                    .chunks
                    .iter()
                    .any(|c| c.source == "kb/sign-in.md" && c.section.contains(code)),
                "no section for {code}"
            );
        }

        let hits = index.search("authenticator code rejected phone clock time", 3);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits.iter().map(|h| h.rank).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(
            hits.iter().any(|h| h.chunk.source == "kb/two-factor.md"
                && h.chunk.section.ends_with("Codes are rejected: check the device clock")),
            "{hits:?}"
        );
    }

    #[test]
    fn error_codes_are_kept_whole() {
        let stems = stems("Error ACCOUNT_LOCKED at sign-in");
        assert!(stems.contains(&"account_locked".to_string()), "{stems:?}");
        assert!(stems.contains(&"accou".to_string()) && stems.contains(&"locke".to_string()));
    }
}

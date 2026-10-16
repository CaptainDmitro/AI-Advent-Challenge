//! The heuristic reranker from lesson 23, as in lesson 31: the second stage
//! after the vector search. It scores every candidate again from three
//! signals, so a chunk that is only vaguely similar to the question can't
//! outrank one that names what it asks (here: the error code).

use std::collections::{HashMap, HashSet};

use crate::index::{Chunk, Hit, stems};

/// Weights of the three signals (lesson 23's grid search): keyword coverage
/// matters most, cosine and the heading break ties.
pub const W_COSINE: f32 = 0.3;
pub const W_COVERAGE: f32 = 0.5;
pub const W_HEADING: f32 = 0.2;
/// The cosine that counts as a full match; cosine is divided by this and
/// capped at 1 to put it on the same 0..1 scale as the other two signals.
pub const COSINE_FULL: f32 = 0.5;
/// A stem found in more than this share of all chunks ("notely", "the")
/// says nothing about relevance, so it isn't counted.
const COMMON_SHARE: f32 = 0.25;

/// One candidate's signals, each 0..1, and the combined score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signals {
    /// The share of the query's terms the chunk contains, weighted by IDF.
    pub coverage: f32,
    /// The same, in the chunk's file name and section heading only.
    pub heading: f32,
    pub score: f32,
}

pub struct Heuristic {
    /// In how many chunks each stem occurs.
    df: HashMap<String, usize>,
    chunks: usize,
}

impl Heuristic {
    pub fn new(chunks: &[Chunk]) -> Heuristic {
        let mut df = HashMap::new();
        for chunk in chunks {
            let unique: HashSet<String> = stems(&chunk.text).into_iter().collect();
            for stem in unique {
                *df.entry(stem).or_insert(0) += 1;
            }
        }
        Heuristic {
            df,
            chunks: chunks.len(),
        }
    }

    /// Rare stems weigh more: matching "sso_required" means more than
    /// matching "sign".
    fn idf(&self, stem: &str) -> f32 {
        let df = self.df.get(stem).copied().unwrap_or(0);
        ((self.chunks + 1) as f32 / (df + 1) as f32).ln() + 1.0
    }

    /// The query's distinct stems, without the too common ones (all of them
    /// if that would leave none).
    pub fn terms(&self, query: &str) -> Vec<String> {
        let mut all: Vec<String> = Vec::new();
        for stem in stems(query) {
            if !all.contains(&stem) {
                all.push(stem);
            }
        }
        let limit = COMMON_SHARE * self.chunks as f32;
        let rare: Vec<String> = all
            .iter()
            .filter(|s| self.df.get(s.as_str()).copied().unwrap_or(0) as f32 <= limit)
            .cloned()
            .collect();
        if rare.is_empty() { all } else { rare }
    }

    pub fn score(&self, terms: &[String], hit: &Hit) -> Signals {
        let text: HashSet<String> = stems(&hit.chunk.text).into_iter().collect();
        let head: HashSet<String> =
            stems(&format!("{} {}", hit.chunk.source, hit.chunk.section))
                .into_iter()
                .collect();
        let total: f32 = terms.iter().map(|t| self.idf(t)).sum();
        let share = |set: &HashSet<String>| {
            if total <= 0.0 {
                return 0.0;
            }
            let found: f32 = terms
                .iter()
                .filter(|t| set.contains(t.as_str()))
                .map(|t| self.idf(t))
                .sum();
            found / total
        };
        let coverage = share(&text);
        let heading = share(&head);
        let cosine = (hit.score / COSINE_FULL).clamp(0.0, 1.0);
        Signals {
            coverage,
            heading,
            score: W_COSINE * cosine + W_COVERAGE * coverage + W_HEADING * heading,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Index, kb_documents, local};

    #[test]
    fn the_heuristic_prefers_the_chunk_that_names_what_is_asked() {
        let index = Index::build(kb_documents(), local());
        let heuristic = Heuristic::new(&index.chunks);

        // "the" is everywhere; "recovery" is rare.
        assert_eq!(heuristic.terms("the recovery"), vec!["recov".to_string()]);
        assert!(heuristic.idf("recov") > heuristic.idf("the"));
        // Only common words: kept, so the query still has terms.
        assert_eq!(heuristic.terms("the and").len(), 2);

        let query = "account locked after too many failed sign-in attempts ACCOUNT_LOCKED";
        let terms = heuristic.terms(query);
        let hits = index.search(query, 20);
        let mut scored: Vec<(Signals, &Hit)> =
            hits.iter().map(|h| (heuristic.score(&terms, h), h)).collect();
        for (s, _) in &scored {
            assert!((0.0..=1.0).contains(&s.coverage) && (0.0..=1.0).contains(&s.heading));
            assert!((0.0..=1.0).contains(&s.score));
        }
        scored.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
        let (best, hit) = scored[0];
        assert_eq!(hit.chunk.source, "kb/sign-in.md", "{:?}", hit.chunk.section);
        assert!(hit.chunk.section.contains("ACCOUNT_LOCKED"), "{:?}", hit.chunk.section);
        assert!(best.heading > 0.0, "{best:?}");
    }
}

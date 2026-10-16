//! The heuristic reranker: the second stage after the vector search. It
//! scores every candidate again from three signals, so a chunk that is only
//! vaguely similar to the question can't outrank one that names what it asks.

use std::collections::{HashMap, HashSet};

use crate::index::{Chunk, Hit, stems};

/// Weights of the three signals. Found by a grid search on the control set
/// (lesson 23): keyword coverage matters most, cosine and the heading
/// break ties.
pub const W_COSINE: f32 = 0.3;
pub const W_COVERAGE: f32 = 0.5;
pub const W_HEADING: f32 = 0.2;
/// The cosine that counts as a full match. With the local embedder even the
/// best chunk rarely scores above 0.5, so cosine is divided by this and capped
/// at 1 to put it on the same 0..1 scale as the other two signals.
pub const COSINE_FULL: f32 = 0.5;
/// A stem found in more than this share of all chunks ("lesson", "the",
/// "агент") says nothing about relevance, so it isn't counted.
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

    /// Rare stems weigh more: matching "linger" means more than matching
    /// "deploy".
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
    use crate::index::{Index, document, local};

    #[tokio::test]
    async fn the_heuristic_prefers_the_chunk_that_names_what_is_asked() {
        let deploy = "# Deployment\n\n## Linger\n\nRun `sudo loginctl enable-linger <user>` once, \
            so user services keep running after the SSH session closes.\n\n## Ports\n\n\
            Every lesson runs on port 4000 plus its number, and the service is restarted \
            after every deploy of the lesson.\n";
        let other = "# Lessons\n\nEvery lesson is a folder. The lesson runs as a service \
            and the session of the lesson keeps its history.\n";
        let docs = vec![
            document("DEPLOYMENT.md", deploy).unwrap(),
            document("docs/lessons.md", other).unwrap(),
            document("docs/a.md", "# A\n\nThe lesson and the service.\n").unwrap(),
            document("docs/b.md", "# B\n\nThe lesson, the deploy and the port.\n").unwrap(),
            document("docs/c.md", "# C\n\nThe lesson again, and the folder.\n").unwrap(),
        ];
        let index = Index::build(docs, local()).await.unwrap();
        let heuristic = Heuristic::new(&index.chunks);

        // "the" and "lesson" are everywhere; "linger" is rare.
        let terms = heuristic.terms("the linger lesson");
        assert_eq!(terms, vec!["linge".to_string()]);
        assert!(heuristic.idf("linge") > heuristic.idf("the"));
        // Only common words: kept, so the query still has terms.
        assert_eq!(heuristic.terms("the lesson").len(), 2);

        let query = "enable linger so user services keep running after the SSH session closes";
        let terms = heuristic.terms(query);
        let hits = index.search(query, 20).await.unwrap();
        let mut scored: Vec<(Signals, &Hit)> =
            hits.iter().map(|h| (heuristic.score(&terms, h), h)).collect();
        for (s, _) in &scored {
            assert!((0.0..=1.0).contains(&s.coverage) && (0.0..=1.0).contains(&s.heading));
            assert!((0.0..=1.0).contains(&s.score));
        }
        scored.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
        let (_, hit) = scored[0];
        assert_eq!(hit.chunk.source, "DEPLOYMENT.md", "{:?}", hit.chunk.section);
        assert!(hit.chunk.text.contains("enable-linger"));
    }
}

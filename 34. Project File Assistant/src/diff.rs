//! Unified diffs (`git diff` style, from the `similar` crate) of one file or
//! of two snapshots of the project.

use std::collections::BTreeSet;

use serde::Serialize;
use similar::{ChangeTag, TextDiff};

use crate::workspace::Snapshot;

#[derive(Serialize, Clone, Debug)]
pub struct FileDiff {
    pub path: String,
    /// `added`, `modified`, `deleted` or `unchanged` (same content written
    /// again; never in `tree_diff`).
    pub status: &'static str,
    /// Lines added and removed.
    pub added: usize,
    pub removed: usize,
    /// The unified diff, empty when nothing changed.
    pub diff: String,
}

/// One file's diff; `None` is a file that doesn't exist on that side.
pub fn file_diff(path: &str, old: Option<&str>, new: Option<&str>) -> FileDiff {
    let (a, b) = (old.unwrap_or(""), new.unwrap_or(""));
    let status = match (old, new) {
        (None, Some(_)) => "added",
        (Some(_), None) => "deleted",
        _ if a == b => "unchanged",
        _ => "modified",
    };
    let diff = TextDiff::from_lines(a, b);
    let (mut added, mut removed) = (0, 0);
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }
    let old_name = old.map_or("/dev/null".to_string(), |_| format!("a/{path}"));
    let new_name = new.map_or("/dev/null".to_string(), |_| format!("b/{path}"));
    let text = if status == "unchanged" {
        String::new()
    } else if a == b {
        // An empty file created or deleted: no lines, but still a change.
        format!("--- {old_name}\n+++ {new_name}\n")
    } else {
        diff.unified_diff()
            .context_radius(3)
            .header(&old_name, &new_name)
            .to_string()
    };
    FileDiff {
        path: path.to_string(),
        status,
        added,
        removed,
        diff: text,
    }
}

/// Every file that differs between two snapshots, sorted by path.
pub fn tree_diff(before: &Snapshot, after: &Snapshot) -> Vec<FileDiff> {
    let paths: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    paths
        .into_iter()
        .filter(|p| before.get(*p) != after.get(*p))
        .map(|p| {
            file_diff(
                p,
                before.get(p).map(String::as_str),
                after.get(p).map(String::as_str),
            )
        })
        .collect()
}

/// All the diffs as one patch, ready for `git apply` / `patch -p1`.
pub fn patch(diffs: &[FileDiff]) -> String {
    diffs.iter().map(|d| d.diff.as_str()).collect()
}

/// `2 files changed: docs/api.md (modified, +4 -0), ...`.
pub fn summary(diffs: &[FileDiff]) -> String {
    if diffs.is_empty() {
        return "No changes.".to_string();
    }
    let files: Vec<String> = diffs
        .iter()
        .map(|d| format!("{} ({}, +{} -{})", d.path, d.status, d.added, d.removed))
        .collect();
    format!(
        "{} file{} changed: {}",
        diffs.len(),
        if diffs.len() == 1 { "" } else { "s" },
        files.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_modified_file_gets_a_unified_diff() {
        let d = file_diff("docs/api.md", Some("a\nb\nc\n"), Some("a\nB\nc\nd\n"));
        assert_eq!(d.status, "modified");
        assert_eq!((d.added, d.removed), (2, 1));
        assert!(d.diff.starts_with("--- a/docs/api.md\n+++ b/docs/api.md\n@@"), "{}", d.diff);
        assert!(d.diff.contains("\n-b\n+B\n") && d.diff.contains("\n+d\n"), "{}", d.diff);
    }

    #[test]
    fn added_unchanged_and_tree_diffs() {
        let d = file_diff("CHANGELOG.md", None, Some("# Changelog\n"));
        assert_eq!((d.status, d.added, d.removed), ("added", 1, 0));
        assert!(d.diff.starts_with("--- /dev/null\n+++ b/CHANGELOG.md\n"), "{}", d.diff);
        let same = file_diff("x.md", Some("same\n"), Some("same\n"));
        assert_eq!((same.status, same.diff.as_str()), ("unchanged", ""));
        let empty = file_diff("tests/__init__.py", None, Some(""));
        assert_eq!(empty.status, "added");
        assert_eq!(empty.diff, "--- /dev/null\n+++ b/tests/__init__.py\n");

        let before: Snapshot = [("a.md", "1\n"), ("b.md", "2\n"), ("c.md", "3\n")]
            .into_iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect();
        let mut after = before.clone();
        after.insert("b.md".into(), "two\n".into());
        after.remove("c.md");
        after.insert("d.md".into(), "4\n".into());
        let diffs = tree_diff(&before, &after);
        let shape: Vec<(&str, &str)> = diffs.iter().map(|d| (d.path.as_str(), d.status)).collect();
        assert_eq!(shape, [("b.md", "modified"), ("c.md", "deleted"), ("d.md", "added")]);
        assert_eq!(
            summary(&diffs),
            "3 files changed: b.md (modified, +1 -1), c.md (deleted, +0 -1), d.md (added, +1 -0)"
        );
        let p = patch(&diffs);
        assert!(p.contains("+++ b/b.md") && p.contains("+++ /dev/null") && p.contains("--- /dev/null"));
        assert_eq!(summary(&[]), "No changes.");
    }
}

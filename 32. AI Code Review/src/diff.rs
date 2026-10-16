//! `git diff` output as data: the changed files, their hunks, and the line
//! numbers of every line on both sides. The model sees the diff with the
//! new-side numbers in front of each line, and a finding may only point at a
//! line the diff actually shows.

use std::collections::BTreeSet;

use serde::Serialize;

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl FileStatus {
    pub fn letter(self) -> char {
        match self {
            FileStatus::Added => 'A',
            FileStatus::Modified => 'M',
            FileStatus::Deleted => 'D',
            FileStatus::Renamed => 'R',
        }
    }

    pub fn word(self) -> &'static str {
        match self {
            FileStatus::Added => "added",
            FileStatus::Modified => "modified",
            FileStatus::Deleted => "deleted",
            FileStatus::Renamed => "renamed",
        }
    }
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LineKind {
    Context,
    Added,
    Removed,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct DiffLine {
    pub kind: LineKind,
    /// The line's number in the old file (`None` for an added line).
    pub old: Option<u32>,
    /// The line's number in the new file (`None` for a removed line).
    pub new: Option<u32>,
    pub text: String,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Hunk {
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    /// What git prints after the second `@@`: usually the enclosing item.
    pub header: String,
    pub lines: Vec<DiffLine>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FileDiff {
    /// The path at the head (for a deleted file, the path it had).
    pub path: String,
    /// The path before a rename.
    pub old_path: Option<String>,
    pub status: FileStatus,
    pub binary: bool,
    pub additions: usize,
    pub deletions: usize,
    #[serde(skip)]
    pub hunks: Vec<Hunk>,
    /// Why the file's patch is not sent to the model (a lock file, a binary,
    /// a deleted file), or `None` when it is reviewed.
    pub skipped: Option<String>,
}

impl FileDiff {
    fn new(path: String) -> FileDiff {
        FileDiff {
            path,
            old_path: None,
            status: FileStatus::Modified,
            binary: false,
            additions: 0,
            deletions: 0,
            hunks: Vec::new(),
            skipped: None,
        }
    }

    /// Every new-side line number the diff shows (added and context lines):
    /// the lines a finding may point at.
    pub fn visible_lines(&self) -> BTreeSet<u32> {
        self.hunks
            .iter()
            .flat_map(|h| h.lines.iter().filter_map(|l| l.new))
            .collect()
    }

    /// The new-side line ranges the hunks cover, inclusive.
    pub fn new_ranges(&self) -> Vec<(u32, u32)> {
        self.hunks
            .iter()
            .filter(|h| h.new_lines > 0)
            .map(|h| (h.new_start, h.new_start + h.new_lines - 1))
            .collect()
    }

    /// The added and removed lines, for building search queries.
    pub fn changed_text(&self, hunk: &Hunk) -> String {
        hunk.lines
            .iter()
            .filter(|l| l.kind != LineKind::Context)
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// `"a/x y.rs"` (git quotes a path with `"`, `\` or control characters)
/// back to `a/x y.rs`.
fn unquote(path: &str) -> String {
    let Some(inner) = path.strip_prefix('"').and_then(|p| p.strip_suffix('"')) else {
        return path.to_string();
    };
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

/// A path from a `--- a/x` / `+++ b/x` line: without the prefix, and without
/// the tab git appends when the path has a space in it.
fn side_path(raw: &str, prefix: &str) -> Option<String> {
    let raw = unquote(raw.trim_end_matches('\t'));
    if raw == "/dev/null" {
        return None;
    }
    Some(raw.strip_prefix(prefix).unwrap_or(&raw).to_string())
}

/// The path from `diff --git a/P b/P`. With spaces in the path this line is
/// ambiguous, but when the two sides are the same path (anything but a
/// rename) it is exactly `a/` + P + ` b/` + P. The `---`/`+++` and
/// `rename` lines that follow give the exact paths anyway.
fn guess_path(rest: &str) -> String {
    let rest = rest.trim_end();
    if rest.len() > 5 && (rest.len() - 5) % 2 == 0 {
        let n = (rest.len() - 5) / 2;
        if rest.is_char_boundary(2 + n) && rest.is_char_boundary(n + 5) {
            let (a, b) = (&rest[2..2 + n], &rest[n + 5..]);
            if a == b && rest.starts_with("a/") && rest[2 + n..].starts_with(" b/") {
                return a.to_string();
            }
        }
    }
    rest.rsplit_once(" b/")
        .map(|(_, b)| b.to_string())
        .unwrap_or_else(|| rest.to_string())
}

/// `12,5` → `(12, 5)`; `12` → `(12, 1)`.
fn range(text: &str) -> Option<(u32, u32)> {
    let (start, len) = text.split_once(',').unwrap_or((text, "1"));
    Some((start.parse().ok()?, len.parse().ok()?))
}

/// `@@ -10,6 +10,8 @@ fn main() {` → a hunk with no lines yet.
fn hunk_header(line: &str) -> Option<Hunk> {
    let rest = line.strip_prefix("@@ -")?;
    let (ranges, header) = rest.split_once(" @@")?;
    let (old, new) = ranges.split_once(" +")?;
    let (old_start, old_lines) = range(old)?;
    let (new_start, new_lines) = range(new)?;
    Some(Hunk {
        old_start,
        old_lines,
        new_start,
        new_lines,
        header: header.trim().to_string(),
        lines: Vec::new(),
    })
}

/// Files whose patch says nothing a reviewer can use.
pub fn skip_reason(file: &FileDiff) -> Option<String> {
    let name = file.path.rsplit('/').next().unwrap_or(&file.path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    if file.binary {
        return Some("binary file".to_string());
    }
    if name.ends_with(".lock") || matches!(name, "package-lock.json" | "pnpm-lock.yaml") {
        return Some("lock file".to_string());
    }
    if matches!(
        ext.as_deref(),
        Some(
            "png" | "jpg" | "jpeg" | "gif" | "ico" | "webp" | "pdf" | "mp4" | "mov" | "woff"
                | "woff2" | "ttf" | "zip" | "gz"
        )
    ) {
        return Some("media or archive".to_string());
    }
    if file.path.starts_with("target/") || file.path.contains("/target/") {
        return Some("build output".to_string());
    }
    if name.ends_with(".min.js") || name.ends_with(".min.css") {
        return Some("minified file".to_string());
    }
    if file.status == FileStatus::Deleted {
        return Some("deleted file (listed, not reviewed)".to_string());
    }
    if file.hunks.is_empty() {
        return Some("no content changes".to_string());
    }
    None
}

fn finish(mut file: FileDiff) -> FileDiff {
    file.skipped = skip_reason(&file);
    file
}

/// Parses the output of `git diff` (unified format, `a/` and `b/` prefixes).
pub fn parse(text: &str) -> Vec<FileDiff> {
    let mut files = Vec::new();
    let mut current: Option<FileDiff> = None;
    // Inside a file, header lines come before the first `@@`; after it a
    // line starting with `---` is a removed line, not a header.
    let mut in_body = false;
    let mut old_side: Option<String> = None;
    let (mut old_no, mut new_no) = (0u32, 0u32);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            if let Some(file) = current.take() {
                files.push(finish(file));
            }
            current = Some(FileDiff::new(guess_path(rest)));
            in_body = false;
            old_side = None;
            continue;
        }
        let Some(file) = current.as_mut() else {
            continue;
        };
        if line.starts_with("@@ ") {
            if let Some(hunk) = hunk_header(line) {
                old_no = hunk.old_start;
                new_no = hunk.new_start;
                file.hunks.push(hunk);
                in_body = true;
            }
            continue;
        }
        if !in_body {
            if line.starts_with("new file mode") {
                file.status = FileStatus::Added;
            } else if line.starts_with("deleted file mode") {
                file.status = FileStatus::Deleted;
            } else if let Some(from) = line.strip_prefix("rename from ") {
                file.status = FileStatus::Renamed;
                file.old_path = Some(unquote(from));
            } else if let Some(to) = line.strip_prefix("rename to ") {
                file.path = unquote(to);
            } else if let Some(old) = line.strip_prefix("--- ") {
                old_side = side_path(old, "a/");
                if old_side.is_none() {
                    file.status = FileStatus::Added;
                }
            } else if let Some(new) = line.strip_prefix("+++ ") {
                match side_path(new, "b/") {
                    Some(path) => file.path = path,
                    None => {
                        file.status = FileStatus::Deleted;
                        if let Some(old) = &old_side {
                            file.path = old.clone();
                        }
                    }
                }
            } else if line.starts_with("Binary files ") || line.starts_with("GIT binary patch") {
                file.binary = true;
            }
            continue;
        }
        let Some(hunk) = file.hunks.last_mut() else {
            continue;
        };
        let (kind, text) = match line.chars().next() {
            Some('+') => (LineKind::Added, &line[1..]),
            Some('-') => (LineKind::Removed, &line[1..]),
            Some(' ') => (LineKind::Context, &line[1..]),
            // "\ No newline at end of file"
            Some('\\') => continue,
            None => (LineKind::Context, ""),
            Some(_) => continue,
        };
        let (old, new) = match kind {
            LineKind::Added => {
                file.additions += 1;
                new_no += 1;
                (None, Some(new_no - 1))
            }
            LineKind::Removed => {
                file.deletions += 1;
                old_no += 1;
                (Some(old_no - 1), None)
            }
            LineKind::Context => {
                old_no += 1;
                new_no += 1;
                (Some(old_no - 1), Some(new_no - 1))
            }
        };
        hunk.lines.push(DiffLine {
            kind,
            old,
            new,
            text: text.to_string(),
        });
    }
    if let Some(file) = current.take() {
        files.push(finish(file));
    }
    files
}

/// One file's patch for the model: a header, then each hunk with the
/// new-side line number in front of every added and context line. Removed
/// lines have no number (a finding can't point at them). Clipped at
/// `max_chars`; the second value says whether it was.
pub fn render(file: &FileDiff, max_chars: usize) -> (String, bool) {
    let mut out = format!(
        "### {} ({}{}, +{} -{})\n",
        file.path,
        file.status.word(),
        file.old_path
            .as_ref()
            .map(|p| format!(" from {p}"))
            .unwrap_or_default(),
        file.additions,
        file.deletions
    );
    let total: usize = file.hunks.iter().map(|h| h.lines.len()).sum();
    let mut shown = 0;
    'hunks: for hunk in &file.hunks {
        let mut head = format!(
            "@@ -{},{} +{},{} @@",
            hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines
        );
        if !hunk.header.is_empty() {
            head.push(' ');
            head.push_str(&hunk.header);
        }
        head.push('\n');
        if out.len() + head.len() > max_chars {
            break;
        }
        out.push_str(&head);
        for line in &hunk.lines {
            let text = match line.kind {
                LineKind::Added => format!("{:>5} + {}\n", line.new.unwrap_or(0), line.text),
                LineKind::Context => format!("{:>5}   {}\n", line.new.unwrap_or(0), line.text),
                LineKind::Removed => format!("{:>5} - {}\n", "", line.text),
            };
            if out.len() + text.len() > max_chars {
                break 'hunks;
            }
            out.push_str(&text);
            shown += 1;
        }
    }
    let clipped = shown < total;
    if clipped {
        out.push_str(&format!(
            "… ({} more diff lines not shown; read the file with read_file if you need them)\n",
            total - shown
        ));
    }
    (out, clipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
diff --git a/07. Lesson/src/main.rs b/07. Lesson/src/main.rs
index 1111111..2222222 100644
--- a/07. Lesson/src/main.rs\t
+++ b/07. Lesson/src/main.rs\t
@@ -10,4 +10,5 @@ fn main() {
 let a = 1;
-let b = 2;
+let b = 3;
+--- not a header
 let c = 4;
@@ -40 +41 @@ fn other() {
-old
+new
diff --git a/docs/new.md b/docs/new.md
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/docs/new.md
@@ -0,0 +1,2 @@
+# New
+text
\\ No newline at end of file
diff --git a/gone.rs b/gone.rs
deleted file mode 100644
index 4444444..0000000
--- a/gone.rs
+++ /dev/null
@@ -1 +0,0 @@
-fn gone() {}
diff --git a/old name.rs b/new name.rs
similarity index 100%
rename from old name.rs
rename to new name.rs
diff --git a/Cargo.lock b/Cargo.lock
index 5555555..6666666 100644
--- a/Cargo.lock
+++ b/Cargo.lock
@@ -1 +1 @@
-a
+b
diff --git a/logo.png b/logo.png
new file mode 100644
index 0000000..7777777
Binary files /dev/null and b/logo.png differ
";

    #[test]
    fn a_diff_is_parsed_into_files_hunks_and_line_numbers() {
        let files = parse(SAMPLE);
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "07. Lesson/src/main.rs",
                "docs/new.md",
                "gone.rs",
                "new name.rs",
                "Cargo.lock",
                "logo.png"
            ]
        );

        let main = &files[0];
        assert_eq!(main.status, FileStatus::Modified);
        assert_eq!((main.additions, main.deletions), (3, 2));
        assert_eq!(main.hunks.len(), 2);
        assert_eq!(main.hunks[0].header, "fn main() {");
        let lines = &main.hunks[0].lines;
        assert_eq!(lines[1].kind, LineKind::Removed);
        assert_eq!((lines[1].old, lines[1].new), (Some(11), None));
        assert_eq!((lines[2].kind, lines[2].new), (LineKind::Added, Some(11)));
        assert_eq!(lines[3].text, "--- not a header");
        assert_eq!((lines[4].old, lines[4].new), (Some(12), Some(13)));
        assert_eq!(main.hunks[1].lines[1].new, Some(41));
        assert_eq!(
            main.visible_lines().into_iter().collect::<Vec<_>>(),
            [10, 11, 12, 13, 41]
        );
        assert_eq!(main.new_ranges(), [(10, 14), (41, 41)]);
        assert!(main.skipped.is_none());

        assert_eq!(files[1].status, FileStatus::Added);
        assert_eq!(files[1].additions, 2);
        assert_eq!(files[2].status, FileStatus::Deleted);
        assert!(files[2].skipped.as_deref().unwrap().contains("deleted"));
        assert_eq!(files[3].status, FileStatus::Renamed);
        assert_eq!(files[3].old_path.as_deref(), Some("old name.rs"));
        assert!(files[3].skipped.is_some(), "a pure rename has no content");
        assert_eq!(files[4].skipped.as_deref(), Some("lock file"));
        assert!(files[5].binary);
        assert_eq!(files[5].skipped.as_deref(), Some("binary file"));
    }

    #[test]
    fn the_rendered_patch_numbers_new_lines_and_is_clipped() {
        let files = parse(SAMPLE);
        let (text, clipped) = render(&files[0], 10_000);
        assert!(!clipped);
        assert!(text.starts_with("### 07. Lesson/src/main.rs (modified, +3 -2)"));
        assert!(text.contains("   11 + let b = 3;"), "{text}");
        assert!(text.contains("      - let b = 2;"), "{text}");
        assert!(text.contains("   10   let a = 1;"), "{text}");
        let (short, clipped) = render(&files[0], 120);
        assert!(clipped);
        assert!(short.contains("more diff lines not shown"), "{short}");
        assert_eq!(guess_path("a/x y/z.rs b/x y/z.rs"), "x y/z.rs");
        assert_eq!(unquote("\"a\\\"b\""), "a\"b");
    }
}

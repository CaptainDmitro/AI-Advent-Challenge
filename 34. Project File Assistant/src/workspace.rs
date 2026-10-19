//! The project the assistant works on: a folder of text files. By default it
//! is the demo project from `sample/` (compiled in, copied to `workspace/` on
//! the first start and put back by reset), so every run starts from the same
//! files. `PROJECT_DIR` points it at any other folder instead.
//!
//! Every path from the outside goes through `normalize`/`file_key`: relative,
//! `/`-separated, no `..`, nothing hidden, a text extension, and never through
//! a symlink out of the root. Files are written with write-then-rename.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Serialize;

/// The demo project: Notely API, a small FastAPI service whose docs lag
/// behind its code and which breaks a few of its own `PROJECT_RULES.md`.
pub const SAMPLE: &[(&str, &str)] = &[
    ("CHANGELOG.md", include_str!("../sample/CHANGELOG.md")),
    ("PROJECT_RULES.md", include_str!("../sample/PROJECT_RULES.md")),
    ("README.md", include_str!("../sample/README.md")),
    ("app/auth.py", include_str!("../sample/app/auth.py")),
    ("app/config.py", include_str!("../sample/app/config.py")),
    ("app/main.py", include_str!("../sample/app/main.py")),
    ("app/routes/notes.py", include_str!("../sample/app/routes/notes.py")),
    ("app/routes/users.py", include_str!("../sample/app/routes/users.py")),
    ("app/storage.py", include_str!("../sample/app/storage.py")),
    (
        "docs/adr/0001-use-fastapi.md",
        include_str!("../sample/docs/adr/0001-use-fastapi.md"),
    ),
    ("docs/api.md", include_str!("../sample/docs/api.md")),
    ("requirements.txt", include_str!("../sample/requirements.txt")),
    ("tests/test_notes.py", include_str!("../sample/tests/test_notes.py")),
];

/// Extensions of the files the assistant may read and write.
pub const TEXT_EXTENSIONS: &[&str] = &[
    "md", "txt", "py", "rs", "toml", "json", "yaml", "yml", "js", "ts", "html", "css", "cfg",
    "ini", "sh", "sql", "csv",
];
/// Text files without one of those extensions.
const TEXT_NAMES: &[&str] = &["Dockerfile", "Makefile", "LICENSE"];
/// Folders never listed or entered (hidden ones neither).
const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__", "venv"];
pub const MAX_FILE_BYTES: usize = 256 * 1024;
const MAX_FILES: usize = 1_000;

/// Every text file of the project: path → content.
pub type Snapshot = BTreeMap<String, String>;

#[derive(Serialize, Clone, Debug)]
pub struct FileInfo {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
}

pub fn sample() -> Snapshot {
    SAMPLE
        .iter()
        .map(|(path, text)| (path.to_string(), text.to_string()))
        .collect()
}

/// A relative path inside the project: `./docs/api.md` → `docs/api.md`.
pub fn normalize(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    let trimmed = trimmed.strip_prefix("./").unwrap_or(trimmed);
    if trimmed.is_empty() {
        return Err("The path is empty.".to_string());
    }
    if trimmed.contains('\\') || trimmed.contains('\0') {
        return Err(format!("`{trimmed}`: use `/` as the separator."));
    }
    if trimmed.starts_with('/') || Path::new(trimmed).is_absolute() {
        return Err(format!(
            "`{trimmed}` is absolute; paths are relative to the project root."
        ));
    }
    let parts: Vec<&str> = trimmed.trim_end_matches('/').split('/').collect();
    for part in &parts {
        if part.is_empty() {
            return Err(format!("`{trimmed}` has an empty path component."));
        }
        if *part == ".." || *part == "." {
            return Err(format!("`{trimmed}`: `..` and `.` are not allowed."));
        }
        if part.starts_with('.') {
            return Err(format!("`{trimmed}`: hidden files and folders are off limits."));
        }
        if SKIP_DIRS.contains(part) {
            return Err(format!("`{trimmed}`: `{part}` is off limits."));
        }
    }
    Ok(parts.join("/"))
}

fn is_text_file(name: &str) -> bool {
    TEXT_NAMES.contains(&name)
        || name.rsplit_once('.').is_some_and(|(stem, ext)| {
            !stem.is_empty() && TEXT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
        })
}

/// A file path the assistant may touch: normalized, with a text extension.
pub fn file_key(path: &str) -> Result<String, String> {
    let key = normalize(path)?;
    let name = key.rsplit('/').next().unwrap_or(&key);
    if !is_text_file(name) {
        return Err(format!(
            "`{key}` is not a text file the assistant may touch (allowed: .{}).",
            TEXT_EXTENSIONS.join(", .")
        ));
    }
    Ok(key)
}

fn relative(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    Some(parts.join("/"))
}

/// Text files under `dir`, depth first. Symlinks are never followed.
fn walk(root: &Path, dir: &Path, out: &mut Vec<FileInfo>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if out.len() >= MAX_FILES {
            return;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            walk(root, &path, out);
        } else if kind.is_file() && is_text_file(&name) {
            let Some(rel) = relative(root, &path) else {
                continue;
            };
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.len() as usize > MAX_FILE_BYTES {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                out.push(FileInfo {
                    path: rel,
                    bytes: text.len(),
                    lines: text.lines().count(),
                });
            }
        }
    }
}

#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    /// What `get_diff` and `/api/diff` compare against: the sample for the
    /// demo project, the files as they were at start for `PROJECT_DIR`.
    baseline: Snapshot,
    demo: bool,
    /// One write at a time.
    writes: Mutex<()>,
}

impl Workspace {
    /// The demo project in `root`, created from `sample/` when the folder is
    /// missing or empty.
    pub fn open_demo(root: PathBuf) -> Result<Workspace, String> {
        let empty = std::fs::read_dir(&root).map_or(true, |mut d| d.next().is_none());
        let ws = Workspace {
            root,
            baseline: sample(),
            demo: true,
            writes: Mutex::new(()),
        };
        if empty {
            std::fs::create_dir_all(&ws.root)
                .map_err(|e| format!("Can't create {}: {e}", ws.root.display()))?;
            for (path, text) in SAMPLE {
                ws.write(path, text)?;
            }
        }
        Ok(ws)
    }

    /// Any existing folder. Its files as they are now become the baseline.
    pub fn open_dir(root: PathBuf) -> Result<Workspace, String> {
        if !root.is_dir() {
            return Err(format!("{} is not a folder.", root.display()));
        }
        let mut ws = Workspace {
            root,
            baseline: Snapshot::new(),
            demo: false,
            writes: Mutex::new(()),
        };
        ws.baseline = ws.snapshot()?;
        Ok(ws)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn is_demo(&self) -> bool {
        self.demo
    }

    pub fn baseline(&self) -> &Snapshot {
        &self.baseline
    }

    fn canonical_root(&self) -> Result<PathBuf, String> {
        self.root
            .canonicalize()
            .map_err(|e| format!("Can't open {}: {e}", self.root.display()))
    }

    /// Text files, sorted by path, optionally only under the folder `dir`.
    pub fn list(&self, dir: Option<&str>) -> Result<Vec<FileInfo>, String> {
        let start = match dir {
            Some(dir) => {
                let key = normalize(dir)?;
                let path = self.root.join(&key);
                if !path.is_dir() {
                    return Err(format!("There is no folder `{key}`."));
                }
                path
            }
            None => self.root.clone(),
        };
        let mut files = Vec::new();
        walk(&self.root, &start, &mut files);
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(files)
    }

    /// A file's content, by its normalized path.
    pub fn read(&self, path: &str) -> Result<(String, String), String> {
        let key = file_key(path)?;
        let full = self.root.join(&key);
        match std::fs::symlink_metadata(&full) {
            Err(_) => return Err(format!("There is no file `{key}`.")),
            Ok(meta) if !meta.is_file() => return Err(format!("`{key}` is not a regular file.")),
            Ok(meta) if meta.len() as usize > MAX_FILE_BYTES => {
                return Err(format!(
                    "`{key}` is larger than {} KB.",
                    MAX_FILE_BYTES / 1024
                ));
            }
            Ok(_) => {}
        }
        let canonical = full.canonicalize().map_err(|e| format!("`{key}`: {e}"))?;
        if !canonical.starts_with(self.canonical_root()?) {
            return Err(format!("`{key}` is outside the project."));
        }
        let text = std::fs::read_to_string(&full)
            .map_err(|_| format!("`{key}` is not a UTF-8 text file."))?;
        Ok((key, text))
    }

    /// Creates or replaces a file (write-then-rename) and returns its
    /// normalized path and the content it had before, if it existed.
    pub fn write(&self, path: &str, text: &str) -> Result<(String, Option<String>), String> {
        let key = file_key(path)?;
        if text.len() > MAX_FILE_BYTES {
            return Err(format!(
                "The content is larger than {} KB.",
                MAX_FILE_BYTES / 1024
            ));
        }
        let _guard = self.writes.lock().unwrap();
        let full = self.root.join(&key);
        let before = if full.exists() {
            Some(self.read(&key)?.1)
        } else {
            None
        };
        let dir = full.parent().unwrap_or(&self.root).to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| format!("Can't create the folder: {e}"))?;
        let canonical = dir.canonicalize().map_err(|e| format!("`{key}`: {e}"))?;
        if !canonical.starts_with(self.canonical_root()?) {
            return Err(format!("`{key}` is outside the project."));
        }
        let name = key.rsplit('/').next().unwrap_or(&key);
        let tmp = dir.join(format!(".{name}.tmp"));
        std::fs::write(&tmp, text).map_err(|e| format!("Can't write `{key}`: {e}"))?;
        std::fs::rename(&tmp, &full).map_err(|e| format!("Can't replace `{key}`: {e}"))?;
        Ok((key, before))
    }

    fn remove(&self, key: &str) -> Result<(), String> {
        let key = file_key(key)?;
        let _guard = self.writes.lock().unwrap();
        std::fs::remove_file(self.root.join(&key)).map_err(|e| format!("Can't remove `{key}`: {e}"))
    }

    /// Every text file's content.
    pub fn snapshot(&self) -> Result<Snapshot, String> {
        let mut snapshot = Snapshot::new();
        for file in self.list(None)? {
            if let Ok((key, text)) = self.read(&file.path) {
                snapshot.insert(key, text);
            }
        }
        Ok(snapshot)
    }

    /// Makes the project look like `target` again: changed files are written
    /// back and files that `target` doesn't have are removed. Returns the
    /// paths it touched.
    pub fn restore(&self, target: &Snapshot) -> Result<Vec<String>, String> {
        let current = self.snapshot()?;
        let mut touched = Vec::new();
        for (path, text) in target {
            if current.get(path) != Some(text) {
                self.write(path, text)?;
                touched.push(path.clone());
            }
        }
        for path in current.keys() {
            if !target.contains_key(path) {
                self.remove(path)?;
                touched.push(path.clone());
            }
        }
        touched.sort();
        Ok(touched)
    }

    /// Back to `sample/`, for the next demo run. Only the demo project can
    /// be reset: a real folder is never rolled back wholesale.
    pub fn reset(&self) -> Result<Vec<String>, String> {
        if !self.demo {
            return Err(
                "Reset only works on the demo project; PROJECT_DIR is never rolled back."
                    .to_string(),
            );
        }
        self.restore(&self.baseline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    pub fn temp_root(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lesson34-ws-{}-{name}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn paths_are_normalized_and_checked() {
        assert_eq!(normalize("./docs/api.md").unwrap(), "docs/api.md");
        assert_eq!(normalize(" app/routes/ ").unwrap(), "app/routes");
        assert_eq!(file_key("app/routes/notes.py").unwrap(), "app/routes/notes.py");
        assert_eq!(file_key("Dockerfile").unwrap(), "Dockerfile");
        for bad in [
            "",
            "../secrets.md",
            "docs/../../x.md",
            "/etc/passwd.txt",
            ".env",
            "app/.hidden/x.py",
            ".git/config.toml",
            "app\\main.py",
            "a//b.md",
            "node_modules/x.js",
        ] {
            assert!(file_key(bad).is_err(), "{bad:?} should be refused");
        }
        for not_text in ["app/run.exe", "image.png", "noext", ".md"] {
            assert!(file_key(not_text).is_err(), "{not_text:?} is not a text file");
        }
    }

    #[test]
    fn the_demo_project_is_created_and_reset() {
        let root = temp_root("demo");
        let ws = Workspace::open_demo(root.clone()).unwrap();
        assert!(ws.is_demo());
        let files = ws.list(None).unwrap();
        assert_eq!(files.len(), SAMPLE.len());
        assert_eq!(ws.snapshot().unwrap(), sample());
        assert_eq!(ws.list(Some("app/routes")).unwrap().len(), 2);
        assert!(ws.list(Some("nope")).is_err());

        let (key, before) = ws.write("./docs/api.md", "# changed\n").unwrap();
        assert_eq!(key, "docs/api.md");
        assert!(before.unwrap().contains("### PUT /notes/{note_id}"));
        let (_, before) = ws.write("docs/adr/0002-new.md", "# ADR 0002\n").unwrap();
        assert!(before.is_none());
        assert_eq!(ws.list(None).unwrap().len(), SAMPLE.len() + 1);
        // The temp file of write-then-rename is never left behind or listed.
        assert!(!root.join("docs/.api.md.tmp").exists());

        // Reopening keeps the changes; reset puts the sample back.
        let reopened = Workspace::open_demo(root.clone()).unwrap();
        assert_eq!(reopened.read("docs/api.md").unwrap().1, "# changed\n");
        let touched = reopened.reset().unwrap();
        assert_eq!(touched, ["docs/adr/0002-new.md", "docs/api.md"]);
        assert_eq!(reopened.snapshot().unwrap(), sample());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_real_folder_keeps_its_baseline_and_is_never_reset() {
        let root = temp_root("dir");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn a() {}\n").unwrap();
        std::fs::write(root.join("logo.png"), [0u8, 1, 2]).unwrap();
        let ws = Workspace::open_dir(root.clone()).unwrap();
        assert!(!ws.is_demo());
        assert_eq!(ws.baseline().keys().collect::<Vec<_>>(), ["src/lib.rs"]);
        assert!(ws.read("logo.png").is_err());
        assert!(ws.reset().is_err());

        // A snapshot taken before a change can still be restored.
        let before = ws.snapshot().unwrap();
        ws.write("src/lib.rs", "pub fn b() {}\n").unwrap();
        ws.write("NOTES.md", "x\n").unwrap();
        assert_eq!(ws.restore(&before).unwrap(), ["NOTES.md", "src/lib.rs"]);
        assert_eq!(ws.snapshot().unwrap(), before);
        assert!(root.join("logo.png").exists(), "non-text files are left alone");
        assert!(Workspace::open_dir(root.join("missing")).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_out_of_the_project_are_refused() {
        let root = temp_root("symlink");
        let outside = temp_root("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.md"), "secret\n").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.md"), root.join("link.md")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("out")).unwrap();
        let ws = Workspace::open_dir(root.clone()).unwrap();
        assert!(ws.read("link.md").is_err());
        assert!(ws.read("out/secret.md").is_err());
        assert!(ws.write("out/new.md", "x").is_err());
        assert!(!outside.join("new.md").exists());
        assert!(ws.list(None).unwrap().is_empty(), "symlinks are not listed");
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&outside).unwrap();
    }

    /// The demo scenarios depend on these: if the sample changes, the demo
    /// (and the README) must change with it.
    #[test]
    fn the_sample_has_the_gaps_the_demo_is_about() {
        let s = sample();
        let notes = &s["app/routes/notes.py"];
        assert!(notes.contains("def delete_note(note_id: str):"), "delete has no require_auth");
        assert!(notes.contains("@router.post(\"/{note_id}/share\")"));
        let api = &s["docs/api.md"];
        assert!(!api.contains("DELETE /notes") && !api.contains("/share"), "docs lag behind");
        assert!(s["app/config.py"].contains("SECRET_KEY = \"dev-secret-123\""));
        assert!(s["app/routes/users.py"].contains("== legacy_hash(data.password)"));
        assert!(s["app/storage.py"].contains("print("));
        assert!(!s.contains_key("tests/test_users.py"));
        assert!(!s["CHANGELOG.md"].contains("## Unreleased"));
        assert_eq!(s["PROJECT_RULES.md"].matches("\n1. ").count(), 1);
    }
}

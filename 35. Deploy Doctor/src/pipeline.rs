//! What a CI/CD run is about: its jobs grouped into subjects (a lesson, or
//! the pipeline itself), and job logs cut down to what the model needs.

use serde::Serialize;

use crate::github::Job;
use crate::vds::Lesson;

pub const PIPELINE: &str = "pipeline";
const MAX_HITS: usize = 40;
const CONTEXT: usize = 2;
const NO_MATCH_TAIL: usize = 40;

/// One thing to diagnose: a lesson whose job failed or that was deployed,
/// or the pipeline when a job that isn't a lesson's (`discover`) failed.
#[derive(Serialize, Clone, Debug)]
pub struct Subject {
    /// `lesson-35` or `pipeline`: the issue's marker.
    pub key: String,
    pub label: String,
    pub folder: Option<String>,
    pub lesson: Option<Lesson>,
    pub jobs: Vec<Job>,
    pub deployed: bool,
    pub failed: bool,
}

impl Subject {
    fn new(key: String, folder: Option<&str>, lesson: Option<Lesson>) -> Subject {
        let label = match (lesson, folder) {
            (Some(lesson), Some(folder)) => format!("lesson {:02} (`{folder}`)", lesson.number()),
            _ => "the pipeline".to_string(),
        };
        Subject {
            key,
            label,
            folder: folder.map(str::to_string),
            lesson,
            jobs: Vec::new(),
            deployed: false,
            failed: false,
        }
    }

    pub fn failed_jobs(&self) -> impl Iterator<Item = &Job> {
        self.jobs.iter().filter(|job| failed(job))
    }
}

pub fn failed(job: &Job) -> bool {
    matches!(
        job.conclusion.as_deref(),
        Some("failure" | "cancelled" | "timed_out" | "startup_failure")
    )
}

pub fn step_failed(conclusion: Option<&str>) -> bool {
    matches!(conclusion, Some("failure" | "cancelled" | "timed_out"))
}

fn succeeded(job: &Job) -> bool {
    job.conclusion.as_deref() == Some("success")
}

/// `deploy (35. Deploy Doctor)` → (`deploy`, `35. Deploy Doctor`): matrix
/// jobs carry the lesson folder in parentheses.
pub fn split_job_name(name: &str) -> (&str, Option<&str>) {
    if let Some((kind, rest)) = name.split_once(" (")
        && let Some(folder) = rest.strip_suffix(')')
    {
        return (kind.trim(), Some(folder));
    }
    (name.trim(), None)
}

pub fn lesson_of(folder: &str) -> Option<Lesson> {
    let digits: String = folder.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok().and_then(Lesson::new)
}

pub fn lesson_key(lesson: Lesson) -> String {
    format!("lesson-{:02}", lesson.number())
}

fn group<'a>(
    groups: &'a mut Vec<Subject>,
    key: String,
    folder: Option<&str>,
    lesson: Option<Lesson>,
) -> &'a mut Subject {
    let at = match groups.iter().position(|s| s.key == key) {
        Some(at) => at,
        None => {
            groups.push(Subject::new(key, folder, lesson));
            groups.len() - 1
        }
    };
    &mut groups[at]
}

/// The subjects to diagnose, and the keys whose jobs all succeeded in this
/// run (their open issues can be closed without asking the model).
pub fn subjects(jobs: &[Job]) -> (Vec<Subject>, Vec<String>) {
    let mut groups: Vec<Subject> = Vec::new();
    let mut discover_ok = false;
    for job in jobs {
        let (kind, folder) = split_job_name(&job.name);
        let Some(folder) = folder else {
            if failed(job) {
                let subject = group(&mut groups, PIPELINE.to_string(), None, None);
                subject.jobs.push(job.clone());
                subject.failed = true;
            } else if kind == "discover" && succeeded(job) {
                discover_ok = true;
            }
            continue;
        };
        let Some(lesson) = lesson_of(folder) else {
            continue;
        };
        let subject = group(&mut groups, lesson_key(lesson), Some(folder), Some(lesson));
        subject.jobs.push(job.clone());
        if kind == "deploy" && job.conclusion.as_deref() != Some("skipped") {
            subject.deployed = true;
        }
        if failed(job) {
            subject.failed = true;
        }
    }
    let mut subjects = Vec::new();
    let mut green = Vec::new();
    for subject in groups {
        if subject.failed || subject.deployed {
            subjects.push(subject);
        } else if subject.jobs.iter().all(succeeded) {
            green.push(subject.key);
        }
    }
    if discover_ok && !subjects.iter().any(|s| s.key == PIPELINE) {
        green.push(PIPELINE.to_string());
    }
    (subjects, green)
}

// ---------------------------------------------------------------------------
// Job logs.
// ---------------------------------------------------------------------------

/// A job log's lines without GitHub's timestamps
/// (`2026-10-19T07:40:41.1234567Z `) or the byte-order mark.
pub fn log_lines(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|line| strip_timestamp(line.trim_start_matches('\u{feff}')).to_string())
        .collect()
}

fn strip_timestamp(line: &str) -> &str {
    let bytes = line.as_bytes();
    if bytes.len() > 20
        && bytes[4] == b'-'
        && bytes[10] == b'T'
        && let Some((stamp, rest)) = line.split_once(' ')
        && stamp.ends_with('Z')
    {
        return rest;
    }
    line
}

fn numbered(lines: &[String], from: usize, to: usize) -> String {
    lines[from..to]
        .iter()
        .enumerate()
        .map(|(k, line)| format!("{:>5}: {line}\n", from + k + 1))
        .collect()
}

/// With `grep`: the matching lines (case-insensitive) with 2 lines of
/// context, gaps marked `…`. Without it: the last `tail` lines.
pub fn excerpt(lines: &[String], grep: Option<&str>, tail: usize) -> String {
    if lines.is_empty() {
        return "The log is empty.".to_string();
    }
    let total = lines.len();
    if let Some(pattern) = grep.map(str::trim).filter(|p| !p.is_empty()) {
        let needle = pattern.to_lowercase();
        let hits: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.to_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect();
        if hits.is_empty() {
            let from = total.saturating_sub(NO_MATCH_TAIL);
            return format!(
                "No line matches `{pattern}` ({total} lines). The last {} lines:\n{}",
                total - from,
                numbered(lines, from, total)
            );
        }
        let mut keep = vec![false; total];
        for &hit in hits.iter().take(MAX_HITS) {
            let end = (hit + CONTEXT).min(total - 1);
            keep[hit.saturating_sub(CONTEXT)..=end].fill(true);
        }
        let shown = if hits.len() > MAX_HITS {
            format!(" (the first {MAX_HITS} shown)")
        } else {
            String::new()
        };
        let mut out = format!(
            "{} of {total} lines match `{pattern}`{shown}:\n",
            hits.len()
        );
        let mut last: Option<usize> = None;
        for (i, line) in lines.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            if let Some(previous) = last
                && i > previous + 1
            {
                out.push_str("    …\n");
            }
            out.push_str(&format!("{:>5}: {line}\n", i + 1));
            last = Some(i);
        }
        return out;
    }
    let from = total.saturating_sub(tail);
    format!(
        "Lines {}-{total} of {total}:\n{}",
        from + 1,
        numbered(lines, from, total)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::github::Step;

    pub fn job(id: u64, name: &str, conclusion: &str) -> Job {
        Job {
            id,
            name: name.to_string(),
            status: "completed".to_string(),
            conclusion: Some(conclusion.to_string()),
            html_url: None,
            steps: vec![Step {
                name: "Set up job".to_string(),
                status: "completed".to_string(),
                conclusion: Some("success".to_string()),
                number: 1,
            }],
        }
    }

    #[test]
    fn job_names_carry_the_lesson_folder() {
        assert_eq!(
            split_job_name("deploy (35. Deploy Doctor)"),
            ("deploy", Some("35. Deploy Doctor"))
        );
        assert_eq!(
            split_job_name("build-test (01. Rust LLM Chat CLI)"),
            ("build-test", Some("01. Rust LLM Chat CLI"))
        );
        assert_eq!(split_job_name("discover"), ("discover", None));
        assert_eq!(lesson_of("04. Model Version Comparison").unwrap().number(), 4);
        assert_eq!(lesson_of("docs"), None);
    }

    #[test]
    fn subjects_are_failures_and_deploys() {
        let jobs = vec![
            job(1, "discover", "success"),
            job(2, "build-test (33. Support Assistant)", "failure"),
            job(3, "build-test (34. Project File Assistant)", "success"),
            job(4, "build-test (35. Deploy Doctor)", "success"),
            job(5, "deploy (35. Deploy Doctor)", "success"),
            // A matrix that never expanded: not a lesson, not a failure.
            job(6, "deploy", "skipped"),
        ];
        let (subjects, green) = subjects(&jobs);
        let keys: Vec<&str> = subjects.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, ["lesson-33", "lesson-35"]);
        assert!(subjects[0].failed && !subjects[0].deployed);
        assert!(!subjects[1].failed && subjects[1].deployed);
        assert_eq!(subjects[1].jobs.len(), 2);
        assert_eq!(subjects[1].label, "lesson 35 (`35. Deploy Doctor`)");
        assert_eq!(green, ["lesson-34", "pipeline"]);
    }

    #[test]
    fn a_failed_discover_is_the_pipelines_problem() {
        let jobs = vec![job(1, "discover", "failure"), job(2, "build-test", "skipped")];
        let (subjects, green) = subjects(&jobs);
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0].key, PIPELINE);
        assert_eq!(subjects[0].label, "the pipeline");
        assert!(subjects[0].failed);
        assert!(green.is_empty());
    }

    #[test]
    fn timestamps_and_the_bom_are_removed() {
        let raw = "\u{feff}2026-10-19T07:40:41.1234567Z ##[group]Run cargo build\n\
                   2026-10-19T07:40:42.0000000Z error[E0425]: cannot find value `x`\n\
                   plain line";
        assert_eq!(
            log_lines(raw),
            ["##[group]Run cargo build", "error[E0425]: cannot find value `x`", "plain line"]
        );
    }

    #[test]
    fn excerpts_grep_with_context_or_tail() {
        let lines: Vec<String> = (1..=30).map(|i| format!("line {i}")).collect();
        let mut with_error = lines.clone();
        with_error[4] = "error: first".to_string();
        with_error[20] = "ERROR: second".to_string();
        let text = excerpt(&with_error, Some("error"), 100);
        assert!(text.starts_with("2 of 30 lines match `error`:"), "{text}");
        assert!(text.contains("    3: line 3\n"));
        assert!(text.contains("    5: error: first\n"));
        assert!(text.contains("    7: line 7\n    …\n   19: line 19\n"));
        assert!(text.contains("   21: ERROR: second\n"));
        assert!(!text.contains("line 8\n"));

        let text = excerpt(&lines, Some("panicked"), 100);
        assert!(text.starts_with("No line matches `panicked` (30 lines). The last 30 lines:"));

        let text = excerpt(&lines, None, 5);
        assert!(text.starts_with("Lines 26-30 of 30:\n   26: line 26\n"));
        assert!(!text.contains("line 25\n"));
        assert_eq!(excerpt(&[], None, 5), "The log is empty.");
    }
}

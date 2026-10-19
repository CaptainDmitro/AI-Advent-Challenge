//! Markdown for the GitHub issues and the Actions job summary.

use serde::Serialize;

use crate::agent::Diagnosis;
use crate::clip;
use crate::github::Run;

/// Every issue the doctor opens starts with `<!-- deploy-doctor:<key> -->`,
/// which is how the next run finds it again.
pub const MARKER_PREFIX: &str = "<!-- deploy-doctor:";
const TITLE_CAUSE_CHARS: usize = 90;
const CALL_RESULT_CHARS: usize = 140;

pub fn marker(key: &str) -> String {
    format!("{MARKER_PREFIX}{key} -->")
}

/// Text from the model or a log, made safe for Markdown: no HTML comments
/// (a fake marker) and no code fences that would swallow the rest.
pub fn sanitize(text: &str) -> String {
    text.replace("<!--", "&lt;!--").replace("```", "'''")
}

pub fn short_sha(sha: &str) -> &str {
    &sha[..sha.len().min(7)]
}

fn badge(status: &str) -> &'static str {
    match status {
        "healthy" => "🟢 Healthy",
        "broken" => "🔴 Broken",
        "known_issue" => "🟡 Known issue",
        _ => "⚪ Not diagnosed",
    }
}

/// What happened to a subject's issue.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct IssueAction {
    pub key: String,
    /// `created`, `commented`, `closed`, `unchanged` or `none`.
    pub action: String,
    pub number: Option<u64>,
    pub url: Option<String>,
    pub error: Option<String>,
}

impl IssueAction {
    pub fn new(key: &str, action: &str, number: Option<u64>, url: Option<String>) -> IssueAction {
        IssueAction {
            key: key.to_string(),
            action: action.to_string(),
            number,
            url,
            error: None,
        }
    }

    pub fn failed(key: &str, error: String) -> IssueAction {
        IssueAction {
            key: key.to_string(),
            action: "failed".to_string(),
            number: None,
            url: None,
            error: Some(error),
        }
    }
}

fn first_sentence(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    let sentence = match line.find(". ") {
        Some(at) => &line[..at],
        None => line.trim_end_matches('.'),
    };
    clip(sentence, TITLE_CAUSE_CHARS)
}

pub fn issue_title(d: &Diagnosis) -> String {
    let who = match d.subject.lesson {
        Some(lesson) => format!("lesson {:02}", lesson.number()),
        None => "pipeline".to_string(),
    };
    let cause = first_sentence(&d.verdict.root_cause);
    if d.verdict.status == "known_issue" {
        format!("Deploy Doctor: {who} — known issue: {cause}")
    } else {
        format!("Deploy Doctor: {who} — {cause}")
    }
}

/// One subject's diagnosis: the issue body, a comment, or a summary section.
pub fn diagnosis_markdown(run: &Run, d: &Diagnosis, model: &str) -> String {
    let v = &d.verdict;
    let mut s = marker(&d.subject.key);
    s.push('\n');
    s.push_str(&format!("### {} — {}\n\n", badge(&v.status), d.subject.label));
    s.push_str(&format!(
        "**Run** [{}]({}) · `{}` · {} · stage `{}`\n\n",
        run.id,
        run.html_url,
        short_sha(&run.head_sha),
        sanitize(&run.display_title),
        v.failed_stage
    ));
    if !v.root_cause.is_empty() {
        s.push_str(&format!("**Root cause.** {}\n\n", sanitize(&v.root_cause)));
    }
    if !v.evidence.is_empty() {
        s.push_str("**Evidence**\n\n");
        for e in &v.evidence {
            s.push_str(&format!("- {}\n", sanitize(e)));
        }
        s.push('\n');
    }
    if !v.suggested_fix.is_empty() {
        s.push_str(&format!("**Suggested fix.** {}\n\n", sanitize(&v.suggested_fix)));
    }
    if !v.commands_to_run.is_empty() {
        s.push_str("```bash\n");
        for c in &v.commands_to_run {
            s.push_str(&sanitize(c));
            s.push('\n');
        }
        s.push_str("```\n\n");
    }
    if let Some(doc) = &v.doc_reference {
        s.push_str(&format!("**Documented in** {}\n\n", sanitize(doc)));
    }
    for note in &v.notes {
        s.push_str(&format!("> ⚠️ {}\n\n", sanitize(note)));
    }
    if !d.calls.is_empty() {
        s.push_str(&format!(
            "<details><summary>How this was diagnosed: {} tool calls in {} rounds</summary>\n\n",
            d.calls.len(),
            d.rounds
        ));
        for c in &d.calls {
            let args = if c.arguments.as_object().is_some_and(|o| o.is_empty()) {
                String::new()
            } else {
                format!(" `{}`", c.arguments.to_string().replace('`', "'"))
            };
            let result = clip(&c.result.split_whitespace().collect::<Vec<_>>().join(" "), CALL_RESULT_CHARS);
            s.push_str(&format!(
                "{}. `{}`{args} → {}{}\n",
                c.step,
                c.tool,
                sanitize(&result),
                if c.is_error { " ⚠️" } else { "" }
            ));
        }
        s.push_str("\n</details>\n\n");
    }
    let by = if v.ai {
        format!(" with `{model}`")
    } else {
        String::new()
    };
    s.push_str(&format!(
        "<sub>Read-only diagnosis by Deploy Doctor{by} in {:.1} s. It never changes the VDS or the repository.</sub>\n",
        d.ms as f64 / 1000.0
    ));
    s
}

/// The whole run: what the CLI prints and appends to the job summary.
pub struct Summary<'a> {
    pub run: Option<&'a Run>,
    pub diagnoses: &'a [Diagnosis],
    pub green: &'a [String],
    pub issues: &'a [IssueAction],
    pub error: Option<&'a str>,
    pub post_error: Option<&'a str>,
    pub model: &'a str,
}

pub fn summary_markdown(sum: &Summary) -> String {
    let mut s = String::from("## 🩺 Deploy Doctor\n\n");
    if let Some(run) = sum.run {
        s.push_str(&format!(
            "CI/CD run [{}]({}) · `{}` · {} · **{}**\n\n",
            run.id,
            run.html_url,
            short_sha(&run.head_sha),
            sanitize(&run.display_title),
            run.conclusion.as_deref().unwrap_or(&run.status)
        ));
    }
    if let Some(e) = sum.error {
        s.push_str(&format!("**The diagnosis failed:** {}\n\n", sanitize(e)));
    } else if sum.diagnoses.is_empty() {
        s.push_str("Nothing to diagnose: no job failed and nothing was deployed in this run.\n\n");
    }
    if let Some(run) = sum.run {
        for d in sum.diagnoses {
            s.push_str(&diagnosis_markdown(run, d, sum.model));
            s.push_str("\n---\n\n");
        }
    }
    if !sum.green.is_empty() {
        s.push_str(&format!(
            "All jobs succeeded for: {}.\n\n",
            sum.green.join(", ")
        ));
    }
    if !sum.issues.is_empty() {
        s.push_str("**Issues**\n\n");
        for a in sum.issues {
            let line = match (&a.error, &a.url) {
                (Some(e), _) => format!("- {}: failed: {}", a.key, sanitize(e)),
                (None, Some(url)) => format!(
                    "- {}: {} [#{}]({url})",
                    a.key,
                    a.action,
                    a.number.unwrap_or_default()
                ),
                (None, None) => format!("- {}: {}", a.key, a.action),
            };
            s.push_str(&line);
            s.push('\n');
        }
        s.push('\n');
    }
    if let Some(e) = sum.post_error {
        s.push_str(&format!("**Nothing was posted:** {}\n", sanitize(e)));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_and_sanitizing() {
        assert_eq!(marker("lesson-35"), "<!-- deploy-doctor:lesson-35 -->");
        assert_eq!(
            sanitize("x <!-- deploy-doctor:lesson-01 --> ```rm```"),
            "x &lt;!-- deploy-doctor:lesson-01 --> '''rm'''"
        );
        assert_eq!(short_sha("abcdef1234"), "abcdef1");
        assert_eq!(short_sha("abc"), "abc");
    }

    #[test]
    fn titles_use_the_first_sentence() {
        assert_eq!(
            first_sentence("OPENAI_API_KEY is empty in .env. The secret is missing."),
            "OPENAI_API_KEY is empty in .env"
        );
        assert_eq!(first_sentence("One sentence."), "One sentence");
        assert_eq!(first_sentence(&"x".repeat(200)).chars().count(), TITLE_CAUSE_CHARS + 1);
    }
}

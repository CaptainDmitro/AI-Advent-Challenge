//! The model's verdict, and the checks it has to pass before it is posted.
//! A verdict that fails a check goes back to the model as the tool's error,
//! so it gets to fix it; one that contradicts the run is corrected here.

use serde::{Deserialize, Serialize};

use crate::clip;
use crate::pipeline::{self, Subject};

pub const STATUSES: [&str; 3] = ["healthy", "broken", "known_issue"];
pub const STAGES: [&str; 6] = ["build", "test", "deploy", "runtime", "pipeline", "none"];
/// Only the doctor sets this: no AI verdict, and no failed job to report.
pub const UNKNOWN: &str = "unknown";
const MAX_ITEMS: usize = 8;
const MAX_ITEM_CHARS: usize = 400;

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub failed_stage: String,
    #[serde(default)]
    pub root_cause: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub doc_reference: Option<String>,
    #[serde(default)]
    pub suggested_fix: String,
    #[serde(default)]
    pub commands_to_run: Vec<String>,
    /// What the doctor changed or wants the reader to know. Never the model's.
    #[serde(default, skip_deserializing)]
    pub notes: Vec<String>,
    /// Whether the model wrote this verdict.
    #[serde(default, skip_deserializing)]
    pub ai: bool,
}

fn tidy(items: Vec<String>) -> Vec<String> {
    items
        .into_iter()
        .map(|item| clip(item.trim(), MAX_ITEM_CHARS))
        .filter(|item| !item.is_empty())
        .take(MAX_ITEMS)
        .collect()
}

pub fn validate(mut v: Verdict, subject: &Subject) -> Result<Verdict, String> {
    v.status = v.status.trim().to_lowercase().replace([' ', '-'], "_");
    if !STATUSES.contains(&v.status.as_str()) {
        return Err(format!(
            "`status` must be one of {}, not `{}`.",
            STATUSES.join(", "),
            v.status
        ));
    }
    v.failed_stage = v.failed_stage.trim().to_lowercase();
    if !STAGES.contains(&v.failed_stage.as_str()) {
        v.failed_stage = infer_stage(subject).to_string();
    }
    v.root_cause = v.root_cause.trim().to_string();
    v.suggested_fix = v.suggested_fix.trim().to_string();
    v.evidence = tidy(v.evidence);
    v.commands_to_run = tidy(v.commands_to_run);
    v.doc_reference = v
        .doc_reference
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty());
    if v.status != "healthy" {
        if v.root_cause.is_empty() {
            return Err("`root_cause` is empty.".to_string());
        }
        if v.evidence.is_empty() {
            return Err("`evidence` is empty: quote what the tools showed.".to_string());
        }
    }
    if v.status == "known_issue" && v.doc_reference.is_none() {
        return Err(
            "A `known_issue` needs `doc_reference`: the file and section that document it."
                .to_string(),
        );
    }
    if v.status == "healthy" && subject.failed {
        let names: Vec<&str> = subject.failed_jobs().map(|j| j.name.as_str()).collect();
        let names = names.join(", ");
        v.notes.push(format!(
            "The model said healthy, but {names} failed in this run, so it is reported as broken."
        ));
        v.status = "broken".to_string();
        if v.failed_stage == "none" {
            v.failed_stage = infer_stage(subject).to_string();
        }
        if v.root_cause.is_empty() {
            v.root_cause = format!("{names} failed.");
        }
    }
    if v.status == "healthy" {
        v.failed_stage = "none".to_string();
    }
    Ok(v)
}

/// The stage the run itself points at.
pub fn infer_stage(subject: &Subject) -> &'static str {
    if subject.key == pipeline::PIPELINE {
        return "pipeline";
    }
    let failed: Vec<_> = subject.failed_jobs().collect();
    if failed
        .iter()
        .any(|job| pipeline::split_job_name(&job.name).0 == "deploy")
    {
        return "deploy";
    }
    if let Some(job) = failed.first() {
        let test = job.steps.iter().any(|step| {
            pipeline::step_failed(step.conclusion.as_deref()) && step.name.contains("test")
        });
        return if test { "test" } else { "build" };
    }
    if subject.deployed { "runtime" } else { "none" }
}

/// What gets reported when the model couldn't diagnose: the failed steps
/// from the run, and where to read the log.
pub fn fallback(subject: &Subject, why: &str) -> Verdict {
    let mut evidence = Vec::new();
    for job in subject.failed_jobs() {
        let steps: Vec<String> = job
            .steps
            .iter()
            .filter(|step| pipeline::step_failed(step.conclusion.as_deref()))
            .map(|step| {
                format!(
                    "{} › {}: {}",
                    job.name,
                    step.name,
                    step.conclusion.as_deref().unwrap_or("?")
                )
            })
            .collect();
        if steps.is_empty() {
            evidence.push(format!(
                "{}: {}",
                job.name,
                job.conclusion.as_deref().unwrap_or("?")
            ));
        }
        evidence.extend(steps);
    }
    let suggested_fix = subject
        .failed_jobs()
        .find_map(|job| job.html_url.clone())
        .map(|url| format!("Open the failed job's log: {url}"))
        .unwrap_or_else(|| "Check the run on GitHub Actions.".to_string());
    Verdict {
        status: (if subject.failed { "broken" } else { UNKNOWN }).to_string(),
        failed_stage: infer_stage(subject).to_string(),
        root_cause: format!("AI diagnosis unavailable: {why}"),
        evidence,
        doc_reference: None,
        suggested_fix,
        commands_to_run: Vec::new(),
        notes: Vec::new(),
        ai: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::github::{Job, Step};

    fn job(name: &str, conclusion: &str, failed_step: Option<&str>) -> Job {
        let mut steps = vec![Step {
            name: "Set up job".into(),
            status: "completed".into(),
            conclusion: Some("success".into()),
            number: 1,
        }];
        if let Some(step) = failed_step {
            steps.push(Step {
                name: step.into(),
                status: "completed".into(),
                conclusion: Some("failure".into()),
                number: 2,
            });
        }
        Job {
            id: 1,
            name: name.into(),
            status: "completed".into(),
            conclusion: Some(conclusion.into()),
            html_url: Some("https://github.test/job/1".into()),
            steps,
        }
    }

    fn subject(jobs: Vec<Job>) -> Subject {
        pipeline::subjects(&jobs).0.remove(0)
    }

    fn verdict(status: &str) -> Verdict {
        Verdict {
            status: status.into(),
            failed_stage: "deploy".into(),
            root_cause: "The key is missing.".into(),
            evidence: vec!["  service_logs: Error  ".into(), "".into()],
            suggested_fix: "Add the secret.".into(),
            ..Verdict::default()
        }
    }

    #[test]
    fn statuses_and_required_fields_are_checked() {
        let s = subject(vec![job("deploy (35. Deploy Doctor)", "failure", Some("Verify service is up"))]);
        assert!(validate(verdict("fine"), &s).unwrap_err().contains("`status` must be one of"));
        let v = validate(verdict("Known issue"), &s);
        assert!(v.unwrap_err().contains("doc_reference"));
        let mut no_evidence = verdict("broken");
        no_evidence.evidence = vec![" ".into()];
        assert!(validate(no_evidence, &s).unwrap_err().contains("evidence"));
        let v = validate(verdict("broken"), &s).unwrap();
        assert_eq!(v.evidence, ["service_logs: Error"]);
        assert!(v.notes.is_empty());
        let mut known = verdict("known-issue");
        known.doc_reference = Some(" DEPLOYMENT.md › Secrets ".into());
        known.failed_stage = "somewhere".into();
        let v = validate(known, &s).unwrap();
        assert_eq!(v.status, "known_issue");
        assert_eq!(v.doc_reference.as_deref(), Some("DEPLOYMENT.md › Secrets"));
        assert_eq!(v.failed_stage, "deploy");
    }

    #[test]
    fn healthy_is_not_accepted_for_a_failed_job() {
        let s = subject(vec![job("build-test (33. Support Assistant)", "failure", Some("cargo test"))]);
        let mut healthy = verdict("healthy");
        healthy.failed_stage = "none".into();
        healthy.root_cause.clear();
        let v = validate(healthy, &s).unwrap();
        assert_eq!(v.status, "broken");
        assert_eq!(v.failed_stage, "test");
        assert_eq!(v.root_cause, "build-test (33. Support Assistant) failed.");
        assert!(v.notes[0].contains("The model said healthy"));

        let ok = subject(vec![job("deploy (35. Deploy Doctor)", "success", None)]);
        let v = validate(verdict("healthy"), &ok).unwrap();
        assert_eq!(v.status, "healthy");
        assert_eq!(v.failed_stage, "none");
    }

    #[test]
    fn the_fallback_reports_the_failed_steps() {
        let s = subject(vec![job("build-test (33. Support Assistant)", "failure", Some("cargo build --release"))]);
        let v = fallback(&s, "LLM request failed");
        assert_eq!(v.status, "broken");
        assert_eq!(v.failed_stage, "build");
        assert!(!v.ai);
        assert_eq!(v.root_cause, "AI diagnosis unavailable: LLM request failed");
        assert_eq!(
            v.evidence,
            ["build-test (33. Support Assistant) › cargo build --release: failure"]
        );
        assert_eq!(v.suggested_fix, "Open the failed job's log: https://github.test/job/1");

        let deployed = subject(vec![job("deploy (35. Deploy Doctor)", "success", None)]);
        let v = fallback(&deployed, "no key");
        assert_eq!(v.status, UNKNOWN);
        assert_eq!(v.failed_stage, "runtime");
        assert!(v.evidence.is_empty());
    }
}

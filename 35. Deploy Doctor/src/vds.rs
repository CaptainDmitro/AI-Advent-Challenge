//! The VDS side of a diagnosis: read-only probes of one lesson's service.
//!
//! The model never writes a command. Every probe is built here from a
//! `Lesson` (a number from 1 to 99, checked when the subject was made), and
//! the only text from the model that reaches a shell is a probe path, which
//! has to pass a strict allowlist first. `.env` is read here and only its
//! shape leaves this module: secret values never reach the model.

use std::time::Duration;

use serde::Serialize;
use tokio::process::Command;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
/// `systemctl --user` and `journalctl --user` need this in a
/// non-interactive ssh session (the same line `ci-cd.yml` uses).
const RUNTIME_DIR: &str = "export XDG_RUNTIME_DIR=/run/user/$(id -u); ";
const MAX_LOG_LINES: usize = 200;
const MAX_PATH_CHARS: usize = 100;
/// `.env` values that aren't secret and help a diagnosis.
const SHOWN: [&str; 3] = ["PORT", "OPENAI_BASE_URL", "OPENAI_MODEL"];
/// What every deployed lesson's `.env` has (see DEPLOYMENT.md).
const EXPECTED: [&str; 2] = ["PORT", "OPENAI_API_KEY"];

/// A lesson number, 1-99: what the service name, port and folder are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Lesson(u8);

impl Lesson {
    pub fn new(n: u32) -> Option<Lesson> {
        (1..=99).contains(&n).then_some(Lesson(n as u8))
    }

    pub fn number(self) -> u8 {
        self.0
    }

    pub fn port(self) -> u16 {
        4000 + u16::from(self.0)
    }

    pub fn service(self) -> String {
        format!("ai-advent-lesson-{:02}", self.0)
    }

    /// Relative to the home folder.
    pub fn app_dir(self) -> String {
        format!("apps/lesson-{:02}", self.0)
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn render(&self) -> String {
        let mut text = format!("exit code {}", self.code);
        if !self.stdout.trim().is_empty() {
            text.push('\n');
            text.push_str(self.stdout.trim_end());
        }
        if !self.stderr.trim().is_empty() {
            text.push_str("\nstderr:\n");
            text.push_str(self.stderr.trim_end());
        }
        text
    }
}

/// Where the probes run.
#[derive(Clone, Debug)]
pub enum Vds {
    /// From GitHub Actions: `ssh -i key user@host`.
    Ssh { host: String, user: String, key: String },
    /// On the VDS itself (the deployed page): `sh -c`.
    Local,
    /// Not configured: every probe answers with the reason.
    Off(String),
    #[cfg(test)]
    Fake(std::sync::Arc<std::sync::Mutex<FakeVds>>),
}

impl Vds {
    pub fn describe(&self) -> String {
        match self {
            Vds::Ssh { host, user, .. } => format!("ssh {user}@{host}"),
            Vds::Local => "local".to_string(),
            Vds::Off(why) => format!("off: {why}"),
            #[cfg(test)]
            Vds::Fake(_) => "fake".to_string(),
        }
    }

    async fn run(&self, script: &str) -> Result<Output, String> {
        let script = format!("{RUNTIME_DIR}{script}");
        let mut command = match self {
            Vds::Ssh { host, user, key } => {
                let mut command = Command::new("ssh");
                command
                    .arg("-i")
                    .arg(key)
                    .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"])
                    .args(["-o", "StrictHostKeyChecking=accept-new"])
                    .arg(format!("{user}@{host}"))
                    .arg(&script);
                command
            }
            Vds::Local => {
                let mut command = Command::new("sh");
                command.arg("-c").arg(&script);
                command
            }
            Vds::Off(why) => return Err(why.clone()),
            #[cfg(test)]
            Vds::Fake(fake) => return Ok(fake.lock().unwrap().answer(&script)),
        };
        command.kill_on_drop(true);
        let out = tokio::time::timeout(COMMAND_TIMEOUT, command.output())
            .await
            .map_err(|_| format!("The command timed out after {}s.", COMMAND_TIMEOUT.as_secs()))?
            .map_err(|e| format!("Can't run the command: {e}"))?;
        Ok(Output {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }

    /// systemd's view of the service, and the binary's size and date.
    pub async fn service_status(&self, lesson: Lesson) -> Result<String, String> {
        let service = lesson.service();
        let dir = lesson.app_dir();
        let script = format!(
            "systemctl --user show {service} \
             -p ActiveState,SubState,Result,NRestarts,ExecMainStatus,ActiveEnterTimestamp,InactiveEnterTimestamp; \
             echo; systemctl --user status {service} --no-pager -l -n 0; \
             echo; ls -l --time-style=long-iso \"$HOME/{dir}/app\""
        );
        Ok(self.run(&script).await?.render())
    }

    /// The last lines of the service's journal.
    pub async fn service_logs(&self, lesson: Lesson, lines: usize) -> Result<String, String> {
        let lines = lines.clamp(10, MAX_LOG_LINES);
        let script = format!(
            "journalctl --user -u {} -n {lines} --no-pager -o short-iso",
            lesson.service()
        );
        Ok(self.run(&script).await?.render())
    }

    /// The names in `.env`, which are empty or missing, and the values of
    /// the few that aren't secret. The file's text never leaves here.
    pub async fn inspect_env(&self, lesson: Lesson) -> Result<String, String> {
        let dir = lesson.app_dir();
        let out = self.run(&format!("cat \"$HOME/{dir}/.env\"")).await?;
        if out.code != 0 {
            return Ok(format!(
                "Can't read ~/{dir}/.env (exit code {}): {}",
                out.code,
                out.stderr.trim()
            ));
        }
        Ok(format!(
            "~/{dir}/.env (values hidden except {}):\n{}",
            SHOWN.join(", "),
            describe_env(&out.stdout)
        ))
    }

    /// An HTTP GET to the service from the VDS itself, so a flaky network
    /// between the runner and the VDS can't make a healthy service look dead.
    pub async fn probe(&self, lesson: Lesson, path: &str) -> Result<String, String> {
        let path = probe_path(path)?;
        let script = format!(
            "curl -sS -o /dev/null -m 5 -w '%{{http_code}} %{{time_total}}s' 'http://127.0.0.1:{}{path}'",
            lesson.port()
        );
        Ok(format!("GET {path}: {}", self.run(&script).await?.render()))
    }
}

/// A path for `probe`: it goes into a shell command, so only `/`, letters,
/// digits, `-`, `_` and `.`, and no `..`.
pub fn probe_path(path: &str) -> Result<String, String> {
    let path = path.trim();
    let path = if path.is_empty() { "/" } else { path };
    let allowed = |c: char| c.is_ascii_alphanumeric() || "/-_.".contains(c);
    if !path.starts_with('/')
        || path.contains("..")
        || path.chars().count() > MAX_PATH_CHARS
        || !path.chars().all(allowed)
    {
        return Err(format!(
            "`{path}` is not a probe path: it starts with `/` and has only letters, digits, `/`, `-`, `_` and `.`."
        ));
    }
    Ok(path.to_string())
}

/// What `.env` holds, without its secrets.
pub fn describe_env(text: &str) -> String {
    let mut out = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    // Not `lines()`: it would hide a CRLF line ending, a classic `.env` bug.
    for (i, raw) in text.split('\n').enumerate() {
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let Some((name, value)) = raw.split_once('=') else {
            out.push(format!("line {}: not NAME=value", i + 1));
            continue;
        };
        let name = name.trim();
        let mut flags = Vec::new();
        if value.ends_with('\r') {
            flags.push("ends with a carriage return (CRLF)");
        }
        let value = value.trim_end_matches('\r');
        if !value.trim().is_empty() && value != value.trim() {
            flags.push("leading or trailing whitespace");
        }
        if seen.iter().any(|s| s == name) {
            flags.push("set twice: the last one wins");
        }
        seen.push(name.to_string());
        let value = value.trim();
        let mut line = if value.is_empty() {
            format!("{name}: EMPTY")
        } else if SHOWN.contains(&name) {
            format!("{name} = {value}")
        } else {
            format!("{name}: set ({} chars, value hidden)", value.chars().count())
        };
        if !flags.is_empty() {
            line.push_str(&format!(" [{}]", flags.join("; ")));
        }
        out.push(line);
    }
    for name in EXPECTED {
        if !seen.iter().any(|s| s == name) {
            out.push(format!("{name}: MISSING"));
        }
    }
    out.join("\n")
}

/// Canned answers for tests, matched by a piece of the script.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct FakeVds {
    pub answers: Vec<(String, Output)>,
    pub scripts: Vec<String>,
}

#[cfg(test)]
impl FakeVds {
    fn answer(&mut self, script: &str) -> Output {
        self.scripts.push(script.to_string());
        self.answers
            .iter()
            .find(|(key, _)| script.contains(key.as_str()))
            .map(|(_, out)| out.clone())
            .unwrap_or(Output {
                code: 127,
                stdout: String::new(),
                stderr: "fake: no answer".to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_lesson_is_a_number_from_1_to_99() {
        assert_eq!(Lesson::new(0), None);
        assert_eq!(Lesson::new(100), None);
        let lesson = Lesson::new(35).unwrap();
        assert_eq!(lesson.port(), 4035);
        assert_eq!(lesson.service(), "ai-advent-lesson-35");
        assert_eq!(lesson.app_dir(), "apps/lesson-35");
        let lesson = Lesson::new(4).unwrap();
        assert_eq!(lesson.service(), "ai-advent-lesson-04");
        assert_eq!(lesson.port(), 4004);
    }

    #[test]
    fn probe_paths_are_checked() {
        assert_eq!(probe_path("").unwrap(), "/");
        assert_eq!(probe_path(" /api/status ").unwrap(), "/api/status");
        for bad in ["api", "/../etc/passwd", "/a b", "/x;rm -rf ~", "/x'y", "/$(id)"] {
            assert!(probe_path(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn env_values_stay_hidden() {
        let text = "PORT=4035\nOPENAI_API_KEY=sk-abcdef\nOPENAI_MODEL=deepseek-v4-flash\r\n\
                    OPENAI_BASE_URL=\n# comment\nbroken line\nOPENAI_API_KEY=sk-second \n";
        let described = describe_env(text);
        assert!(!described.contains("sk-"), "{described}");
        assert!(described.contains("PORT = 4035"));
        assert!(described.contains("OPENAI_API_KEY: set (9 chars, value hidden)"));
        assert!(described.contains(
            "OPENAI_MODEL = deepseek-v4-flash [ends with a carriage return (CRLF)]"
        ));
        assert!(described.contains("OPENAI_BASE_URL: EMPTY"));
        assert!(described.contains("line 6: not NAME=value"));
        assert!(described.contains("leading or trailing whitespace; set twice"));
        assert!(!described.contains("MISSING"));
        assert_eq!(
            describe_env("OPENAI_API_KEY=\n"),
            "OPENAI_API_KEY: EMPTY\nPORT: MISSING"
        );
    }

    #[tokio::test]
    async fn probes_are_built_from_the_lesson_only() {
        let fake = Arc::new(Mutex::new(FakeVds::default()));
        let vds = Vds::Fake(fake.clone());
        let lesson = Lesson::new(35).unwrap();
        vds.service_status(lesson).await.unwrap();
        vds.service_logs(lesson, 5000).await.unwrap();
        vds.inspect_env(lesson).await.unwrap();
        vds.probe(lesson, "/api/status").await.unwrap();
        assert!(vds.probe(lesson, "/x;reboot").await.is_err());
        let scripts = fake.lock().unwrap().scripts.clone();
        assert_eq!(scripts.len(), 4);
        assert!(scripts.iter().all(|s| s.starts_with(RUNTIME_DIR)));
        assert!(scripts[0].contains("systemctl --user show ai-advent-lesson-35"));
        assert!(scripts[1].contains("journalctl --user -u ai-advent-lesson-35 -n 200"));
        assert!(scripts[2].contains("cat \"$HOME/apps/lesson-35/.env\""));
        assert!(scripts[3].contains("'http://127.0.0.1:4035/api/status'"));
        assert!(Vds::Off("no".into()).probe(lesson, "/").await.is_err());
    }
}

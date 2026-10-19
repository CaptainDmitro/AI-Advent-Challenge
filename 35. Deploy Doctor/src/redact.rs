//! Masks secrets in everything a tool returns, before the model sees it.
//!
//! GitHub already masks registered secrets in job logs, but the journal on
//! the VDS doesn't, and an app may print its own key. Three layers: the
//! doctor's own secrets by value, `NAME=value` where the name says it is a
//! secret, and things that look like keys (`sk-…`, `Bearer …`).

const MIN_SECRET_CHARS: usize = 8;
const MIN_KEY_CHARS: usize = 20;
const SENSITIVE: [&str; 5] = ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD"];
pub const MASK: &str = "***";

pub struct Redactor {
    secrets: Vec<String>,
}

impl Redactor {
    pub fn new(secrets: impl IntoIterator<Item = String>) -> Redactor {
        let mut secrets: Vec<String> = secrets
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.chars().count() >= MIN_SECRET_CHARS)
            .collect();
        // The longest first, so a secret that contains another is masked whole.
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Redactor { secrets }
    }

    pub fn redact(&self, text: &str) -> String {
        let mut text = text.to_string();
        for secret in &self.secrets {
            if text.contains(secret.as_str()) {
                text = text.replace(secret.as_str(), MASK);
            }
        }
        text.split('\n').map(redact_line).collect::<Vec<_>>().join("\n")
    }
}

fn sensitive(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    SENSITIVE.iter().any(|word| name.contains(word))
}

fn redact_line(line: &str) -> String {
    let words: Vec<&str> = line.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let after_bearer = i > 0 && words[i - 1].eq_ignore_ascii_case("bearer");
        let redacted = if after_bearer && !word.is_empty() {
            MASK.to_string()
        } else if let Some(at) = word.find("sk-")
            && word[at..].chars().count() >= MIN_KEY_CHARS
        {
            format!("{}sk-{MASK}", &word[..at])
        } else if let Some((name, value)) = word.split_once('=')
            && sensitive(name)
            && !value.is_empty()
            && value != MASK
        {
            format!("{name}={MASK}")
        } else {
            word.to_string()
        };
        out.push(redacted);
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_secrets_are_masked_by_value() {
        let r = Redactor::new(vec!["ghs_0123456789".to_string(), "short".to_string()]);
        assert_eq!(r.redact("token ghs_0123456789 here"), "token *** here");
        // Too short to mask by value: it would mangle ordinary words.
        assert_eq!(r.redact("a short line"), "a short line");
    }

    #[test]
    fn things_that_look_like_secrets_are_masked() {
        let r = Redactor::new(Vec::new());
        assert_eq!(
            r.redact("Authorization: Bearer abc.def.ghi"),
            "Authorization: Bearer ***"
        );
        assert_eq!(
            r.redact("key=\"sk-0123456789abcdefghij\""),
            "key=\"sk-***"
        );
        assert_eq!(
            r.redact("OPENAI_API_KEY=sk-1 GITHUB_TOKEN=x PORT=4035"),
            "OPENAI_API_KEY=*** GITHUB_TOKEN=*** PORT=4035"
        );
        assert_eq!(r.redact("OPENAI_API_KEY="), "OPENAI_API_KEY=");
        let plain = "error[E0425]: cannot find value `x` in this scope\n  --> src/main.rs:3:5";
        assert_eq!(r.redact(plain), plain);
        assert_eq!(r.redact("ask-me about sk-short"), "ask-me about sk-short");
    }
}

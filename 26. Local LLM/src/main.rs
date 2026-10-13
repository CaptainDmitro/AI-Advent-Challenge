use std::io::{self, BufRead, BufReader, Write};
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "qwen2.5:3b";

#[derive(Serialize, Clone, Debug, PartialEq)]
struct Message {
    role: String,
    content: String,
}

impl Message {
    fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

// Ollama's native /api/chat, not the OpenAI-compatible /v1: only the native
// endpoint reports model load time and generation time separately.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Options>,
}

#[derive(Serialize)]
struct Options {
    temperature: f32,
}

// One line of the NDJSON stream. The last line has `done: true` and the counters
// (durations are in nanoseconds).
#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    total_duration: u64,
}

#[derive(Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Deserialize)]
struct VersionResponse {
    version: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    name: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Default)]
struct Stats {
    prompt_tokens: u64,
    completion_tokens: u64,
    eval_secs: f64,
    load_secs: f64,
    total_secs: f64,
    first_token_secs: f64,
}

impl Stats {
    fn tokens_per_sec(&self) -> f64 {
        if self.eval_secs > 0.0 {
            self.completion_tokens as f64 / self.eval_secs
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} → {} tok · {:.1} tok/s · first token {:.2}s · total {:.2}s, of which model load {:.2}s]",
            self.prompt_tokens,
            self.completion_tokens,
            self.tokens_per_sec(),
            self.first_token_secs,
            self.total_secs,
            self.load_secs,
        )
    }
}

struct Reply {
    text: String,
    stats: Stats,
}

struct Ollama {
    http: reqwest::blocking::Client,
    host: String,
    model: String,
}

fn secs(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}

fn api_error(status: reqwest::StatusCode, body: &str) -> String {
    let detail = serde_json::from_str::<ErrorResponse>(body)
        .map(|e| e.error)
        .unwrap_or_else(|_| body.to_string());
    format!("Ollama error ({status}): {detail}")
}

impl Ollama {
    fn new(host: &str, model: &str) -> Self {
        // Generous: on CPU a long answer from a cold model can take minutes.
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            host: host.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let response = self
            .http
            .get(format!("{}{path}", self.host))
            .send()
            .map_err(|e| format!("Can't reach Ollama at {}: {e}", self.host))?;
        let status = response.status();
        let body = response.text().unwrap_or_default();
        if !status.is_success() {
            return Err(api_error(status, &body));
        }
        serde_json::from_str(&body).map_err(|e| format!("Failed to parse {path}: {e}"))
    }

    fn version(&self) -> Result<String, String> {
        self.get::<VersionResponse>("/api/version").map(|v| v.version)
    }

    fn models(&self) -> Result<Vec<ModelInfo>, String> {
        self.get::<TagsResponse>("/api/tags").map(|t| t.models)
    }

    /// Streams the reply: `on_token` gets every piece as it arrives, and the
    /// whole text plus the server's counters are returned at the end.
    fn chat(
        &self,
        messages: &[Message],
        temperature: Option<f32>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Reply, String> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            stream: true,
            options: temperature.map(|temperature| Options { temperature }),
        };
        let started = Instant::now();
        let response = self
            .http
            .post(format!("{}/api/chat", self.host))
            .json(&request)
            .send()
            .map_err(|e| format!("Can't reach Ollama at {}: {e}", self.host))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(api_error(status, &body));
        }

        let mut text = String::new();
        let mut first_token = None;
        for line in BufReader::new(response).lines() {
            let line = line.map_err(|e| format!("Stream interrupted: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let chunk: ChatChunk = serde_json::from_str(&line)
                .map_err(|e| format!("Failed to parse stream line: {e}"))?;
            if let Some(error) = chunk.error {
                return Err(format!("Ollama error: {error}"));
            }
            if let Some(message) = &chunk.message {
                if !message.content.is_empty() {
                    first_token.get_or_insert_with(|| started.elapsed());
                    on_token(&message.content);
                    text.push_str(&message.content);
                }
            }
            if chunk.done {
                let stats = Stats {
                    prompt_tokens: chunk.prompt_eval_count,
                    completion_tokens: chunk.eval_count,
                    eval_secs: secs(chunk.eval_duration),
                    load_secs: secs(chunk.load_duration),
                    total_secs: secs(chunk.total_duration),
                    first_token_secs: first_token.unwrap_or_default().as_secs_f64(),
                };
                return Ok(Reply { text, stats });
            }
        }
        Err("Stream ended before the model finished.".to_string())
    }
}

/// `qwen2.5` and `qwen2.5:latest` are the same model to Ollama.
fn model_installed(installed: &[String], model: &str) -> bool {
    installed
        .iter()
        .any(|name| name == model || *name == format!("{model}:latest"))
}

/// Accepts the same shapes Ollama's own `OLLAMA_HOST` does: with or without a scheme.
fn normalize_host(host: &str) -> String {
    let host = host.trim().trim_end_matches('/');
    if host.is_empty() {
        DEFAULT_HOST.to_string()
    } else if host.contains("://") {
        host.to_string()
    } else {
        format!("http://{host}")
    }
}

fn missing_keywords<'a>(answer: &str, expected: &[&'a str]) -> Vec<&'a str> {
    let answer = answer.to_lowercase();
    expected
        .iter()
        .copied()
        .filter(|keyword| !answer.contains(&keyword.to_lowercase()))
        .collect()
}

struct Preset {
    level: &'static str,
    prompt: &'static str,
    expected: &'static [&'static str],
}

// Three requests of growing difficulty: a fact, an explanation, and a word
// problem that needs real arithmetic (9:00–10:00 the first train covers 60 km,
// 240 km remain at a closing speed of 150 km/h → 1.6 h → 11:36).
const PRESETS: [Preset; 3] = [
    Preset {
        level: "easy",
        prompt: "Какая столица Франции? Ответь одним словом.",
        expected: &["Париж"],
    },
    Preset {
        level: "medium",
        prompt: "Объясни в трёх предложениях, чем TCP отличается от UDP, и назови по одному \
                 протоколу или сервису, который работает поверх каждого из них.",
        expected: &["TCP", "UDP"],
    },
    Preset {
        level: "hard",
        prompt: "Поезд выехал из города A в 9:00 со скоростью 60 км/ч. Второй поезд выехал из \
                 города B ему навстречу в 10:00 со скоростью 90 км/ч. Расстояние между городами \
                 300 км. Во сколько они встретятся? Реши по шагам, а в последней строке напиши \
                 только время в формате ЧЧ:ММ.",
        expected: &["11:36"],
    },
];

fn print_token(token: &str) {
    print!("{token}");
    let _ = io::stdout().flush();
}

fn preflight(ollama: &Ollama) -> Result<(), String> {
    let version = ollama.version()?;
    let models = ollama.models()?;
    println!("Ollama {version} at {}", ollama.host);
    println!("Installed models:");
    for model in &models {
        println!("  {} ({:.1} GB)", model.name, model.size as f64 / 1e9);
    }
    let names: Vec<String> = models.into_iter().map(|m| m.name).collect();
    if !model_installed(&names, &ollama.model) {
        return Err(format!(
            "model {} is not pulled. Run: ollama pull {}",
            ollama.model, ollama.model
        ));
    }
    println!("Using: {}\n", ollama.model);
    Ok(())
}

fn run_demo(ollama: &Ollama) -> Result<(), String> {
    preflight(ollama)?;
    let mut rows = Vec::new();
    for (i, preset) in PRESETS.iter().enumerate() {
        println!("── {}/{} · {} ──", i + 1, PRESETS.len(), preset.level);
        println!("> {}\n", preset.prompt);
        // Temperature 0, so a rerun of the demo gives the same answers.
        let reply = ollama.chat(
            &[Message::new("user", preset.prompt)],
            Some(0.0),
            &mut print_token,
        )?;
        println!("\n");
        let missing = missing_keywords(&reply.text, preset.expected);
        if missing.is_empty() {
            println!("✓ contains: {}", preset.expected.join(", "));
        } else {
            println!("✗ missing: {}", missing.join(", "));
        }
        println!("{}\n", reply.stats);
        rows.push((preset.level, missing.is_empty(), reply.stats));
    }

    println!(
        "{:<8} {:>6} {:>7} {:>8} {:>12} {:>9}",
        "level", "check", "tokens", "tok/s", "1st token,s", "total,s"
    );
    for (level, ok, stats) in &rows {
        println!(
            "{:<8} {:>6} {:>7} {:>8.1} {:>12.2} {:>9.2}",
            level,
            if *ok { "✓" } else { "✗" },
            stats.completion_tokens,
            stats.tokens_per_sec(),
            stats.first_token_secs,
            stats.total_secs,
        );
    }
    let passed = rows.iter().filter(|(_, ok, _)| *ok).count();
    println!("\n{passed}/{} answers contain what was expected.", rows.len());
    Ok(())
}

fn run_ask(ollama: &Ollama, prompt: &str) -> Result<(), String> {
    let reply = ollama.chat(&[Message::new("user", prompt)], None, &mut print_token)?;
    println!("\n\n{}", reply.stats);
    Ok(())
}

fn run_chat(ollama: &Ollama) -> Result<(), String> {
    preflight(ollama)?;
    println!("Chat with {}. /reset clears the history, /quit exits.\n", ollama.model);
    let mut history: Vec<Message> = Vec::new();
    loop {
        print!("> ");
        if io::stdout().flush().is_err() {
            break;
        }
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF (e.g. Ctrl+D)
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {e}");
                continue;
            }
        }
        let input = input.trim();
        match input {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                history.clear();
                println!("History cleared.\n");
                continue;
            }
            _ => {}
        }

        history.push(Message::new("user", input));
        match ollama.chat(&history, None, &mut print_token) {
            Ok(reply) => {
                println!("\n{}\n", reply.stats);
                history.push(Message::new("assistant", &reply.text));
            }
            Err(e) => {
                eprintln!("\n{e}\n");
                history.pop();
            }
        }
    }
    Ok(())
}

fn print_usage() {
    println!(
        "Usage:\n\
         \x20 local-llm               three requests of growing difficulty, with checks and timings\n\
         \x20 local-llm ask <prompt>  one request\n\
         \x20 local-llm chat          interactive chat with history"
    );
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let host = normalize_host(&env_or("OLLAMA_HOST", DEFAULT_HOST));
    let model = env_or("OLLAMA_MODEL", DEFAULT_MODEL);
    let ollama = Ollama::new(&host, &model);

    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None | Some("demo") => run_demo(&ollama),
        Some("chat") => run_chat(&ollama),
        Some("ask") if args.len() > 1 => run_ask(&ollama, &args[1..].join(" ")),
        _ => {
            print_usage();
            return;
        }
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use std::thread;

    struct Recorded {
        path: String,
        body: String,
    }

    /// A one-shot HTTP server standing in for Ollama: answers each incoming
    /// connection with the next canned response and records what it was sent.
    fn fake_ollama(
        responses: Vec<(u16, &'static str, String)>,
    ) -> (String, thread::JoinHandle<Vec<Recorded>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, content_type, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut request_body = vec![0; length];
                reader.read_exact(&mut request_body).unwrap();
                seen.push(Recorded {
                    path: request_line
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .to_string(),
                    body: String::from_utf8(request_body).unwrap(),
                });
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
            seen
        });
        (host, handle)
    }

    #[test]
    fn reads_version_and_installed_models() {
        let (host, server) = fake_ollama(vec![
            (200, "application/json", r#"{"version":"0.40.3"}"#.to_string()),
            (
                200,
                "application/json",
                r#"{"models":[{"name":"qwen2.5:3b","size":1929912432},{"name":"llama3.2:latest","size":2019393189}]}"#
                    .to_string(),
            ),
        ]);
        let ollama = Ollama::new(&host, "qwen2.5:3b");
        assert_eq!(ollama.version().unwrap(), "0.40.3");
        let names: Vec<String> = ollama
            .models()
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(names, ["qwen2.5:3b", "llama3.2:latest"]);
        assert!(model_installed(&names, "qwen2.5:3b"));
        assert!(model_installed(&names, "llama3.2"));
        assert!(!model_installed(&names, "qwen2.5:7b"));

        let seen = server.join().unwrap();
        let paths: Vec<&str> = seen.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(paths, ["/api/version", "/api/tags"]);
    }

    #[test]
    fn streams_a_reply_token_by_token_and_reads_the_counters() {
        let stream = [
            r#"{"model":"qwen2.5:3b","message":{"role":"assistant","content":"Па"},"done":false}"#,
            r#"{"model":"qwen2.5:3b","message":{"role":"assistant","content":"риж"},"done":false}"#,
            r#"{"model":"qwen2.5:3b","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":12,"eval_count":2,"eval_duration":500000000,"load_duration":250000000,"total_duration":1000000000}"#,
        ]
        .join("\n")
            + "\n";
        let (host, server) = fake_ollama(vec![(200, "application/x-ndjson", stream)]);
        let ollama = Ollama::new(&host, "qwen2.5:3b");

        let mut tokens: Vec<String> = Vec::new();
        let reply = ollama
            .chat(
                &[Message::new("user", "Столица Франции?")],
                Some(0.0),
                &mut |t: &str| tokens.push(t.to_string()),
            )
            .unwrap();
        assert_eq!(tokens, ["Па", "риж"]);
        assert_eq!(reply.text, "Париж");
        assert_eq!(reply.stats.prompt_tokens, 12);
        assert_eq!(reply.stats.completion_tokens, 2);
        assert!((reply.stats.tokens_per_sec() - 4.0).abs() < 1e-9);
        assert!((reply.stats.load_secs - 0.25).abs() < 1e-9);
        assert!((reply.stats.total_secs - 1.0).abs() < 1e-9);

        let seen = server.join().unwrap();
        assert_eq!(seen[0].path, "/api/chat");
        let body: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
        assert_eq!(body["model"], "qwen2.5:3b");
        assert_eq!(body["stream"], true);
        assert_eq!(body["options"]["temperature"], 0.0);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Столица Франции?");
    }

    #[test]
    fn reports_errors_before_during_and_instead_of_the_stream() {
        let (host, server) = fake_ollama(vec![
            (
                404,
                "application/json",
                r#"{"error":"model \"nope\" not found, try pulling it first"}"#.to_string(),
            ),
            (
                200,
                "application/x-ndjson",
                "{\"error\":\"model requires more system memory\"}\n".to_string(),
            ),
            (
                200,
                "application/x-ndjson",
                "{\"message\":{\"content\":\"обры\"},\"done\":false}\n".to_string(),
            ),
        ]);
        let ollama = Ollama::new(&host, "nope");
        let ask = || ollama.chat(&[Message::new("user", "hi")], None, &mut |_: &str| {});

        let err = ask().unwrap_err();
        assert!(err.contains("404") && err.contains("not found"), "{err}");
        let err = ask().unwrap_err();
        assert!(err.contains("more system memory"), "{err}");
        let err = ask().unwrap_err();
        assert!(err.contains("ended before"), "{err}");
        server.join().unwrap();

        // The server above is gone now, so its port refuses connections.
        let err = Ollama::new(&host, "nope").version().unwrap_err();
        assert!(err.contains("Can't reach Ollama"), "{err}");
    }

    #[test]
    fn checks_answers_for_expected_keywords() {
        assert!(missing_keywords("Париж.", PRESETS[0].expected).is_empty());
        assert!(
            missing_keywords(
                "tcp устанавливает соединение (HTTP), udp — нет (DNS).",
                PRESETS[1].expected
            )
            .is_empty()
        );
        assert!(missing_keywords("...\n11:36", PRESETS[2].expected).is_empty());
        assert_eq!(
            missing_keywords("...\n13:00", PRESETS[2].expected),
            ["11:36"]
        );
        assert_eq!(missing_keywords("Только TCP.", PRESETS[1].expected), ["UDP"]);
    }

    #[test]
    fn normalizes_ollama_host_like_ollama_does() {
        assert_eq!(normalize_host(""), DEFAULT_HOST);
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(
            normalize_host("http://gpu-box:11434/"),
            "http://gpu-box:11434"
        );
    }
}

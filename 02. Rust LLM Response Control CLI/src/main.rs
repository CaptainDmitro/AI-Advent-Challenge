use std::io::{self, Write};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

#[derive(Default, Clone)]
struct Settings {
    format: Option<String>,
    max_tokens: Option<u32>,
    stop: Option<String>,
}

fn show_setting(value: &Option<impl std::fmt::Display>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "(none)".to_string(),
    }
}

// The format instruction is injected fresh into every request rather than stored in
// history, so toggling it only affects requests sent after the change.
fn build_messages(base: &[Message], format: &Option<String>) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(format) = format {
        messages.push(Message {
            role: "system".to_string(),
            content: format.clone(),
        });
    }
    messages.extend(base.iter().cloned());
    messages
}

fn send_request(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: Vec<Message>,
    settings: &Settings,
) -> Result<String, String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages,
        max_tokens: settings.max_tokens,
        stop: settings.stop.clone(),
    };

    let response = client
        .post(endpoint)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .map_err(|e| format!("Request failed: {e}"))?;

    let status = response.status();
    let body = response.text().unwrap_or_default();

    if !status.is_success() {
        let detail = serde_json::from_str::<ErrorResponse>(&body)
            .map(|e| e.error.message)
            .unwrap_or(body);
        return Err(format!("API error ({status}): {detail}"));
    }

    let parsed: ChatResponse =
        serde_json::from_str(&body).map_err(|e| format!("Failed to parse response: {e}"))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "No response returned from the model.".to_string())
}

fn print_help() {
    println!(
        "Commands:\n\
         \x20 :set format <text>|none      explicit response-format instruction\n\
         \x20 :set max_tokens <n>|none     API max_tokens limit\n\
         \x20 :set stop <sequence>|none    API stop sequence\n\
         \x20 :show                        show current settings\n\
         \x20 :resend                      resend the last question with current settings\n\
         \x20 :reset                       clear format, max_tokens, and stop\n\
         \x20 :help                        show this list\n"
    );
}

fn main() {
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Error: OPENAI_API_KEY environment variable is not set.");
            std::process::exit(1);
        }
    };

    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let client = reqwest::blocking::Client::new();
    let mut history: Vec<Message> = Vec::new();
    let mut settings = Settings::default();
    let mut last_user_message: Option<String> = None;

    println!("LLM chat ({model}). Type :help for commands, Ctrl+C to exit.\n");

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
        if input.is_empty() {
            continue;
        }

        if let Some(command) = input.strip_prefix(':') {
            let mut parts = command.trim().splitn(3, ' ');
            match parts.next().unwrap_or("") {
                "help" => print_help(),
                "show" => {
                    println!("format:     {}", show_setting(&settings.format));
                    println!("max_tokens: {}", show_setting(&settings.max_tokens));
                    println!("stop:       {}\n", show_setting(&settings.stop));
                }
                "reset" => {
                    settings = Settings::default();
                    println!("format, max_tokens, and stop cleared.\n");
                }
                "set" => {
                    let key = parts.next().unwrap_or("");
                    let value = parts.next().unwrap_or("").trim();
                    match key {
                        "format" => {
                            settings.format = if value.is_empty() || value == "none" {
                                None
                            } else {
                                Some(value.to_string())
                            };
                            println!("format set to: {}\n", show_setting(&settings.format));
                        }
                        "max_tokens" => {
                            if value.is_empty() || value == "none" {
                                settings.max_tokens = None;
                                println!("max_tokens cleared.\n");
                            } else {
                                match value.parse::<u32>() {
                                    Ok(n) => {
                                        settings.max_tokens = Some(n);
                                        println!("max_tokens set to: {n}\n");
                                    }
                                    Err(_) => println!("Invalid number: {value}\n"),
                                }
                            }
                        }
                        "stop" => {
                            settings.stop = if value.is_empty() || value == "none" {
                                None
                            } else {
                                Some(value.to_string())
                            };
                            println!("stop set to: {}\n", show_setting(&settings.stop));
                        }
                        _ => println!("Unknown setting. Use format, max_tokens, or stop.\n"),
                    }
                }
                "resend" => match &last_user_message {
                    None => println!("No previous question to resend.\n"),
                    Some(question) => {
                        // Sent standalone (not the running history) so the comparison
                        // reflects only the current settings, not prior conversation turns.
                        let messages = build_messages(
                            &[Message {
                                role: "user".to_string(),
                                content: question.clone(),
                            }],
                            &settings.format,
                        );
                        match send_request(&client, &endpoint, &api_key, &model, messages, &settings)
                        {
                            Ok(answer) => println!("{answer}\n"),
                            Err(e) => eprintln!("{e}\n"),
                        }
                    }
                },
                other => println!("Unknown command: {other}. Type :help for a list.\n"),
            }
            continue;
        }

        last_user_message = Some(input.to_string());
        history.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let messages = build_messages(&history, &settings.format);
        match send_request(&client, &endpoint, &api_key, &model, messages, &settings) {
            Ok(answer) => {
                println!("{answer}\n");
                history.push(Message {
                    role: "assistant".to_string(),
                    content: answer,
                });
            }
            Err(e) => {
                eprintln!("{e}\n");
                history.pop();
            }
        }
    }
}

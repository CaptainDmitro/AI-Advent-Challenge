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

    println!("LLM chat ({model}). Press Ctrl+C to exit.\n");

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

        history.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let request = ChatRequest {
            model: model.clone(),
            messages: history.clone(),
        };

        let response = client
            .post(&endpoint)
            .bearer_auth(&api_key)
            .json(&request)
            .send();

        match response {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();

                if !status.is_success() {
                    let detail = serde_json::from_str::<ErrorResponse>(&body)
                        .map(|e| e.error.message)
                        .unwrap_or(body);
                    eprintln!("API error ({status}): {detail}\n");
                    history.pop();
                    continue;
                }

                match serde_json::from_str::<ChatResponse>(&body) {
                    Ok(parsed) => {
                        if let Some(choice) = parsed.choices.into_iter().next() {
                            let answer = choice.message.content;
                            println!("{answer}\n");
                            history.push(Message {
                                role: "assistant".to_string(),
                                content: answer,
                            });
                        } else {
                            eprintln!("No response returned from the model.\n");
                            history.pop();
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to parse response: {e}\n");
                        history.pop();
                    }
                }
            }
            Err(e) => {
                eprintln!("Request failed: {e}\n");
                history.pop();
            }
        }
    }
}

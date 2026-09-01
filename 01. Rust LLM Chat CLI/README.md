# 01. Rust LLM Chat CLI

A simple Rust command-line app that continuously chats with any OpenAI-compatible LLM API: it prompts for input, sends the running conversation to the `/chat/completions` endpoint, prints the reply, and repeats until interrupted with `Ctrl+C`.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

## Configuration

Set via environment variables:

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | no | `gpt-4o-mini` |

`OPENAI_BASE_URL` can point at any OpenAI-compatible server (e.g. a local or self-hosted one).

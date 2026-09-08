# AI Advent Challenge

Practice projects from an AI-focused course, one folder per lesson.

## Table of contents

| # | Lesson | Description |
|---|---|---|
| 01 | [Rust LLM Chat CLI](./01.%20Rust%20LLM%20Chat%20CLI) | Continuous CLI chat loop against an OpenAI-compatible LLM API, written in Rust |
| 02 | [Rust LLM Response Control CLI](./02.%20Rust%20LLM%20Response%20Control%20CLI) | Live commands to control response format, length, and stop conditions, to compare restricted vs. unrestricted answers |
| 03 | [Different Reasoning Approaches](./03.%20Different%20Reasoning%20Approaches) | Solving one task four ways (direct, step-by-step, self-authored prompt, expert panel) and comparing the results |
| 04 | [Model Version Comparison](./04.%20Model%20Version%20Comparison) | Same prompt sent to a weak, medium, and strong model in parallel, comparing response time, tokens, and cost |
| 05 | [Temperature Comparison](./05.%20Temperature%20Comparison) | Same prompt sampled 3x at temperature 0 / 0.7 / 1.2 to compare accuracy, creativity, and diversity |
| 06 | [First Agent](./06.%20First%20Agent) | A minimal chat agent that encapsulates the LLM request/response cycle behind an `Agent` entity |

## CI/CD

Every push to `master` runs [`.github/workflows/ci-cd.yml`](./.github/workflows/ci-cd.yml), which discovers every lesson crate automatically:

1. **build-test** — `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt --check` for every lesson.
2. **deploy** — for lessons that are web apps (depend on `axum`), cross-compiles a static `x86_64-unknown-linux-musl` binary and deploys it as a systemd user service on a VDS, reachable at `http://<VDS host>:400N` for lesson `N`.

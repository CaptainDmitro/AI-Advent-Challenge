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
| 07 | [Context Persistence](./07.%20Context%20Persistence) | The same agent, now saving and restoring its conversation history to a JSON file so it survives a restart |
| 08 | [Token Counting](./08.%20Token%20Counting) | The same agent, now reporting estimated and actual token usage per turn, a running history total, and cost |

## CI/CD

Every push to `master` runs [`.github/workflows/ci-cd.yml`](./.github/workflows/ci-cd.yml): it builds, tests, and lints every lesson automatically, then deploys whichever lesson is newest (web-app lessons only) to a VDS. See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full architecture, the secrets/variables inventory, and how to redeploy an older lesson on demand.

## Working with AI coding agents

See [`AGENTS.md`](./AGENTS.md) for the conventions, environment constraints, and known-deferred issues an agent (or a new contributor) should know before touching this repo.

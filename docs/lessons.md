# Lessons

Every lesson folder with its crate name, its kind, the port it runs on when
deployed, and what it builds on. "Web" means an axum app (deployable);
"CLI" means a terminal program (never deployed). Ports are `4000 + NN`.
The root `README.md` has a longer description of each lesson.

| # | Folder | Crate | Kind | Port | Topic | Builds on |
|---|---|---|---|---|---|---|
| 01 | `01. Rust LLM Chat CLI` | `llm-chat` | CLI | — | a chat loop against an OpenAI-compatible API | — |
| 02 | `02. Rust LLM Response Control CLI` | `llm-response-control` | CLI | — | commands for response format, length and stop sequences | 01 |
| 03 | `03. Different Reasoning Approaches` | `llm-reasoning-web` | Web | 4003 | one task solved four ways and compared | — |
| 04 | `04. Model Version Comparison` | `model-version-comparison` | Web | 4004 | weak / medium / strong model side by side (uses `WEAK_*`/`MEDIUM_*`/`STRONG_*`; its deploy fails on purpose) | — |
| 05 | `05. Temperature Comparison` | `temperature-comparison` | Web | 4005 | the same prompt at temperature 0 / 0.7 / 1.2 | — |
| 06 | `06. First Agent` | `first-agent` | Web | 4006 | the `Agent` struct around the request/response cycle | — |
| 07 | `07. Context Persistence` | `context-persistence` | Web | 4007 | conversation saved to a JSON file | 06 |
| 08 | `08. Token Counting` | `token-counting` | Web | 4008 | estimated and actual tokens, cost | 07 |
| 09 | `09. Context Compression` | `context-compression` | Web | 4009 | rolling summary of older messages | 08 |
| 10 | `10. Context Management Strategies` | `context-strategies` | Web | 4010 | sliding window, sticky facts, branching | — |
| 11 | `11. Agent Memory Model` | `agent-memory-layers` | Web | 4011 | short-term, working and long-term memory | 10 |
| 12 | `12. Personalization` | `agent-personalization` | Web | 4012 | an explicit user profile in every request | 11 |
| 13 | `13. Task State Machine` | `agent-task-fsm` | Web | 4013 | task stages as a finite state machine | 12 |
| 14 | `14. Invariant Guardrails` | `agent-invariants` | Web | 4014 | binding invariants checked before the model | — |
| 15 | `15. Controlled State Transitions` | `agent-state-transitions` | Web | 4015 | plan-approved and validation-passed gates | 13 |
| 16 | `16. MCP Connection` | `mcp-connection` | Web | 4016 | an `rmcp` client: handshake and `tools/list` | — |
| 17 | `17. First MCP Tool` | `first-mcp-tool` | Web | 4017 | an own MCP server with one tool, used by an agent | 16 |
| 18 | `18. Scheduler and Background Tasks` | `scheduler-background-tasks` | Web | 4018 | deferred and periodic MCP jobs | 17 |
| 19 | `19. MCP Tool Composition` | `mcp-tool-composition` | Web | 4019 | chained tools: search → summarize → save | 18 |
| 20 | `20. MCP Orchestration` | `mcp-orchestration` | Web | 4020 | three MCP servers and a routing orchestrator | 19 |
| 21 | `21. Document Indexing` | `document-indexing` | Web | 4021 | chunking strategies, embeddings, an index | — |
| 22 | `22. First RAG Query` | `first-rag-query` | Web | 4022 | RAG vs. no RAG on control questions | 21 |
| 23 | `23. Reranking and Filtering` | `reranking-and-filtering` | Web | 4023 | heuristic reranker and relevance threshold | 22 |
| 24 | `24. Citations and Sources` | `citations-and-sources` | Web | 4024 | quotes checked word for word | 23 |
| 25 | `25. RAG Chat with Task Memory` | `rag-chat-with-task-memory` | Web | 4025 | RAG chat with sessions and task state | 24 |
| 26 | `26. Local LLM` | `local-llm` | Web | 4026 | Ollama + Qwen 2.5 3B on the Mac, reached through an SSH tunnel | 06 |
| 27 | — | — | — | — | no folder for day 27 | — |
| 28 | `28. Local RAG` | `local-rag` | Web | 4028 | lesson 22's RAG with a local model | 22, 26 |
| 29 | `29. Local LLM Optimization` | `local-llm-optimization` | Web | 4029 | five tuning profiles for the 3B model | 28 |
| 30 | `30. Private LLM Service` | `private-llm-service` | Web | 4030 | an authenticated, rate-limited gateway to Ollama | 26 |
| 31 | `31. Developer Assistant` | `developer-assistant` | Web | 4031 | `/help` over the repo's docs (RAG) and its git checkout (MCP) | 20, 25 |
| 32 | `32. AI Code Review` | `ai-code-review` | Web + CLI | 4032 | AI review of every PR: a GitHub Action, RAG over the repo's docs and code, MCP, a PR comment | 31 |

## Which lessons use what

- **MCP (`rmcp`)**: 16, 17, 18, 19, 20, 31, 32.
- **RAG index (structural chunks + local hashed TF-IDF)**: 21, 22, 23, 24,
  25, 28, 29, 31, 32 (32 indexes code as well as docs).
- **Ollama instead of a hosted API**: 26, 28, 29, 30.
- **Files written at runtime** (relative to the service's working directory
  `~/apps/lesson-NN/`): 07 and the agent lessons after it (conversation and memory JSON), 18 (jobs), 19
  and 20 (notes), 21 (index), 25 (sessions), 30 (`api_keys.txt`), 31 and 32
  (the git checkout in `repo/`; 32 also fetches PRs into it and makes
  temporary worktrees in the system temp folder).
- **A GitHub Action of its own**: 32 (`.github/workflows/ai-review.yml`).

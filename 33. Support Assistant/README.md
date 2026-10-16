# 33. Support Assistant

## Задание

🔥 День 33. Ассистент поддержки пользователей

Сделайте мини-сервис для поддержки пользователей.

Ассистент должен:

👉 отвечать на вопросы о продукте
👉 использовать RAG (FAQ, документация)
👉 учитывать контекст пользователя или тикета

Через MCP:

👉 подключите CRM
👉 или JSON с пользователями / тикетами

Пример:

👉 «Почему не работает авторизация?»
👉 ассистент отвечает с учётом данных тикета

Результат:

AI-ассистент для поддержки пользователей

Формат:

Видео + Код

## Demo

_(video to be added after recording)_

## What this is

A support assistant for **Notely**, a made-up note-taking app (sync across
devices, plans Free / Pro / Team). A support agent picks a ticket and asks a
question. The assistant reads the ticket and the customer's account from a
CRM through an MCP server, searches Notely's FAQ and help center through
RAG, and answers from both.

```
question + selected ticket (T-1001)
   │
   ├─▶ MCP  crm server (/mcp/crm) ─ get_ticket ─ get_user ──────▶ "Anna Petrova, plan pro · locked: YES, until the password
   │        always first, before the model runs                   is reset · last sign-ins: ACCOUNT_LOCKED, WRONG_PASSWORD…"
   │
   ├─▶ RAG  kb/*.md (FAQ + help center, compiled in)
   │        query = English rewrite + the ticket's tags + its error codes
   │        structural chunks ─▶ top-20 ─▶ heuristic rerank ─▶ ≥ 0.30 ─▶ up to 5 fragments [1]..[5]
   │
   └─▶ LLM  system: ticket context + fragments + rules       ┌───────────────────────────────────────────────────┐
            tools:  the crm server's 7 tools       ◀────────▶│ read:  get_ticket · get_user · get_login_events   │
            ≤ 8 rounds of tool calls                          │        list_tickets · search_users                │
                                                              │ write: add_ticket_note · set_ticket_status        │
            ─▶ the cause (from the CRM) + the steps [n]      └───────────────────────────────────────────────────┘
```

### The same question, four different answers

The demo data has four tickets in which the customer wrote almost the same
thing: "I can't sign in". The account data tells them apart:

| Ticket | Customer | What the CRM shows | What the assistant answers |
|---|---|---|---|
| `T-1001` | Anna Petrova, Pro | 10 failed sign-ins in 24 h, `locked: until the password is reset`, then `ACCOUNT_LOCKED` | the account is locked and waiting won't help: reset the password through "Forgot password?" |
| `T-1002` | Boris Ivanov, Free, signed up yesterday | `email verified: NO`, every attempt `EMAIL_NOT_VERIFIED` | confirm the email, "Resend confirmation email", check Spam |
| `T-1003` | Chen Wei, Team, BigCorp | SSO via Okta **enforced** for bigcorp.com since yesterday, `SSO_REQUIRED`, the reset email wasn't sent | the company turned on SSO: use "Sign in with SSO" with the work email |
| `T-1004` | Elena Smirnova, Pro, 2FA on | password right, `INVALID_2FA_CODE` with "clock skew about −4 min" | the phone's clock is off: turn on automatic time, sync the authenticator |

Ask «Почему не работает авторизация?» on each of them, and then with no
ticket at all, where the assistant can only list the usual causes and asks
for the account's email.

### RAG: the FAQ and the help center

`kb/` is Notely's knowledge base: `faq.md`, `sign-in.md` (one section per
sign-in error code), `two-factor.md`, `sso.md`, `sync.md`,
`plans-and-billing.md` and `account.md`. It is compiled into the binary and
indexed at start (29 chunks, a few milliseconds). The index is the one from
lessons 21-25 and 31: structural chunks (one per Markdown section), the
local hashed TF-IDF embedder (no network, no embeddings key), and lesson
23's heuristic reranker with the 0.30 threshold.

Two things make the search follow the ticket:

- **The ticket goes into the query.** The query is the question plus the
  ticket's tags plus the error codes of the customer's recent failed
  sign-ins. Error codes are kept whole by the tokenizer (`account_locked`),
  and every code has a section of its own in `sign-in.md` with the code in
  its heading. So the same question finds `ACCOUNT_LOCKED: too many failed
  sign-in attempts` for Anna and `SSO_REQUIRED: your organization requires
  single sign-on` for Chen.
- **A Russian question is searched in English.** The knowledge base is in
  English and the index is lexical, so Cyrillic words can't match anything.
  A question in Russian is rewritten into an English search query first (one
  short LLM call). The answer is still in the language of the question.

### MCP: the `crm` server

`/mcp/crm` is an MCP server (rmcp, Streamable HTTP, the same setup as
lesson 31) over `crm.json`, a file with users and tickets:

| Tool | Kind | What it does |
|---|---|---|
| `get_ticket` | read | a ticket: subject, status, priority, tags, the customer, the conversation, the internal notes |
| `get_user` | read | an account by id or email: plan, devices, security (email confirmed, 2FA, SSO, lock), the last 5 sign-in attempts with error codes, their tickets |
| `get_login_events` | read | the sign-in history, newest first, optionally failed only |
| `list_tickets` | read | tickets by status or by user, most recently updated first |
| `search_users` | read | users whose name, email, company or id contains a text |
| `add_ticket_note` | write | an internal note on a ticket, signed `assistant` |
| `set_ticket_status` | write | `open` / `pending` / `resolved` / `closed`, optionally with a note |

The assistant connects to it as an MCP client over HTTP, like to any remote
server: `initialize`, `tools/list`, then the tools go to the LLM as
functions. **When a ticket is selected, `get_ticket` and then `get_user` for
its customer are called on every question, before the model runs.** Their
results go into the system prompt and into the reply's `ticket` and `user`
fields, so the customer card on the page comes from MCP, not from the
model. The page's ticket list is `list_tickets` through the same server.

The model can call more tools itself: `get_login_events` for a longer
history, `search_users` when a message names an email, and the two write
tools. The system prompt allows writes only when the message asks for one
(«Запиши диагноз в тикет и переведи его в pending»). A write goes to the
file at once with write-then-rename, the reply says `changed: true`, and the
page reloads the ticket. **reset CRM** on the page (`POST /api/reset`)
puts `data/crm.json` back for the next run.

Any MCP client can use the server too:

```bash
npx @modelcontextprotocol/inspector   # -> Streamable HTTP, http://<host>:4033/mcp/crm
```

Ids are forgiving (`1001`, `#1001`, `t1001` are `T-1001`). An unknown id, an
empty note or an unknown status is a tool result with `isError: true` that
the model can read and react to, not a protocol error.

### The answer

The system prompt tells the model to:

- first work out the cause from the ticket context (plan, security settings,
  the error codes and details of the recent sign-ins) and name the data it
  comes from;
- then give only the steps that apply to this customer, from the knowledge
  base, citing fragments as `[n]`;
- say when neither the knowledge base nor the CRM covers the question, and
  offer to hand the ticket to a person, rather than invent;
- change the ticket only when asked, and never claim a change that didn't
  succeed;
- answer in the question's language, addressed to the customer.

The page has three columns. On the left is the ticket list. In the middle is
the chat, with example questions. On the right are the customer card
(account, security flags in red, the last sign-ins with their codes, the
conversation and notes), every MCP call of the last answer (automatic ones
marked `auto`, writes marked `write`), the sources with the cited ones
highlighted, and the retrieval candidates with what the ticket added to the
query.

### Verification

`cargo test` needs no network and no API key. The real app runs on a random
port over a temp copy of the CRM, against a scripted LLM, and everything
goes over real HTTP and a real MCP connection.

- `crm_mcp_server_reads_and_writes_the_crm`: the server lists exactly the 7
  tools, each with a description. `get_ticket("1001")` is `T-1001`, and an
  unknown ticket is an error. `get_user` by a differently-cased email finds
  Chen with SSO enforced, the newest event first and his ticket. Anna's card
  says she is locked until the password is reset, with 5 recent events.
  `get_login_events(failed_only)` returns her 12 failed attempts.
  `list_tickets(open)` returns 5, most recently updated first, and `lost` is
  refused. `search_users("BigCorp")` returns 2. `add_ticket_note` and
  `set_ticket_status("Pending")` are in the file on disk right after the
  call. An empty note, an unknown ticket, the status `deleted` and an unknown
  tool are refused.
- `the_same_question_is_answered_from_each_tickets_data`: «Почему не работает
  авторизация?» on `T-1001`…`T-1004`. Each time the calls are
  `get_ticket` and `get_user` (both `auto`). The query is the English
  rewrite plus that ticket's error code. A fragment from that code's own
  section passes the threshold. The model's system prompt carries that
  ticket, the fact that explains it (`locked … until the password is reset`,
  `email verified: NO`, `ENFORCED since`, `clock skew`) and the fragments,
  and its request carries the 7 tools.
- `the_assistant_writes_to_the_ticket_when_asked`: the model calls
  `add_ticket_note` and `set_ticket_status`. The reply says `changed`, the
  CRM has the note and `pending`, the results went back to the model, and
  `/api/tickets` shows the new status. `/api/reset` restores `open`.
- `without_a_ticket_it_answers_from_the_knowledge_base`: no MCP calls before
  the model, the prompt says no ticket is selected, an English question is
  searched as it is, and the sync / plans docs are retrieved and cited.
- `an_unknown_ticket_is_reported_to_the_model`: `get_ticket` fails, there is
  no `get_user`, and the system prompt says so.
- `status_tickets_and_context_endpoints`, and in the modules: id
  normalization, the seed's four sign-in cases, writes that survive a reopen
  and a failed change that leaves nothing behind, the timestamp format, every
  `kb/` file in the index with a section per error code, retrieval of the
  clock-skew section, and the reranker.

## Run

```bash
OPENAI_API_KEY=sk-... cargo run
```

Then open http://localhost:3000, pick a ticket and click an example. The
first start creates `crm.json` in the lesson folder from `data/crm.json`.

With curl:

```bash
curl -s -X POST localhost:3000/api/ask -H 'Content-Type: application/json' \
  -d '{"message": "Почему не работает авторизация?", "ticket_id": "T-1003"}' \
  | jq '{answer, hints: .context_hints, sources: [.sources[] | {n, section, cited}], calls: [.calls[] | {tool, auto}]}'
```

```bash
curl -s localhost:3000/api/tickets | jq '.tickets[] | {id, status, subject, customer}'
```

```bash
curl -s -X POST localhost:3000/api/reset
```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.deepseek.com` |
| `OPENAI_MODEL` | no | `deepseek-v4-flash` |
| `CRM_FILE` | no | `crm.json` in the working directory, created from `data/crm.json` when missing |
| `MCP_URL` | no | `http://127.0.0.1:$PORT/mcp/crm`: the MCP server the assistant connects to |
| `PORT` | no | `3000` |

The model has to support tool calling (DeepSeek and OpenAI do).

## Deploy

GitHub Actions deploys this lesson automatically on every push to `master`,
because it's now the highest-numbered lesson (see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md)). It runs at `http://<VDS host>:4033`
and the MCP server at `http://<VDS host>:4033/mcp/crm`, with the `OPENAI_*`
secrets the pipeline already supplies.

The CRM lives in `~/apps/lesson-33/crm.json`, created on the first start. A
redeploy replaces only the binary and `.env`, so notes and statuses written
by the assistant stay until **reset CRM** is clicked (or the file is
deleted).

## Conclusion

_(to fill in after recording the demo)_

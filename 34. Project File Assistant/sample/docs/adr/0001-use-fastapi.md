# ADR 0001: Use FastAPI for the HTTP API

- Status: accepted
- Date: 2026-03-02

## Context

Notely needs a small HTTP API for accounts and notes. The team writes Python,
wants request validation and generated OpenAPI docs, and needs neither a
template engine nor an admin panel.

## Decision

Build the API on FastAPI, with a Pydantic model for every request body, and
keep storage behind a `Storage` class so it can move to a database later.

## Consequences

- Request validation and `/docs` come for free.
- Auth is a FastAPI dependency, `require_auth`, that every protected endpoint
  declares. Forgetting it leaves the endpoint open, so `PROJECT_RULES.md`
  makes it a rule.
- Storage stays in memory until a later ADR picks a database.

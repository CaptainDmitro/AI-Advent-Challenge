# Notely API

All endpoints take and return JSON. Endpoints marked **Auth: required** need
an `Authorization: Bearer <token>` header with a token from `POST /users/login`.

## Service

### GET /health

Returns `{"status": "ok", "version": "..."}`. Auth: none.

## Users

### POST /users/register

Creates an account. Auth: none.

Body: `{"email": "...", "password": "...", "name": "..."}`. Returns
`{"id", "email", "name"}` with `201`, or `409` when the email is taken.

### POST /users/login

Signs in. Auth: none.

Body: `{"email": "...", "password": "..."}`. Returns `{"token": "..."}`, or
`401`.

### GET /users/me

The signed-in user's profile. Auth: required.

## Notes

### GET /notes

The user's notes. Auth: required.

### GET /notes/{note_id}

One note. Auth: required. Returns `404` when the note doesn't exist or isn't
the user's.

### POST /notes

Creates a note. Auth: required.

Body: `{"title": "...", "body": "..."}` (`body` is optional). Returns the new
note with `201`, or `413` when the body is longer than the limit.

### PUT /notes/{note_id}

Replaces the title and body of one of the user's own notes. Auth: required.

Body: `{"title": "...", "body": "..."}`. Returns the note, or `404`.

# Notely API

The backend of Notely, a note-taking app: a small FastAPI service with
accounts, notes and sharing.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Layout

- `app/main.py`: the FastAPI app and its routers
- `app/auth.py`: tokens, password hashing, the `require_auth` dependency
- `app/routes/notes.py`: the notes endpoints
- `app/routes/users.py`: sign-up, sign-in and the profile
- `app/storage.py`: in-memory storage
- `app/config.py`: settings
- `docs/api.md`: the HTTP API
- `docs/adr/`: architecture decision records
- `PROJECT_RULES.md`: the rules every change must follow
- `CHANGELOG.md`: what changed in each version

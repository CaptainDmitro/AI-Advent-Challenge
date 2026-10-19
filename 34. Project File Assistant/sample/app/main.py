"""The Notely API application."""

import logging

from fastapi import FastAPI

from app.config import APP_NAME, VERSION
from app.routes import notes, users

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, version=VERSION)
app.include_router(users.router)
app.include_router(notes.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}

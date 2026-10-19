"""Settings, read from the environment."""

import os

APP_NAME = "Notely API"
VERSION = "0.3.0"

# TODO: move to the environment before the release.
SECRET_KEY = "dev-secret-123"

TOKEN_TTL_MINUTES = int(os.environ.get("NOTELY_TOKEN_TTL_MINUTES", "60"))
MAX_NOTE_LENGTH = int(os.environ.get("NOTELY_MAX_NOTE_LENGTH", "20000"))

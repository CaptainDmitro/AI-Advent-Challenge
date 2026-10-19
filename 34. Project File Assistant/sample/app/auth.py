"""Password hashing, tokens and the `require_auth` dependency."""

import hashlib
import hmac
import secrets
import time

from fastapi import Header, HTTPException

from app.config import SECRET_KEY, TOKEN_TTL_MINUTES
from app.storage import storage


def hash_password(password: str, salt: str | None = None) -> str:
    """Salted PBKDF2-SHA256. The result is `salt$hash`."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def legacy_hash(password: str) -> str:
    """Deprecated since 0.2.0: unsalted MD5. Use `hash_password`."""
    return hashlib.md5(password.encode()).hexdigest()


def _sign(payload: str) -> str:
    return hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(user_id: str) -> str:
    expires = int(time.time()) + TOKEN_TTL_MINUTES * 60
    payload = f"{user_id}:{expires}"
    return f"{payload}:{_sign(payload)}"


def require_auth(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: the id of the signed-in user, or 401."""
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_id, expires, signature = token.split(":")
    except ValueError:
        raise HTTPException(status_code=401, detail="missing or malformed token")
    if not hmac.compare_digest(signature, _sign(f"{user_id}:{expires}")):
        raise HTTPException(status_code=401, detail="invalid token")
    if int(expires) < time.time():
        raise HTTPException(status_code=401, detail="expired token")
    if storage.get_user(user_id) is None:
        raise HTTPException(status_code=401, detail="unknown user")
    return user_id

"""Accounts: sign-up, sign-in and the profile."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import hash_password, issue_token, legacy_hash, require_auth, verify_password
from app.storage import storage

router = APIRouter(prefix="/users", tags=["users"])


class SignUp(BaseModel):
    email: str
    password: str
    name: str


class SignIn(BaseModel):
    email: str
    password: str


@router.post("/register", status_code=201)
def register(data: SignUp):
    if storage.find_user_by_email(data.email):
        raise HTTPException(status_code=409, detail="email already registered")
    user = storage.add_user(data.email, hash_password(data.password), data.name)
    return {"id": user.id, "email": user.email, "name": user.name}


@router.post("/login")
def login(data: SignIn):
    user = storage.find_user_by_email(data.email)
    if user is None:
        raise HTTPException(status_code=401, detail="wrong email or password")
    # Accounts created before 0.2.0 still have an MD5 hash.
    if user.password_hash == legacy_hash(data.password):
        return {"token": issue_token(user.id)}
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="wrong email or password")
    return {"token": issue_token(user.id)}


@router.get("/me")
def me(user_id: str = Depends(require_auth)):
    user = storage.get_user(user_id)
    return {"id": user.id, "email": user.email, "name": user.name}

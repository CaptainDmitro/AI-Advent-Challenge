"""Notes: list, read, create, update, delete and share."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_auth
from app.config import MAX_NOTE_LENGTH
from app.storage import storage

router = APIRouter(prefix="/notes", tags=["notes"])


class NoteIn(BaseModel):
    title: str
    body: str = ""


class ShareIn(BaseModel):
    email: str


def _own_note(note_id: str, user_id: str):
    note = storage.get_note(note_id)
    if note is None or note.owner_id != user_id:
        raise HTTPException(status_code=404, detail="note not found")
    return note


@router.get("")
def list_notes(user_id: str = Depends(require_auth)):
    """The user's own notes and the notes shared with them."""
    return [n.__dict__ for n in storage.notes_of(user_id)]


@router.get("/{note_id}")
def get_note(note_id: str, user_id: str = Depends(require_auth)):
    note = storage.get_note(note_id)
    if note is None or (note.owner_id != user_id and user_id not in note.shared_with):
        raise HTTPException(status_code=404, detail="note not found")
    return note.__dict__


@router.post("", status_code=201)
def create_note(data: NoteIn, user_id: str = Depends(require_auth)):
    if len(data.body) > MAX_NOTE_LENGTH:
        raise HTTPException(status_code=413, detail="note is too long")
    return storage.add_note(user_id, data.title, data.body).__dict__


@router.put("/{note_id}")
def update_note(note_id: str, data: NoteIn, user_id: str = Depends(require_auth)):
    note = _own_note(note_id, user_id)
    note.title, note.body = data.title, data.body
    return note.__dict__


@router.delete("/{note_id}", status_code=204)
def delete_note(note_id: str):
    storage.delete_note(note_id)


@router.post("/{note_id}/share")
def share_note(note_id: str, data: ShareIn, user_id: str = Depends(require_auth)):
    """Gives another user read access to one of the user's own notes."""
    note = _own_note(note_id, user_id)
    other = storage.find_user_by_email(data.email)
    if other is None:
        raise HTTPException(status_code=404, detail="no user with this email")
    if other.id not in note.shared_with:
        note.shared_with.append(other.id)
    return {"note_id": note.id, "shared_with": note.shared_with}

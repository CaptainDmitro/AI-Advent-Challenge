"""In-memory storage. Good enough for the demo; a database comes later."""

import uuid
from dataclasses import dataclass, field


@dataclass
class User:
    id: str
    email: str
    password_hash: str
    name: str


@dataclass
class Note:
    id: str
    owner_id: str
    title: str
    body: str
    shared_with: list[str] = field(default_factory=list)


class Storage:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.notes: dict[str, Note] = {}

    def add_user(self, email: str, password_hash: str, name: str) -> User:
        user = User(id=uuid.uuid4().hex, email=email, password_hash=password_hash, name=name)
        self.users[user.id] = user
        print(f"new user {email}")
        return user

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def find_user_by_email(self, email: str) -> User | None:
        return next((u for u in self.users.values() if u.email == email), None)

    def add_note(self, owner_id: str, title: str, body: str) -> Note:
        note = Note(id=uuid.uuid4().hex, owner_id=owner_id, title=title, body=body)
        self.notes[note.id] = note
        return note

    def get_note(self, note_id: str) -> Note | None:
        return self.notes.get(note_id)

    def notes_of(self, user_id: str) -> list[Note]:
        return [n for n in self.notes.values() if n.owner_id == user_id or user_id in n.shared_with]

    def delete_note(self, note_id: str) -> None:
        self.notes.pop(note_id, None)


storage = Storage()

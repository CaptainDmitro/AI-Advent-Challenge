from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sign_in(email="anna@example.com"):
    client.post("/users/register", json={"email": email, "password": "pw-123456", "name": "Anna"})
    token = client.post("/users/login", json={"email": email, "password": "pw-123456"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_create_and_list_notes():
    headers = sign_in()
    created = client.post("/notes", json={"title": "Groceries", "body": "milk"}, headers=headers)
    assert created.status_code == 201
    notes = client.get("/notes", headers=headers).json()
    assert [n["title"] for n in notes] == ["Groceries"]


def test_notes_need_a_token():
    assert client.get("/notes").status_code == 401

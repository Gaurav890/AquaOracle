"""End-to-end tests for the auth API: signup, login, logout, session checks.

Uses a fresh scratch SQLite file per test (via tmp_path), and resets the
lazily-created engine/session singletons in src.core.db between tests so
each test gets its own isolated database — this must never touch the real
project's data/metadata.db.
"""

import pytest
from fastapi.testclient import TestClient

from src.core.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "metadata_db_path", tmp_path / "test_metadata.db")
    monkeypatch.setattr(settings, "secret_key", "test-secret-key-not-a-real-fernet-key")

    import src.core.db as db_module
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    from src.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_signup_creates_user_and_logs_in(client):
    r = client.post("/api/auth/signup", json={
        "email": "alice@example.com", "password": "correcthorse123", "display_name": "Alice",
    })
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "alice@example.com"
    assert body["display_name"] == "Alice"
    assert "password" not in body and "password_hash" not in body


def test_signup_sets_a_working_session_cookie(client):
    client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "correcthorse123"})

    r = client.get("/api/auth/me")

    assert r.status_code == 200
    assert r.json()["email"] == "bob@example.com"


def test_duplicate_email_signup_is_rejected(client):
    client.post("/api/auth/signup", json={"email": "dup@example.com", "password": "correcthorse123"})

    r = client.post("/api/auth/signup", json={"email": "dup@example.com", "password": "anotherpass123"})

    assert r.status_code == 409


def test_login_with_correct_credentials(client):
    client.post("/api/auth/signup", json={"email": "carol@example.com", "password": "correcthorse123"})
    client.post("/api/auth/logout")

    r = client.post("/api/auth/login", json={"email": "carol@example.com", "password": "correcthorse123"})

    assert r.status_code == 200
    assert r.json()["email"] == "carol@example.com"


def test_login_with_wrong_password_is_rejected(client):
    client.post("/api/auth/signup", json={"email": "dave@example.com", "password": "correcthorse123"})

    r = client.post("/api/auth/login", json={"email": "dave@example.com", "password": "wrongpassword"})

    assert r.status_code == 401


def test_login_with_unknown_email_is_rejected(client):
    r = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever123"})

    assert r.status_code == 401


def test_me_without_session_is_unauthorized(client):
    r = client.get("/api/auth/me")

    assert r.status_code == 401


def test_logout_invalidates_the_session(client):
    client.post("/api/auth/signup", json={"email": "erin@example.com", "password": "correcthorse123"})

    logout_resp = client.post("/api/auth/logout")
    me_resp = client.get("/api/auth/me")

    assert logout_resp.status_code == 204
    assert me_resp.status_code == 401


def test_passwords_are_hashed_not_stored_in_plaintext(client, tmp_path):
    client.post("/api/auth/signup", json={"email": "frank@example.com", "password": "correcthorse123"})

    import sqlite3
    conn = sqlite3.connect(tmp_path / "test_metadata.db")
    row = conn.execute("SELECT password_hash FROM users WHERE email = ?", ("frank@example.com",)).fetchone()
    conn.close()

    assert row is not None
    assert row[0] != "correcthorse123"
    assert row[0].startswith("$2b$")  # bcrypt hash prefix

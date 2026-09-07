"""Tests for chat CRUD, ownership isolation, and message send/streaming.

Uses the same isolated-DB fixture pattern as test_auth.py, plus an isolated
scratch vector store path and mocked `ollama` calls so these never touch the
real project data or require a live Ollama server.
"""

import ollama
import pytest
from fastapi.testclient import TestClient

from src.core.config import settings
from src.indexing.metadata_store import MetadataStore


def _add_legacy_shared_document(doc_id: str = "legacy_doc") -> None:
    """A document with no owner is visible to every chat (see
    src/chat/scoping.py) — this is what CLI-ingested documents look like.
    Message-send tests need at least one visible document, or the chat
    short-circuits with a "no documents" note instead of generating."""
    MetadataStore(settings.full_metadata_db_path).add_document(
        doc_id, {"file_name": "legacy.pdf", "file_path": "/legacy.pdf", "page_count": 1}
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "metadata_db_path", tmp_path / "test_metadata.db")
    monkeypatch.setattr(settings, "vector_store_path", tmp_path / "test_vector_store")
    monkeypatch.setattr(settings, "secret_key", "test-secret-key-not-a-real-fernet-key")

    import src.core.db as db_module
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    from src.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def other_client(client, tmp_path, monkeypatch):
    """A second authenticated client sharing the same app/DB, for isolation tests."""
    # Reuse the same app instance the `client` fixture already built (same
    # settings overrides are still in effect), just sign up a second user.
    from src.api.main import create_app

    app2 = create_app()
    with TestClient(app2) as c2:
        c2.post("/api/auth/signup", json={"email": "other@example.com", "password": "correcthorse123"})
        yield c2


def _signed_up(client, email="owner@example.com"):
    client.post("/api/auth/signup", json={"email": email, "password": "correcthorse123"})
    return client


def test_create_and_list_chats(client):
    client = _signed_up(client)

    create_resp = client.post("/api/chats", json={"title": "My first chat"})
    list_resp = client.get("/api/chats")

    assert create_resp.status_code == 201
    assert create_resp.json()["title"] == "My first chat"
    assert create_resp.json()["provider"] == "ollama"
    chats = list_resp.json()
    assert len(chats) == 1
    assert chats[0]["id"] == create_resp.json()["id"]


def test_create_chat_without_title_gets_a_default(client):
    client = _signed_up(client)

    r = client.post("/api/chats", json={})

    assert r.status_code == 201
    assert r.json()["title"] == "New chat"


def test_rename_chat(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    r = client.patch(f"/api/chats/{chat_id}", json={"title": "Renamed"})

    assert r.status_code == 200
    assert r.json()["title"] == "Renamed"


def test_delete_chat_removes_it_and_its_messages(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    delete_resp = client.delete(f"/api/chats/{chat_id}")
    get_resp = client.get(f"/api/chats/{chat_id}")

    assert delete_resp.status_code == 204
    assert get_resp.status_code == 404


def test_chats_are_isolated_between_users(client, other_client):
    client = _signed_up(client)
    my_chat_id = client.post("/api/chats", json={"title": "Private"}).json()["id"]

    r = other_client.get(f"/api/chats/{my_chat_id}")

    assert r.status_code == 404


def test_other_user_cannot_delete_my_chat(client, other_client):
    client = _signed_up(client)
    my_chat_id = client.post("/api/chats", json={"title": "Private"}).json()["id"]

    r = other_client.delete(f"/api/chats/{my_chat_id}")
    still_there = client.get(f"/api/chats/{my_chat_id}")

    assert r.status_code == 404
    assert still_there.status_code == 200


def test_unauthenticated_request_is_rejected(client):
    r = client.post("/api/auth/logout")  # not signed up yet, but exercise a chats route
    r = client.get("/api/chats")

    assert r.status_code == 401


def test_send_message_with_no_documents_short_circuits_gracefully(client):
    """A chat with nothing to search (no chat-scoped uploads, no shared/legacy
    docs) should get a friendly note, not silently search the whole index or
    crash on an empty Qdrant filter."""
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "doesn't have any documents" in body

    history = client.get(f"/api/chats/{chat_id}/messages").json()
    assert len(history) == 2
    assert "doesn't have any documents" in history[1]["content"]


def test_send_message_streams_and_persists(client, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama,
        "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "Hello"}, {"response": " there"}])
            if stream
            else {"response": "Hello there"}
        ),
    )

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: token" in body or "event: done" in body

    history = client.get(f"/api/chats/{chat_id}/messages").json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hi"
    assert history[1]["role"] == "assistant"


def test_new_chat_title_becomes_the_first_message(client, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama,
        "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "ok"}]) if stream else {"response": "ok"}
        ),
    )

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "What is Legionella?"}):
        pass

    chat = client.get(f"/api/chats/{chat_id}").json()
    assert chat["title"] == "What is Legionella?"


def test_set_chat_provider(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    r = client.patch(f"/api/chats/{chat_id}/provider", json={"provider": "openai", "model": "gpt-4o"})

    assert r.status_code == 200
    assert r.json()["provider"] == "openai"
    assert r.json()["model"] == "gpt-4o"

    chat = client.get(f"/api/chats/{chat_id}").json()
    assert chat["provider"] == "openai"
    assert chat["model"] == "gpt-4o"


def test_set_chat_provider_rejects_unknown_provider(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]

    r = client.patch(f"/api/chats/{chat_id}/provider", json={"provider": "not-a-real-provider"})

    assert r.status_code == 400


def test_send_message_on_cloud_provider_without_api_key_gets_a_friendly_error(client):
    """Selecting OpenAI/Anthropic with no key on file must not leak a raw SDK
    exception to the client or crash the stream — it should behave like any
    other generation failure: a clear 'error' SSE event, persisted so a
    reload still shows why the message failed."""
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()
    client.patch(f"/api/chats/{chat_id}/provider", json={"provider": "openai"})

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: error" in body
    assert "No OpenAI API key on file" in body

    history = client.get(f"/api/chats/{chat_id}/messages").json()
    assert len(history) == 2
    assert "No OpenAI API key on file" in history[1]["content"]

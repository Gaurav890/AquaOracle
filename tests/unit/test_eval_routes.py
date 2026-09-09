"""Tests for the /api/eval/* endpoints: per-user scoping, admin-only
cross-user access, and summary aggregation math."""

import ollama
import pytest
from fastapi.testclient import TestClient

from src.auth.models import User
from src.core.config import settings
from src.indexing.metadata_store import MetadataStore


def _add_legacy_shared_document(doc_id: str = "legacy_doc") -> None:
    MetadataStore(settings.full_metadata_db_path).add_document(
        doc_id, {"file_name": "legacy.pdf", "file_path": "/legacy.pdf", "page_count": 1}
    )


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "metadata_db_path", tmp_path / "test_metadata.db")
    monkeypatch.setattr(settings, "vector_store_path", tmp_path / "test_vector_store")
    monkeypatch.setattr(settings, "secret_key", "test-secret-key-not-a-real-fernet-key")

    import src.core.db as db_module
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    from src.api.main import create_app

    return create_app


@pytest.fixture
def client(app_factory):
    with TestClient(app_factory()) as c:
        yield c


def _signed_up(client, email="owner@example.com"):
    client.post("/api/auth/signup", json={"email": email, "password": "correcthorse123"})
    return client


def _send_a_message(client, monkeypatch, provider="ollama"):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()
    if provider != "ollama":
        client.patch(f"/api/chats/{chat_id}/provider", json={"provider": provider})

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama, "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "Answer [1]"}]) if stream else {"response": "Answer [1]"}
        ),
    )
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass
    return chat_id


def test_summary_scope_mine_only_shows_own_queries(client, monkeypatch):
    client = _signed_up(client)
    _send_a_message(client, monkeypatch)

    r = client.get("/api/eval/summary?scope=mine")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["provider"] == "ollama"
    assert body[0]["count"] == 1


def test_summary_scope_all_forbidden_for_non_admin(client, monkeypatch):
    client = _signed_up(client)
    _send_a_message(client, monkeypatch)

    r = client.get("/api/eval/summary?scope=all")
    assert r.status_code == 403


def test_summary_scope_all_shows_every_users_data_for_admin(app_factory, monkeypatch):
    with TestClient(app_factory()) as client:
        client = _signed_up(client, "alice@example.com")
        _send_a_message(client, monkeypatch)

    with TestClient(app_factory()) as admin_client:
        admin_client = _signed_up(admin_client, "bob@example.com")
        _send_a_message(admin_client, monkeypatch)

        import src.core.db as db_module
        session = db_module._SessionLocal()
        session.query(User).filter(User.email == "bob@example.com").update({"is_admin": True})
        session.commit()
        session.close()

        r = admin_client.get("/api/eval/summary?scope=all")
        assert r.status_code == 200
        assert r.json()[0]["count"] == 2  # both alice's and bob's queries


def test_invalid_scope_rejected(client):
    client = _signed_up(client)
    r = client.get("/api/eval/summary?scope=sideways")
    assert r.status_code == 400


def test_logs_endpoint_includes_all_six_dimensions(client, monkeypatch):
    client = _signed_up(client)
    chat_id = _send_a_message(client, monkeypatch)

    r = client.get("/api/eval/logs?scope=mine")
    assert r.status_code == 200
    logs = r.json()
    assert len(logs) == 1
    log = logs[0]
    assert log["chat_id"] == chat_id
    assert "doc_ids_allowed" in log["scope"]
    assert "external_call" in log["authority"]
    assert "prompt_tokens" in log["budget"]
    assert "ungrounded_citation_indices" in log["verification"]
    assert log["containment"] is None


def test_logs_endpoint_respects_limit(client, monkeypatch):
    client = _signed_up(client)
    _send_a_message(client, monkeypatch)
    _send_a_message(client, monkeypatch)

    r = client.get("/api/eval/logs?scope=mine&limit=1")
    assert len(r.json()) == 1


def test_summary_groundedness_rate_reflects_ungrounded_citations(client, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    # Cites [7], which can't exist — only 1 legacy doc's chunks are in scope.
    monkeypatch.setattr(
        ollama, "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "Answer [7]"}]) if stream else {"response": "Answer [7]"}
        ),
    )
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass

    r = client.get("/api/eval/summary?scope=mine")
    assert r.json()[0]["groundedness_pass_rate"] == 0.0

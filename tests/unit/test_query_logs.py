"""Tests for QueryLog persistence: every send_message call — success or
deviation — must produce exactly one row with the right shape, and the
feedback endpoint must update it correctly.
"""

import json

import ollama
import pytest
from fastapi.testclient import TestClient

from src.core.config import settings
from src.eval.models import QueryLog
from src.indexing.metadata_store import MetadataStore


def _add_legacy_shared_document(doc_id: str = "legacy_doc") -> None:
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


def _signed_up(client, email="owner@example.com"):
    client.post("/api/auth/signup", json={"email": email, "password": "correcthorse123"})
    return client


def _query_logs():
    import src.core.db as db_module

    session = db_module._SessionLocal()
    try:
        return session.query(QueryLog).all()
    finally:
        session.close()


def test_successful_message_creates_one_query_log_with_full_shape(client, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama,
        "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "Answer [1]"}, {"response": "", "done": True, "prompt_eval_count": 40, "eval_count": 8}])
            if stream
            else {"response": "Answer [1]"}
        ),
    )

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass

    logs = _query_logs()
    assert len(logs) == 1
    log = logs[0]
    assert log.chat_id == chat_id
    assert log.provider == "ollama"
    assert log.containment_json is None

    scope = json.loads(log.scope_json)
    assert scope["doc_ids_allowed"] == ["legacy_doc"]

    authority = json.loads(log.authority_json)
    assert authority == {"provider": "ollama", "external_call": False, "persisted": True}

    budget = json.loads(log.budget_json)
    assert budget["prompt_tokens"] == 40
    assert budget["completion_tokens"] == 8
    assert budget["estimated_cost_usd"] == 0.0  # ollama is always free

    verification = json.loads(log.verification_json)
    assert verification["cited_indices"] == [1]

    assert log.raw_metadata_json is not None
    assert log.user_feedback is None


def test_no_documents_short_circuit_records_containment_event(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    # No document added — chat has nothing visible.

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass

    logs = _query_logs()
    assert len(logs) == 1
    containment = json.loads(logs[0].containment_json)
    assert containment["event"] == "no_documents"


def test_missing_api_key_records_containment_event(client):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()
    client.patch(f"/api/chats/{chat_id}/provider", json={"provider": "openai"})

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass

    logs = _query_logs()
    assert len(logs) == 1
    containment = json.loads(logs[0].containment_json)
    assert containment["event"] == "missing_api_key"
    assert logs[0].provider == "openai"


def test_feedback_updates_the_query_log(client, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama, "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "ok"}]) if stream else {"response": "ok"}
        ),
    )

    message_id = None
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:") and "message_id" in line:
                message_id = json.loads(line[len("data:"):])["message_id"]

    assert message_id is not None

    r = client.post(f"/api/messages/{message_id}/feedback", json={"rating": "up", "comment": "nice"})
    assert r.status_code == 204

    logs = _query_logs()
    assert logs[0].user_feedback == "up"
    assert logs[0].user_feedback_comment == "nice"
    assert logs[0].feedback_at is not None


def test_feedback_rejects_invalid_rating(client):
    client = _signed_up(client)
    r = client.post("/api/messages/does-not-exist/feedback", json={"rating": "sideways"})
    assert r.status_code == 422


def test_feedback_404s_for_unknown_message(client):
    client = _signed_up(client)
    r = client.post("/api/messages/does-not-exist/feedback", json={"rating": "up"})
    assert r.status_code == 404


def test_feedback_404s_for_another_users_message(client, tmp_path, monkeypatch):
    client = _signed_up(client)
    chat_id = client.post("/api/chats", json={}).json()["id"]
    _add_legacy_shared_document()

    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.1] * 768})
    monkeypatch.setattr(
        ollama, "generate",
        lambda model, prompt="", system=None, options=None, stream=False, **kw: (
            iter([{"response": "ok"}]) if stream else {"response": "ok"}
        ),
    )
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"message": "Hi"}):
        pass

    message_id = _query_logs()[0].message_id

    from src.api.main import create_app
    other = TestClient(create_app())
    other.post("/api/auth/signup", json={"email": "other@example.com", "password": "correcthorse123"})

    r = other.post(f"/api/messages/{message_id}/feedback", json={"rating": "up"})
    assert r.status_code == 404

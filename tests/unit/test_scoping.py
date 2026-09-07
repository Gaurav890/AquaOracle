"""Tests for per-chat document scoping resolution."""

import pytest

from src.auth.models import User
from src.chat.models import Chat, ChatDocument
from src.chat.scoping import get_allowed_doc_ids
from src.core.config import settings
from src.indexing.metadata_store import MetadataStore


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "metadata_db_path", tmp_path / "test_metadata.db")

    import src.core.db as db_module
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    db_module.init_db()
    db_module._get_engine()
    session = db_module._SessionLocal()
    yield session
    session.close()


@pytest.fixture
def metadata_store(tmp_path):
    return MetadataStore(tmp_path / "test_metadata.db")


def _make_chat(db_session, chat_id: str, user_id: str) -> None:
    """chat_documents.chat_id and chats.user_id both have real FK
    constraints (foreign_keys=ON) — a valid User and Chat row must exist
    first."""
    if db_session.get(User, user_id) is None:
        db_session.add(User(id=user_id, email=f"{user_id}@example.com", password_hash="x"))
    db_session.add(Chat(id=chat_id, user_id=user_id))
    db_session.commit()


def test_chat_scoped_upload_is_visible_to_that_chat(db_session, metadata_store):
    _make_chat(db_session, "chat1", "u1")
    metadata_store.add_document(
        "docA", {"file_name": "a.pdf", "file_path": "/a", "page_count": 1, "owner_user_id": "u1"}
    )
    db_session.add(ChatDocument(chat_id="chat1", doc_id="docA"))
    db_session.commit()

    allowed = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat1", user_id="u1")

    assert allowed == ["docA"]


def test_chat_scoped_upload_is_not_visible_to_a_different_chat(db_session, metadata_store):
    _make_chat(db_session, "chat1", "u1")
    _make_chat(db_session, "chat2", "u1")
    metadata_store.add_document(
        "docA", {"file_name": "a.pdf", "file_path": "/a", "page_count": 1, "owner_user_id": "u1"}
    )
    db_session.add(ChatDocument(chat_id="chat1", doc_id="docA"))
    db_session.commit()

    allowed = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat2", user_id="u1")

    assert allowed == []


def test_shared_document_is_visible_to_every_chat(db_session, metadata_store):
    metadata_store.add_document(
        "docShared",
        {"file_name": "s.pdf", "file_path": "/s", "page_count": 1, "owner_user_id": "u1", "is_shared": True},
    )

    allowed_chat1 = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat1", user_id="u1")
    allowed_chat2 = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat2", user_id="u1")

    assert allowed_chat1 == ["docShared"]
    assert allowed_chat2 == ["docShared"]


def test_legacy_cli_ingested_doc_is_visible_to_every_chat(db_session, metadata_store):
    metadata_store.add_document("legacy", {"file_name": "l.pdf", "file_path": "/l", "page_count": 1})

    allowed = get_allowed_doc_ids(db_session, metadata_store, chat_id="any_chat", user_id="u1")

    assert allowed == ["legacy"]


def test_chat_with_its_own_document_excludes_the_legacy_corpus(db_session, metadata_store):
    """Regression test: a chat scoped to a specific uploaded document must
    NOT also pull in the (potentially huge) legacy/shared corpus — a small,
    specific upload was getting drowned out by thousands of unrelated
    legacy chunks for any query that didn't happen to match it exactly."""
    _make_chat(db_session, "chat1", "u1")
    metadata_store.add_document(
        "docA", {"file_name": "a.pdf", "file_path": "/a", "page_count": 1, "owner_user_id": "u1"}
    )
    db_session.add(ChatDocument(chat_id="chat1", doc_id="docA"))
    db_session.commit()
    metadata_store.add_document("legacy", {"file_name": "l.pdf", "file_path": "/l", "page_count": 1})

    allowed = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat1", user_id="u1")

    assert allowed == ["docA"]


def test_another_users_private_document_is_not_visible(db_session, metadata_store):
    metadata_store.add_document(
        "theirs", {"file_name": "t.pdf", "file_path": "/t", "page_count": 1, "owner_user_id": "u2"}
    )

    allowed = get_allowed_doc_ids(db_session, metadata_store, chat_id="chat1", user_id="u1")

    assert allowed == []

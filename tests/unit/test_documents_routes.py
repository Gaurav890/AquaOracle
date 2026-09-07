"""Tests for the document-file-serving endpoint: visibility rules and path
resolution (CLI-ingested legacy docs store a path relative to the repo
root; web uploads store an absolute one — both must resolve correctly
regardless of the server process's cwd)."""

import pytest
from fastapi.testclient import TestClient

from src.core.config import settings
from src.indexing.metadata_store import MetadataStore


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
        c.post("/api/auth/signup", json={"email": "owner@example.com", "password": "correcthorse123"})
        yield c


def _metadata_store():
    return MetadataStore(settings.full_metadata_db_path)


def test_owner_can_fetch_their_own_document(client, tmp_path):
    me = client.get("/api/auth/me").json()
    pdf_path = tmp_path / "mine.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    _metadata_store().add_document(
        "docA",
        {"file_name": "mine.pdf", "file_path": str(pdf_path), "page_count": 1, "owner_user_id": me["id"]},
    )

    r = client.get("/api/documents/docA/file")

    assert r.status_code == 200
    assert r.content == b"%PDF-1.4 fake content"
    assert r.headers["content-type"] == "application/pdf"
    assert "inline" in r.headers["content-disposition"]


def test_legacy_document_with_no_owner_is_visible_to_anyone(client, tmp_path, monkeypatch):
    """Mirrors how rag ingest stores file_path: relative to the repo root,
    not absolute — must still resolve correctly."""
    fake_root = tmp_path / "repo"
    (fake_root / "soc").mkdir(parents=True)
    (fake_root / "soc" / "legacy.pdf").write_bytes(b"%PDF legacy content")
    monkeypatch.setattr(settings, "project_root", fake_root)

    _metadata_store().add_document(
        "legacy", {"file_name": "legacy.pdf", "file_path": "soc/legacy.pdf", "page_count": 1}
    )
    r = client.get("/api/documents/legacy/file")

    assert r.status_code == 200
    assert r.content == b"%PDF legacy content"


def test_another_users_private_document_returns_404(client, tmp_path):
    pdf_path = tmp_path / "theirs.pdf"
    pdf_path.write_bytes(b"%PDF theirs")
    _metadata_store().add_document(
        "theirs", {"file_name": "theirs.pdf", "file_path": str(pdf_path), "page_count": 1, "owner_user_id": "u2"}
    )

    r = client.get("/api/documents/theirs/file")

    assert r.status_code == 404


def test_nonexistent_document_returns_404(client):
    r = client.get("/api/documents/does-not-exist/file")

    assert r.status_code == 404


def test_document_missing_from_disk_returns_404(client):
    _metadata_store().add_document(
        "gone", {"file_name": "gone.pdf", "file_path": "/nonexistent/gone.pdf", "page_count": 1}
    )

    r = client.get("/api/documents/gone/file")

    assert r.status_code == 404

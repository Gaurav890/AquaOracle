"""End-to-end tests for provider API key management: store/list/delete,
masking, and that the raw key is never echoed back.

Uses a real Fernet key (unlike test_auth.py/test_chats.py's placeholder
string) since these tests actually exercise encrypt/decrypt.
"""

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src.core.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "metadata_db_path", tmp_path / "test_metadata.db")
    monkeypatch.setattr(settings, "secret_key", Fernet.generate_key().decode())

    import src.core.db as db_module
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_SessionLocal", None)

    from src.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        c.post("/api/auth/signup", json={"email": "byok@example.com", "password": "correcthorse123"})
        yield c


def test_no_keys_configured_initially(client):
    assert client.get("/api/settings/api-keys").json() == []


def test_set_and_list_api_key_never_returns_raw_value(client):
    r = client.put("/api/settings/api-keys", json={"provider": "openai", "api_key": "sk-supersecretvalue"})
    assert r.status_code == 204

    listed = client.get("/api/settings/api-keys").json()
    assert len(listed) == 1
    assert listed[0]["provider"] == "openai"
    assert "sk-supersecretvalue" not in listed[0]["masked_key"]
    assert listed[0]["masked_key"].startswith("sk-s")


def test_setting_key_twice_updates_rather_than_duplicates(client):
    client.put("/api/settings/api-keys", json={"provider": "openai", "api_key": "sk-firstvalue"})
    client.put("/api/settings/api-keys", json={"provider": "openai", "api_key": "sk-secondvalue"})

    listed = client.get("/api/settings/api-keys").json()
    assert len(listed) == 1
    assert listed[0]["masked_key"] == "sk-s...alue"  # from "sk-secondvalue", not the first-set value


def test_delete_api_key(client):
    client.put("/api/settings/api-keys", json={"provider": "anthropic", "api_key": "sk-ant-value"})
    r = client.delete("/api/settings/api-keys/anthropic")
    assert r.status_code == 204
    assert client.get("/api/settings/api-keys").json() == []


def test_delete_nonexistent_key_returns_404(client):
    r = client.delete("/api/settings/api-keys/openai")
    assert r.status_code == 404


def test_setting_ollama_as_a_provider_is_rejected(client):
    """Ollama never has a stored key — it's always local, no key needed."""
    r = client.put("/api/settings/api-keys", json={"provider": "ollama", "api_key": "irrelevant"})
    assert r.status_code == 400


def test_setting_unknown_provider_is_rejected(client):
    r = client.put("/api/settings/api-keys", json={"provider": "not-a-real-provider", "api_key": "x"})
    assert r.status_code == 400


def test_api_keys_are_isolated_per_user(client, tmp_path):
    client.put("/api/settings/api-keys", json={"provider": "openai", "api_key": "sk-alice-key"})

    from src.api.main import create_app

    app2 = create_app()
    with TestClient(app2) as other:
        other.post("/api/auth/signup", json={"email": "bob@example.com", "password": "correcthorse123"})
        assert other.get("/api/settings/api-keys").json() == []

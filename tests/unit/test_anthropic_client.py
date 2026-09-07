"""Tests for AnthropicClient, with the underlying SDK call monkeypatched so
these never make a real network request."""

from types import SimpleNamespace

from src.generation.anthropic_client import AnthropicClient


def _fake_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeStreamContext:
    def __init__(self, pieces):
        self.text_stream = iter(pieces)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_generate_returns_joined_text_blocks(monkeypatch):
    client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-5")
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: _fake_response("Hello there"))

    assert client.generate("hi") == "Hello there"


def test_generate_passes_system_and_user_message(monkeypatch):
    client = AnthropicClient(api_key="sk-ant-test")
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return _fake_response("ok")

    monkeypatch.setattr(client._client.messages, "create", fake_create)
    client.generate("hi", system_prompt="be terse")

    assert captured["system"] == "be terse"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_stream_yields_text_pieces(monkeypatch):
    client = AnthropicClient(api_key="sk-ant-test")
    monkeypatch.setattr(client._client.messages, "stream", lambda **kw: _FakeStreamContext(["Hel", "lo"]))

    assert list(client.generate_stream("hi")) == ["Hel", "lo"]

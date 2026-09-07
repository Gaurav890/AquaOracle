"""Tests for OpenAIClient, with the underlying SDK call monkeypatched so
these never make a real network request."""

from types import SimpleNamespace

from src.generation.openai_client import OpenAIClient


def _fake_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _fake_stream(pieces):
    for piece in pieces:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


def test_generate_returns_message_content(monkeypatch):
    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kw: _fake_response("Hello there")
    )

    assert client.generate("hi") == "Hello there"


def test_generate_passes_system_prompt_as_message(monkeypatch):
    client = OpenAIClient(api_key="sk-test")
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return _fake_response("ok")

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    client.generate("hi", system_prompt="be terse")

    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["messages"][1] == {"role": "user", "content": "hi"}


def test_generate_stream_yields_text_deltas(monkeypatch):
    client = OpenAIClient(api_key="sk-test")
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kw: _fake_stream(["Hel", "lo"])
    )

    assert list(client.generate_stream("hi")) == ["Hel", "lo"]


def test_generate_stream_skips_empty_deltas(monkeypatch):
    """The SDK sends a final chunk with delta.content=None to signal end of
    stream — that must not be yielded as a literal "None" string."""
    client = OpenAIClient(api_key="sk-test")
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kw: _fake_stream(["Hi", None])
    )

    assert list(client.generate_stream("hi")) == ["Hi"]

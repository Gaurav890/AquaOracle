"""Tests for AnthropicClient, with the underlying SDK call monkeypatched so
these never make a real network request."""

from types import SimpleNamespace

from src.generation.anthropic_client import AnthropicClient


def _fake_response(text: str, usage=None):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)


class _FakeStreamContext:
    def __init__(self, pieces, final_usage=None):
        self.text_stream = iter(pieces)
        self._final_usage = final_usage

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(usage=self._final_usage)


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


def test_generate_reports_usage_via_callback(monkeypatch):
    client = AnthropicClient(api_key="sk-ant-test")
    usage = SimpleNamespace(input_tokens=80, output_tokens=15)
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: _fake_response("ok", usage=usage))

    captured = []
    client.generate("hi", on_usage=captured.append)

    assert len(captured) == 1
    assert captured[0].prompt_tokens == 80
    assert captured[0].completion_tokens == 15


def test_generate_stream_reports_usage_via_callback_after_stream_ends(monkeypatch):
    client = AnthropicClient(api_key="sk-ant-test")
    usage = SimpleNamespace(input_tokens=40, output_tokens=8)
    monkeypatch.setattr(
        client._client.messages, "stream", lambda **kw: _FakeStreamContext(["Hi"], final_usage=usage)
    )

    captured = []
    pieces = list(client.generate_stream("hi", on_usage=captured.append))

    assert pieces == ["Hi"]
    assert len(captured) == 1
    assert captured[0].prompt_tokens == 40
    assert captured[0].completion_tokens == 8

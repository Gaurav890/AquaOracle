"""Tests for OpenAIClient, with the underlying SDK call monkeypatched so
these never make a real network request."""

from types import SimpleNamespace

from src.generation.openai_client import OpenAIClient


def _fake_response(text: str, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)


def _fake_stream(pieces, final_usage=None):
    for piece in pieces:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))], usage=None)
    if final_usage:
        # Real behavior with stream_options={"include_usage": True}: a final
        # chunk with empty choices carrying only usage.
        yield SimpleNamespace(choices=[], usage=final_usage)


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


def test_generate_reports_usage_via_callback(monkeypatch):
    client = OpenAIClient(api_key="sk-test")
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: _fake_response("ok", usage=usage))

    captured = []
    client.generate("hi", on_usage=captured.append)

    assert len(captured) == 1
    assert captured[0].prompt_tokens == 100
    assert captured[0].completion_tokens == 20


def test_generate_stream_reports_usage_via_callback_from_final_chunk(monkeypatch):
    client = OpenAIClient(api_key="sk-test")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10)
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kw: _fake_stream(["Hi"], final_usage=usage)
    )

    captured = []
    pieces = list(client.generate_stream("hi", on_usage=captured.append))

    assert pieces == ["Hi"]
    assert len(captured) == 1
    assert captured[0].prompt_tokens == 50
    assert captured[0].completion_tokens == 10


def test_generate_stream_requests_usage_in_stream_options(monkeypatch):
    """Regression test: OpenAI only populates a final usage chunk when
    stream_options={"include_usage": True} is explicitly requested."""
    client = OpenAIClient(api_key="sk-test")
    captured = {}

    def fake_create(**kw):
        captured.update(kw)
        return _fake_stream(["Hi"])

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    list(client.generate_stream("hi"))

    assert captured["stream_options"] == {"include_usage": True}


def test_generate_without_callback_does_not_touch_usage(monkeypatch):
    """Regression test: accessing chunk.usage unconditionally (before
    checking on_usage) would raise on any object that doesn't define it."""
    client = OpenAIClient(api_key="sk-test")

    class NoUsageAttr:
        choices = [SimpleNamespace(delta=SimpleNamespace(content="Hi"))]
        # deliberately no `usage` attribute

    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: iter([NoUsageAttr()]))

    assert list(client.generate_stream("hi")) == ["Hi"]

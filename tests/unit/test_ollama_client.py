"""Tests for OllamaClient's usage-reporting: the `ollama` package's raw
response already includes prompt_eval_count/eval_count, but the client used
to discard them entirely."""

import ollama
import pytest

from src.generation.ollama_client import OllamaClient


@pytest.fixture
def client(monkeypatch):
    # OllamaClient._test_connection() calls ollama.generate() at construction
    # time — must be mocked before the client is built.
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": "ok"})
    return OllamaClient(model="llama3.1:8b")


def test_generate_reports_usage_via_callback(client, monkeypatch):
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: {"response": "Hello there", "prompt_eval_count": 120, "eval_count": 30},
    )

    captured = []
    answer = client.generate("hi", on_usage=captured.append)

    assert answer == "Hello there"
    assert len(captured) == 1
    assert captured[0].prompt_tokens == 120
    assert captured[0].completion_tokens == 30


def test_generate_without_usage_fields_reports_none(client, monkeypatch):
    """Older Ollama versions/models might not report these — must not raise."""
    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": "Hello there"})

    captured = []
    client.generate("hi", on_usage=captured.append)

    assert captured[0].prompt_tokens is None
    assert captured[0].completion_tokens is None


def test_generate_stream_reports_usage_from_final_chunk(client, monkeypatch):
    def fake_generate(**kw):
        return iter([
            {"response": "Hel"},
            {"response": "lo"},
            {"response": "", "done": True, "prompt_eval_count": 90, "eval_count": 12},
        ])

    monkeypatch.setattr(ollama, "generate", fake_generate)

    captured = []
    pieces = list(client.generate_stream("hi", on_usage=captured.append))

    assert pieces == ["Hel", "lo", ""]
    assert len(captured) == 1
    assert captured[0].prompt_tokens == 90
    assert captured[0].completion_tokens == 12


def test_generate_stream_without_callback_does_not_raise(client, monkeypatch):
    monkeypatch.setattr(
        ollama, "generate",
        lambda **kw: iter([{"response": "Hi"}, {"response": "", "done": True, "eval_count": 5}]),
    )

    assert list(client.generate_stream("hi")) == ["Hi", ""]

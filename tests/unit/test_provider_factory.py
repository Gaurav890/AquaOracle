"""Tests for the provider-dispatch factory: correct client type per provider,
and a clear error rather than a confusing SDK exception when a cloud
provider is selected with no API key on file."""

import pytest

from src.generation.anthropic_client import AnthropicClient
from src.generation.ollama_client import OllamaClient
from src.generation.openai_client import OpenAIClient
from src.generation.provider_factory import MissingApiKeyError, create_llm_client


def test_ollama_provider_returns_ollama_client(monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "generate", lambda **kw: {"response": "ok"})
    client = create_llm_client("ollama", model="llama3.1:8b")
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.1:8b"


def test_openai_provider_returns_openai_client():
    client = create_llm_client("openai", api_key="sk-test", model="gpt-4o-mini")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"


def test_openai_provider_without_key_raises_missing_api_key_error():
    with pytest.raises(MissingApiKeyError):
        create_llm_client("openai")


def test_anthropic_provider_returns_anthropic_client():
    client = create_llm_client("anthropic", api_key="sk-ant-test", model="claude-sonnet-5")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5"


def test_anthropic_provider_without_key_raises_missing_api_key_error():
    with pytest.raises(MissingApiKeyError):
        create_llm_client("anthropic")


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        create_llm_client("not-a-real-provider")

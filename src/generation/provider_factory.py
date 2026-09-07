"""Dispatches a provider name to a live LLMClient.

Called fresh per chat-send request — cheap for cloud providers since neither
OpenAIClient nor AnthropicClient does an eager connection test.
"""

from typing import Optional

from src.core.config import settings
from src.generation.anthropic_client import AnthropicClient
from src.generation.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from src.generation.base_client import LLMClient
from src.generation.ollama_client import OllamaClient
from src.generation.openai_client import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from src.generation.openai_client import OpenAIClient

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")


class MissingApiKeyError(ValueError):
    """Raised when a cloud provider is selected but no API key is on file."""


def create_llm_client(
    provider: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(
            model=model or settings.ollama_llm_model,
            host=settings.ollama_host,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        if not api_key:
            raise MissingApiKeyError("No OpenAI API key on file. Add one in Settings.")
        return OpenAIClient(
            api_key=api_key,
            model=model or OPENAI_DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        if not api_key:
            raise MissingApiKeyError("No Anthropic API key on file. Add one in Settings.")
        return AnthropicClient(
            api_key=api_key,
            model=model or ANTHROPIC_DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: {provider!r} (expected one of {SUPPORTED_PROVIDERS})")

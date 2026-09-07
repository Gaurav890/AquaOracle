"""Anthropic LLM client (opt-in BYOK — only the live prompt for this request
is sent to Anthropic; documents, embeddings, and chat history stay local)."""

from typing import Iterator, Optional

from anthropic import Anthropic
from loguru import logger

from src.generation.base_client import LLMClient

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicClient(LLMClient):
    """Client for Anthropic message generation.

    Like OpenAIClient, does not eagerly test the connection on construction.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.logger = logger.bind(name="AnthropicClient")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                model=self.model,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise

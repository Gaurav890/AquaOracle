"""OpenAI LLM client (opt-in BYOK — only the live prompt for this request is
sent to OpenAI; documents, embeddings, and chat history stay local)."""

from typing import Iterator, Optional

from loguru import logger
from openai import OpenAI

from src.generation.base_client import LLMClient

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(LLMClient):
    """Client for OpenAI chat completion generation.

    Unlike OllamaClient, this does NOT eagerly test the connection on
    construction — that would cost the user a billed API call just to build
    the object, and a fresh client is built per chat-send request.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.logger = logger.bind(name="OpenAIClient")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def _messages(self, prompt: str, system_prompt: Optional[str]) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system_prompt),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content or ""
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
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system_prompt),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise

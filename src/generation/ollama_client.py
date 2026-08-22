"""Ollama LLM client for text generation."""

from typing import Optional, Iterator
import ollama
from loguru import logger


class OllamaClient:
    """Client for Ollama LLM generation."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize Ollama client.

        Args:
            model: LLM model name
            host: Ollama server host
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.logger = logger.bind(name="OllamaClient")
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            # Test with a simple prompt
            response = ollama.generate(
                model=self.model,
                prompt="Hello",
                options={"num_predict": 10}
            )
            self.logger.info(f"Connected to Ollama with model {self.model}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Generated text
        """
        try:
            # Prepare options
            options = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }

            # Generate response
            if system_prompt:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    options=options,
                )
            else:
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options=options,
                )

            return response['response']

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature

        Yields:
            Generated text chunks
        """
        try:
            options = {
                "temperature": temperature or self.temperature,
            }

            if system_prompt:
                stream = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    options=options,
                    stream=True,
                )
            else:
                stream = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options=options,
                    stream=True,
                )

            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise

    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        try:
            options = {
                "temperature": temperature or self.temperature,
            }

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=options,
            )

            return response['message']['content']

        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            raise

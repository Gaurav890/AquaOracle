"""Abstract LLM client interface.

ResponseGenerator only ever calls .generate()/.generate_stream() and reads
.model (confirmed against src/generation/response_generator.py) — so this is
the entire surface a provider client needs to implement.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class LLMClient(ABC):
    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        ...

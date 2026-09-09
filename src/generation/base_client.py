"""Abstract LLM client interface.

ResponseGenerator only ever calls .generate()/.generate_stream() and reads
.model (confirmed against src/generation/response_generator.py) — so this is
the entire surface a provider client needs to implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass
class UsageInfo:
    """Token usage for one generate()/generate_stream() call, when the
    provider reports it. Either field may be None (e.g. an older Ollama
    that doesn't return eval counts)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# Called once, after generation completes, with that call's usage — a plain
# closure scoped to one request, not stored state, so it's safe to pass even
# when the underlying client is a shared warm singleton (see OllamaClient's
# module-level reuse in src/api/services/chat_service.py).
UsageCallback = Callable[[UsageInfo], None]


class LLMClient(ABC):
    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_usage: Optional[UsageCallback] = None,
    ) -> str:
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_usage: Optional[UsageCallback] = None,
    ) -> Iterator[str]:
        ...

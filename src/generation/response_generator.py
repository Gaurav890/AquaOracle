"""Complete RAG response generation."""

import re
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_templates import SYSTEM_PROMPT, build_rag_prompt
from src.utils.citations import format_page_citation


@dataclass
class RAGResponse:
    """Complete RAG response with answer and citations."""

    answer: str
    sources: list
    formatted_sources: str
    metadata: Dict[str, Any]
    retrieval_result: Optional[RetrievalResult] = None


class ResponseGenerator:
    """
    Complete RAG response generator.

    Orchestrates retrieval and generation to produce
    answers with citations.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: OllamaClient,
    ):
        """
        Initialize response generator.

        Args:
            retrieval_pipeline: Retrieval pipeline instance
            llm_client: LLM client for generation
        """
        self.logger = logger.bind(name="ResponseGenerator")
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        top_k: int = 50,
        top_n: int = 10,
        doc_filter: Optional[Dict[str, Any]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> RAGResponse:
        """
        Generate complete RAG response.

        Args:
            question: User's question
            top_k: Vector search top K
            top_n: Re-ranking top N
            doc_filter: Filter by document
            on_token: If given, stream generation and call this with each
                text chunk as it arrives (e.g. to print progressively)
                instead of blocking until the full answer is ready. Total
                generation time is unchanged either way — this only affects
                when the caller sees output.

        Returns:
            RAGResponse with answer and citations
        """
        self.logger.info(f"Generating response for: {question[:50]}...")

        # Step 1: Retrieve relevant context
        retrieval_result = self.retrieval_pipeline.retrieve(
            query=question,
            top_k=top_k,
            top_n=top_n,
            doc_filter=doc_filter,
        )

        # Step 2: Build prompt
        prompt = build_rag_prompt(
            question=question,
            context_chunks=retrieval_result.chunks,
        )

        # Step 3: Generate answer
        self.logger.info("Generating answer with LLM")
        if on_token:
            pieces = []
            for piece in self.llm_client.generate_stream(prompt=prompt, system_prompt=SYSTEM_PROMPT):
                pieces.append(piece)
                on_token(piece)
            answer = "".join(pieces)
        else:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

        # Step 4: Format sources — only the ones the answer actually cites,
        # not every chunk that was fed to the LLM as context. Most retrieved
        # chunks end up unused for any given question, so listing all of
        # them under "Sources:" looks like a citation list but isn't one.
        # Original [n] numbering is preserved so it still matches the
        # citations the LLM wrote inline in the answer.
        cited_indices = self._extract_cited_indices(answer)
        sources, formatted_sources = self._format_sources(
            retrieval_result.chunks,
            retrieval_result.citation_map,
            only_indices=cited_indices,
        )

        # Step 5: Build metadata
        metadata = {
            "question": question,
            "retrieved_chunks": len(retrieval_result.chunks),
            "total_tokens": retrieval_result.total_tokens,
            "retrieval_metadata": retrieval_result.metadata,
            "model": self.llm_client.model,
        }

        response = RAGResponse(
            answer=answer,
            sources=sources,
            formatted_sources=formatted_sources,
            metadata=metadata,
            retrieval_result=retrieval_result,
        )

        self.logger.info("Response generation complete")
        return response

    def _extract_cited_indices(self, answer: str) -> set:
        """
        Find which [n] citation markers the answer's prose actually cites.

        The prompt asks the model to end with its own "Sources:" recap, and
        it sometimes lists every candidate there — including ones it
        annotates as "(not used in this answer)". Scanning past that
        heading would count those as cited too, so only the text before it
        is considered.
        """
        body = re.split(r'\n\s*Sources:', answer, maxsplit=1, flags=re.IGNORECASE)[0]
        # Matches both "[1]" and multi-citation brackets like "[1, 5]".
        indices = set()
        for group in re.findall(r'\[([\d,\s]+)\]', body):
            indices.update(int(n) for n in re.findall(r'\d+', group))
        return indices

    def _format_sources(
        self,
        chunks: list,
        citation_map: Dict[str, Any],
        only_indices: Optional[set] = None,
    ) -> tuple:
        """
        Format sources for display.

        Args:
            chunks: Retrieved chunks
            citation_map: Citation mapping
            only_indices: If given, only include chunks whose 1-based
                position is in this set (i.e. the ones actually cited in the
                answer). Numbering is kept as the original [n] so it still
                matches what the answer references. Falls back to all
                chunks if empty/None (e.g. the model cited nothing).

        Returns:
            Tuple of (sources_list, formatted_string)
        """
        sources = []
        formatted_lines = ["Sources:"]

        for i, chunk in enumerate(chunks, 1):
            if only_indices and i not in only_indices:
                continue

            doc_id = chunk.get("doc_id", "Unknown")
            page_numbers = chunk.get("page_numbers", [])
            page_str = format_page_citation(page_numbers, chunk.get("line_ranges"))

            # Extract section title if available
            section = chunk.get("section_title", "")
            if section:
                section_str = f': "{section}"'
            else:
                section_str = ""

            source_entry = {
                "index": i,
                "doc_id": doc_id,
                "page_numbers": page_numbers,
                "line_ranges": chunk.get("line_ranges") or {},
                "section": section,
            }
            sources.append(source_entry)

            # Format for display
            formatted_lines.append(f"[{i}] {doc_id} - {page_str}{section_str}")

        formatted_sources = "\n".join(formatted_lines)

        return sources, formatted_sources

"""Complete RAG response generation."""

import re
import time
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult
from src.generation.base_client import UsageInfo
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
        usage_holder: Dict[str, Optional[UsageInfo]] = {"usage": None}

        def _capture_usage(usage: UsageInfo) -> None:
            usage_holder["usage"] = usage

        generation_start = time.perf_counter()
        if on_token:
            pieces = []
            for piece in self.llm_client.generate_stream(
                prompt=prompt, system_prompt=SYSTEM_PROMPT, on_usage=_capture_usage
            ):
                pieces.append(piece)
                on_token(piece)
            answer = "".join(pieces)
        else:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                on_usage=_capture_usage,
            )
        generation_ms = round((time.perf_counter() - generation_start) * 1000, 1)

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

        # Step 5: Verification signals — cheap, automatic checks of whether
        # the answer is actually grounded in what was retrieved.
        verification = self._compute_verification(cited_indices, retrieval_result.chunks, answer, sources)

        # Step 6: Build metadata
        retrieval_timing = retrieval_result.metadata.get("timing_ms", {})
        usage = usage_holder["usage"]
        metadata = {
            "question": question,
            "retrieved_chunks": len(retrieval_result.chunks),
            "retrieved_doc_ids": sorted({c.get("doc_id") for c in retrieval_result.chunks if c.get("doc_id")}),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieval_result.chunks if c.get("chunk_id")],
            "total_tokens": retrieval_result.total_tokens,
            "retrieval_metadata": retrieval_result.metadata,
            "model": self.llm_client.model,
            "usage": usage,
            "timing_ms": {
                "vector_search": retrieval_timing.get("vector_search"),
                "rerank": retrieval_timing.get("rerank"),
                "generation": generation_ms,
                "total": round(
                    sum(v for v in retrieval_timing.values() if v is not None) + generation_ms, 1
                ),
            },
            "verification": verification,
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
                "rerank_score": chunk.get("rerank_score"),
            }
            sources.append(source_entry)

            # Format for display
            formatted_lines.append(f"[{i}] {doc_id} - {page_str}{section_str}")

        formatted_sources = "\n".join(formatted_lines)

        return sources, formatted_sources

    def _compute_verification(
        self,
        cited_indices: set,
        chunks: list,
        answer: str,
        sources: list,
    ) -> Dict[str, Any]:
        """
        Cheap, automatic groundedness signals — no LLM-judge call, just
        cross-checking the answer against what was actually retrieved.

        - ungrounded_citation_indices: [n] markers the model wrote that
          don't correspond to any chunk that was really retrieved (e.g. it
          says "[7]" but only 6 chunks were in context). _format_sources
          silently drops these rather than flagging them, so this is the
          only place that catches a hallucinated citation number.
        - avg_rerank_score_of_cited: retrieval-confidence proxy for the
          chunks the answer actually used, not everything retrieved.
        - no_citations_flag: a substantial answer that cites nothing is a
          proxy for "answered from the model's own knowledge instead of
          the retrieved context" — worth flagging even though it's not
          proof either way.
        """
        valid_indices = set(range(1, len(chunks) + 1))
        ungrounded = sorted(cited_indices - valid_indices)

        rerank_scores = [s["rerank_score"] for s in sources if s.get("rerank_score") is not None]
        avg_rerank_score = round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else None

        return {
            "cited_indices": sorted(cited_indices),
            "ungrounded_citation_indices": ungrounded,
            "avg_rerank_score_of_cited": avg_rerank_score,
            "no_citations_flag": len(cited_indices) == 0 and len(answer) > 200,
        }

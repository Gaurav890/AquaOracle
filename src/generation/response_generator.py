"""Complete RAG response generation."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.retrieval.retrieval_pipeline import RetrievalPipeline, RetrievalResult
from src.generation.ollama_client import OllamaClient
from src.generation.prompt_templates import SYSTEM_PROMPT, build_rag_prompt


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
    ) -> RAGResponse:
        """
        Generate complete RAG response.

        Args:
            question: User's question
            top_k: Vector search top K
            top_n: Re-ranking top N
            doc_filter: Filter by document

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
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        # Step 4: Format sources
        sources, formatted_sources = self._format_sources(
            retrieval_result.chunks,
            retrieval_result.citation_map,
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

    def _format_sources(
        self,
        chunks: list,
        citation_map: Dict[str, Any],
    ) -> tuple:
        """
        Format sources for display.

        Args:
            chunks: Retrieved chunks
            citation_map: Citation mapping

        Returns:
            Tuple of (sources_list, formatted_string)
        """
        sources = []
        formatted_lines = ["Sources:"]

        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("doc_id", "Unknown")
            page_numbers = chunk.get("page_numbers", [])

            # Format page numbers
            if page_numbers:
                if len(page_numbers) == 1:
                    page_str = f"Page {page_numbers[0]}"
                else:
                    page_str = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
            else:
                page_str = "Page Unknown"

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
                "section": section,
            }
            sources.append(source_entry)

            # Format for display
            formatted_lines.append(f"[{i}] {doc_id} - {page_str}{section_str}")

        formatted_sources = "\n".join(formatted_lines)

        return sources, formatted_sources

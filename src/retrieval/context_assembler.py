"""Assemble final context from retrieved chunks."""

from typing import List, Dict, Any, Tuple
from loguru import logger


class ContextAssembler:
    """Assemble final context from retrieved and expanded chunks."""

    def __init__(
        self,
        max_context_tokens: int = 8192,
        include_metadata: bool = True,
    ):
        """
        Initialize context assembler.

        Args:
            max_context_tokens: Maximum tokens in assembled context
            include_metadata: Include chunk metadata in context
        """
        self.logger = logger.bind(name="ContextAssembler")
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata

    def assemble(
        self,
        chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Assemble final context from chunks.

        Args:
            chunks: List of chunks to assemble
            query: Original query (for reference)

        Returns:
            Tuple of (assembled_chunks, citation_map)
        """
        self.logger.info(f"Assembling context from {len(chunks)} chunks")

        # Remove duplicates by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        # Sort chunks
        sorted_chunks = self._sort_chunks(unique_chunks)

        # Trim to fit context window
        final_chunks = self._trim_to_context_window(sorted_chunks)

        # Build citation map
        citation_map = self._build_citation_map(final_chunks)

        self.logger.info(f"Final context: {len(final_chunks)} chunks")
        return final_chunks, citation_map

    def _sort_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort chunks by relevance, document, and page number.

        Args:
            chunks: List of chunks

        Returns:
            Sorted chunks
        """
        def sort_key(chunk):
            # Primary: relevance score (rerank_score or original score)
            score = chunk.get("rerank_score", chunk.get("score", 0.0))

            # Secondary: document ID
            doc_id = chunk.get("doc_id", "")

            # Tertiary: page number
            page_numbers = chunk.get("page_numbers", [])
            page = page_numbers[0] if page_numbers else 0

            return (-score, doc_id, page)  # Negative score for descending

        return sorted(chunks, key=sort_key)

    def _trim_to_context_window(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Trim chunks to fit within context window.

        Args:
            chunks: Sorted chunks

        Returns:
            Trimmed chunks
        """
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        total_tokens = 0
        trimmed = []

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_tokens = len(text) // 4

            if total_tokens + chunk_tokens <= self.max_context_tokens:
                trimmed.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Stop adding chunks
                break

        if len(trimmed) < len(chunks):
            self.logger.info(
                f"Trimmed from {len(chunks)} to {len(trimmed)} chunks "
                f"to fit {self.max_context_tokens} token limit"
            )

        return trimmed

    def _build_citation_map(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build citation map for chunks.

        Args:
            chunks: Final chunks

        Returns:
            Citation map with document and page info
        """
        citation_map = {}

        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("doc_id", "Unknown")
            page_numbers = chunk.get("page_numbers", [])

            citation_key = f"[{i}]"

            citation_map[citation_key] = {
                "doc_id": doc_id,
                "page_numbers": page_numbers,
                "chunk_id": chunk.get("chunk_id", ""),
                "text_preview": chunk.get("text", "")[:100] + "...",
            }

        return citation_map

    def format_for_llm(
        self,
        chunks: List[Dict[str, Any]],
        include_sources: bool = True,
    ) -> str:
        """
        Format chunks as text for LLM input.

        Args:
            chunks: Chunks to format
            include_sources: Include source markers

        Returns:
            Formatted text
        """
        formatted_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")

            if include_sources:
                doc_id = chunk.get("doc_id", "Unknown")
                page_numbers = chunk.get("page_numbers", [])

                if page_numbers:
                    if len(page_numbers) == 1:
                        page_str = f"Page {page_numbers[0]}"
                    else:
                        page_str = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
                else:
                    page_str = "Page Unknown"

                source_marker = f"[{i}] Source: {doc_id}, {page_str}"
                formatted_parts.append(f"{source_marker}\n{text}")
            else:
                formatted_parts.append(text)

        return "\n\n---\n\n".join(formatted_parts)

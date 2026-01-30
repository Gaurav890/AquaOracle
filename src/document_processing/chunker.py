"""Semantic chunking for document text."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from loguru import logger
import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    doc_id: str
    chunk_index: int
    page_numbers: List[int]
    section_title: Optional[str]
    token_count: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any]


class Chunker:
    """
    Semantic chunker with hybrid strategy.

    Combines semantic boundaries (paragraphs, sections) with
    sliding window to ensure optimal chunk sizes.
    """

    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 128,
        overlap: float = 0.2,
        preserve_tables: bool = True,
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            overlap: Overlap ratio between chunks (0.0-1.0)
            preserve_tables: Keep tables together in single chunks
        """
        self.logger = logger.bind(name="Chunker")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.preserve_tables = preserve_tables

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.logger.warning("Could not load tiktoken, using character-based estimation")
            self.tokenizer = None

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        page_numbers: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk a document into semantically meaningful pieces.

        Args:
            text: Document text
            doc_id: Document identifier
            page_numbers: List of page numbers in document
            metadata: Additional metadata

        Returns:
            List of chunks
        """
        self.logger.debug(f"Chunking document {doc_id}")

        if not text or not text.strip():
            return []

        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)

        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Handle oversized paragraphs (split them)
            if para_tokens > self.max_chunk_size:
                # Finish current chunk if any
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        doc_id,
                        chunk_index,
                        page_numbers,
                        metadata or {}
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split oversized paragraph
                sub_chunks = self._split_oversized_text(para, para_tokens)
                for sub_text in sub_chunks:
                    chunks.append(self._create_chunk(
                        [sub_text],
                        doc_id,
                        chunk_index,
                        page_numbers,
                        metadata or {}
                    ))
                    chunk_index += 1

            # Add paragraph to current chunk if it fits
            elif current_tokens + para_tokens <= self.max_chunk_size:
                current_chunk.append(para)
                current_tokens += para_tokens

            # Start new chunk if current is full
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        doc_id,
                        chunk_index,
                        page_numbers,
                        metadata or {}
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Include last paragraph(s) for overlap
                    overlap_tokens = int(self.max_chunk_size * self.overlap)
                    overlap_text = []
                    overlap_token_count = 0

                    for para in reversed(current_chunk):
                        para_tokens = self._count_tokens(para)
                        if overlap_token_count + para_tokens <= overlap_tokens:
                            overlap_text.insert(0, para)
                            overlap_token_count += para_tokens
                        else:
                            break

                    current_chunk = overlap_text + [para]
                    current_tokens = overlap_token_count + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                doc_id,
                chunk_index,
                page_numbers,
                metadata or {}
            ))

        self.logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or paragraph markers
        paragraphs = re.split(r'\n\s*\n', text)

        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _split_oversized_text(self, text: str, token_count: int) -> List[str]:
        """Split oversized text into smaller pieces."""
        # Split into sentences
        sentences = re.split(r'([.!?]+\s+)', text)

        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            if current_tokens + sent_tokens <= self.max_chunk_size:
                current.append(sentence)
                current_tokens += sent_tokens
            else:
                if current:
                    chunks.append("".join(current).strip())
                current = [sentence]
                current_tokens = sent_tokens

        if current:
            chunks.append("".join(current).strip())

        return chunks

    def _create_chunk(
        self,
        paragraphs: List[str],
        doc_id: str,
        chunk_index: int,
        page_numbers: List[int],
        metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a Chunk object from paragraphs."""
        text = "\n\n".join(paragraphs)
        token_count = self._count_tokens(text)

        return Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=chunk_index,
            page_numbers=page_numbers,
            section_title=metadata.get("section_title"),
            token_count=token_count,
            char_start=0,  # Could be calculated if needed
            char_end=len(text),
            metadata=metadata,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass

        # Fallback: estimate 1 token ≈ 4 characters
        return len(text) // 4

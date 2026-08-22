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
        pages: List[tuple],
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk a document into semantically meaningful pieces.

        Args:
            pages: List of (page_number, page_text) tuples, in document order
            doc_id: Document identifier
            metadata: Additional metadata

        Returns:
            List of chunks, each tagged with the actual page(s) its text came from
        """
        self.logger.debug(f"Chunking document {doc_id}")

        # Split into paragraphs per-page so every paragraph keeps a precise
        # page number instead of being tagged with the whole document's range.
        paragraphs = []  # List[(page_number, text)]
        for page_number, page_text in pages:
            page_text = self._strip_layout_noise(page_text)
            for para in self._split_paragraphs(page_text):
                paragraphs.append((page_number, para))

        if not paragraphs:
            return []

        # Group paragraphs into chunks
        chunks = []
        current_chunk = []  # List[(page_number, text)]
        current_tokens = 0
        chunk_index = 0

        for page_number, para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Handle oversized paragraphs (split them)
            if para_tokens > self.max_chunk_size:
                # Finish current chunk if any
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        doc_id,
                        chunk_index,
                        metadata or {}
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split oversized paragraph
                sub_chunks = self._split_oversized_text(para, para_tokens)
                for sub_text in sub_chunks:
                    chunks.append(self._create_chunk(
                        [(page_number, sub_text)],
                        doc_id,
                        chunk_index,
                        metadata or {}
                    ))
                    chunk_index += 1

            # Add paragraph to current chunk if it fits
            elif current_tokens + para_tokens <= self.max_chunk_size:
                current_chunk.append((page_number, para))
                current_tokens += para_tokens

            # Start new chunk if current is full
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        doc_id,
                        chunk_index,
                        metadata or {}
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Include last paragraph(s) for overlap
                    overlap_tokens = int(self.max_chunk_size * self.overlap)
                    overlap_text = []
                    overlap_token_count = 0

                    for prev_page, prev_para in reversed(current_chunk):
                        prev_tokens = self._count_tokens(prev_para)
                        if overlap_token_count + prev_tokens <= overlap_tokens:
                            overlap_text.insert(0, (prev_page, prev_para))
                            overlap_token_count += prev_tokens
                        else:
                            break

                    current_chunk = overlap_text + [(page_number, para)]
                    current_tokens = overlap_token_count + para_tokens
                else:
                    current_chunk = [(page_number, para)]
                    current_tokens = para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                doc_id,
                chunk_index,
                metadata or {}
            ))

        self.logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks

    _LAYOUT_NOISE_RE = re.compile(r'[.\xa0]{4,}')
    _MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')

    def _strip_layout_noise(self, text: str) -> str:
        """
        Collapse PDF layout artifacts that carry no semantic content but can
        make a chunk far longer, in tokens, than its character count
        suggests — most commonly table-of-contents dot leaders
        ("Section 4.1 .......................... 42") and runs of repeated
        non-breaking spaces used for alignment. A real 1024-cl100k-token
        chunk of prose fits comfortably in nomic-embed-text's context
        window, but a dot-leader-heavy TOC page of the same cl100k length
        tokenizes far less efficiently under nomic's own tokenizer and can
        exceed it — this was the actual cause of a batch of chunks failing
        to embed with "input length exceeds the context length".
        """
        text = self._LAYOUT_NOISE_RE.sub(' ', text)
        text = self._MULTI_SPACE_RE.sub(' ', text)
        return text

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
            if not sentence:
                continue

            sent_tokens = self._count_tokens(sentence)

            # A single "sentence" can itself exceed max_chunk_size — most
            # often because there's no sentence-ending punctuation to split
            # on at all (e.g. a dense reference list or table dump), so the
            # whole text comes back as one unsplit segment. Left alone, that
            # produces one chunk too long for the embedding model's context
            # window. Fall back to a token-bounded split for just that piece.
            if sent_tokens > self.max_chunk_size:
                if current:
                    chunks.append("".join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(self._split_by_tokens(sentence))
                continue

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

        return [c for c in chunks if c]

    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Last-resort splitter for a single segment that has no punctuation to
        break on. Slices the raw token stream so every piece is guaranteed to
        fit max_chunk_size, unlike splitting on whitespace (a single very
        long "word" could still exceed the limit).
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return [
                    self.tokenizer.decode(tokens[i:i + self.max_chunk_size]).strip()
                    for i in range(0, len(tokens), self.max_chunk_size)
                ]
            except Exception:
                self.logger.warning("Tokenizer failed during oversized-text split; using character estimate")

        # Fallback: character-based estimate (~4 chars/token), matching the
        # heuristic _count_tokens uses when no tokenizer is available.
        max_chars = self.max_chunk_size * 4
        return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

    def _create_chunk(
        self,
        paragraphs: List[tuple],
        doc_id: str,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a Chunk object from a list of (page_number, text) paragraphs."""
        text = "\n\n".join(para for _, para in paragraphs)
        token_count = self._count_tokens(text)
        page_numbers = sorted(set(page for page, _ in paragraphs))

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

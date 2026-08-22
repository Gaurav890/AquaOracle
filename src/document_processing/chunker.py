"""Semantic chunking for document text."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
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
    # page_number -> (first_line, last_line), 1-indexed within that page as
    # extracted from the PDF. Lets citations point at "Page 33, Lines 12-18"
    # instead of just "Page 33". Chunks are kept small (see chunking.yaml)
    # specifically so this range stays tight rather than covering most of a
    # page.
    line_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)


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
        # page number instead of being tagged with the whole document's
        # range, and a precise line range within that page.
        paragraphs = []  # List[(page_number, line_start, line_end, text)]
        for page_number, page_text in pages:
            page_text = self._strip_layout_noise(page_text)
            for text, line_start, line_end in self._split_paragraphs_with_lines(page_text):
                paragraphs.append((page_number, line_start, line_end, text))

        if not paragraphs:
            return []

        # Group paragraphs into chunks
        chunks = []
        current_chunk = []  # List[(page_number, line_start, line_end, text)]
        current_tokens = 0
        chunk_index = 0

        for page_number, line_start, line_end, para in paragraphs:
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

                # Split oversized paragraph, keeping each sub-piece's own
                # line range (relative offsets from _split_oversized_text,
                # rebased onto the parent paragraph's absolute start line).
                sub_chunks = self._split_oversized_text(para, para_tokens)
                for sub_text, rel_start, rel_end in sub_chunks:
                    chunks.append(self._create_chunk(
                        [(page_number, line_start + rel_start, line_start + rel_end, sub_text)],
                        doc_id,
                        chunk_index,
                        metadata or {}
                    ))
                    chunk_index += 1

            # Add paragraph to current chunk if it fits
            elif current_tokens + para_tokens <= self.max_chunk_size:
                current_chunk.append((page_number, line_start, line_end, para))
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

                    for prev_entry in reversed(current_chunk):
                        prev_tokens = self._count_tokens(prev_entry[3])
                        if overlap_token_count + prev_tokens <= overlap_tokens:
                            overlap_text.insert(0, prev_entry)
                            overlap_token_count += prev_tokens
                        else:
                            break

                    current_chunk = overlap_text + [(page_number, line_start, line_end, para)]
                    current_tokens = overlap_token_count + para_tokens
                else:
                    current_chunk = [(page_number, line_start, line_end, para)]
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

    def _split_paragraphs_with_lines(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split page text into paragraphs, each tagged with the (first_line,
        last_line) it spans within the page — 1-indexed over the page's
        extracted lines, matching how PyMuPDF's text extraction lays them
        out. A blank (or whitespace-only) line ends the current paragraph,
        the same boundary `re.split(r'\\n\\s*\\n', ...)` used before, just
        tracked line-by-line so the line range survives.
        """
        lines = text.split('\n')
        paragraphs = []
        current_lines: List[str] = []
        start_line: Optional[int] = None

        for i, line in enumerate(lines, 1):
            if line.strip():
                if start_line is None:
                    start_line = i
                current_lines.append(line)
            elif current_lines:
                paragraphs.append(('\n'.join(current_lines).strip(), start_line, i - 1))
                current_lines = []
                start_line = None

        if current_lines:
            paragraphs.append(('\n'.join(current_lines).strip(), start_line, len(lines)))

        return [(p, s, e) for p, s, e in paragraphs if p]

    def _split_oversized_text(self, text: str, token_count: int) -> List[Tuple[str, int, int]]:
        """
        Split oversized text into smaller pieces, each tagged with a
        0-indexed (start_line, end_line) offset relative to the start of
        `text` — the caller rebases these onto the parent paragraph's
        absolute page line number. Without this, every sub-piece of a
        multi-line oversized paragraph would fall back to the whole
        paragraph's line range, defeating line-level citation precision for
        exactly the paragraphs most likely to need splitting.
        """
        sentences = [s for s in re.split(r'([.!?]+\s+)', text) if s]

        pieces: List[Tuple[str, int, int]] = []
        current: List[str] = []
        current_tokens = 0
        lines_consumed = 0  # newlines consumed across all prior sentences
        current_start_line = 0

        def flush():
            if not current:
                return
            joined = "".join(current)
            piece_text = joined.strip()
            if piece_text:
                pieces.append((piece_text, current_start_line, current_start_line + joined.count('\n')))

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            sent_lines = sentence.count('\n')

            # A single "sentence" can itself exceed max_chunk_size — most
            # often because there's no sentence-ending punctuation to split
            # on at all (e.g. a dense reference list or table dump), so the
            # whole text comes back as one unsplit segment. Fall back to a
            # token-bounded split; its sub-pieces all inherit this one
            # sentence's line span since we're already splitting on tokens
            # rather than lines at that point.
            if sent_tokens > self.max_chunk_size:
                flush()
                current = []
                current_tokens = 0
                sentence_start_line = lines_consumed
                for sub_text in self._split_by_tokens(sentence):
                    pieces.append((sub_text, sentence_start_line, sentence_start_line + sent_lines))
                lines_consumed += sent_lines
                current_start_line = lines_consumed
                continue

            if current_tokens + sent_tokens <= self.max_chunk_size:
                if not current:
                    current_start_line = lines_consumed
                current.append(sentence)
                current_tokens += sent_tokens
            else:
                flush()
                current_start_line = lines_consumed
                current = [sentence]
                current_tokens = sent_tokens

            lines_consumed += sent_lines

        flush()
        return pieces

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
        paragraphs: List[Tuple[int, int, int, str]],
        doc_id: str,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> Chunk:
        """Create a Chunk from (page_number, line_start, line_end, text) paragraphs."""
        text = "\n\n".join(para for _, _, _, para in paragraphs)
        token_count = self._count_tokens(text)
        page_numbers = sorted(set(page for page, _, _, _ in paragraphs))

        line_ranges: Dict[int, Tuple[int, int]] = {}
        for page, line_start, line_end, _ in paragraphs:
            if page in line_ranges:
                prev_start, prev_end = line_ranges[page]
                line_ranges[page] = (min(prev_start, line_start), max(prev_end, line_end))
            else:
                line_ranges[page] = (line_start, line_end)

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
            line_ranges=line_ranges,
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

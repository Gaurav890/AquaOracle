"""Tests for the semantic chunker, especially per-chunk page attribution."""

from src.document_processing.chunker import Chunker


def test_empty_pages_returns_no_chunks():
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=128, overlap=0.2)
    assert chunker.chunk_document(pages=[], doc_id="doc1", metadata={}) == []


def test_single_page_chunk_gets_that_pages_number():
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=128, overlap=0.2)
    pages = [(1, "First paragraph.\n\nSecond paragraph.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) == 1
    assert chunks[0].page_numbers == [1]
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text


def test_chunks_are_tagged_with_their_real_page_not_the_whole_document():
    """Regression test: chunks used to be tagged with the full document page
    range (e.g. [1, 2, 3, ..., N]) instead of the specific page(s) their text
    came from. Each page's paragraph here is ~10 tokens; a 15-token budget
    fits one page but not two, so each chunk should be scoped to a single
    page, not to the full 5-page document.
    """
    chunker = Chunker(max_chunk_size=15, min_chunk_size=1, overlap=0.0)
    pages = [(i, f"This is the unique content of page {i}.") for i in range(1, 6)]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    all_page_numbers = [pn for chunk in chunks for pn in chunk.page_numbers]
    assert all_page_numbers == [1, 2, 3, 4, 5]
    for chunk in chunks:
        assert chunk.page_numbers != [1, 2, 3, 4, 5]


def test_oversized_paragraph_is_split_and_keeps_its_page_number():
    chunker = Chunker(max_chunk_size=10, min_chunk_size=1, overlap=0.0)
    # Sentence punctuation is what the splitter uses to break up an oversized
    # paragraph, so the text needs real sentences rather than just many words.
    long_paragraph = "This is a sentence. " * 50
    pages = [(3, long_paragraph)]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) > 1
    assert all(chunk.page_numbers == [3] for chunk in chunks)


def test_chunk_spanning_a_page_boundary_lists_both_pages():
    # Small max_chunk_size forces page 1's and page 2's paragraphs together.
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=1, overlap=0.0)
    pages = [(1, "Short page one text."), (2, "Short page two text.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) == 1
    assert chunks[0].page_numbers == [1, 2]


def test_punctuation_free_oversized_text_is_still_split():
    """Regression test: a paragraph with no sentence-ending punctuation used
    to come back as a single unsplit blob from the sentence-based splitter,
    producing a chunk too long for the embedding model's context window
    (this caused real "input length exceeds context length" failures during
    ingestion). Every resulting chunk must now respect max_chunk_size.
    """
    chunker = Chunker(max_chunk_size=50, min_chunk_size=1, overlap=0.0)
    huge_no_punctuation = " ".join(f"word{i}" for i in range(500))
    pages = [(7, huge_no_punctuation)]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert all(chunk.page_numbers == [7] for chunk in chunks)


def test_toc_dot_leaders_are_stripped_before_chunking():
    """Regression test: table-of-contents pages full of dot leaders
    ("Section 4.1 .......................... 42") and repeated non-breaking
    spaces tokenize far less efficiently under the embedding model's own
    tokenizer than under the chunker's tiktoken-based size estimate. A real
    document page like this produced chunks tiktoken measured as ~700
    tokens (within budget) that nomic-embed-text rejected as exceeding its
    context length. Stripping the noise should sharply shrink the text.
    """
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=1, overlap=0.0)
    toc_page = (
        "4.5.5.5 \xa0ADA Accessibility " + "." * 60 + " 64\xa0\xa0\xa0\xa0\n"
        "4.5.5.7 \xa0Dimensions " + "." * 60 + " 64\xa0\xa0\xa0\xa0\n"
    )

    chunks = chunker.chunk_document(pages=[(9, toc_page)], doc_id="doc1", metadata={})

    combined_text = " ".join(c.text for c in chunks)
    assert "." * 10 not in combined_text
    assert "\xa0\xa0\xa0" not in combined_text
    assert "ADA Accessibility" in combined_text


def test_line_ranges_are_tight_for_a_single_page_chunk():
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=1, overlap=0.0)
    pages = [(4, "Line one of the paragraph.\nLine two continues here.\nLine three ends it.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) == 1
    assert chunks[0].line_ranges == {4: (1, 3)}


def test_line_ranges_track_separate_paragraphs_on_one_page():
    # Small max_chunk_size forces each ~10-token paragraph into its own chunk.
    chunker = Chunker(max_chunk_size=15, min_chunk_size=1, overlap=0.0)
    page_text = "First short paragraph line.\n\nSecond short paragraph line.\n\nThird short paragraph line."
    pages = [(1, page_text)]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    ranges = [c.line_ranges[1] for c in chunks]
    assert ranges == sorted(ranges)  # line ranges advance monotonically
    assert len(set(ranges)) == len(ranges)  # no two chunks share the same range


def test_line_ranges_stay_tight_when_a_paragraph_is_split_as_oversized():
    """Regression test: splitting an oversized paragraph used to give every
    resulting sub-chunk the SAME line range as the whole parent paragraph
    (e.g. lines 1-78 for all of them), which defeats line-level citation
    precision for exactly the paragraphs big enough to need splitting.
    """
    chunker = Chunker(max_chunk_size=15, min_chunk_size=1, overlap=0.0)
    sentences = [f"This is sentence number {i} in a long paragraph." for i in range(20)]
    long_paragraph = "\n".join(sentences)  # one paragraph, no blank lines inside
    pages = [(2, long_paragraph)]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) > 1
    ranges = [c.line_ranges[2] for c in chunks]
    assert len(set(ranges)) > 1, "all sub-chunks got the same line range"
    full_span = ranges[0][0], ranges[-1][1]
    assert any(r != full_span for r in ranges), "every sub-chunk still spans the whole paragraph"


def test_line_ranges_have_separate_entries_per_page_for_a_cross_page_chunk():
    chunker = Chunker(max_chunk_size=1024, min_chunk_size=1, overlap=0.0)
    pages = [(1, "Short page one text."), (2, "Short page two text.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert len(chunks) == 1
    assert chunks[0].line_ranges == {1: (1, 1), 2: (1, 1)}


def test_chunk_index_increments_in_order():
    chunker = Chunker(max_chunk_size=5, min_chunk_size=1, overlap=0.0)
    pages = [(1, "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

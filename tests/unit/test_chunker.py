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


def test_chunk_index_increments_in_order():
    chunker = Chunker(max_chunk_size=5, min_chunk_size=1, overlap=0.0)
    pages = [(1, "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota.")]

    chunks = chunker.chunk_document(pages=pages, doc_id="doc1", metadata={})

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

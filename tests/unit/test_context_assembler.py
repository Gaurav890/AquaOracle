"""Tests for context assembly: dedup, sorting, token trimming, citations."""

from src.retrieval.context_assembler import ContextAssembler


def make_chunk(chunk_id, score, doc_id="doc1", page_numbers=None, text="x" * 40):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "rerank_score": score,
        "page_numbers": page_numbers or [1],
    }


def test_deduplicates_by_chunk_id():
    assembler = ContextAssembler(max_context_tokens=10_000)
    chunks = [make_chunk("c1", 0.9), make_chunk("c1", 0.9), make_chunk("c2", 0.5)]

    final_chunks, _ = assembler.assemble(chunks)

    assert [c["chunk_id"] for c in final_chunks] == ["c1", "c2"]


def test_sorts_by_score_descending():
    assembler = ContextAssembler(max_context_tokens=10_000)
    chunks = [make_chunk("low", 0.1), make_chunk("high", 0.9), make_chunk("mid", 0.5)]

    final_chunks, _ = assembler.assemble(chunks)

    assert [c["chunk_id"] for c in final_chunks] == ["high", "mid", "low"]


def test_trims_to_context_window():
    # Each 40-char chunk is ~10 tokens (chars // 4); budget for one, not two.
    assembler = ContextAssembler(max_context_tokens=15)
    chunks = [make_chunk("c1", 0.9, text="a" * 40), make_chunk("c2", 0.5, text="b" * 40)]

    final_chunks, _ = assembler.assemble(chunks)

    assert len(final_chunks) == 1
    assert final_chunks[0]["chunk_id"] == "c1"


def test_citation_map_indexes_from_one_and_includes_doc_and_pages():
    assembler = ContextAssembler(max_context_tokens=10_000)
    chunks = [make_chunk("c1", 0.9, doc_id="CDC_2017", page_numbers=[3, 4])]

    final_chunks, citation_map = assembler.assemble(chunks)

    assert citation_map["[1]"]["doc_id"] == "CDC_2017"
    assert citation_map["[1]"]["page_numbers"] == [3, 4]


def test_empty_input_returns_empty_output():
    assembler = ContextAssembler(max_context_tokens=10_000)

    final_chunks, citation_map = assembler.assemble([])

    assert final_chunks == []
    assert citation_map == {}

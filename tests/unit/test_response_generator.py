"""Tests for citation filtering in response generation."""

from src.generation.base_client import UsageInfo
from src.generation.response_generator import ResponseGenerator
from src.retrieval.retrieval_pipeline import RetrievalResult


def make_generator():
    # _extract_cited_indices/_format_sources don't touch these collaborators.
    return ResponseGenerator(retrieval_pipeline=None, llm_client=None)


def make_chunk(doc_id, page_numbers, line_ranges=None, rerank_score=None):
    chunk = {"doc_id": doc_id, "page_numbers": page_numbers}
    if line_ranges is not None:
        chunk["line_ranges"] = line_ranges
    if rerank_score is not None:
        chunk["rerank_score"] = rerank_score
    return chunk


def test_extract_cited_indices_finds_bracket_numbers():
    gen = make_generator()

    indices = gen._extract_cited_indices("Legionella spreads via water droplets [1]. See also [1] and [3].")

    assert indices == {1, 3}


def test_extract_cited_indices_handles_comma_grouped_citations():
    """Regression test: the model often cites multiple sources in one
    bracket, e.g. "[1, 5]" rather than "[1] [5]". The naive \\[(\\d+)\\]
    pattern doesn't match "[1, 5]" at all (the comma+space breaks it),
    silently dropping every multi-source citation.
    """
    gen = make_generator()

    indices = gen._extract_cited_indices("Legionella is found in water systems [1, 5] and spreads via aerosols [2].")

    assert indices == {1, 2, 5}


def test_extract_cited_indices_ignores_the_models_own_trailing_sources_list():
    """Regression test: the model sometimes ends its answer with its own
    "Sources:" recap listing every candidate, including ones annotated as
    "(not used in this answer)". Those shouldn't count as cited just for
    appearing in that recap.
    """
    gen = make_generator()
    answer = (
        "Legionella spreads via water droplets [1, 5].\n\n"
        "Sources:\n"
        "[1] CDC_2017\n"
        "[2] CMS_2017\n"
        "[3] MIAC_Australia_2010 (not used in this answer)\n"
        "[5] CDC_2017\n"
    )

    assert gen._extract_cited_indices(answer) == {1, 5}


def test_extract_cited_indices_empty_when_no_citations():
    gen = make_generator()

    assert gen._extract_cited_indices("No citations here.") == set()


def test_format_sources_only_includes_cited_chunks():
    """Regression test: the CLI used to print every retrieved chunk under
    "Sources:" regardless of whether the answer actually cited it, which
    looked like a citation list of 10 items when the model had only used
    one of them.
    """
    gen = make_generator()
    chunks = [make_chunk("USEPA_1985", [15, 16]), make_chunk("CDC_2017", [33]), make_chunk("NASEM_2019", [59])]

    sources, formatted = gen._format_sources(chunks, {}, only_indices={1})

    assert len(sources) == 1
    assert sources[0]["doc_id"] == "USEPA_1985"
    assert "[1] USEPA_1985" in formatted
    assert "CDC_2017" not in formatted


def test_format_sources_keeps_original_numbering_for_non_contiguous_citations():
    gen = make_generator()
    chunks = [make_chunk("A", [1]), make_chunk("B", [2]), make_chunk("C", [3])]

    sources, formatted = gen._format_sources(chunks, {}, only_indices={1, 3})

    assert [s["index"] for s in sources] == [1, 3]
    assert "[1] A" in formatted
    assert "[3] C" in formatted
    assert "[2]" not in formatted


def test_format_sources_includes_line_ranges_for_frontend_deep_linking():
    """Regression test: line_ranges was computed and used to build the
    plain-text formatted string, but silently dropped from the structured
    `sources` list the web UI actually renders from — so citation chips
    could only ever show a page number, never the line range, even though
    the data existed the whole time."""
    gen = make_generator()
    chunks = [make_chunk("A", [12], line_ranges={12: (5, 9)})]

    sources, _ = gen._format_sources(chunks, {}, only_indices={1})

    assert sources[0]["line_ranges"] == {12: (5, 9)}


def test_format_sources_falls_back_to_all_when_no_indices_given():
    gen = make_generator()
    chunks = [make_chunk("A", [1]), make_chunk("B", [2])]

    sources, _ = gen._format_sources(chunks, {}, only_indices=None)

    assert len(sources) == 2


def test_format_sources_includes_rerank_score():
    gen = make_generator()
    chunks = [make_chunk("A", [1], rerank_score=0.87)]

    sources, _ = gen._format_sources(chunks, {}, only_indices={1})

    assert sources[0]["rerank_score"] == 0.87


def test_verification_flags_citation_index_outside_retrieved_range():
    """Regression test: _format_sources silently drops an out-of-range
    citation (e.g. the model wrote "[7]" but only 3 chunks were retrieved)
    rather than flagging it — this is the only place that catches a
    hallucinated citation number."""
    gen = make_generator()
    chunks = [make_chunk("A", [1]), make_chunk("B", [2]), make_chunk("C", [3])]
    sources, _ = gen._format_sources(chunks, {}, only_indices={1, 7})

    verification = gen._compute_verification(cited_indices={1, 7}, chunks=chunks, answer="x" * 300, sources=sources)

    assert verification["ungrounded_citation_indices"] == [7]


def test_verification_averages_rerank_score_of_cited_sources_only():
    gen = make_generator()
    chunks = [
        make_chunk("A", [1], rerank_score=0.9),
        make_chunk("B", [2], rerank_score=0.3),
        make_chunk("C", [3], rerank_score=0.6),
    ]
    sources, _ = gen._format_sources(chunks, {}, only_indices={1, 3})

    verification = gen._compute_verification(cited_indices={1, 3}, chunks=chunks, answer="x" * 300, sources=sources)

    # Only chunks 1 and 3 were cited (0.9 and 0.6) — chunk 2's 0.3 must not pull the average down.
    assert verification["avg_rerank_score_of_cited"] == 0.75


def test_verification_avg_rerank_score_is_none_when_no_scores_available():
    gen = make_generator()
    chunks = [make_chunk("A", [1])]
    sources, _ = gen._format_sources(chunks, {}, only_indices={1})

    verification = gen._compute_verification(cited_indices={1}, chunks=chunks, answer="x" * 300, sources=sources)

    assert verification["avg_rerank_score_of_cited"] is None


def test_verification_flags_substantial_answer_with_no_citations():
    gen = make_generator()

    verification = gen._compute_verification(cited_indices=set(), chunks=[], answer="x" * 300, sources=[])

    assert verification["no_citations_flag"] is True


def test_verification_does_not_flag_short_uncited_answer():
    """A short "I don't know" / no-match answer citing nothing isn't the
    same failure mode as a long answer that ignores its sources."""
    gen = make_generator()

    verification = gen._compute_verification(
        cited_indices=set(), chunks=[], answer="Not found in the documents.", sources=[]
    )

    assert verification["no_citations_flag"] is False


class _FakeRetrievalPipeline:
    def retrieve(self, query, top_k, top_n, doc_filter=None):
        return RetrievalResult(
            chunks=[make_chunk("A", [1], rerank_score=0.8)],
            citation_map={},
            total_tokens=42,
            metadata={"query": query, "stages": [], "timing_ms": {"vector_search": 12.0, "rerank": 3.0}},
        )


class _FakeLLMClient:
    model = "fake-model"

    def generate(self, prompt, system_prompt=None, on_usage=None, **kw):
        if on_usage:
            on_usage(UsageInfo(prompt_tokens=500, completion_tokens=50))
        return "The answer is X [1]."


def test_generate_metadata_includes_timing_usage_and_verification():
    """End-to-end (mocked collaborators) check that Phase A's new
    instrumentation actually reaches RAGResponse.metadata, not just the
    individual helper methods in isolation."""
    gen = ResponseGenerator(retrieval_pipeline=_FakeRetrievalPipeline(), llm_client=_FakeLLMClient())

    response = gen.generate(question="What is X?")

    assert response.metadata["usage"].prompt_tokens == 500
    assert response.metadata["usage"].completion_tokens == 50
    assert response.metadata["timing_ms"]["vector_search"] == 12.0
    assert response.metadata["timing_ms"]["rerank"] == 3.0
    assert response.metadata["timing_ms"]["generation"] is not None
    assert response.metadata["timing_ms"]["total"] >= 15.0
    assert response.metadata["verification"]["cited_indices"] == [1]
    assert response.metadata["verification"]["ungrounded_citation_indices"] == []
    assert response.metadata["retrieved_doc_ids"] == ["A"]

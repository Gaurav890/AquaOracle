"""Tests for citation filtering in response generation."""

from src.generation.response_generator import ResponseGenerator


def make_generator():
    # _extract_cited_indices/_format_sources don't touch these collaborators.
    return ResponseGenerator(retrieval_pipeline=None, llm_client=None)


def make_chunk(doc_id, page_numbers):
    return {"doc_id": doc_id, "page_numbers": page_numbers}


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


def test_format_sources_falls_back_to_all_when_no_indices_given():
    gen = make_generator()
    chunks = [make_chunk("A", [1]), make_chunk("B", [2])]

    sources, _ = gen._format_sources(chunks, {}, only_indices=None)

    assert len(sources) == 2

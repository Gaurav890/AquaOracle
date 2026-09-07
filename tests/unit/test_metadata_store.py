"""Tests for the SQLite metadata store, including table/figure persistence."""

import pytest

from src.indexing.metadata_store import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(tmp_path / "metadata.db")


def test_add_and_get_document(store):
    store.add_document("doc1", {
        "file_name": "doc1.pdf",
        "file_path": "/path/doc1.pdf",
        "title": "Doc One",
        "organization": "CDC",
        "year": 2017,
        "page_count": 10,
    })

    doc = store.get_document("doc1")

    assert doc["title"] == "Doc One"
    assert doc["organization"] == "CDC"
    assert doc["year"] == 2017


def test_add_chunk_updates_document_chunk_count(store):
    store.add_document("doc1", {"file_name": "doc1.pdf", "file_path": "/p", "page_count": 1})

    store.add_chunk("doc1_chunk_0", "doc1", {
        "chunk_index": 0, "text": "hello", "page_numbers": [1],
    })
    store.add_chunk("doc1_chunk_1", "doc1", {
        "chunk_index": 1, "text": "world", "page_numbers": [1],
    })

    assert store.get_document("doc1")["chunk_count"] == 2


def test_chunk_line_ranges_round_trip(store):
    store.add_document("doc1", {"file_name": "doc1.pdf", "file_path": "/p", "page_count": 1})

    store.add_chunk("doc1_chunk_0", "doc1", {
        "chunk_index": 0, "text": "hello", "page_numbers": [3],
        "line_ranges": {3: (12, 18)},
    })

    chunk = store.get_chunk("doc1_chunk_0")

    assert chunk["line_ranges"] == {"3": [12, 18]}  # JSON round-trip: int keys/tuples -> str/list


def test_add_table_and_get_tables(store):
    """Regression test: table extraction results used to be discarded —
    only a count was stored, never the table content."""
    store.add_document("doc1", {"file_name": "doc1.pdf", "file_path": "/p", "page_count": 1})

    store.add_table("doc1", {
        "page_number": 3,
        "row_count": 2,
        "col_count": 2,
        "has_header": True,
        "data": [["A", "B"], ["1", "2"]],
    })

    tables = store.get_tables("doc1")

    assert len(tables) == 1
    assert tables[0]["page_number"] == 3
    assert tables[0]["data"] == [["A", "B"], ["1", "2"]]


def test_add_figure_and_get_figures(store):
    store.add_document("doc1", {"file_name": "doc1.pdf", "file_path": "/p", "page_count": 1})

    store.add_figure("doc1", {
        "page_number": 5, "image_index": 0, "caption": "Figure 1: a chart",
        "width": 100.0, "height": 50.0,
    })

    figures = store.get_figures("doc1")

    assert len(figures) == 1
    assert figures[0]["caption"] == "Figure 1: a chart"


def test_clear_tables_and_figures_only_affects_that_document(store):
    store.add_document("doc1", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1})
    store.add_document("doc2", {"file_name": "d2.pdf", "file_path": "/p2", "page_count": 1})
    store.add_table("doc1", {"page_number": 1, "data": [["x"]]})
    store.add_table("doc2", {"page_number": 1, "data": [["y"]]})

    store.clear_tables_and_figures("doc1")

    assert store.get_tables("doc1") == []
    assert len(store.get_tables("doc2")) == 1


def test_get_stats_counts_tables_and_figures(store):
    store.add_document("doc1", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 2})
    store.add_table("doc1", {"page_number": 1, "data": [["x"]]})
    store.add_figure("doc1", {"page_number": 1, "caption": "fig"})

    stats = store.get_stats()

    assert stats["document_count"] == 1
    assert stats["table_count"] == 1
    assert stats["figure_count"] == 1
    assert stats["total_pages"] == 2


def test_clear_all_removes_everything(store):
    store.add_document("doc1", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1})
    store.add_chunk("doc1_chunk_0", "doc1", {"chunk_index": 0, "text": "hi"})
    store.add_table("doc1", {"page_number": 1, "data": [["x"]]})
    store.add_figure("doc1", {"page_number": 1, "caption": "fig"})

    store.clear_all()

    stats = store.get_stats()
    assert stats == {
        "document_count": 0, "chunk_count": 0, "table_count": 0,
        "figure_count": 0, "total_pages": 0,
    }


def test_list_documents_returns_all(store):
    store.add_document("doc1", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1})
    store.add_document("doc2", {"file_name": "d2.pdf", "file_path": "/p2", "page_count": 1})

    docs = store.list_documents()

    assert {d["doc_id"] for d in docs} == {"doc1", "doc2"}


def test_list_documents_for_a_user_excludes_other_users_private_docs(store):
    store.add_document("legacy_doc", {"file_name": "d0.pdf", "file_path": "/p0", "page_count": 1})
    store.add_document("mine", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1, "owner_user_id": "u1"})
    store.add_document(
        "theirs", {"file_name": "d2.pdf", "file_path": "/p2", "page_count": 1, "owner_user_id": "u2"}
    )

    docs = store.list_documents(owner_user_id="u1")

    doc_ids = {d["doc_id"] for d in docs}
    assert doc_ids == {"legacy_doc", "mine"}


def test_list_shared_doc_ids_includes_legacy_and_own_shared_docs(store):
    store.add_document("legacy_doc", {"file_name": "d0.pdf", "file_path": "/p0", "page_count": 1})
    store.add_document(
        "shared_mine",
        {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1, "owner_user_id": "u1", "is_shared": True},
    )
    store.add_document(
        "private_mine",
        {"file_name": "d2.pdf", "file_path": "/p2", "page_count": 1, "owner_user_id": "u1", "is_shared": False},
    )
    store.add_document(
        "shared_theirs",
        {"file_name": "d3.pdf", "file_path": "/p3", "page_count": 1, "owner_user_id": "u2", "is_shared": True},
    )

    shared = store.list_shared_doc_ids(owner_user_id="u1")

    assert set(shared) == {"legacy_doc", "shared_mine"}


def test_set_shared_toggles_visibility(store):
    store.add_document(
        "doc1", {"file_name": "d1.pdf", "file_path": "/p1", "page_count": 1, "owner_user_id": "u1"}
    )

    result = store.set_shared("doc1", True)

    assert result is True
    assert store.list_shared_doc_ids(owner_user_id="u1") == ["doc1"]

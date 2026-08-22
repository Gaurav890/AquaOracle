"""Tests for the Qdrant-backed vector store, especially cross-document ID safety."""

import pytest

from src.indexing.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    vs = VectorStore(path=tmp_path / "vector_store", embedding_dim=4)
    yield vs
    vs.close()


def test_vectors_from_different_documents_do_not_collide(store):
    """Regression test: point IDs used to be a per-call counter starting at 0,
    so a second document's chunks would silently overwrite a first document's
    vectors at the same IDs. IDs must now be stable across documents.
    """
    doc_a_ids = ["docA_chunk_0", "docA_chunk_1"]
    doc_a_vectors = [[1, 0, 0, 0], [0, 1, 0, 0]]
    doc_a_payloads = [{"doc_id": "docA"}, {"doc_id": "docA"}]

    doc_b_ids = ["docB_chunk_0", "docB_chunk_1", "docB_chunk_2"]
    doc_b_vectors = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]]
    doc_b_payloads = [{"doc_id": "docB"}] * 3

    assert store.add_embeddings(doc_a_ids, doc_a_vectors, doc_a_payloads)
    assert store.add_embeddings(doc_b_ids, doc_b_vectors, doc_b_payloads)

    assert store.get_stats()["total_vectors"] == 5


def test_reingesting_the_same_chunk_id_updates_in_place(store):
    """Re-embedding a chunk with the same chunk_id should upsert, not duplicate."""
    store.add_embeddings(["doc_chunk_0"], [[1, 0, 0, 0]], [{"doc_id": "doc"}])
    store.add_embeddings(["doc_chunk_0"], [[0, 1, 0, 0]], [{"doc_id": "doc"}])

    assert store.get_stats()["total_vectors"] == 1


def test_search_returns_added_vectors(store):
    store.add_embeddings(["c1", "c2"], [[1, 0, 0, 0], [0, 1, 0, 0]], [
        {"doc_id": "doc", "text": "first", "page_numbers": [1]},
        {"doc_id": "doc", "text": "second", "page_numbers": [2]},
    ])

    results = store.search(query_vector=[1, 0, 0, 0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].text == "first"


def test_clear_removes_all_vectors(store):
    store.add_embeddings(["c1"], [[1, 0, 0, 0]], [{"doc_id": "doc"}])
    assert store.get_stats()["total_vectors"] == 1

    assert store.clear()

    assert store.get_stats()["total_vectors"] == 0


def test_close_releases_the_embedded_lock_for_another_instance(tmp_path):
    """Regression test: a VectorStore opened to clear/inspect the index must
    release Qdrant's exclusive file lock via close() so a later VectorStore
    (e.g. one opened per document during ingestion) can open the same path.
    """
    path = tmp_path / "vector_store"
    first = VectorStore(path=path, embedding_dim=4)
    first.close()

    second = VectorStore(path=path, embedding_dim=4)  # must not raise
    second.close()

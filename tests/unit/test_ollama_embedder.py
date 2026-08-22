"""Tests for OllamaEmbedder batching and failure handling — fully mocked,
no real Ollama server required.
"""

import ollama
import pytest

from src.embedding.ollama_embedder import OllamaEmbedder


@pytest.fixture
def embedder(monkeypatch):
    # _test_connection() calls ollama.embeddings() during __init__.
    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [0.0] * 4})
    return OllamaEmbedder(model="fake-model", host="http://fake:11434")


def test_embed_documents_uses_one_batched_call_per_batch(monkeypatch, embedder):
    """Regression test: embedding used to make one HTTP call per chunk. It
    should now make a single batched call per batch_size group of texts.
    """
    calls = []

    def fake_embed(model, input):
        calls.append(list(input))
        return {"embeddings": [[float(i)] * 4 for i in range(len(input))]}

    monkeypatch.setattr(ollama, "embed", fake_embed)

    texts = [f"chunk {i}" for i in range(5)]
    result = embedder.embed_documents(texts, batch_size=2, show_progress=False)

    assert len(result) == 5
    assert all(v is not None for v in result)
    assert len(calls) == 3  # batches of 2, 2, 1
    assert calls[0] == ["chunk 0", "chunk 1"]


def test_failed_item_becomes_none_not_a_fabricated_vector(monkeypatch, embedder):
    """Regression test: a failed embedding used to silently become a
    [0.0]*768 vector, indistinguishable from a real (if unlikely) embedding.
    It must now surface as None so callers can detect and skip it.
    """
    def failing_batch(model, input):
        raise RuntimeError("simulated batch failure")

    def fake_single(model, prompt):
        if prompt == "bad chunk":
            raise RuntimeError("simulated single-item failure")
        return {"embedding": [1.0] * 4}

    monkeypatch.setattr(ollama, "embed", failing_batch)
    monkeypatch.setattr(ollama, "embeddings", fake_single)

    result = embedder.embed_documents(["good chunk", "bad chunk"], batch_size=2, show_progress=False)

    assert result[0] == [1.0] * 4
    assert result[1] is None


def test_batch_failure_falls_back_to_per_item_embedding(monkeypatch, embedder):
    def failing_batch(model, input):
        raise RuntimeError("batch endpoint unavailable")

    monkeypatch.setattr(ollama, "embed", failing_batch)
    monkeypatch.setattr(ollama, "embeddings", lambda model, prompt: {"embedding": [2.0] * 4})

    result = embedder.embed_batch(["a", "b", "c"])

    assert result == [[2.0] * 4, [2.0] * 4, [2.0] * 4]


def test_embed_text_raises_on_failure(monkeypatch, embedder):
    def fake_embeddings(model, prompt):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ollama, "embeddings", fake_embeddings)

    with pytest.raises(RuntimeError):
        embedder.embed_text("some query")

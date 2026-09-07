"""Builds a ResponseGenerator for one request, reusing warm singletons for
the expensive-to-reload collaborators (embedder, reranker, LLM client) the
same way src/web/app.py does — but see that file's module docstring for why
VectorStore itself is deliberately NOT kept warm (Qdrant's embedded mode
holds an exclusive lock on its storage path).

The singletons are created lazily (on first real use, not at import time) so
that importing this module — and building the FastAPI app, which imports
every route module up front — doesn't require a live Ollama server. That
matters for tests that only exercise auth/chat-CRUD endpoints and never
actually generate an answer.
"""

import threading
from typing import Optional, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from src.auth.models import UserApiKey
from src.core.config import settings, load_model_config, load_retrieval_config
from src.core.crypto import decrypt_api_key
from src.embedding.ollama_embedder import OllamaEmbedder
from src.generation.base_client import LLMClient
from src.generation.ollama_client import OllamaClient
from src.generation.provider_factory import create_llm_client
from src.generation.response_generator import ResponseGenerator
from src.indexing.vector_store import VectorStore
from src.retrieval.context_assembler import ContextAssembler
from src.retrieval.reranker import Reranker
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.vector_retriever import VectorRetriever

log = logger.bind(name="ChatService")

model_config = load_model_config()
retrieval_config = load_retrieval_config()

if retrieval_config.graph_expansion_enabled:
    log.warning(
        "retrieval.yaml sets stage3_graph_expansion.enabled=true, but graph "
        "expansion isn't implemented yet — ignoring."
    )

# Serializes all access to the embedded Qdrant store within this process.
vector_store_lock = threading.Lock()

_embedder: Optional[OllamaEmbedder] = None
_reranker: Optional[Reranker] = None
_ollama_llm_client: Optional[OllamaClient] = None


def _get_embedder() -> OllamaEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder(
            model=settings.ollama_embed_model,
            host=settings.ollama_host,
            batch_size=model_config.embed_batch_size,
        )
    return _embedder


def _get_reranker() -> Optional[Reranker]:
    global _reranker
    if not retrieval_config.rerank_enabled:
        return None
    if _reranker is None:
        _reranker = Reranker(model_name=model_config.reranker_model, top_n=retrieval_config.rerank_top_n)
    return _reranker


def _get_ollama_client() -> OllamaClient:
    global _ollama_llm_client
    if _ollama_llm_client is None:
        _ollama_llm_client = OllamaClient(
            model=settings.ollama_llm_model,
            host=settings.ollama_host,
            temperature=model_config.llm_temperature,
            max_tokens=model_config.llm_max_tokens,
        )
    return _ollama_llm_client


def get_decrypted_api_key(db: Session, user_id: str, provider: str) -> Optional[str]:
    """The caller's own decrypted API key for `provider`, or None if they
    haven't added one. Irrelevant for "ollama" (never has a stored key)."""
    row = db.query(UserApiKey).filter_by(user_id=user_id, provider=provider).first()
    if row is None:
        return None
    return decrypt_api_key(row.encrypted_key)


def _get_llm_client(provider: str, model: Optional[str], api_key: Optional[str]) -> LLMClient:
    """Reuse the warm Ollama singleton for the common case (default provider,
    no per-chat model override); build a fresh client for everything else —
    a custom Ollama model, or a cloud provider (which also needs a decrypted
    per-user API key that can change between requests)."""
    if provider == "ollama" and (model is None or model == settings.ollama_llm_model):
        return _get_ollama_client()
    return create_llm_client(
        provider,
        model=model,
        api_key=api_key,
        temperature=model_config.llm_temperature,
        max_tokens=model_config.llm_max_tokens,
    )


def build_response_generator(
    *, provider: str = "ollama", model: Optional[str] = None, api_key: Optional[str] = None
) -> Tuple[ResponseGenerator, VectorStore]:
    """
    Open a fresh VectorStore-backed pipeline for one request. Caller must
    call vector_store.close() when done (use vector_store_lock around the
    whole open-use-close cycle).

    `api_key` is the caller's already-decrypted key for `provider`, required
    for "openai"/"anthropic" and ignored for "ollama".
    """
    # Resolve the LLM client first — it can raise MissingApiKeyError, and
    # VectorStore must not be opened (and left holding Qdrant's embedded
    # file lock) if generation is about to fail anyway.
    llm_client = _get_llm_client(provider, model, api_key)

    vector_store = VectorStore(
        path=settings.full_vector_store_path,
        embedding_dim=model_config.embed_dimensions,
    )
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=_get_embedder(),
        top_k=retrieval_config.vector_top_k,
    )
    context_assembler = ContextAssembler(max_context_tokens=retrieval_config.max_context_tokens)
    retrieval_pipeline = RetrievalPipeline(
        vector_retriever=vector_retriever,
        reranker=_get_reranker(),
        context_assembler=context_assembler,
        rerank_enabled=retrieval_config.rerank_enabled,
        graph_expansion_enabled=False,
    )
    response_gen = ResponseGenerator(retrieval_pipeline=retrieval_pipeline, llm_client=llm_client)
    return response_gen, vector_store

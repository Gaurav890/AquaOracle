"""Vector retriever using Qdrant."""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.indexing.vector_store import VectorStore
from src.embedding.ollama_embedder import OllamaEmbedder


class VectorRetriever:
    """Retrieve relevant chunks using vector similarity search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: OllamaEmbedder,
        top_k: int = 50,
    ):
        """
        Initialize vector retriever.

        Args:
            vector_store: Vector store instance
            embedder: Embedding model
            top_k: Number of results to retrieve
        """
        self.logger = logger.bind(name="VectorRetriever")
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results (overrides default)
            doc_filter: Filter by document metadata
            score_threshold: Minimum similarity score

        Returns:
            List of retrieved chunks with scores and metadata
        """
        top_k = top_k or self.top_k

        self.logger.info(f"Retrieving top {top_k} results for query")

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            doc_filter=doc_filter,
            score_threshold=score_threshold,
        )

        # Convert to dict format
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append({
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "text": result.text,
                "score": result.score,
                "page_numbers": result.page_numbers,
                "metadata": result.metadata,
            })

        self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks

    def retrieve_by_doc_id(
        self,
        query: str,
        doc_id: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document.

        Args:
            query: Search query
            doc_id: Document ID to filter by
            top_k: Number of results

        Returns:
            List of retrieved chunks from the specified document
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            doc_filter={"doc_id": doc_id},
        )

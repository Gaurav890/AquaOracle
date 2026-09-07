"""Qdrant vector store for embeddings."""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from loguru import logger


@dataclass
class SearchResult:
    """Vector search result."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    page_numbers: List[int]
    metadata: Dict[str, Any]


class VectorStore:
    """Qdrant-based vector store."""

    def __init__(
        self,
        collection_name: str = "documents",
        path: Optional[Path] = None,
        embedding_dim: int = 768,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
            path: Path to store data (embedded mode)
            embedding_dim: Embedding dimension
        """
        self.logger = logger.bind(name="VectorStore")
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Initialize Qdrant client in embedded mode
        if path:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.logger.info(f"Creating collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema="keyword",
            )

            self.logger.info(f"Collection {self.collection_name} created")
        else:
            self.logger.info(f"Collection {self.collection_name} already exists")

    def add_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """
        Add embeddings to vector store.

        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
            payloads: List of metadata payloads

        Returns:
            True if successful
        """
        try:
            # Point IDs are derived deterministically from chunk_id (not a local
            # per-call counter) so that ingesting multiple documents doesn't
            # collide on the same small integer IDs and silently overwrite each
            # other's vectors. This also makes re-ingesting a chunk idempotent.
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk_id,
                        **payload,
                    }
                )
                for chunk_id, embedding, payload in zip(chunk_ids, embeddings, payloads)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            self.logger.info(f"Added {len(points)} embeddings to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add embeddings: {e}")
            return False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        doc_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            doc_filter: Filter by document metadata
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

            # Build Qdrant filter if needed. A list value means "any of
            # these" (e.g. doc_id in [chat's own docs + shared docs]) —
            # everything else is a plain equality match.
            qdrant_filter = None
            if doc_filter:
                conditions = []
                for k, v in doc_filter.items():
                    if isinstance(v, (list, tuple, set)):
                        conditions.append(FieldCondition(key=k, match=MatchAny(any=list(v))))
                    else:
                        conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
                qdrant_filter = Filter(must=conditions)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            ).points

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                payload = result.payload

                search_results.append(SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", ""),
                    score=result.score,
                    page_numbers=payload.get("page_numbers", []),
                    metadata=payload,
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all vectors for a document."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )

            self.logger.info(f"Deleted vectors for document {doc_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "total_vectors": collection_info.points_count,
                "vector_dim": self.embedding_dim,
                "collection_name": self.collection_name,
            }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self) -> None:
        """
        Release the embedded Qdrant client and its file lock.

        Qdrant's embedded (local) mode holds an exclusive lock on the storage
        path for as long as the client is open, so any code that opens a
        short-lived VectorStore (e.g. to clear the collection) must close it
        before another VectorStore can open the same path.
        """
        self.client.close()

    def clear(self) -> bool:
        """Clear all vectors from collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            self.logger.info(f"Cleared collection {self.collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False

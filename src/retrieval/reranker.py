"""Re-ranking retrieved chunks using cross-encoder."""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from loguru import logger


class Reranker:
    """Re-rank retrieved chunks using cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 10,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            top_n: Number of top results to keep after re-ranking
        """
        self.logger = logger.bind(name="Reranker")
        self.top_n = top_n

        try:
            self.model = CrossEncoder(model_name)
            self.logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks based on query relevance.

        Args:
            query: Search query
            chunks: List of retrieved chunks
            top_n: Number of top results to return

        Returns:
            Re-ranked list of chunks
        """
        if not self.model:
            self.logger.warning("Reranker not available, returning original order")
            return chunks[:top_n or self.top_n]

        top_n = top_n or self.top_n

        if len(chunks) == 0:
            return []

        self.logger.info(f"Re-ranking {len(chunks)} chunks")

        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, chunk["text"]) for chunk in chunks]

            # Get scores from cross-encoder
            scores = self.model.predict(pairs)

            # Add scores to chunks and sort
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
                chunk["original_score"] = chunk.get("score", 0.0)

            # Sort by rerank score
            reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

            # Return top N
            result = reranked[:top_n]

            self.logger.info(f"Returning top {len(result)} re-ranked chunks")
            return result

        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            return chunks[:top_n]

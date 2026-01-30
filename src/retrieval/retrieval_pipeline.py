"""Complete retrieval pipeline orchestration."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""

    chunks: List[Dict[str, Any]]
    citation_map: Dict[str, Any]
    total_tokens: int
    metadata: Dict[str, Any]


class RetrievalPipeline:
    """
    Complete multi-stage retrieval pipeline.

    Stages:
    1. Vector search → Top K candidates
    2. Re-ranking → Top N chunks
    3. Graph expansion → Add related context (TODO)
    4. Context assembly → Build final context
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        reranker: Optional[Reranker] = None,
        context_assembler: Optional[ContextAssembler] = None,
        rerank_enabled: bool = True,
        graph_expansion_enabled: bool = False,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            vector_retriever: Vector retrieval component
            reranker: Re-ranking component
            context_assembler: Context assembly component
            rerank_enabled: Enable re-ranking stage
            graph_expansion_enabled: Enable graph expansion stage
        """
        self.logger = logger.bind(name="RetrievalPipeline")
        self.vector_retriever = vector_retriever
        self.reranker = reranker or Reranker()
        self.context_assembler = context_assembler or ContextAssembler()
        self.rerank_enabled = rerank_enabled
        self.graph_expansion_enabled = graph_expansion_enabled

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        top_n: int = 10,
        doc_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Execute full retrieval pipeline.

        Args:
            query: Search query
            top_k: Number of candidates from vector search
            top_n: Number of final chunks after re-ranking
            doc_filter: Filter by document metadata

        Returns:
            RetrievalResult with chunks and citation map
        """
        self.logger.info(f"Starting retrieval pipeline for query: {query[:50]}...")

        metadata = {
            "query": query,
            "stages": [],
        }

        # Stage 1: Vector Search
        self.logger.info(f"Stage 1: Vector search (top_k={top_k})")
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
            doc_filter=doc_filter,
        )
        metadata["stages"].append({
            "stage": "vector_search",
            "results_count": len(vector_results),
            "top_k": top_k,
        })

        current_results = vector_results

        # Stage 2: Re-ranking
        if self.rerank_enabled and self.reranker:
            self.logger.info(f"Stage 2: Re-ranking (top_n={top_n})")
            current_results = self.reranker.rerank(
                query=query,
                chunks=current_results,
                top_n=top_n,
            )
            metadata["stages"].append({
                "stage": "reranking",
                "results_count": len(current_results),
                "top_n": top_n,
            })

        # Stage 3: Graph Expansion (TODO)
        if self.graph_expansion_enabled:
            self.logger.info("Stage 3: Graph expansion (not yet implemented)")
            # TODO: Implement graph expansion
            pass

        # Stage 4: Context Assembly
        self.logger.info("Stage 4: Context assembly")
        final_chunks, citation_map = self.context_assembler.assemble(
            chunks=current_results,
            query=query,
        )

        # Calculate total tokens
        total_tokens = sum(len(chunk.get("text", "")) // 4 for chunk in final_chunks)

        metadata["stages"].append({
            "stage": "context_assembly",
            "final_chunks": len(final_chunks),
            "total_tokens": total_tokens,
        })

        result = RetrievalResult(
            chunks=final_chunks,
            citation_map=citation_map,
            total_tokens=total_tokens,
            metadata=metadata,
        )

        self.logger.info(f"Retrieval complete: {len(final_chunks)} chunks, {total_tokens} tokens")
        return result

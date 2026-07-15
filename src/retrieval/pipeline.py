"""
pipeline.py
-----------
Defines the main RetrievalPipeline combining hybrid retrieval and cross-encoder reranking.
"""

from src.retrieval.fusion import HybridRetriever
from src.retrieval.rerank import Reranker


class RetrievalPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever = None, reranker: Reranker = None):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or Reranker()

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[dict]:
        """
        Full retrieval pipeline:
        1. Hybrid retrieve top-20 candidates (RRF of dense and sparse).
        2. Rerank candidates using CrossEncoder.
        3. Return top_k.
        """
        # Retrieve candidates from the hybrid retriever
        pool_size = max(top_k, 20)
        candidates = self.hybrid_retriever.retrieve(query, corpus_id=corpus_id, top_k=pool_size)

        # Rerank the candidates and return top_k
        return self.reranker.rerank(query, candidates, top_k=top_k)

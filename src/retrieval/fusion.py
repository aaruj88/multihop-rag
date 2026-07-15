"""
fusion.py
---------
Combines dense and sparse retriever results using Reciprocal Rank Fusion (RRF).
"""

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever = None, sparse_retriever: SparseRetriever = None):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves top_k chunks using Reciprocal Rank Fusion (RRF) over dense and sparse results.
        """
        # Retrieve candidates from both retrievers.
        # We retrieve a larger candidate pool (e.g. max(top_k * 2, 50)) to get high-quality fusion.
        pool_k = max(top_k * 2, 50)
        dense_results = self.dense_retriever.retrieve(query, corpus_id=corpus_id, top_k=pool_k)
        sparse_results = self.sparse_retriever.retrieve(query, corpus_id=corpus_id, top_k=pool_k)

        k = 60
        scores = {}
        chunks_map = {}

        # Apply RRF for dense results
        for rank, res in enumerate(dense_results, 1):
            cid = res["chunk_id"]
            if cid not in chunks_map:
                chunks_map[cid] = res
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

        # Apply RRF for sparse results
        for rank, res in enumerate(sparse_results, 1):
            cid = res["chunk_id"]
            if cid not in chunks_map:
                chunks_map[cid] = res
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

        # Sort the documents based on their RRF scores
        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        # Construct the final retrieved list
        fused_results = []
        for cid in sorted_cids[:top_k]:
            chunk_copy = dict(chunks_map[cid])
            chunk_copy["score"] = scores[cid]
            fused_results.append(chunk_copy)

        return fused_results

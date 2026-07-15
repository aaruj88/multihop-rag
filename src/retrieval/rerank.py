"""
rerank.py
---------
Reranks retrieved candidate chunks using BAAI/bge-reranker-base cross-encoder.
"""

import os


class MockCrossEncoder:
    def predict(self, pairs, **kwargs):
        import numpy as np
        return np.zeros(len(pairs), dtype=np.float32)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        if os.getenv("MOCK_MODELS", "false").lower() == "true":
            self.model = MockCrossEncoder()
        else:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cpu" if os.getenv("FORCE_CPU", "true").lower() == "true" or not torch.cuda.is_available() else "cuda"
            self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reranks a list of candidate chunks for a query using the cross-encoder.
        """
        if not candidates:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Predict scores (higher score means more relevant)
        scores = self.model.predict(pairs)

        # Update candidate scores and sort
        reranked = []
        for c, score in zip(candidates, scores):
            c_copy = dict(c)
            c_copy["score"] = float(score)
            reranked.append(c_copy)

        # Sort by score descending
        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked[:top_k]

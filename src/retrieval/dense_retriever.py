"""
dense_retriever.py
------------------
Retrieves the most relevant chunks from Qdrant using dense vector search.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class MockSentenceTransformer:
    def encode(self, texts, **kwargs):
        import numpy as np
        if isinstance(texts, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(texts), 384).astype(np.float32)


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", collection_name: str = "papers"):
        self.collection_name = collection_name
        
        if os.getenv("MOCK_MODELS", "false").lower() == "true":
            self.model = MockSentenceTransformer()
        else:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cpu" if os.getenv("FORCE_CPU", "true").lower() == "true" or not torch.cuda.is_available() else "cuda"
            if device == "cpu":
                torch.set_num_threads(1)
            self.model = SentenceTransformer(model_name, device=device)
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[dict]:
        """
        Embed query, perform cosine similarity search in Qdrant, and return top_k chunks.
        """
        # BGE models can benefit from a query instruction, but bge-small-en-v1.5 works well directly
        query_vector = self.model.encode(query).tolist()
        
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="corpus_id",
                    match=MatchValue(value=corpus_id)
                )
            ]
        )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        ).points
        
        retrieved = []
        for r in results:
            retrieved.append({
                "score": r.score,
                "chunk_id": r.id,
                "source_file": r.payload.get("source_file"),
                "page_number": r.payload.get("page_number"),
                "text": r.payload.get("text"),
                "char_start": r.payload.get("char_start"),
                "char_end": r.payload.get("char_end"),
                "corpus_id": r.payload.get("corpus_id")
            })
            
        return retrieved

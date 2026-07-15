"""
sparse_retriever.py
------------------
Retrieves relevant chunks using BM25 (rank_bm25) over the processed chunk corpus.
Caches the BM25 index to disk to avoid rebuilding on every run.
"""

import json
import os
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def tokenize(text: str) -> list[str]:
    """
    Tokenizes text by lowercasing and splitting on word boundaries.
    """
    return re.findall(r"\b\w+\b", text.lower())


class SparseRetriever:
    def __init__(self):
        # In-memory cache for BM25 indices keyed by corpus_id
        # corpus_id -> (bm25, chunks)
        self.indices = {}

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def _build_index_for_corpus(self, corpus_id: str) -> tuple[BM25Okapi | None, list[dict]]:
        """
        Lazily fetches all chunks for corpus_id from Qdrant and builds the BM25 index.
        """
        chunks = []
        offset = None
        collection_name = "papers"

        if not self.client.collection_exists(collection_name):
            return None, []

        while True:
            records, next_offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="corpus_id",
                            match=MatchValue(value=corpus_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for r in records:
                payload = r.payload or {}
                chunks.append({
                    "chunk_id": r.id,
                    "source_file": payload.get("source_file"),
                    "page_number": payload.get("page_number"),
                    "text": payload.get("text", ""),
                    "char_start": payload.get("char_start"),
                    "char_end": payload.get("char_end"),
                    "corpus_id": payload.get("corpus_id")
                })
            if next_offset is None:
                break
            offset = next_offset

        if not chunks:
            return None, []

        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, chunks

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves top_k chunks for the given query and corpus_id using BM25 scoring.
        """
        if corpus_id not in self.indices:
            bm25, chunks = self._build_index_for_corpus(corpus_id)
            self.indices[corpus_id] = (bm25, chunks)

        bm25, chunks = self.indices[corpus_id]
        if not bm25 or not chunks:
            return []

        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            scored_chunks.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "corpus_id": chunk["corpus_id"]
            })

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def clear_cache(self, corpus_id: str) -> None:
        """
        Clears the cached BM25 index for the corpus.
        """
        if corpus_id in self.indices:
            del self.indices[corpus_id]

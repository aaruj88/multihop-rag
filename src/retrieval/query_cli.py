"""
query_cli.py
------------
Simple command-line interface to test retrieval modes (dense, sparse, hybrid, or multihop) on Qdrant,
with optional cross-encoder reranking.
"""

import argparse
import sys
import time
from dotenv import load_dotenv

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.fusion import HybridRetriever
from src.retrieval.rerank import Reranker
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.multihop import MultiHopRetriever

load_dotenv()


def main():
    # Configure UTF-8 encoding for stdout on Windows to prevent UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Command-line interface to test multihop RAG retrieval modes."
    )
    parser.add_argument(
        "query",
        type=str,
        help="The query string to search for."
    )
    parser.add_argument(
        "top_k",
        type=int,
        nargs="?",
        default=5,
        help="Number of top results to return (default: 5)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "sparse", "hybrid", "multihop"],
        default="hybrid",
        help="Retrieval mode to use: 'dense', 'sparse', 'hybrid', or 'multihop' (default: 'hybrid')."
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking using BAAI/bge-reranker-base."
    )

    args = parser.parse_args()
    query = args.query
    top_k = args.top_k
    mode = args.mode
    rerank_enabled = args.rerank

    print(f"Querying in '{mode}' mode (rerank={rerank_enabled}): '{query}' (top_k={top_k})...\n")

    t0 = time.perf_counter()

    if mode == "multihop":
        if rerank_enabled:
            retriever = MultiHopRetriever()
        else:
            retriever = MultiHopRetriever(pipeline=HybridRetriever())
        results = retriever.retrieve(query, top_k=top_k)
    elif rerank_enabled:
        if mode == "hybrid":
            retriever = RetrievalPipeline()
            results = retriever.retrieve(query, top_k=top_k)
        else:
            if mode == "dense":
                base_retriever = DenseRetriever()
            else:
                base_retriever = SparseRetriever()
            candidates = base_retriever.retrieve(query, top_k=max(top_k, 20))
            reranker = Reranker()
            results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        if mode == "dense":
            retriever = DenseRetriever()
        elif mode == "sparse":
            retriever = SparseRetriever()
        else:
            retriever = HybridRetriever()
        results = retriever.retrieve(query, top_k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not results:
        print("No results found.")
        print(f"\nLatency: {elapsed_ms:.2f} ms")
        return

    for i, res in enumerate(results, 1):
        print(f"[{i}] Score: {res['score']:.4f} | Source: {res['source_file']} (Page {res['page_number']})")
        if "matched_queries" in res:
            print(f"    Matched Queries: {res['matched_queries']}")
        print(f"    Text: {res['text'][:300]}...")
        print("-" * 80)

    print(f"\nLatency: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()

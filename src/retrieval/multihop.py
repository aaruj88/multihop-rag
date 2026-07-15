"""
multihop.py
-----------
Orchestrates multi-hop retrieval by decomposing queries, retrieving chunks
for each sub-query, deduplicating them, and labeling each chunk with its
corresponding sub-queries.
"""

from src.generation.decompose import decompose_query
from src.retrieval.pipeline import RetrievalPipeline


class MultiHopRetriever:
    def __init__(self, pipeline: RetrievalPipeline = None):
        self.pipeline = pipeline or RetrievalPipeline()

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[dict]:
        """
        Performs multi-hop retrieval:
        1. Decomposes the query into sub-questions.
        2. Retrieves top_k chunks for each sub-question.
        3. Merges and deduplicates chunks across all sub-questions.
        4. Annotates each chunk with a 'matched_queries' list.
        5. Returns the top_k sorted by score.
        """
        # 1. Decompose the query
        sub_queries = decompose_query(query)

        # 2. Retrieve for each sub-query
        merged_results = {}
        for sub_q in sub_queries:
            sub_results = self.pipeline.retrieve(sub_q, corpus_id=corpus_id, top_k=top_k)
            for res in sub_results:
                cid = res["chunk_id"]
                if cid not in merged_results:
                    res_copy = dict(res)
                    res_copy["matched_queries"] = [sub_q]
                    merged_results[cid] = res_copy
                else:
                    # Deduplicate and label
                    if sub_q not in merged_results[cid]["matched_queries"]:
                        merged_results[cid]["matched_queries"].append(sub_q)
                    # Keep the highest relevance score
                    if res["score"] > merged_results[cid]["score"]:
                        merged_results[cid]["score"] = res["score"]

        # 3. Sort by score descending and return top_k
        sorted_results = sorted(merged_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

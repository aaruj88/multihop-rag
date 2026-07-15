# Evaluation Results: RAG Configurations Comparison

Evaluated on a set of 25 questions containing a mix of single-hop and multi-hop queries from the paper corpus.

## Overall Configuration Metrics (Averages)

| Configuration | Faithfulness | Answer Relevance | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | 1.0000 | 0.8439 | 0.4036 | 0.6522 |
| **Hybrid + Rerank** | 1.0000 | 0.8356 | 0.6703 | 0.6957 |
| **Full Multihop Pipeline** | 1.0000 | 0.8358 | 0.6355 | 0.7609 |

## Single-Hop Questions Performance

| Configuration | Faithfulness | Answer Relevance | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | 1.0000 | 0.8310 | 0.4522 | 0.8000 |
| **Hybrid + Rerank** | 1.0000 | 0.8219 | 0.8000 | 0.8000 |
| **Full Multihop Pipeline** | 1.0000 | 0.8243 | 0.7133 | 0.8667 |

## Multi-Hop Questions Performance

| Configuration | Faithfulness | Answer Relevance | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Only** | 1.0000 | 0.8681 | 0.3125 | 0.3750 |
| **Hybrid + Rerank** | 1.0000 | 0.8612 | 0.4271 | 0.5000 |
| **Full Multihop Pipeline** | 1.0000 | 0.8572 | 0.4896 | 0.5625 |

## Key Insights

1. **Decomposition Impact**: The **Full Multihop Pipeline** (which utilizes query decomposition) significantly outperforms the other configurations on **multi-hop questions**, particularly in **Context Recall** and **Context Precision**. This is because decomposing a complex query into independent sub-questions allows the retriever to retrieve relevant documents across different hops that a single joint query fails to fetch.
2. **Hybrid + Reranking**: Combining dense and sparse retrieval with cross-encoder reranking yields substantial gains over pure dense retrieval on single-hop questions, improving precision.
3. **Faithfulness**: Faithfulness scores remain high across all configurations due to the extractive local synthesis mechanism, ensuring that all claims in generated answers are strictly grounded in retrieved chunks.

"""
tests/test_retrieval.py
-----------------------
Unit and integration tests for SparseRetriever and HybridRetriever.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.fusion import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever, tokenize
from src.retrieval.rerank import Reranker
from src.retrieval.pipeline import RetrievalPipeline


@pytest.fixture
def temp_corpus_file():
    """Create a temporary chunks.jsonl file with sample content."""
    chunks = [
        {
            "chunk_id": "chunk-1",
            "source_file": "doc1.pdf",
            "page_number": 1,
            "text": "The quick brown fox jumps over the lazy dog.",
            "char_start": 0,
            "char_end": 43,
        },
        {
            "chunk_id": "chunk-2",
            "source_file": "doc1.pdf",
            "page_number": 1,
            "text": "Artificial intelligence and machine learning are expanding.",
            "char_start": 44,
            "char_end": 103,
        },
        {
            "chunk_id": "chunk-3",
            "source_file": "doc2.pdf",
            "page_number": 2,
            "text": "Deep neural networks are used for natural language processing.",
            "char_start": 0,
            "char_end": 62,
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks_path = Path(tmpdir) / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        yield chunks_path


def test_tokenize():
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_sparse_retriever_caching_and_search():
    with patch("src.retrieval.sparse_retriever.QdrantClient") as mock_qdrant_class:
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        mock_client.collection_exists.return_value = True
        
        # Mock scroll to return our chunks
        mock_record1 = MagicMock()
        mock_record1.id = "chunk-1"
        mock_record1.payload = {
            "source_file": "doc1.pdf",
            "page_number": 1,
            "text": "The quick brown fox jumps over the lazy dog.",
            "char_start": 0,
            "char_end": 43,
            "corpus_id": "test-corpus"
        }
        
        mock_record2 = MagicMock()
        mock_record2.id = "chunk-2"
        mock_record2.payload = {
            "source_file": "doc2.pdf",
            "page_number": 2,
            "text": "Deep neural networks are used for natural language processing.",
            "char_start": 0,
            "char_end": 62,
            "corpus_id": "test-corpus"
        }
        
        mock_record3 = MagicMock()
        mock_record3.id = "chunk-3"
        mock_record3.payload = {
            "source_file": "doc3.pdf",
            "page_number": 3,
            "text": "Some other totally unrelated text here.",
            "char_start": 0,
            "char_end": 39,
            "corpus_id": "test-corpus"
        }
        
        mock_client.scroll.side_effect = [
            ([mock_record1, mock_record2, mock_record3], None)
        ]
        
        retriever = SparseRetriever()
        results = retriever.retrieve("neural networks", corpus_id="test-corpus", top_k=2)
        
        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk-2"
        assert results[0]["corpus_id"] == "test-corpus"
        assert "score" in results[0]
        
        # Verify schema
        first = results[0]
        required_keys = {"score", "chunk_id", "source_file", "page_number", "text", "char_start", "char_end", "corpus_id"}
        assert required_keys.issubset(first.keys())


def test_hybrid_retriever_rrf_logic():
    # Mock dense and sparse retrievers to return deterministic ranked lists
    dense_mock = MagicMock()
    sparse_mock = MagicMock()

    # Mock return values (schema matches retrieved chunks)
    dense_mock.retrieve.return_value = [
        {"chunk_id": "chunk-a", "source_file": "a.pdf", "page_number": 1, "text": "A", "char_start": 0, "char_end": 1, "corpus_id": "test-corpus"},
        {"chunk_id": "chunk-b", "source_file": "b.pdf", "page_number": 1, "text": "B", "char_start": 0, "char_end": 1, "corpus_id": "test-corpus"},
    ]
    sparse_mock.retrieve.return_value = [
        {"chunk_id": "chunk-b", "source_file": "b.pdf", "page_number": 1, "text": "B", "char_start": 0, "char_end": 1, "corpus_id": "test-corpus"},
        {"chunk_id": "chunk-c", "source_file": "c.pdf", "page_number": 1, "text": "C", "char_start": 0, "char_end": 1, "corpus_id": "test-corpus"},
    ]

    hybrid = HybridRetriever(dense_retriever=dense_mock, sparse_retriever=sparse_mock)
    results = hybrid.retrieve("query text", corpus_id="test-corpus", top_k=3)

    # Assert retrieved count
    assert len(results) == 3

    assert results[0]["chunk_id"] == "chunk-b"
    assert results[1]["chunk_id"] == "chunk-a"
    assert results[2]["chunk_id"] == "chunk-c"

    # Verify score
    assert pytest.approx(results[0]["score"]) == (1 / 61 + 1 / 62)
    assert pytest.approx(results[1]["score"]) == (1 / 61)
    assert pytest.approx(results[2]["score"]) == (1 / 62)


@patch("sentence_transformers.CrossEncoder")
def test_reranker(mock_cross_encoder_class):
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.1, 0.9]

    reranker = Reranker()
    candidates = [
        {"chunk_id": "c1", "text": "text 1"},
        {"chunk_id": "c2", "text": "text 2"},
    ]
    results = reranker.rerank("query", candidates, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"  # higher score first
    assert results[0]["score"] == 0.9
    assert results[1]["chunk_id"] == "c1"


@patch("sentence_transformers.CrossEncoder")
def test_pipeline(mock_cross_encoder_class):
    mock_model = MagicMock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [float(i) for i in range(20)]

    hybrid_mock = MagicMock()
    hybrid_mock.retrieve.return_value = [
        {"chunk_id": f"chunk-{i}", "text": f"text {i}"} for i in range(20)
    ]

    reranker = Reranker()
    pipeline = RetrievalPipeline(hybrid_retriever=hybrid_mock, reranker=reranker)
    results = pipeline.retrieve("query", corpus_id="test-corpus", top_k=5)

    assert len(results) == 5
    assert results[0]["chunk_id"] == "chunk-19"
    assert results[0]["score"] == 19.0
    hybrid_mock.retrieve.assert_called_once_with("query", corpus_id="test-corpus", top_k=20)


@patch("src.generation.decompose.Groq")
@patch("src.generation.decompose.os.getenv")
def test_decompose_query_multihop(mock_getenv, mock_groq_class):
    mock_getenv.return_value = "mock_key"
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"sub_queries": ["sub-q1", "sub-q2"]}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    from src.generation.decompose import decompose_query
    sub_qs = decompose_query("Compare A and B")
    assert sub_qs == ["sub-q1", "sub-q2"]


@patch("src.retrieval.multihop.decompose_query")
def test_multihop_retriever(mock_decompose):
    mock_decompose.return_value = ["sub-q1", "sub-q2"]

    pipeline_mock = MagicMock()
    pipeline_mock.retrieve.side_effect = [
        [
            {"chunk_id": "chunk-1", "score": 0.9, "text": "text 1", "source_file": "doc1.pdf", "page_number": 1, "char_start": 0, "char_end": 10, "corpus_id": "test-corpus"},
            {"chunk_id": "chunk-2", "score": 0.8, "text": "text 2", "source_file": "doc1.pdf", "page_number": 1, "char_start": 10, "char_end": 20, "corpus_id": "test-corpus"},
        ],
        [
            {"chunk_id": "chunk-2", "score": 0.95, "text": "text 2", "source_file": "doc1.pdf", "page_number": 1, "char_start": 10, "char_end": 20, "corpus_id": "test-corpus"},
            {"chunk_id": "chunk-3", "score": 0.7, "text": "text 3", "source_file": "doc2.pdf", "page_number": 2, "char_start": 0, "char_end": 10, "corpus_id": "test-corpus"},
        ]
    ]

    from src.retrieval.multihop import MultiHopRetriever
    retriever = MultiHopRetriever(pipeline=pipeline_mock)
    results = retriever.retrieve("Compare A and B", corpus_id="test-corpus", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-2"
    assert results[0]["score"] == 0.95
    assert set(results[0]["matched_queries"]) == {"sub-q1", "sub-q2"}

    assert results[1]["chunk_id"] == "chunk-1"
    assert results[1]["score"] == 0.9
    assert results[1]["matched_queries"] == ["sub-q1"]


def test_reranker_disabled():
    with patch.dict("os.environ", {"DISABLE_RERANKER": "true"}):
        reranker = Reranker()
        assert reranker.disabled is True
        assert reranker.model is None
        
        candidates = [
            {"chunk_id": "chunk-1", "text": "text 1", "score": 0.9},
            {"chunk_id": "chunk-2", "text": "text 2", "score": 0.8},
            {"chunk_id": "chunk-3", "text": "text 3", "score": 0.7},
        ]
        
        res = reranker.rerank("query", candidates, top_k=2)
        assert len(res) == 2
        assert res[0]["chunk_id"] == "chunk-1"
        assert res[1]["chunk_id"] == "chunk-2"


def test_reranker_auto_disabled_on_render():
    with patch.dict("os.environ", {"RENDER": "true"}):
        reranker = Reranker()
        assert reranker.disabled is True
        assert reranker.model is None





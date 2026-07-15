import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Force a clean environment limit if needed, or rely on default env
os.environ["MAX_ACTIVE_CORPORA"] = "3"

from src.api.main import app, limiter, cleanup_expired_corpora_job
from src.generation.synthesize import AnswerSchema

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_limiter():
    limiter.reset()

@patch("src.api.main.get_corpus_status")
@patch("src.api.main.decompose_query")
@patch("src.api.main.synthesize_answer")
def test_byok_rate_limiting_no_key(mock_synth, mock_decomp, mock_status):
    """Verify that without a key, the 6th rapid request is blocked by rate limiting (5/min)."""
    mock_status.return_value = {"status": "ready", "chunk_count": 5, "file_count": 1}
    mock_decomp.return_value = ["sub-query"]
    mock_synth.return_value = AnswerSchema(
        answer_text="Mock answer.",
        citations=["chunk-1"],
        unanswered_aspects=[]
    )

    # First 5 requests should succeed
    for _ in range(5):
        response = client.post(
            "/query",
            json={"question": "What is attention?", "corpus_id": "test-corpus", "top_k": 3}
        )
        assert response.status_code == 200, f"Failed at request: {response.text}"

    # 6th request should fail with 429
    response = client.post(
        "/query",
        json={"question": "What is attention?", "corpus_id": "test-corpus", "top_k": 3}
    )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text


@patch("src.api.main.get_corpus_status")
@patch("src.api.main.decompose_query")
@patch("src.api.main.synthesize_answer")
def test_byok_rate_limiting_with_key(mock_synth, mock_decomp, mock_status):
    """Verify that with a key, we can make up to 10 rapid requests successfully (limit is 20/min)."""
    mock_status.return_value = {"status": "ready", "chunk_count": 5, "file_count": 1}
    mock_decomp.return_value = ["sub-query"]
    mock_synth.return_value = AnswerSchema(
        answer_text="Mock answer.",
        citations=["chunk-1"],
        unanswered_aspects=[]
    )

    # 10 requests should succeed because BYOK rate limit is 20/minute
    for _ in range(10):
        response = client.post(
            "/query",
            json={
                "question": "What is attention?",
                "corpus_id": "test-corpus",
                "top_k": 3,
                "groq_api_key": "some_test_key"
            }
        )
        assert response.status_code == 200, f"Failed at request: {response.text}"


@patch("src.api.main.get_total_active_corpora")
def test_global_corpus_cap(mock_get_total):
    """Verify new uploads are rejected if active corpora exceeds cap (MAX_ACTIVE_CORPORA=3 in test env)."""
    # Active corpora is 3, which equals the cap of 3
    mock_get_total.return_value = 3

    # Attempt to upload new corpus
    files = [("files", ("test.pdf", b"pdf content", "application/pdf"))]
    response = client.post(
        "/corpus",
        files=files,
        data={"groq_api_key": "some_key"}
    )

    assert response.status_code == 400
    assert "Global corpus limit reached" in response.json()["detail"]


@patch("src.api.main.get_expired_corpora")
@patch("src.api.main.delete_corpus_helper")
def test_cleanup_expired_corpora_job(mock_delete, mock_expired):
    """Verify cleanup job finds and deletes expired corpora older than 24 hours."""
    mock_expired.return_value = ["corpus-expired-1", "corpus-expired-2"]

    cleanup_expired_corpora_job()

    assert mock_delete.call_count == 2
    mock_delete.assert_any_call("corpus-expired-1")
    mock_delete.assert_any_call("corpus-expired-2")


@patch("src.api.main.get_corpus_status")
@patch("src.api.main.decompose_query")
@patch("src.api.main.synthesize_answer")
def test_key_passing_to_llm_calls(mock_synth, mock_decomp, mock_status):
    """Verify that groq_api_key provided in query is passed to decompose_query and synthesize_answer."""
    mock_status.return_value = {"status": "ready", "chunk_count": 5, "file_count": 1}
    mock_decomp.return_value = ["sub-query"]
    mock_synth.return_value = AnswerSchema(
        answer_text="Mock answer.",
        citations=["chunk-1"],
        unanswered_aspects=[]
    )

    test_key = "special_secret_byok_key"
    
    # We patch retriever.pipeline.retrieve to return dummy chunks
    with patch("src.api.main.retriever.pipeline.retrieve") as mock_retrieve:
        mock_retrieve.return_value = [{"chunk_id": "chunk-1", "corpus_id": "test-corpus", "text": "mock text", "source_file": "file.pdf", "page_number": 1, "score": 0.9}]
        response = client.post(
            "/query",
            json={
                "question": "What is attention?",
                "corpus_id": "test-corpus",
                "top_k": 3,
                "groq_api_key": test_key
            }
        )
    assert response.status_code == 200

    # Verify decompose_query called with api_key
    mock_decomp.assert_called_once_with("What is attention?", api_key=test_key)

    # Verify synthesize_answer called with api_key
    expected_chunks = [
        {
            "chunk_id": "chunk-1",
            "corpus_id": "test-corpus",
            "text": "mock text",
            "source_file": "file.pdf",
            "page_number": 1,
            "score": 0.9,
            "matched_queries": ["sub-query"]
        }
    ]
    mock_synth.assert_called_once_with(
        "What is attention?",
        expected_chunks,
        corpus_id="test-corpus",
        api_key=test_key
    )

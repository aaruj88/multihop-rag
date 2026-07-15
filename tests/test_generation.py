"""
tests/test_generation.py
------------------------
Unit and integration tests for the generation/synthesis module.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.generation.synthesize import AnswerSchema, synthesize_answer, run_pipeline


def test_answer_schema_parsing():
    """Verify that AnswerSchema behaves correctly as a Pydantic model."""
    data = {
        "answer_text": "This is a test [1].",
        "citations": ["chunk-uuid-123"],
        "unanswered_aspects": ["some aspect"]
    }
    schema = AnswerSchema(**data)
    assert schema.answer_text == "This is a test [1]."
    assert schema.citations == ["chunk-uuid-123"]
    assert schema.unanswered_aspects == ["some aspect"]


@patch("src.generation.synthesize.os.getenv")
def test_synthesize_answer_missing_key(mock_getenv):
    """Verify fallback behavior when GROQ_API_KEY is not set."""
    mock_getenv.return_value = None
    
    query = "What is attention?"
    chunks = [{"chunk_id": "uuid-1", "text": "attention context", "corpus_id": "test-corpus"}]
    
    result = synthesize_answer(query, chunks, corpus_id="test-corpus")
    assert "GROQ_API_KEY not configured" in result.answer_text
    assert result.citations == []
    assert result.unanswered_aspects == [query]


@patch("src.generation.synthesize.os.getenv")
def test_synthesize_answer_empty_chunks(mock_getenv):
    """Verify fallback behavior when retrieved chunks is empty."""
    mock_getenv.return_value = "mock_key"
    
    query = "What is attention?"
    result = synthesize_answer(query, [], corpus_id="test-corpus")
    assert result.answer_text == "I cannot answer this from the given context."
    assert result.citations == []
    assert result.unanswered_aspects == [query]


@patch("src.generation.synthesize.Groq")
@patch("src.generation.synthesize.os.getenv")
def test_synthesize_answer_success(mock_getenv, mock_groq_class):
    """Verify synthesize_answer handles successful Groq API call with structured schema."""
    mock_getenv.return_value = "mock_key"
    
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "answer_text": "Attention maps a query to keys and values [1].",
        "citations": ["chunk-uuid-1"],
        "unanswered_aspects": []
    })
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
 
    query = "How does attention work?"
    chunks = [
        {
            "chunk_id": "chunk-uuid-1",
            "source_file": "doc1.pdf",
            "page_number": 3,
            "text": "Attention mapping function maps query and key-value pairs.",
            "corpus_id": "test-corpus"
        }
    ]
 
    result = synthesize_answer(query, chunks, corpus_id="test-corpus")
    assert result.answer_text == "Attention maps a query to keys and values [1]."
    assert result.citations == ["chunk-uuid-1"]
    assert result.unanswered_aspects == []


@patch("src.generation.synthesize.Groq")
@patch("src.generation.synthesize.os.getenv")
def test_synthesize_answer_api_error(mock_getenv, mock_groq_class):
    """Verify synthesize_answer handles Groq API failure gracefully."""
    mock_getenv.return_value = "mock_key"
    
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API quota exceeded")
 
    query = "How does attention work?"
    chunks = [
        {
            "chunk_id": "chunk-uuid-1",
            "source_file": "doc1.pdf",
            "page_number": 3,
            "text": "Attention mapping function maps query and key-value pairs.",
            "corpus_id": "test-corpus"
        }
    ]
 
    result = synthesize_answer(query, chunks, corpus_id="test-corpus")
    assert "error during synthesis" in result.answer_text
    assert result.citations == []
    assert result.unanswered_aspects == [query]


@patch("src.generation.synthesize.synthesize_answer")
@patch("src.generation.synthesize.MultiHopRetriever")
def test_run_pipeline(mock_retriever_class, mock_synthesize):
    """Verify run_pipeline correctly wires retrieval and synthesis."""
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    
    mock_chunks = [{"chunk_id": "uuid-1", "text": "retrieved text", "corpus_id": "test-corpus"}]
    mock_retriever.retrieve.return_value = mock_chunks
    
    mock_answer = AnswerSchema(
        answer_text="Synthesized answer [1].",
        citations=["uuid-1"],
        unanswered_aspects=[]
    )
    mock_synthesize.return_value = mock_answer

    query = "Run RAG pipeline"
    result = run_pipeline(query, corpus_id="test-corpus", top_k=3)
    
    mock_retriever.retrieve.assert_called_once_with(query, corpus_id="test-corpus", top_k=3)
    mock_synthesize.assert_called_once_with(query, mock_chunks, corpus_id="test-corpus")
    assert result.answer_text == "Synthesized answer [1]."
    assert result.citations == ["uuid-1"]
    assert result.unanswered_aspects == []

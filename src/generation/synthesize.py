"""
synthesize.py
-------------
Handles answer synthesis by prompting Google Gemini with a query and retrieved chunks.
Returns a structured AnswerSchema containing the synthesized text, citations, and unanswered sub-questions.
"""

import json
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

from src.retrieval.multihop import MultiHopRetriever

load_dotenv()


class AnswerSchema(BaseModel):
    """
    Response schema for answer synthesis.
    """
    answer_text: str = Field(
        ...,
        description="The detailed synthesized answer text based ONLY on the provided chunks. Cite the numeric chunk ID (e.g. [1], [2]) after every claim."
    )
    citations: List[str] = Field(
        ...,
        description="List of chunk IDs (UUID strings) of the chunks that were actually used to support the answer."
    )
    unanswered_aspects: List[str] = Field(
        ...,
        description="List of sub-questions or aspects of the query that the provided chunks did not cover."
    )


def synthesize_answer(query: str, retrieved_chunks: List[Dict[str, Any]], corpus_id: str, api_key: str = None) -> AnswerSchema:
    """
    Prompts Gemini with the query and retrieved chunks to synthesize a structured answer.
    """
    for chunk in retrieved_chunks:
        assert chunk.get("corpus_id") == corpus_id, f"Security Violation: leaked chunk {chunk.get('chunk_id')} belongs to different corpus than requested {corpus_id}"

    effective_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not effective_api_key or effective_api_key == "your_groq_api_key_here":
        print("Warning: GROQ_API_KEY is not configured. Returning fallback response.")
        return AnswerSchema(
            answer_text="I cannot answer this from the given context (GROQ_API_KEY not configured).",
            citations=[],
            unanswered_aspects=[query]
        )

    if not retrieved_chunks:
        return AnswerSchema(
            answer_text="I cannot answer this from the given context.",
            citations=[],
            unanswered_aspects=[query]
        )

    # 1. Format the retrieved chunks for the prompt
    chunks_formatted = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        chunk_info = (
            f"[{idx}] Chunk ID: {chunk.get('chunk_id')}\n"
            f"Source: {chunk.get('source_file')} (Page {chunk.get('page_number')})\n"
            f"Text: {chunk.get('text')}\n"
        )
        chunks_formatted.append(chunk_info)
    chunks_str = "\n".join(chunks_formatted)

    # 2. Build the instruction prompt
    prompt = (
        "You are an expert academic assistant. Your task is to synthesize an answer to the query "
        "using ONLY the provided retrieved chunks. Follow these guidelines strictly:\n\n"
        "Guidelines:\n"
        "a) Only make claims that are directly supported by the provided chunks. Do NOT assume, extrapolate, or guess.\n"
        "b) Cite the chunk citation ID (e.g. [1], [2], etc.) in the text after every single claim or fact you mention.\n"
        "c) If any sub-question or aspect of the query is not covered by the chunks, do not try to answer it. "
        "Instead, list the unanswered aspect or sub-question in the 'unanswered_aspects' list.\n"
        "d) If the entire query cannot be answered from the provided chunks, return 'I cannot answer this from the given context' "
        "in the 'answer_text' field, leave 'citations' empty, and put the query in the 'unanswered_aspects' list.\n\n"
        f"Original Query: \"{query}\"\n\n"
        "Retrieved Chunks:\n"
        f"{chunks_str}\n\n"
        "You must respond with a JSON object strictly matching this schema:\n"
        "{\n"
        '  "answer_text": "string (The detailed synthesized answer text based ONLY on the chunks. Cite chunk ID e.g. [1], [2] after every claim)",\n'
        '  "citations": ["string (List of chunk IDs/UUIDs actually used to support the answer)"],\n'
        '  "unanswered_aspects": ["string (List of sub-questions/aspects of the query not covered by the chunks)"]\n'
        "}"
    )

    try:
        client = Groq(api_key=effective_api_key)

        # Call Groq with structured JSON output configuration
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        res_text = response.choices[0].message.content.strip()
        data = json.loads(res_text)

        # Validate structured response keys
        return AnswerSchema(
            answer_text=data.get("answer_text", "I cannot answer this from the given context"),
            citations=data.get("citations", []),
            unanswered_aspects=data.get("unanswered_aspects", [])
        )

    except Exception as e:
        print(f"Error during answer synthesis: {e}. Falling back to default response.")
        return AnswerSchema(
            answer_text="I cannot answer this from the given context (error during synthesis).",
            citations=[],
            unanswered_aspects=[query]
        )


def run_pipeline(query: str, corpus_id: str, top_k: int = 5) -> AnswerSchema:
    """
    End-to-end pipeline: query -> decompose -> retrieve per sub-question -> synthesize.
    """
    retriever = MultiHopRetriever()
    retrieved_chunks = retriever.retrieve(query, corpus_id=corpus_id, top_k=top_k)
    return synthesize_answer(query, retrieved_chunks, corpus_id=corpus_id)


if __name__ == "__main__":
    # Command line interface to test the pipeline
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python -m src.generation.synthesize \"<query>\" [top_k]")
        sys.exit(1)

    query_str = sys.argv[1]
    top_k_val = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Running end-to-end RAG pipeline for query: '{query_str}'...\n")
    result = run_pipeline(query_str, top_k=top_k_val)

    print("=" * 60)
    print("ANSWER TEXT:")
    print(result.answer_text)
    print("-" * 60)
    print("CITATIONS (Chunk UUIDs):")
    for cit in result.citations:
        print(f" - {cit}")
    print("-" * 60)
    print("UNANSWERED ASPECTS:")
    for asp in result.unanswered_aspects:
        print(f" - {asp}")
    print("=" * 60)

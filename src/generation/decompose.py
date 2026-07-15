"""
decompose.py
------------
Handles query decomposition by calling Google Gemini.
Decomposes complex multi-hop queries into 2-4 self-contained sub-questions.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def decompose_query(query: str, api_key: str = None) -> list[str]:
    """
    Decomposes a query into 2-4 sub-queries if it requires multi-hop retrieval,
    otherwise returns the original query unchanged in a list of one element.
    """
    effective_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not effective_api_key or effective_api_key == "your_groq_api_key_here":
        print("Warning: GROQ_API_KEY is not configured. Returning original query.")
        return [query]

    try:
        client = Groq(api_key=effective_api_key)

        prompt = (
            "You are a search query decomposition engine. Your task is to analyze the input query and "
            "decide if it requires information from multiple distinct papers or sections to answer (multi-hop) "
            "or is answerable from a single retrieval (single-hop).\n\n"
            "Format requirements:\n"
            "You must return a JSON object strictly matching this schema:\n"
            "{\n"
            '  "sub_queries": ["string"]\n'
            "}\n\n"
            "Guidelines:\n"
            "- If the query is single-hop (requires only one topic or paper), return the original query as a "
            "single-element array under 'sub_queries': [\"original query\"].\n"
            "- If the query is multi-hop (e.g. comparing two different systems, papers, methods, or requirements), "
            "decompose it into 2 to 4 independent, self-contained sub-questions. Each sub-question must contain all "
            "relevant context (such as paper names, model names, or methods) so that it can be searched for and "
            "retrieved on its own without context from the other sub-questions.\n\n"
            f"Query: \"{query}\"\n\n"
            "JSON Object:"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        res_text = response.choices[0].message.content.strip()
        data = json.loads(res_text)
        sub_queries = data.get("sub_queries", [])

        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            return sub_queries

        print(f"Warning: Unexpected response format from Groq: {res_text}. Returning original query.")
        return [query]

    except Exception as e:
        print(f"Error during query decomposition: {e}. Falling back to original query.")
        return [query]

"""
build_eval_set.py
-----------------
Generates a labeled evaluation dataset (eval_set.jsonl) of ~25 questions
(mix of single-hop and multi-hop) based on processed paper chunks,
using Google Gemini to generate the questions and ground truth answers.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EVAL_OUT_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_OUT_FILE = EVAL_OUT_DIR / "eval_set.jsonl"


class EvalPair(BaseModel):
    question: str = Field(..., description="A specific search question.")
    ground_truth: str = Field(..., description="Detailed, factual ground truth answer based strictly on the supporting chunks.")
    supporting_chunk_ids: List[str] = Field(..., description="The chunk IDs (UUIDs) that contain the facts to answer the question.")
    question_type: str = Field(..., description="The type of the question: 'single-hop' or 'multi-hop'.")


class EvalPairBatch(BaseModel):
    pairs: List[EvalPair]


def load_chunks() -> List[Dict[str, Any]]:
    """Loads chunks from data/processed/chunks.jsonl."""
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} not found. Run ingestion pipeline first.")
        sys.exit(1)
        
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def filter_good_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters out very short chunks or chunks that look like references/acknowledgements."""
    filtered = []
    for c in chunks:
        text = c.get("text", "").strip()
        # Ensure length is reasonable (at least 600 chars) and doesn't contain reference markers
        if len(text) > 600 and "references" not in text[:200].lower() and "acknowledgments" not in text[:200].lower():
            filtered.append(c)
    return filtered


def generate_single_hop_batch(client: Groq, chunks_batch: List[Dict[str, Any]]) -> List[EvalPair]:
    """Generates single-hop questions for a batch of chunks in a single API call."""
    chunks_str = ""
    for idx, c in enumerate(chunks_batch, 1):
        chunks_str += (
            f"--- CHUNK {idx} ---\n"
            f"ID: {c['chunk_id']}\n"
            f"Source: {c['source_file']} (Page {c['page_number']})\n"
            f"Content: {c['text']}\n\n"
        )

    prompt = (
        "You are building a RAG evaluation set. I will provide you with a list of chunks from academic papers. "
        "For EACH chunk, generate exactly one highly specific 'single-hop' question that can be answered "
        "solely by the facts in that chunk, along with the detailed ground truth answer.\n\n"
        "Guidelines:\n"
        "1. The question must be specific (e.g. mention the specific method, paper, or system names if present).\n"
        "2. The question must be answerable using ONLY the provided chunk.\n"
        "3. The ground_truth answer must be detailed, factual, and fully supported by the chunk.\n"
        "4. The supporting_chunk_ids list for each question must contain only the ID of the chunk it was generated from.\n"
        "5. The question_type must be 'single-hop'.\n\n"
        f"{chunks_str}\n"
        "Format as a JSON object matching this schema:\n"
        "{\n"
        '  "pairs": [\n'
        "    {\n"
        '      "question": "string (the specific question)",\n'
        '      "ground_truth": "string (the detailed ground truth answer based strictly on the chunk)",\n'
        '      "supporting_chunk_ids": ["string (the ID of the chunk used)"],\n'
        '      "question_type": "single-hop"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096
        )
        data = json.loads(response.choices[0].message.content.strip())
        pairs = [EvalPair(**p) for p in data.get("pairs", [])]
        return pairs
    except Exception as e:
        print(f"Error generating single-hop batch: {e}")
        return []


def generate_multi_hop_batch(client: Groq, pairs_batch: List[tuple[Dict[str, Any], Dict[str, Any]]]) -> List[EvalPair]:
    """Generates multi-hop questions for a batch of chunk pairs in a single API call."""
    pairs_str = ""
    for idx, (c1, c2) in enumerate(pairs_batch, 1):
        pairs_str += (
            f"--- PAIR {idx} ---\n"
            f"Chunk A ID: {c1['chunk_id']}\n"
            f"Chunk A Source: {c1['source_file']} (Page {c1['page_number']})\n"
            f"Chunk A Content: {c1['text']}\n\n"
            f"Chunk B ID: {c2['chunk_id']}\n"
            f"Chunk B Source: {c2['source_file']} (Page {c2['page_number']})\n"
            f"Chunk B Content: {c2['text']}\n\n"
        )

    prompt = (
        "You are building a RAG evaluation set. I will provide you with pairs of text chunks (Chunk A and Chunk B). "
        "For EACH pair, generate exactly one challenging 'multi-hop' question that requires reasoning or combining "
        "information across BOTH Chunk A and Chunk B to answer, along with the detailed ground truth answer.\n\n"
        "Guidelines:\n"
        "1. The question must require combining facts from BOTH chunks. It must NOT be answerable by Chunk A alone or Chunk B alone.\n"
        "2. The question must be specific (e.g. comparing methods, combining an architecture detail from Chunk A with performance from Chunk B, etc.).\n"
        "3. The ground_truth answer must be detailed, factual, and fully supported by both chunks.\n"
        "4. The supporting_chunk_ids list for each question must contain exactly both chunk IDs: [Chunk A ID, Chunk B ID].\n"
        "5. The question_type must be 'multi-hop'.\n\n"
        f"{pairs_str}\n"
        "Format as a JSON object matching this schema:\n"
        "{\n"
        '  "pairs": [\n'
        "    {\n"
        '      "question": "string (the specific question requiring both chunks)",\n'
        '      "ground_truth": "string (the detailed ground truth answer combining facts from both chunks)",\n'
        '      "supporting_chunk_ids": ["string (Chunk A ID)", "string (Chunk B ID)"],\n'
        '      "question_type": "multi-hop"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=4096
        )
        data = json.loads(response.choices[0].message.content.strip())
        pairs = [EvalPair(**p) for p in data.get("pairs", [])]
        return pairs
    except Exception as e:
        print(f"Error generating multi-hop batch: {e}")
        return []


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("Error: GROQ_API_KEY is not configured. Cannot run generation.")
        sys.exit(1)

    print("Loading chunks from database...")
    chunks = load_chunks()
    good_chunks = filter_good_chunks(chunks)
    print(f"Total chunks: {len(chunks)}, Chunks suitable for eval set: {len(good_chunks)}")

    if len(good_chunks) < 30:
        print("Error: Not enough good chunks in the database to generate 25 questions.")
        sys.exit(1)

    # Initialize Groq Client
    client = Groq(api_key=api_key)

    print("Generating evaluation set (~15 single-hop and ~10 multi-hop questions)...")
    
    # 1. Select chunks for single-hop questions
    # Randomly shuffle and select 15 chunks
    random.seed(42)  # For reproducibility
    single_hop_chunks = random.sample(good_chunks, 15)
    
    eval_pairs: List[EvalPair] = []
    
    # Generate single-hop questions in batches of 5
    print("Generating single-hop questions...")
    for i in range(0, len(single_hop_chunks), 5):
        batch = single_hop_chunks[i : i + 5]
        print(f"  Processing single-hop batch {i//5 + 1}...")
        pairs = generate_single_hop_batch(client, batch)
        eval_pairs.extend(pairs)
        time.sleep(2)  # Pause to avoid rate limits

    # 2. Select chunks for multi-hop questions
    # Let's pair related documents or same-document chunks
    # Group good chunks by document
    docs = {}
    for c in good_chunks:
        docs.setdefault(c["source_file"], []).append(c)

    multi_hop_pairs = []
    
    # Paper pairings for multi-hop across different papers
    related_pairs_configs = [
        ("2205.14135.pdf", "2307.08691.pdf"),  # FlashAttention and FlashAttention-2
        ("2001.04451.pdf", "2006.16362.pdf"),  # Reformer and Linformer
        ("1706.03762.pdf", "2104.09864.pdf"),  # Transformer and RoFormer
        ("2001.04451.pdf", "2004.05150.pdf"),  # Reformer and Longformer
        ("2205.14135.pdf", "1706.03762.pdf"),  # FlashAttention and Transformer
    ]

    # Generate 5 pairs from different papers
    for doc1_name, doc2_name in related_pairs_configs:
        if doc1_name in docs and doc2_name in docs:
            c1 = random.choice(docs[doc1_name])
            c2 = random.choice(docs[doc2_name])
            multi_hop_pairs.append((c1, c2))

    # Generate 5 pairs from the same paper (different sections)
    doc_keys = list(docs.keys())
    while len(multi_hop_pairs) < 10:
        doc_name = random.choice(doc_keys)
        if len(docs[doc_name]) >= 2:
            c1, c2 = random.sample(docs[doc_name], 2)
            # Ensure they are not too close (different pages if possible)
            if c1["page_number"] != c2["page_number"]:
                multi_hop_pairs.append((c1, c2))

    # Generate multi-hop questions in batches of 2 to ensure high quality and prevent token limits
    print("Generating multi-hop questions...")
    for i in range(0, len(multi_hop_pairs), 2):
        batch = multi_hop_pairs[i : i + 2]
        print(f"  Processing multi-hop batch {i//2 + 1}...")
        pairs = generate_multi_hop_batch(client, batch)
        eval_pairs.extend(pairs)
        time.sleep(2)  # Pause to avoid rate limits

    # Save to eval_set.jsonl
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EVAL_OUT_FILE, "w", encoding="utf-8") as f:
        for p in eval_pairs:
            f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Successfully generated {len(eval_pairs)} evaluation pairs!")
    print(f"Saved to: {EVAL_OUT_FILE}")
    print("=" * 60)
    print("WARNING / ACTION REQUIRED:")
    print("This file contains automatically generated ground truth data.")
    print("Please review and manually correct any errors in data/eval/eval_set.jsonl")
    print("before running the evaluation script to ensure exact ground truth.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

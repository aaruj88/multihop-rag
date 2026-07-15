"""
embed.py
--------
Load chunks from data/processed/chunks.jsonl, compute dense embeddings using
BAAI/bge-small-en-v1.5, and index them into Qdrant collection "papers".
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "papers"

def main():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Reading chunks from {CHUNKS_FILE}...")
    if not CHUNKS_FILE.exists():
        print(f"Error: {CHUNKS_FILE} does not exist. Run run_ingestion.py first.")
        return
        
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    if not chunks:
        print("No chunks to process.")
        return
        
    print(f"Loaded {len(chunks)} chunks. Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    print("Connecting to Qdrant...")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Re-create collection
    vector_size = len(embeddings[0])
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Re-created collection '{COLLECTION_NAME}' with vector size {vector_size}.")
    
    print("Upserting points to Qdrant...")
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],  # using chunk's UUID
                vector=vector.tolist(),
                payload={
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"]
                }
            )
        )
        
    # Batch upsert
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting batches"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size]
        )
        
    print(f"Successfully indexed {len(points)} chunks into collection '{COLLECTION_NAME}'!")


if __name__ == "__main__":
    main()

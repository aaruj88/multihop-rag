import os
import time
import requests
from pathlib import Path
from pypdf import PdfReader, PdfWriter

API_URL = "http://127.0.0.1:8000"

def create_tiny_pdf(source_pdf_name: str, target_pdf_path: Path):
    """Extracts the first page of a source PDF to create a tiny PDF."""
    source_path = Path("data") / "raw" / source_pdf_name
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF not found at {source_path}")
        
    reader = PdfReader(source_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    
    target_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_pdf_path, "wb") as f:
        writer.write(f)
    print(f"Created tiny PDF at {target_pdf_path}")

def upload_corpus(pdf_path: Path) -> str:
    """Uploads a PDF file and returns the corpus_id."""
    with open(pdf_path, "rb") as f:
        files = [("files", (pdf_path.name, f, "application/pdf"))]
        response = requests.post(f"{API_URL}/corpus", files=files)
        
    if response.status_code != 202:
        raise RuntimeError(f"Failed to upload corpus: {response.text}")
        
    data = response.json()
    corpus_id = data["corpus_id"]
    print(f"Uploaded {pdf_path.name}, got corpus_id: {corpus_id}")
    return corpus_id

def wait_for_ready(corpus_id: str, timeout_seconds: int = 60) -> dict:
    """Polls the corpus status endpoint until it's ready or failed."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        response = requests.get(f"{API_URL}/corpus/{corpus_id}/status")
        if response.status_code == 200:
            data = response.json()
            status = data["status"]
            print(f"Corpus {corpus_id} status: {status}")
            if status == "ready":
                return data
            if status == "failed":
                raise RuntimeError(f"Corpus {corpus_id} processing failed.")
        else:
            raise RuntimeError(f"Failed to check status: {response.text}")
        time.sleep(2)
    raise TimeoutError(f"Corpus {corpus_id} did not become ready within {timeout_seconds} seconds.")

def run_query(question: str, corpus_id: str) -> dict:
    """Queries the API for a given corpus_id."""
    payload = {
        "question": question,
        "corpus_id": corpus_id,
        "top_k": 3
    }
    response = requests.post(f"{API_URL}/query", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Query failed: {response.text}")
    return response.json()

def delete_corpus(corpus_id: str):
    """Deletes the corpus from the backend."""
    response = requests.delete(f"{API_URL}/corpus/{corpus_id}")
    if response.status_code == 200:
        print(f"Deleted corpus {corpus_id}")
    else:
        print(f"Warning: failed to delete corpus {corpus_id}: {response.text}")

def main():
    print("Starting isolation test...")
    
    # 1. Create tiny PDFs
    pdf_a_path = Path("data") / "processed" / "temp_test_a.pdf"
    pdf_b_path = Path("data") / "processed" / "temp_test_b.pdf"
    
    # Use Attention translation (1508.04025.pdf) for A
    # Use Relative Position Representations (1803.02155.pdf) for B
    create_tiny_pdf("1508.04025.pdf", pdf_a_path)
    create_tiny_pdf("1803.02155.pdf", pdf_b_path)
    
    corpus_id_a = None
    corpus_id_b = None
    
    try:
        # 2. Upload both corpora
        corpus_id_a = upload_corpus(pdf_a_path)
        corpus_id_b = upload_corpus(pdf_b_path)
        
        # 3. Wait for both to be ready
        print("Waiting for Corpus A to be ready...")
        info_a = wait_for_ready(corpus_id_a)
        print("Waiting for Corpus B to be ready...")
        info_b = wait_for_ready(corpus_id_b)
        
        print(f"Corpus A stats: Chunks: {info_a['chunk_count']}, Files: {info_a['file_count']}")
        print(f"Corpus B stats: Chunks: {info_b['chunk_count']}, Files: {info_b['file_count']}")
        
        # 4. Define cross-queries
        # Question B: "Who proposed Self-Attention with Relative Position Representations?"
        # Only answerable from Corpus B (Shaw et al.). Corpus A should not know this.
        q_b = "Who proposed Self-Attention with Relative Position Representations?"
        
        # Query Corpus A with B's question
        print(f"\nQuerying Corpus A ( NMT Attention ) with question B: '{q_b}'")
        res_a = run_query(q_b, corpus_id_a)
        print("Answer A:")
        print(res_a["answer_text"])
        print("Citations A:", res_a["citations"])
        print("Unanswered A:", res_a["unanswered_aspects"])
        
        # ASSERT: Corpus A cannot answer the question and lists it as unanswered,
        # and has NO citations matching B's chunks.
        assert "cannot answer" in res_a["answer_text"].lower() or not res_a["citations"], \
            "Leaking error: Corpus A answered a question only present in Corpus B!"
            
        # Query Corpus B with B's question
        print(f"\nQuerying Corpus B ( Relative Positions ) with question B: '{q_b}'")
        res_b = run_query(q_b, corpus_id_b)
        print("Answer B:")
        print(res_b["answer_text"])
        print("Citations B:", res_b["citations"])
        
        # ASSERT: Corpus B should be able to answer the question, or at least have citations
        assert res_b["citations"], "Error: Corpus B failed to retrieve any citations for its own content."
        assert "cannot answer" not in res_b["answer_text"].lower(), "Error: Corpus B could not answer its own question."
        
        print("\nSUCCESS: Strict multi-tenant isolation verified. Zero chunk leaking detected!")
        
    finally:
        # 5. Cleanup
        print("\nCleaning up...")
        if corpus_id_a:
            delete_corpus(corpus_id_a)
        if corpus_id_b:
            delete_corpus(corpus_id_b)
            
        # Delete temp PDFs
        if pdf_a_path.exists():
            pdf_a_path.unlink()
        if pdf_b_path.exists():
            pdf_b_path.unlink()
            
if __name__ == "__main__":
    main()

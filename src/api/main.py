import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, BackgroundTasks, Form
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from qdrant_client import QdrantClient
from apscheduler.schedulers.background import BackgroundScheduler
from limits import parse as parse_limit

from src.generation.synthesize import AnswerSchema, synthesize_answer
from src.generation.decompose import decompose_query
from src.retrieval.multihop import MultiHopRetriever
from src.api.db import init_db, set_corpus_status, get_corpus_status, delete_corpus_status, get_expired_corpora, get_total_active_corpora

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize slowapi Limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize MultiHopRetriever once globally at startup to reuse model weights across requests
retriever = MultiHopRetriever()

app = FastAPI(
    title="Multi-Hop RAG API",
    description="API for multi-hop academic paper question answering with user-uploaded corpora.",
    version="1.1.0"
)

from fastapi.middleware.cors import CORSMiddleware

allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire up rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def get_qdrant_client(timeout: float = None) -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout)
    else:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        return QdrantClient(host=qdrant_host, port=qdrant_port, timeout=timeout)

scheduler = BackgroundScheduler()

def delete_corpus_helper(corpus_id: str):
    """
    Deletes all chunks matching corpus_id from Qdrant, clears the cached BM25 index,
    and removes the status row from SQLite.
    """
    client = get_qdrant_client()
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    collection_name = "papers"
    if client.collection_exists(collection_name):
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="corpus_id",
                        match=MatchValue(value=corpus_id)
                    )
                ]
            )
        )
        
    # Clear cached BM25 index
    retriever.pipeline.hybrid_retriever.sparse_retriever.clear_cache(corpus_id)
    
    # Delete from SQLite status DB
    delete_corpus_status(corpus_id)

def cleanup_expired_corpora_job():
    try:
        expired_ids = get_expired_corpora(hours=24)
        if expired_ids:
            logger.info(f"Scheduled Cleanup: Found {len(expired_ids)} expired corpora older than 24 hours: {expired_ids}")
            for cid in expired_ids:
                try:
                    delete_corpus_helper(cid)
                    logger.info(f"Scheduled Cleanup: Successfully deleted corpus {cid}")
                except Exception as ex:
                    logger.error(f"Scheduled Cleanup: Error deleting corpus {cid}: {ex}")
        else:
            logger.info("Scheduled Cleanup: No expired corpora found.")
    except Exception as e:
        logger.error(f"Scheduled Cleanup: Error running cleanup job: {e}")

@app.on_event("startup")
async def startup_event():
    # Initialize the SQLite status database
    init_db()
    
    # Try to initialize the keyword payload index on Qdrant if the collection exists
    try:
        client = get_qdrant_client(timeout=2.0)
        collection_name = "papers"
        if client.collection_exists(collection_name):
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=collection_name,
                field_name="corpus_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
    except Exception as e:
        print(f"Startup warning: could not initialize Qdrant payload index: {e}")
        
    # Start scheduled hourly cleanup job
    scheduler.add_job(cleanup_expired_corpora_job, "interval", hours=1)
    scheduler.start()
    logger.info("Background scheduler started hourly cleanup job.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("Background scheduler shutdown.")



class QueryRequest(BaseModel):
    question: str = Field(..., description="The query to send through the RAG pipeline.")
    corpus_id: str = Field(..., description="The ID of the corpus to query.")
    top_k: int = Field(5, description="Number of context chunks to retrieve.")
    groq_api_key: Optional[str] = Field(None, description="Optional custom Groq API key (BYOK).")



class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    text: str
    score: float
    corpus_id: str

class QueryResponse(BaseModel):
    answer_text: str
    citations: list[str]
    unanswered_aspects: list[str]
    latency_seconds: float
    decomposition_triggered: bool
    retrieved_chunks: Optional[List[RetrievedChunkSchema]] = None



def process_corpus_task(corpus_id: str, files_saved: List[Path]):
    """
    Background task to parse PDFs, chunk them, compute embeddings, and upsert them to Qdrant.
    """
    try:
        from src.ingestion.parse import parse_pdf
        corpus = {}
        for pdf_path in files_saved:
            try:
                pages = parse_pdf(pdf_path)
                if pages:
                    corpus[pdf_path.name] = pages
            except Exception as e:
                print(f"Error parsing PDF {pdf_path}: {e}")
                
        if not corpus:
            set_corpus_status(corpus_id, "failed")
            return
            
        from src.ingestion.chunk import chunk_corpus
        chunks = chunk_corpus(corpus)
        if not chunks:
            set_corpus_status(corpus_id, "failed")
            return
            
        # Get SentenceTransformer model from global retriever
        model = retriever.pipeline.hybrid_retriever.dense_retriever.model
        
        texts = [c.text for c in chunks]
        # Encode chunks
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        
        # Connect to Qdrant
        client = get_qdrant_client()
        
        collection_name = "papers"
        vector_size = len(embeddings[0])
        
        from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
        
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="corpus_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector.tolist(),
                    payload={
                        "corpus_id": corpus_id,
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end
                    }
                )
            )
            
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection_name,
                points=points[i : i + batch_size]
            )
            
        # Update SQLite status to ready
        set_corpus_status(corpus_id, "ready", chunk_count=len(chunks), file_count=len(files_saved))
        
    except Exception as e:
        print(f"Error in process_corpus_task for {corpus_id}: {e}")
        set_corpus_status(corpus_id, "failed")
    finally:
        # Clean up temp folder and files
        for f in files_saved:
            try:
                f.unlink()
            except Exception:
                pass
        try:
            if files_saved:
                files_saved[0].parent.rmdir()
        except Exception:
            pass


@app.post("/corpus", status_code=202)
async def upload_corpus(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    groq_api_key: Optional[str] = Form(None)
):
    """
    Accepts multipart upload of one or more PDFs (cap: max 10 files, max 15MB each).
    Generates a new corpus_id (UUID), starts ingestion in background, and returns immediately.
    """
    # 1. Dynamic rate limiting (no-key: 5/min, BYOK: 20/min)
    has_key = bool(groq_api_key)
    limit_str = "20/minute" if has_key else "5/minute"
    limit = parse_limit(limit_str)
    ip = get_remote_address(request)
    namespace = f"corpus:{ip}"
    if not limiter.limiter.hit(limit, namespace):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Global cap check: reject new uploads if active corpora limit is reached
    max_active = int(os.getenv("MAX_ACTIVE_CORPORA", "200"))
    if get_total_active_corpora() >= max_active:
        raise HTTPException(
            status_code=400,
            detail=f"Global corpus limit reached. Max limit is {max_active} active corpora."
        )


    # 1. Cap: max 10 files
    if len(files) > 10:
        raise HTTPException(status_code=413, detail="Maximum of 10 files allowed.")

        
    # 2. Check file size cap: max 15MB each (15 * 1024 * 1024 bytes)
    max_size = 15 * 1024 * 1024
    for f in files:
        # Read the file to determine its size
        content = await f.read()
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail=f"File {f.filename} exceeds maximum size of 15MB.")
        await f.seek(0)
        
    # Generate new corpus_id
    corpus_id = str(uuid.uuid4())
    
    # Save the files temporarily
    temp_dir = Path("data") / "raw" / corpus_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    files_saved = []
    for f in files:
        file_path = temp_dir / f.filename
        content = await f.read()
        with open(file_path, "wb") as out:
            out.write(content)
        files_saved.append(file_path)
        
    # Initialize status in SQLite
    set_corpus_status(corpus_id, "processing", chunk_count=0, file_count=len(files_saved))
    
    # Kick off background task
    background_tasks.add_task(process_corpus_task, corpus_id, files_saved)
    
    return {
        "corpus_id": corpus_id,
        "status": "processing"
    }


@app.get("/corpus/{corpus_id}/status")
async def get_status(corpus_id: str):
    """
    Returns corpus status ("processing", "ready", or "failed"), chunk count, and file count.
    """
    status_info = get_corpus_status(corpus_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Corpus not found.")
    return status_info


@app.delete("/corpus/{corpus_id}")
async def delete_corpus(corpus_id: str):
    """
    Deletes all chunks matching corpus_id from Qdrant and clears the cached BM25 index.
    """
    # 1. Verify existence in database
    status_info = get_corpus_status(corpus_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Corpus not found.")
        
    # 2. Perform deletion
    delete_corpus_helper(corpus_id)
    
    return {"message": f"Corpus {corpus_id} deleted successfully."}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(query_request: QueryRequest, request: Request):
    """
    Accepts a question and corpus_id, runs the isolated RAG pipeline,
    and returns the structured answer.
    """
    # 1. Dynamic rate limiting (no-key: 5/min, BYOK: 20/min)
    has_key = bool(query_request.groq_api_key)
    limit_str = "20/minute" if has_key else "5/minute"
    limit = parse_limit(limit_str)
    ip = get_remote_address(request)
    namespace = f"query:{ip}"
    if not limiter.limiter.hit(limit, namespace):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    question = query_request.question
    corpus_id = query_request.corpus_id
    top_k = query_request.top_k
    
    # 1. Verify corpus exists and is ready
    status_info = get_corpus_status(corpus_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Corpus not found.")
    if status_info["status"] == "processing":
        raise HTTPException(status_code=400, detail="Corpus is still processing.")
    if status_info["status"] == "failed":
        raise HTTPException(status_code=400, detail="Corpus processing failed.")
        
    start_time = time.time()
    
    try:
        # 1. Decompose the query
        sub_queries = decompose_query(question, api_key=query_request.groq_api_key)
        
        # 2. Check if decomposition actually triggered
        decomp_triggered = len(sub_queries) > 1 and (len(sub_queries) != 1 or sub_queries[0] != question)
        
        # 3. Retrieve chunks filtered by corpus_id
        merged_results = {}
        for sub_q in sub_queries:
            sub_results = retriever.pipeline.retrieve(sub_q, corpus_id=corpus_id, top_k=top_k)
            for res in sub_results:
                cid = res["chunk_id"]
                if cid not in merged_results:
                    res_copy = dict(res)
                    res_copy["matched_queries"] = [sub_q]
                    merged_results[cid] = res_copy
                else:
                    if sub_q not in merged_results[cid]["matched_queries"]:
                        merged_results[cid]["matched_queries"].append(sub_q)
                    if res["score"] > merged_results[cid]["score"]:
                        merged_results[cid]["score"] = res["score"]
                        
        sorted_results = sorted(merged_results.values(), key=lambda x: x["score"], reverse=True)
        retrieved_chunks = sorted_results[:top_k]
        
        # 4. Synthesize Answer
        answer_schema = synthesize_answer(
            question, 
            retrieved_chunks, 
            corpus_id=corpus_id, 
            api_key=query_request.groq_api_key
        )
        
        latency = time.time() - start_time
        
        # 5. Log the query patterns locally
        log_dir = os.path.join("data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "api_usage.log")
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": question,
            "latency_seconds": round(latency, 4),
            "decomposition_triggered": decomp_triggered
        }
        
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        return QueryResponse(
            answer_text=answer_schema.answer_text,
            citations=answer_schema.citations,
            unanswered_aspects=answer_schema.unanswered_aspects,
            latency_seconds=round(latency, 4),
            decomposition_triggered=decomp_triggered,
            retrieved_chunks=[
                RetrievedChunkSchema(
                    chunk_id=c.get("chunk_id", ""),
                    source_file=c.get("source_file", ""),
                    page_number=c.get("page_number", 0),
                    text=c.get("text", ""),
                    score=c.get("score", 0.0),
                    corpus_id=c.get("corpus_id", "")
                )
                for c in retrieved_chunks
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Performs a basic health check and verifies connectivity to the Qdrant database.
    """
    try:
        # Check connectivity using the qdrant-client
        client = get_qdrant_client(timeout=2.0)
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"disconnected: {str(e)}"
        
    status = "healthy" if "disconnected" not in qdrant_status else "degraded"
    
    return {
        "status": status,
        "qdrant_connectivity": qdrant_status
    }

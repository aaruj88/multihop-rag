# multihop-rag

> **Multi-hop Retrieval-Augmented Generation with end-to-end evaluation**

A research engineering project that builds a full RAG pipeline capable of
answering questions that require reasoning across **multiple documents**
(multi-hop questions). The system ingests academic PDFs, indexes them in a
vector database, retrieves relevant passages across multiple reasoning steps,
generates answers with Google Gemini, and evaluates quality with RAGAS.

🚀 **[Live Demo](https://your-frontend-domain.vercel.app)** | 📖 **[Deployment Guide](DEPLOY.md)**

---

## Table of Contents

- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running the Pipeline](#running-the-pipeline)
- [API Server](#api-server)
- [Evaluation](#evaluation)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)

---

## Live Demo

You can try the live deployment of this project at:
- **Frontend App (Vercel)**: [https://your-frontend-domain.vercel.app](https://your-frontend-domain.vercel.app) *(placeholder)*
- **Backend API (Render)**: [https://your-backend-domain.onrender.com/health](https://your-backend-domain.onrender.com/health) *(placeholder)*

See the [Deployment Guide](DEPLOY.md) for full instructions on deploying your own instances.

---

## Architecture

### System Flow
```
PDFs  →  Ingestion  →  Qdrant (vector store)
                               ↓
          Query  →  Multi-hop Retriever  →  Gemini  →  Answer
                                                   ↓
                                              RAGAS eval
```

### Production Infrastructure Diagram
```
  [ Browser (User Interface) ]
              │
              ▼ (HTTPS)
     [ Vercel (React App) ]
              │
              ▼ (REST API / JSON)
   [ Render (FastAPI Docker Web Service) ]
              │
      ┌───────┴────────────────────────┐
      ▼ (gRPC/REST)                    ▼ (REST)
[ Qdrant Cloud (Vector DB) ]     [ Gemini API / Groq ]
```

| Component | Technology |
|-----------|-----------|
| Vector store | Qdrant (Docker locally / Qdrant Cloud in production) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| LLM | Google Gemini (`google-generativeai`) |
| API | FastAPI + Uvicorn |
| Evaluation | RAGAS |
| PDF parsing | pypdf |

---

## Project Structure

```
multihop-rag/
├── data/
│   ├── raw/               # Source PDFs (downloaded by fetch_papers.py)
│   ├── processed/         # Parsed & chunked outputs (JSON)
│   └── arxiv_ids.txt      # arXiv paper IDs to download
├── src/
│   ├── ingestion/         # PDF download, parsing, chunking, indexing
│   ├── retrieval/         # Dense + sparse + multi-hop retrieval logic
│   ├── generation/        # Prompt construction & Gemini calls
│   ├── eval/              # RAGAS evaluation harness
│   └── api/               # FastAPI application
├── tests/                 # Pytest test suite
├── docker-compose.yml     # Qdrant local instance
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Docker & Docker Compose | any recent version |

---

## Quick Start

### 1. Clone & enter the repo

```bash
git clone <your-repo-url>
cd multihop-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your GEMINI_API_KEY
```

### 5. Start Qdrant

```bash
docker-compose up -d
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine","version":"..."}
```

Or open <http://localhost:6333/dashboard> in your browser.

---

## Running the Pipeline

### Step 1 – Download papers

```bash
python -m src.ingestion.fetch_papers
```

Optional arguments:

```
--ids-file  PATH   Path to arXiv IDs file (default: data/arxiv_ids.txt)
--out-dir   PATH   Output directory for PDFs   (default: data/raw/)
```

To download a specific subset:

```bash
python -m src.ingestion.fetch_papers --ids-file data/arxiv_ids.txt --out-dir data/raw
```

### Step 2 – Parse & chunk PDFs *(coming soon)*

```bash
python -m src.ingestion.parse_and_chunk
```

### Step 3 – Index into Qdrant *(coming soon)*

```bash
python -m src.ingestion.index_documents
```

### Step 4 – Run a multi-hop query *(coming soon)*

```bash
python -m src.retrieval.query "What methods improve multi-hop reasoning in RAG systems?"
```

---

## API Server

Start the API server locally:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Once running, the API features:
* **POST /query**: Runs the full RAG pipeline (decomposition, retrieval, synthesis) on a query.
* **GET /health**: Verifies API status and database connectivity to Qdrant.
* **Local Logging**: Logs query latency, timestamp, query string, and whether decomposition was triggered to `data/logs/api_usage.log`.
* **Rate Limiting**: Limits requests to 10 requests per minute per IP using `slowapi`.

Interactive API documentation is available at <http://localhost:8000/docs>.

### Example curl Request

#### Query API:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Compare FlashAttention and Linformer in terms of complexity.", "top_k": 3}'
```

Example Response:

```json
{
  "answer_text": "FlashAttention reduces memory accesses but retains O(N^2) complexity, whereas Linformer projects the key-value states to achieve O(N) complexity...",
  "citations": ["chunk-uuid-1", "chunk-uuid-2"],
  "unanswered_aspects": [],
  "latency_seconds": 1.4520,
  "decomposition_triggered": true
}
```

#### Health Check:

```bash
curl http://localhost:8000/health
```

Example Response:

```json
{
  "status": "healthy",
  "qdrant_connectivity": "connected"
}
```

---

## Evaluation

```bash
pytest tests/
```

To run a RAGAS evaluation on a saved Q&A dataset:

```bash
python -m src.eval.run_ragas --dataset data/processed/qa_pairs.json
```

---

## Cost & Abuse Controls

To protect server resources, prevent unexpected token costs, and keep storage unbounded, the API enforces several guardrails:

1. **Bring-Your-Own-Key (BYOK) Dynamic Rate Limiting**:
   - **No Key Path**: If no custom `groq_api_key` is provided in the request body, calls use the server's default API key and are rate-limited to **5 requests/minute** per IP. This prevents anonymous users from draining the server's API credits.
   - **BYOK Path**: If an optional `groq_api_key` is supplied in the request body, requests are allowed up to **20 requests/minute** per IP. The key is only used in-flight to make the LLM requests and is never logged or stored.
2. **Global Active Corpora Cap**:
   - The total number of active corpora uploaded to the server is capped at a configurable limit (configured via `MAX_ACTIVE_CORPORA`, defaulting to `200`). Uploading new corpora beyond this cap is rejected with an HTTP 400 error. This prevents the host disk from filling up.
3. **Scheduled Ingestion Cleanup**:
   - A background scheduler (using APScheduler) runs **hourly** and automatically deletes any corpus (including its Qdrant chunks and SQLite status records) that is older than **24 hours**. This ensures transient corpora uploaded for testing or single-session use are garbage-collected daily.

---

## Environment Variables


| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `QDRANT_HOST` | optional | Qdrant host (default: `localhost`) |
| `QDRANT_PORT` | optional | Qdrant REST port (default: `6333`) |
| `QDRANT_COLLECTION` | optional | Collection name (default: `multihop_rag`) |
| `MAX_ACTIVE_CORPORA`| optional | Maximum active corpora allowed globally (default: `200`) |


---

## Contributing

1. Fork the repo and create a feature branch.
2. Write tests for any new logic.
3. Run `pytest` before opening a PR.
4. Follow [Conventional Commits](https://www.conventionalcommits.org/).

---

## License

MIT

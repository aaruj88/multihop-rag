# Stage 1: Builder stage to install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only to keep size optimized, then install requirements
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final minimal runtime stage
FROM python:3.11-slim as runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    HF_HOME="/home/appuser/.cache/huggingface"

# Create a non-root user and prepare required directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/processed /app/data/raw /app/data/logs /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appuser src/ /app/src/

USER appuser

# Pre-cache Hugging Face embedding model in the Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; \
print('Pre-downloading BAAI/bge-small-en-v1.5...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8000

# Bind to PORT env var (Render/Fly.io support), fallback to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

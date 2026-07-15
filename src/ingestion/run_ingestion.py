"""
run_ingestion.py
----------------
Orchestrates the full ingestion pipeline:

    PDF files  →  parse.py  →  chunk.py  →  data/processed/chunks.jsonl

Each line of the output JSONL file is a JSON object representing one chunk::

    {
        "chunk_id":    "uuid-...",
        "source_file": "1706.03762.pdf",
        "page_number": 3,
        "text":        "...",
        "char_start":  1024,
        "char_end":    3021
    }

Usage
-----
    # From the repo root (with venv active):
    python -m src.ingestion.run_ingestion

    # Custom paths:
    python -m src.ingestion.run_ingestion \\
        --raw-dir   data/raw \\
        --out-file  data/processed/chunks.jsonl \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

from src.ingestion.chunk import Chunk, chunk_corpus, estimate_tokens
from src.ingestion.parse import parse_directory

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_OUT_FILE = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_jsonl(chunks: list[Chunk], out_file: Path) -> None:
    """Serialise *chunks* to a JSONL file, one chunk per line."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), out_file)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def report_stats(chunks: list[Chunk], parse_failures: list[str]) -> None:
    """Print a summary table to stdout."""
    if not chunks:
        logger.warning("No chunks produced – nothing to report.")
        return

    token_counts = [estimate_tokens(c.text) for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_chars = sum(len(c.text) for c in chunks) / len(chunks)
    max_tokens = max(token_counts)
    docs = {c.source_file for c in chunks}

    lines = [
        "",
        "=" * 60,
        "  INGESTION SUMMARY",
        "=" * 60,
        f"  Documents processed : {len(docs)}",
        f"  Total chunks        : {len(chunks):,}",
        f"  Avg chunk length    : {avg_chars:,.0f} chars  (~{avg_tokens:.0f} tokens)",
        f"  Max chunk tokens    : {max_tokens}",
        f"  Parse failures      : {len(parse_failures)}",
    ]
    if parse_failures:
        lines.append("  Failed files:")
        for f in parse_failures:
            lines.append(f"    - {f}")
    lines.append("=" * 60)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_ingestion(raw_dir: Path, out_file: Path) -> tuple[list[Chunk], list[str]]:
    """
    Full pipeline: parse → chunk → write JSONL.

    Returns
    -------
    (chunks, parse_failures)
    """
    t0 = time.perf_counter()
    logger.info("Starting ingestion pipeline")
    logger.info("  Raw PDF dir : %s", raw_dir)
    logger.info("  Output file : %s", out_file)

    # ── Step 1: Parse ──────────────────────────────────────────────────────
    logger.info("Step 1/3: Parsing PDFs …")
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in %s. Run fetch_papers.py first.", raw_dir)
        sys.exit(1)

    logger.info("Found %d PDF files", len(pdf_files))

    parse_failures: list[str] = []
    corpus: dict = {}

    for pdf_path in pdf_files:
        try:
            from src.ingestion.parse import parse_pdf  # local import for clarity
            pages = parse_pdf(pdf_path)
            if pages:
                corpus[pdf_path.name] = pages
            else:
                logger.warning("No content extracted from %s (scanned?)", pdf_path.name)
                parse_failures.append(pdf_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAIL %s: %s", pdf_path.name, exc)
            parse_failures.append(pdf_path.name)

    logger.info(
        "Parsed %d/%d documents successfully", len(corpus), len(pdf_files)
    )

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    logger.info("Step 2/3: Chunking documents …")
    chunks = chunk_corpus(corpus)
    logger.info("Produced %d chunks total", len(chunks))

    # ── Step 3: Write JSONL ─────────────────────────────────────────────────
    logger.info("Step 3/3: Writing JSONL output …")
    write_jsonl(chunks, out_file)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1f s", elapsed)

    report_stats(chunks, parse_failures)
    return chunks, parse_failures


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full PDF → JSONL ingestion pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing source PDFs (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=DEFAULT_OUT_FILE,
        help=f"Output JSONL file (default: {DEFAULT_OUT_FILE})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)
    chunks, failures = run_ingestion(args.raw_dir, args.out_file)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

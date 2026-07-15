"""
fetch_papers.py
---------------
Download arXiv PDFs listed in data/arxiv_ids.txt into data/raw/.

Usage:
    python -m src.ingestion.fetch_papers                     # uses default config
    python -m src.ingestion.fetch_papers --ids-file path/to/ids.txt --out-dir data/raw

Each non-blank, non-comment line in the IDs file is treated as an arXiv paper ID
(e.g. "2005.11401"). The script skips IDs whose PDF already exists locally.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults (relative to the repo root)
# ---------------------------------------------------------------------------
DEFAULT_IDS_FILE = Path(__file__).resolve().parents[2] / "data" / "arxiv_ids.txt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
REQUEST_DELAY_SECONDS = 2  # be polite to arXiv servers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_arxiv_ids(ids_file: Path) -> list[str]:
    """Read arXiv IDs from a text file, ignoring blank lines and comments."""
    if not ids_file.exists():
        logger.error("IDs file not found: %s", ids_file)
        sys.exit(1)

    ids: list[str] = []
    with ids_file.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.append(stripped)

    logger.info("Loaded %d arXiv IDs from %s", len(ids), ids_file)
    return ids


def download_pdf(arxiv_id: str, out_dir: Path, session: requests.Session) -> Path | None:
    """
    Download a single arXiv PDF.

    Returns the local Path on success, or None on failure.
    """
    out_path = out_dir / f"{arxiv_id.replace('/', '_')}.pdf"

    if out_path.exists():
        logger.info("[SKIP] %s – already downloaded (%s)", arxiv_id, out_path.name)
        return out_path

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    logger.info("[FETCH] %s → %s", url, out_path.name)

    try:
        response = session.get(url, timeout=60, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with out_path.open("wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))

        logger.info("[OK]   Saved %s (%.1f KB)", out_path.name, out_path.stat().st_size / 1024)
        return out_path

    except requests.HTTPError as exc:
        logger.error("[FAIL] HTTP %s for %s: %s", exc.response.status_code, arxiv_id, exc)
    except requests.RequestException as exc:
        logger.error("[FAIL] Network error for %s: %s", arxiv_id, exc)

    # Remove partial file if it exists
    if out_path.exists():
        out_path.unlink()
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(ids_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arxiv_ids = load_arxiv_ids(ids_file)

    succeeded: list[str] = []
    failed: list[str] = []

    with requests.Session() as session:
        session.headers.update({"User-Agent": "multihop-rag-fetcher/1.0 (research project)"})

        for i, arxiv_id in enumerate(arxiv_ids):
            result = download_pdf(arxiv_id, out_dir, session)
            if result is not None:
                succeeded.append(arxiv_id)
            else:
                failed.append(arxiv_id)

            # Polite delay between requests (skip after last item)
            if i < len(arxiv_ids) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

    # Summary
    logger.info("─" * 60)
    logger.info("Done. %d succeeded, %d failed.", len(succeeded), len(failed))
    if failed:
        logger.warning("Failed IDs: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download arXiv PDFs listed in a config file."
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=DEFAULT_IDS_FILE,
        help=f"Path to the arXiv IDs text file (default: {DEFAULT_IDS_FILE})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to save PDFs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    main(args.ids_file, args.out_dir)

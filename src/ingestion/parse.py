"""
parse.py
--------
Extract text from PDF files using pypdf, page by page.

Key behaviours
--------------
* Preserves page numbers (1-indexed).
* Detects reference / bibliography sections via common heading patterns
  and stops extracting at that point, so downstream chunks stay content-only.
* Returns a list of :class:`ParsedPage` dataclass instances; the caller
  never needs to touch pypdf directly.

Usage (standalone)
------------------
    from src.ingestion.parse import parse_pdf
    pages = parse_pdf(Path("data/raw/1706.03762.pdf"))
    for p in pages:
        print(p.page_number, p.text[:80])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference-section detection
# ---------------------------------------------------------------------------
# Patterns that indicate the start of a bibliography / references section.
# We look for them as isolated headings (possibly with surrounding whitespace
# or numbering like "7  References").
_REF_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\d+[\.\s]+)?\s*"
    r"(References|Bibliography|Works\s+Cited|Literature\s+Cited|"
    r"Bibliographie|Literatur|Related\s+Works?)\s*$",
    re.IGNORECASE,
)


def _find_ref_cutoff(text: str) -> int | None:
    """
    Return the character offset where the references section begins,
    or None if no such heading is found.
    """
    match = _REF_HEADING_RE.search(text)
    return match.start() if match else None


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedPage:
    """A single page extracted from a PDF."""

    source_file: str   # basename, e.g. "1706.03762.pdf"
    page_number: int   # 1-indexed page number in the original PDF
    text: str          # cleaned page text (may be empty for scanned images)


# ---------------------------------------------------------------------------
# Core parsing function
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path) -> list[ParsedPage]:
    """
    Parse *pdf_path* and return a list of :class:`ParsedPage` objects,
    one per non-empty content page.

    Pages that come **after** a detected references/bibliography heading
    are silently dropped.  If the heading appears mid-page, only the text
    before it is retained.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[ParsedPage]
        Possibly empty if the PDF is scanned / image-only.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    pypdf.errors.PdfReadError
        On corrupted / encrypted PDF files.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[ParsedPage] = []
    found_refs = False

    with pdf_path.open("rb") as fh:
        try:
            reader = pypdf.PdfReader(fh)
        except Exception as exc:
            logger.error("Cannot open %s: %s", pdf_path.name, exc)
            raise

        n_pages = len(reader.pages)
        logger.debug("Parsing %s (%d pages)", pdf_path.name, n_pages)

        for page_idx, pdf_page in enumerate(reader.pages):
            if found_refs:
                break

            # pypdf returns None for image-only pages
            raw = pdf_page.extract_text() or ""

            # Normalise whitespace but keep paragraph breaks (double newlines)
            text = _normalise_text(raw)

            # Check for a references heading on this page
            cutoff = _find_ref_cutoff(text)
            if cutoff is not None:
                text = text[:cutoff].strip()
                found_refs = True
                logger.debug(
                    "References section detected on page %d of %s – stopping.",
                    page_idx + 1,
                    pdf_path.name,
                )

            if text:
                pages.append(
                    ParsedPage(
                        source_file=pdf_path.name,
                        page_number=page_idx + 1,
                        text=text,
                    )
                )

    logger.info(
        "Parsed %s → %d content pages (references %s)",
        pdf_path.name,
        len(pages),
        "truncated" if found_refs else "not found / kept",
    )
    return pages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_text(raw: str) -> str:
    """
    Light-touch normalisation:
    * Collapse runs of 3+ newlines into two (preserve paragraph breaks).
    * Remove carriage-returns.
    * Strip leading/trailing whitespace.
    pypdf sometimes inserts hyphenation artefacts; we leave them intact so
    the chunker can decide whether to merge lines.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Convenience: parse an entire directory
# ---------------------------------------------------------------------------

def parse_directory(raw_dir: Path) -> dict[str, list[ParsedPage]]:
    """
    Parse all PDFs in *raw_dir*.

    Returns
    -------
    dict[str, list[ParsedPage]]
        Keys are PDF basenames; values are their parsed pages.
        Files that fail to parse are logged and skipped.
    """
    results: dict[str, list[ParsedPage]] = {}
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", raw_dir)
        return results

    for pdf_path in pdf_files:
        try:
            results[pdf_path.name] = parse_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("SKIP %s – parse error: %s", pdf_path.name, exc)

    return results

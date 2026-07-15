"""
chunk.py
--------
Recursive character-level text splitter that produces fixed-size, overlapping
chunks with rich metadata.

Design decisions
----------------
* **Token estimation**: 1 token ≈ 4 characters (GPT-family heuristic).
  This avoids a full tokenizer dependency while staying within ±20 % of the
  real token count for English scientific text.
* **Separator cascade**: the splitter tries progressively finer boundaries
  (section break → paragraph → newline → sentence → word → character) so
  chunks almost never cut a sentence in half.
* **Overlap**: the last *OVERLAP_CHARS* characters of a chunk are prepended
  to the next one to preserve cross-boundary context.
* **Page attribution**: each chunk carries the page number of the PDF page
  where its first character originated.

Usage
-----
    from src.ingestion.parse import parse_pdf
    from src.ingestion.chunk import chunk_document

    pages = parse_pdf(Path("data/raw/1706.03762.pdf"))
    chunks = chunk_document(pages)
    for c in chunks[:3]:
        print(c.chunk_id, c.page_number, len(c.text))
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from src.ingestion.parse import ParsedPage

# ---------------------------------------------------------------------------
# Sizing constants
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN: int = 4          # rough approximation
TARGET_TOKENS: int = 500
OVERLAP_TOKENS: int = 50

TARGET_CHARS: int = TARGET_TOKENS * CHARS_PER_TOKEN   # 2 000
OVERLAP_CHARS: int = OVERLAP_TOKENS * CHARS_PER_TOKEN  # 200
HARD_MAX_CHARS: int = 600 * CHARS_PER_TOKEN            # 2 400 (test guard)

# Separators tried in order, from coarsest to finest
SEPARATORS: list[str] = [
    "\n\n\n",   # multiple blank lines (section boundaries in raw PDF text)
    "\n\n",     # paragraph break
    "\n",       # single newline
    ". ",       # sentence end (English)
    "! ",
    "? ",
    "; ",
    ", ",
    " ",        # word boundary
    "",         # character (last resort)
]


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""

    chunk_id: str        # UUID string – unique across the entire corpus
    source_file: str     # PDF basename, e.g. "1706.03762.pdf"
    page_number: int     # page where this chunk's first character originated
    text: str            # chunk content (stripped)
    char_start: int      # character offset in the concatenated document text
    char_end: int        # exclusive character offset


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Approximate GPT-style token count: 1 token ≈ 4 characters."""
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Recursive splitter (core algorithm)
# ---------------------------------------------------------------------------

def _split_recursive(
    text: str,
    separators: list[str],
    max_chars: int,
) -> list[str]:
    """
    Recursively split *text* into pieces no longer than *max_chars*.

    The algorithm:
    1. Try the first separator in *separators*.
    2. If splitting produces pieces that still exceed *max_chars*, recurse
       on those pieces with the remaining (finer) separators.
    3. Merge the resulting small pieces back together (greedy, left-to-right)
       so that each merged chunk is as large as possible without exceeding
       *max_chars*.
    """
    if len(text) <= max_chars:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:]

    # Split on current separator
    if sep:
        raw_splits = text.split(sep)
        # Re-attach the separator so we don't lose it in the final text
        splits = [s + sep for s in raw_splits[:-1]] + [raw_splits[-1]]
    else:
        # Character-level fallback
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    # Recursively handle any oversized pieces
    fine_splits: list[str] = []
    for piece in splits:
        if len(piece) > max_chars and remaining_seps:
            fine_splits.extend(_split_recursive(piece, remaining_seps, max_chars))
        else:
            fine_splits.append(piece)

    # Greedily merge fine splits back up to max_chars
    merged: list[str] = []
    current = ""
    for piece in fine_splits:
        if len(current) + len(piece) <= max_chars:
            current += piece
        else:
            if current.strip():
                merged.append(current)
            current = piece
    if current.strip():
        merged.append(current)

    return merged if merged else [text[:max_chars]]


# ---------------------------------------------------------------------------
# Overlap injection
# ---------------------------------------------------------------------------

def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """
    Prepend the tail of the previous chunk to each chunk (except the first).
    The overlap is trimmed to a word boundary where possible.
    """
    if len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap_chars:]
        # Snap to the nearest word boundary (find first space)
        ws = tail.find(" ")
        if 0 < ws < len(tail):
            tail = tail[ws + 1 :]
        result.append(tail + chunks[i])

    return result


# ---------------------------------------------------------------------------
# Page-number lookup table
# ---------------------------------------------------------------------------

def _build_page_index(pages: Sequence[ParsedPage]) -> list[tuple[int, int]]:
    """
    Return a list of ``(char_offset, page_number)`` tuples representing the
    start of each page in the concatenated document string.

    The concatenated string is formed by joining page texts with ``"\\n\\n"``.
    """
    index: list[tuple[int, int]] = []
    offset = 0
    SEP = "\n\n"
    for p in pages:
        index.append((offset, p.page_number))
        offset += len(p.text) + len(SEP)
    return index


def _page_at_offset(index: list[tuple[int, int]], char_offset: int) -> int:
    """Binary-search *index* to find which page *char_offset* belongs to."""
    lo, hi = 0, len(index) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if index[mid][0] <= char_offset:
            lo = mid
        else:
            hi = mid - 1
    return index[lo][1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    pages: Sequence[ParsedPage],
    *,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    hard_max_chars: int = HARD_MAX_CHARS,
) -> list[Chunk]:
    """
    Split a list of parsed pages into overlapping chunks.

    Parameters
    ----------
    pages:
        Output of :func:`src.ingestion.parse.parse_pdf`.
    target_chars:
        Soft upper bound on chunk size in characters.
    overlap_chars:
        How many characters from the end of the previous chunk to prepend
        to the next one.
    hard_max_chars:
        Absolute ceiling; oversized chunks produced by extremely long words
        or separator-free text are truncated to this length.

    Returns
    -------
    list[Chunk]
    """
    if not pages:
        return []

    source_file = pages[0].source_file
    SEP = "\n\n"

    # Concatenate all page texts into a single document string
    full_text = SEP.join(p.text for p in pages)
    page_index = _build_page_index(pages)

    # 1. Recursively split into raw pieces
    raw_pieces = _split_recursive(full_text, SEPARATORS, target_chars)

    # 2. Add overlap between consecutive pieces
    overlapped = _add_overlap(raw_pieces, overlap_chars)

    # 3. Build Chunk objects, computing char offsets by scanning the full text
    chunks: list[Chunk] = []
    search_start = 0  # avoid O(n²) by advancing forward

    for piece in overlapped:
        stripped = piece.strip()
        if not stripped:
            continue

        # Enforce hard maximum (safety net for pathological inputs)
        if len(stripped) > hard_max_chars:
            stripped = stripped[:hard_max_chars].rsplit(" ", 1)[0]

        # Locate this piece in the full_text
        pos = full_text.find(stripped[:50], search_start)
        if pos == -1:
            # Fallback: search from the beginning (shouldn't happen often)
            pos = full_text.find(stripped[:50])
        char_start = pos if pos != -1 else search_start
        char_end = char_start + len(stripped)

        # Advance search cursor (with a small back-step to handle overlaps)
        search_start = max(search_start, char_end - overlap_chars)

        page_number = _page_at_offset(page_index, char_start)

        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                source_file=source_file,
                page_number=page_number,
                text=stripped,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks


def chunk_corpus(
    corpus: dict[str, list[ParsedPage]],
    **kwargs,
) -> list[Chunk]:
    """
    Chunk every document in *corpus* (output of
    :func:`src.ingestion.parse.parse_directory`).

    Returns a flat list of all chunks across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc_name, pages in corpus.items():
        doc_chunks = chunk_document(pages, **kwargs)
        all_chunks.extend(doc_chunks)
    return all_chunks

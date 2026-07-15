"""
tests/test_chunking.py
----------------------
Pytest suite for the ingestion pipeline (parse.py + chunk.py).

Test coverage
-------------
1. Token ceiling         – no chunk exceeds ~600 tokens (~2 400 characters).
2. Metadata completeness – every chunk carries all required metadata fields
                           with the correct types and non-empty values.
3. Page-number continuity– within a single document, page numbers are
                           monotonically non-decreasing.
4. Overlap present       – consecutive chunks from the same document share
                           at least some overlapping text (validates the
                           overlap injection logic).
5. Empty-input guard     – chunk_document([]) returns [].
6. Reference truncation  – pages after a "References" heading are dropped.
7. Integration smoke     – if chunks.jsonl already exists, spot-check it.

Running
-------
    pytest tests/test_chunking.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.chunk import (
    HARD_MAX_CHARS,
    Chunk,
    chunk_document,
    estimate_tokens,
)
from src.ingestion.parse import ParsedPage, parse_pdf, _find_ref_cutoff

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKEN_HARD_LIMIT = 600   # test enforces this ceiling
REQUIRED_FIELDS: set[str] = {
    "chunk_id",
    "source_file",
    "page_number",
    "text",
    "char_start",
    "char_end",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_JSONL = REPO_ROOT / "data" / "processed" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pages(
    texts: list[str],
    source: str = "test_doc.pdf",
    start_page: int = 1,
) -> list[ParsedPage]:
    """Helper: build ParsedPage list from a list of text strings."""
    return [
        ParsedPage(source_file=source, page_number=start_page + i, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def short_pages() -> list[ParsedPage]:
    """Three short pages – each fits in a single chunk."""
    return _make_pages(
        [
            "This is the introduction of the paper. It covers the background.",
            "The method section describes how we do things. Experiments follow.",
            "Results show significant improvements over the baseline approach.",
        ]
    )


@pytest.fixture
def long_page() -> list[ParsedPage]:
    """One very long page that must be split into multiple chunks."""
    # ~4 000 characters – forces at least 2 chunks at TARGET_CHARS=2 000
    sentence = "The quick brown fox jumps over the lazy dog. " * 90
    return _make_pages([sentence])


@pytest.fixture
def multi_doc_pages() -> list[ParsedPage]:
    """Pages from a 'document' with many pages to test continuity."""
    texts = [f"Content of page {i}. " * 30 for i in range(1, 11)]
    return _make_pages(texts, start_page=1)


# ---------------------------------------------------------------------------
# 1. Token ceiling
# ---------------------------------------------------------------------------

class TestTokenCeiling:
    def test_short_pages_under_limit(self, short_pages):
        chunks = chunk_document(short_pages)
        assert chunks, "Expected at least one chunk"
        for c in chunks:
            tokens = estimate_tokens(c.text)
            assert tokens <= TOKEN_HARD_LIMIT, (
                f"Chunk {c.chunk_id} has {tokens} tokens (limit {TOKEN_HARD_LIMIT})\n"
                f"text[:100]: {c.text[:100]!r}"
            )

    def test_long_page_under_limit(self, long_page):
        chunks = chunk_document(long_page)
        assert len(chunks) >= 2, "Long page should produce multiple chunks"
        for c in chunks:
            tokens = estimate_tokens(c.text)
            assert tokens <= TOKEN_HARD_LIMIT, (
                f"Chunk exceeded {TOKEN_HARD_LIMIT} tokens: {tokens}"
            )

    def test_multi_page_under_limit(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        violations = [
            (c.chunk_id, estimate_tokens(c.text))
            for c in chunks
            if estimate_tokens(c.text) > TOKEN_HARD_LIMIT
        ]
        assert not violations, f"Chunks exceeding token limit: {violations}"


# ---------------------------------------------------------------------------
# 2. Metadata completeness
# ---------------------------------------------------------------------------

class TestMetadataCompleteness:
    def _assert_chunk_metadata(self, chunk: Chunk) -> None:
        d = chunk.__dict__
        # All required fields present
        missing = REQUIRED_FIELDS - d.keys()
        assert not missing, f"Missing fields in chunk {chunk.chunk_id}: {missing}"

        # chunk_id: non-empty string
        assert isinstance(chunk.chunk_id, str) and chunk.chunk_id, \
            "chunk_id must be a non-empty string"

        # source_file: non-empty string ending with .pdf
        assert isinstance(chunk.source_file, str) and chunk.source_file, \
            "source_file must be a non-empty string"
        assert chunk.source_file.endswith(".pdf"), \
            f"source_file should be a PDF basename, got {chunk.source_file!r}"

        # page_number: positive int
        assert isinstance(chunk.page_number, int) and chunk.page_number >= 1, \
            f"page_number must be a positive int, got {chunk.page_number!r}"

        # text: non-empty string
        assert isinstance(chunk.text, str) and chunk.text.strip(), \
            "text must be a non-empty string"

        # char offsets: non-negative, consistent
        assert isinstance(chunk.char_start, int) and chunk.char_start >= 0, \
            f"char_start must be >= 0, got {chunk.char_start}"
        assert isinstance(chunk.char_end, int) and chunk.char_end > chunk.char_start, \
            f"char_end ({chunk.char_end}) must be > char_start ({chunk.char_start})"

    def test_all_chunks_have_complete_metadata(self, short_pages):
        for chunk in chunk_document(short_pages):
            self._assert_chunk_metadata(chunk)

    def test_long_page_chunks_have_metadata(self, long_page):
        for chunk in chunk_document(long_page):
            self._assert_chunk_metadata(chunk)

    def test_chunk_ids_are_unique(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"

    def test_source_file_consistent_within_doc(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        sources = {c.source_file for c in chunks}
        assert sources == {"test_doc.pdf"}, \
            f"All chunks should share the same source_file, got {sources}"


# ---------------------------------------------------------------------------
# 3. Page-number continuity
# ---------------------------------------------------------------------------

class TestPageNumberContinuity:
    def test_page_numbers_non_decreasing(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        assert chunks, "Expected chunks from multi-page document"
        for prev, curr in zip(chunks, chunks[1:]):
            assert curr.page_number >= prev.page_number, (
                f"Page number went backwards: chunk {prev.chunk_id} "
                f"(page {prev.page_number}) → chunk {curr.chunk_id} "
                f"(page {curr.page_number})"
            )

    def test_page_numbers_within_document_range(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        max_page = len(multi_doc_pages)
        for c in chunks:
            assert 1 <= c.page_number <= max_page, (
                f"page_number {c.page_number} out of range [1, {max_page}]"
            )

    def test_first_chunk_on_first_page(self, multi_doc_pages):
        chunks = chunk_document(multi_doc_pages)
        assert chunks[0].page_number == 1, \
            f"First chunk should be on page 1, got page {chunks[0].page_number}"


# ---------------------------------------------------------------------------
# 4. Overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_consecutive_chunks_share_text(self, long_page):
        """Adjacent chunks should share some text due to overlap injection."""
        chunks = chunk_document(long_page)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        shared = False
        for prev, curr in zip(chunks, chunks[1:]):
            # Take the last 100 chars of prev and check if any of it
            # appears at the start of curr
            tail = prev.text[-100:].strip()[:40]
            if tail and tail in curr.text[:300]:
                shared = True
                break
        assert shared, "Expected some text overlap between consecutive chunks"


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_returns_empty_list(self):
        assert chunk_document([]) == []

    def test_single_empty_page_returns_empty_list(self):
        pages = _make_pages(["   \n  \n  "])  # whitespace only
        chunks = chunk_document(pages)
        assert chunks == [], f"Expected [], got {chunks}"

    def test_very_long_word_truncated(self):
        """A chunk containing a single word longer than HARD_MAX_CHARS is safe."""
        monster = "a" * (HARD_MAX_CHARS + 500)
        pages = _make_pages([monster])
        chunks = chunk_document(pages)
        for c in chunks:
            assert len(c.text) <= HARD_MAX_CHARS + 10, \
                f"Chunk is {len(c.text)} chars, hard max is {HARD_MAX_CHARS}"


# ---------------------------------------------------------------------------
# 6. Reference-section truncation
# ---------------------------------------------------------------------------

class TestReferenceTruncation:
    def test_ref_heading_detected(self):
        text_with_refs = "Some content.\n\nReferences\n\n[1] Author et al."
        cutoff = _find_ref_cutoff(text_with_refs)
        assert cutoff is not None, "Should detect 'References' heading"
        assert "References" not in text_with_refs[:cutoff]

    def test_bibliography_heading_detected(self):
        text = "Content.\n\nBibliography\n\n[1] Some paper."
        assert _find_ref_cutoff(text) is not None

    def test_numbered_ref_heading_detected(self):
        text = "Content.\n\n7  References\n\n[1] Paper."
        assert _find_ref_cutoff(text) is not None

    def test_no_false_positive_in_body(self):
        text = "We reference prior work on attention.\nSee section 3."
        # "reference" inside a sentence should NOT trigger
        assert _find_ref_cutoff(text) is None


# ---------------------------------------------------------------------------
# 7. Integration smoke test (requires chunks.jsonl to exist)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CHUNKS_JSONL.exists(),
    reason="chunks.jsonl not found – run run_ingestion.py first",
)
class TestIntegrationChunksJsonl:
    @pytest.fixture(scope="class")
    @classmethod
    def all_chunks(cls) -> list[dict]:
        with CHUNKS_JSONL.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_file_is_non_empty(self, all_chunks):
        assert len(all_chunks) > 0, "chunks.jsonl should contain at least one chunk"

    def test_all_required_fields_present(self, all_chunks):
        for i, chunk in enumerate(all_chunks):
            missing = REQUIRED_FIELDS - chunk.keys()
            assert not missing, f"Chunk #{i} missing fields: {missing}"

    def test_no_chunk_exceeds_token_limit(self, all_chunks):
        violations = [
            (c["chunk_id"], estimate_tokens(c["text"]))
            for c in all_chunks
            if estimate_tokens(c["text"]) > TOKEN_HARD_LIMIT
        ]
        assert not violations, (
            f"{len(violations)} chunks exceed {TOKEN_HARD_LIMIT} tokens. "
            f"First few: {violations[:5]}"
        )

    def test_page_continuity_per_document(self, all_chunks):
        from collections import defaultdict

        doc_chunks: dict[str, list[dict]] = defaultdict(list)
        for c in all_chunks:
            doc_chunks[c["source_file"]].append(c)

        for doc, chunks in doc_chunks.items():
            pages = [c["page_number"] for c in chunks]
            for i in range(len(pages) - 1):
                assert pages[i + 1] >= pages[i], (
                    f"{doc}: page number went backwards at chunk index {i}: "
                    f"{pages[i]} → {pages[i+1]}"
                )

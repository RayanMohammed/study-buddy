"""End-to-end orchestration: PDF file(s) -> extracted -> chunked -> embedded -> stored.

This module has no CLI or argument-parsing concerns; it's the library-level
entry point that study_buddy.cli (and tests) call into.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from study_buddy.chunking import Chunk, chunk_pages
from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.extraction import extract_pdf_pages
from study_buddy.vectorstore import store_chunks


@dataclass
class IngestResult:
    filename: str
    page_count: int
    chunk_count: int
    chunks: list[Chunk]


def ingest_pdf(pdf_path: Path, settings: Settings = DEFAULT_SETTINGS) -> IngestResult:
    """Run one PDF through the full pipeline and persist its chunks."""
    pages = extract_pdf_pages(pdf_path, settings)
    chunks = chunk_pages(pages, settings)
    store_chunks(chunks, settings)
    return IngestResult(
        filename=pdf_path.name,
        page_count=len(pages),
        chunk_count=len(chunks),
        chunks=chunks,
    )


def ingest_folder(folder: Path, settings: Settings = DEFAULT_SETTINGS) -> list[IngestResult]:
    """Ingest every PDF directly inside `folder` (non-recursive)."""
    pdf_paths = sorted(p for p in folder.glob("*.pdf"))
    return [ingest_pdf(pdf_path, settings) for pdf_path in pdf_paths]

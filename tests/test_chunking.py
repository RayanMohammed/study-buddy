from __future__ import annotations

from pathlib import Path

from study_buddy.chunking import chunk_pages
from study_buddy.config import Settings
from study_buddy.extraction import extract_pdf_pages


def test_chunks_split_on_heading_and_page_boundaries(sample_pdf: Path, isolated_settings: Settings):
    pages = extract_pdf_pages(sample_pdf, isolated_settings)
    assert len(pages) == 2

    chunks = chunk_pages(pages, isolated_settings)

    # Page 1 has two headings -> two chunks; page 2 has one -> one chunk.
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page1_chunks) == 2
    assert len(page2_chunks) == 1

    topics = {c.topic for c in page1_chunks}
    assert "Photosynthesis Overview" in topics
    assert "Light-Dependent Reactions" in topics


def test_chunk_metadata_preserves_filename_page_and_topic(sample_pdf: Path, isolated_settings: Settings):
    pages = extract_pdf_pages(sample_pdf, isolated_settings)
    chunks = chunk_pages(pages, isolated_settings)

    for chunk in chunks:
        metadata = chunk.to_metadata()
        assert metadata["filename"] == "biology_slides.pdf"
        assert isinstance(metadata["page_number"], int)
        assert metadata["topic"]
        assert metadata["extraction_method"] == "pymupdf"

    calvin_chunk = next(c for c in chunks if c.topic == "The Calvin Cycle")
    assert "carbon dioxide" in calvin_chunk.text


def test_chunk_ids_are_unique_and_deterministic(sample_pdf: Path, isolated_settings: Settings):
    pages = extract_pdf_pages(sample_pdf, isolated_settings)
    chunks_a = chunk_pages(pages, isolated_settings)
    chunks_b = chunk_pages(pages, isolated_settings)

    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert len(ids_a) == len(set(ids_a))  # all unique
    assert ids_a == ids_b  # re-chunking the same pages is deterministic

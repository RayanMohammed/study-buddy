"""End-to-end tests: ingest a real PDF and confirm chunks come back out
of ChromaDB via semantic search (not just that they were written)."""

from __future__ import annotations

from pathlib import Path

from study_buddy.config import Settings
from study_buddy.ingest import ingest_pdf
from study_buddy.vectorstore import query_chunks


def test_ingested_chunks_are_retrievable_by_semantic_query(sample_pdf: Path, isolated_settings: Settings):
    result = ingest_pdf(sample_pdf, isolated_settings)

    assert result.page_count == 2
    assert result.chunk_count == 3

    results = query_chunks("carbon dioxide fixation in plants", n_results=1, settings=isolated_settings)
    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]

    assert "Calvin cycle" in top_doc
    assert top_meta["topic"] == "The Calvin Cycle"
    assert top_meta["filename"] == "biology_slides.pdf"
    assert top_meta["page_number"] == 2


def test_reingesting_same_pdf_does_not_duplicate_chunks(sample_pdf: Path, isolated_settings: Settings):
    from study_buddy.vectorstore import get_collection

    ingest_pdf(sample_pdf, isolated_settings)
    ingest_pdf(sample_pdf, isolated_settings)

    collection = get_collection(isolated_settings)
    assert collection.count() == 3


def test_query_top_result_differs_by_topic(sample_pdf: Path, isolated_settings: Settings):
    ingest_pdf(sample_pdf, isolated_settings)

    photosynthesis_results = query_chunks("what is chlorophyll used for", n_results=1, settings=isolated_settings)
    assert photosynthesis_results["metadatas"][0][0]["topic"] == "Photosynthesis Overview"

    calvin_results = query_chunks("stroma of the chloroplast", n_results=1, settings=isolated_settings)
    assert calvin_results["metadatas"][0][0]["topic"] == "The Calvin Cycle"

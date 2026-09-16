"""End-to-end tests: ingest a real PDF and confirm chunks come back out
of Postgres (via pgvector) through semantic search (not just that they
were written)."""

from __future__ import annotations

from pathlib import Path

from study_buddy.config import Settings
from study_buddy.db import SlideChunk, get_session_factory
from study_buddy.ingest import ingest_pdf
from study_buddy.vectorstore import query_chunks


def test_ingested_chunks_are_retrievable_by_semantic_query(sample_pdf: Path, db_settings: Settings):
    result = ingest_pdf(sample_pdf, db_settings)

    assert result.page_count == 2
    assert result.chunk_count == 3

    results = query_chunks("carbon dioxide fixation in plants", n_results=1, settings=db_settings)
    top = results[0]

    assert "Calvin cycle" in top["text"]
    assert top["topic"] == "The Calvin Cycle"
    assert top["filename"] == "biology_slides.pdf"
    assert top["page_number"] == 2


def test_reingesting_same_pdf_does_not_duplicate_chunks(sample_pdf: Path, db_settings: Settings):
    ingest_pdf(sample_pdf, db_settings)
    ingest_pdf(sample_pdf, db_settings)

    session_factory = get_session_factory(db_settings)
    with session_factory() as session:
        count = session.query(SlideChunk).filter_by(filename="biology_slides.pdf").count()
    assert count == 3


def test_query_top_result_differs_by_topic(sample_pdf: Path, db_settings: Settings):
    ingest_pdf(sample_pdf, db_settings)

    photosynthesis_results = query_chunks("what is chlorophyll used for", n_results=1, settings=db_settings)
    assert photosynthesis_results[0]["topic"] == "Photosynthesis Overview"

    calvin_results = query_chunks("stroma of the chloroplast", n_results=1, settings=db_settings)
    assert calvin_results[0]["topic"] == "The Calvin Cycle"

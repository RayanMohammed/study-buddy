"""Shared pytest fixtures: build small real PDFs on the fly with PyMuPDF
so tests don't depend on checked-in binary fixture files."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf as fitz
import pytest
from sqlalchemy import delete

from study_buddy.config import Settings
from study_buddy.db import QuestionInteraction, SlideChunk, init_db


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF: each page has a large-font heading and normal-font body,
    enough plain text to avoid tripping the math-heavy fallback heuristic."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Photosynthesis Overview", fontsize=18)
    page1.insert_text(
        (72, 110),
        "Photosynthesis converts light energy into chemical energy.\n"
        "Plants use chlorophyll to capture sunlight for this process.",
        fontsize=11,
    )
    page1.insert_text((72, 200), "Light-Dependent Reactions", fontsize=18)
    page1.insert_text(
        (72, 230),
        "These reactions occur in the thylakoid membrane and produce ATP.\n"
        "Water molecules are split, releasing oxygen as a byproduct.",
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_text((72, 72), "The Calvin Cycle", fontsize=18)
    page2.insert_text(
        (72, 110),
        "The Calvin cycle fixes carbon dioxide into organic molecules.\n"
        "This cycle takes place in the stroma of the chloroplast.",
        fontsize=11,
    )

    pdf_path = tmp_path / "biology_slides.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def isolated_settings() -> Settings:
    """Plain settings for tests that only exercise extraction/chunking
    logic and never open a database connection."""
    return Settings()


@pytest.fixture
def db_settings() -> Settings:
    """Settings pointed at a dedicated Neon test branch (TEST_DATABASE_URL)
    — never the application's real DATABASE_URL.

    slide_chunks and question_interactions are truncated before every test
    that requests this fixture, so each test starts from a clean, empty
    database. This is safe only because TEST_DATABASE_URL is expected to
    point at a disposable branch, not production data.
    """
    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip(
            "TEST_DATABASE_URL is not set — point it at a dedicated Neon "
            "test branch (with the pgvector extension available) to run "
            "database-backed tests."
        )

    settings = Settings(database_url=test_url)
    engine = init_db(settings)
    with engine.begin() as conn:
        conn.execute(delete(SlideChunk))
        conn.execute(delete(QuestionInteraction))
    return settings

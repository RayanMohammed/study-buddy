"""Shared pytest fixtures: build small real PDFs on the fly with PyMuPDF
so tests don't depend on checked-in binary fixture files."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from study_buddy.config import Settings


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
def isolated_settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway chroma dir and sqlite file per test."""
    return Settings(
        chroma_dir=tmp_path / "chroma",
        chroma_collection="test_collection",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )

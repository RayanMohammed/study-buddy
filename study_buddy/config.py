"""Central configuration for the ingestion pipeline.

All paths and model names live here so the CLI, the pipeline, and the tests
agree on where things are stored without repeating string literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Where ChromaDB persists its on-disk collection.
    chroma_dir: Path = Path("./data/chroma")
    chroma_collection: str = "study_material"

    # sentence-transformers model used to embed every chunk.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # SQLAlchemy connection string for the interactions database.
    # Swap for a postgres:// URL to move to Postgres without code changes.
    database_url: str = "sqlite:///./data/study_buddy.db"

    # A page is routed to the marker-pdf fallback when pymupdf's extracted
    # text looks too sparse/garbled relative to how much content the page
    # visually holds (see extraction.is_math_heavy_or_low_quality).
    min_chars_per_page: int = 40
    math_symbol_density_threshold: float = 0.04

    # Chunking: a heading/topic must be shorter than this to be treated as
    # a section title rather than a wrapped paragraph line.
    max_heading_chars: int = 120

    # A line is treated as a section heading when its font size is at least
    # this many times the page's dominant (body) font size.
    heading_font_ratio: float = 1.15


DEFAULT_SETTINGS = Settings()

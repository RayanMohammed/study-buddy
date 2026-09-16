"""Central configuration for the ingestion pipeline.

All paths and model names live here so the CLI, the pipeline, and the tests
agree on where things are stored without repeating string literals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load the project-root .env file explicitly (rather than relying on
# python-dotenv's upward directory search) so DATABASE_URL is found
# regardless of the current working directory a script is run from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _database_url_from_env() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to a .env file at the project "
            "root (or export it) — it should point at a Postgres instance "
            "with the pgvector extension available, e.g. a Neon database."
        )
    return url


@dataclass(frozen=True)
class Settings:
    # sentence-transformers model used to embed every chunk.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # SQLAlchemy connection string for the single Postgres database that
    # holds both slide chunks (with pgvector embeddings) and quiz
    # interactions. Sourced from the DATABASE_URL environment variable.
    database_url: str = field(default_factory=_database_url_from_env)

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

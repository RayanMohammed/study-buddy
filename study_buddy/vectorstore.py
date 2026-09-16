"""Step 5: persist embedded chunks in Postgres via pgvector.

Chunks live in the same Postgres database as quiz interactions (see
study_buddy.db.SlideChunk for the schema), but the reads/writes here are
raw SQL executed via SQLAlchemy's text() — not the ORM query builder — so
the actual statements sent to Postgres are plain, readable SQL.

Bypassing the ORM has one consequence worth calling out: pgvector's
SQLAlchemy column type (used only for schema/DDL in db.py) knows how to
convert a Python list of floats into Postgres's `vector` wire format
automatically. Raw text() SQL does not get that conversion for free — a
Python list handed to a bind parameter has no defined `vector`
representation, so a naive `:embedding` bind would arrive at Postgres as an
opaque value with no way to compare it against the vector column. Instead,
each embedding is formatted here as pgvector's text literal syntax
(`"[0.1,0.2,0.3]"`, a plain string) and cast to `vector` explicitly in the
SQL via `CAST(:embedding AS vector)`.
"""

from __future__ import annotations

from sqlalchemy import text

from study_buddy.chunking import Chunk
from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.db import get_session_factory
from study_buddy.embeddings import embed_texts


def _vector_literal(vector: list[float]) -> str:
    """Format a Python list of floats as pgvector's text input syntax.

    pgvector accepts a vector literal as a bracketed, comma-separated
    string, e.g. "[0.1,0.2,0.3]" — this is what lets a plain string bind
    parameter be cast to `vector` with `CAST(:param AS vector)` below.
    """
    return "[" + ",".join(repr(component) for component in vector) + "]"


_UPSERT_SQL = text(
    """
    INSERT INTO slide_chunks
        (chunk_id, filename, page_number, topic, text, extraction_method, embedding)
    VALUES
        (:chunk_id, :filename, :page_number, :topic, :text, :extraction_method,
         CAST(:embedding AS vector))
    ON CONFLICT (chunk_id) DO UPDATE SET
        filename = EXCLUDED.filename,
        page_number = EXCLUDED.page_number,
        topic = EXCLUDED.topic,
        text = EXCLUDED.text,
        extraction_method = EXCLUDED.extraction_method,
        embedding = EXCLUDED.embedding
    """
)

_QUERY_SQL = text(
    """
    SELECT
        chunk_id,
        text,
        filename,
        page_number,
        topic,
        embedding <=> CAST(:query_embedding AS vector) AS distance
    FROM slide_chunks
    ORDER BY distance ASC
    LIMIT :n_results
    """
)


def store_chunks(chunks: list[Chunk], settings: Settings = DEFAULT_SETTINGS) -> int:
    """Embed and upsert a list of chunks into the slide_chunks table.

    Upsert (rather than insert) makes re-ingesting the same PDF idempotent:
    a chunk's deterministic id (filename + page + position) means re-running
    the CLI on an unchanged file updates its row in place instead of
    duplicating it.
    """
    if not chunks:
        return 0

    vectors = embed_texts([c.text for c in chunks], settings)
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "topic": chunk.topic,
            "text": chunk.text,
            "extraction_method": chunk.extraction_method,
            "embedding": _vector_literal(vector),
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    # A text() statement executed with a *list* of parameter dicts runs as
    # a single executemany round trip — one INSERT per row, batched.
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        session.execute(_UPSERT_SQL, rows)
        session.commit()
    return len(chunks)


def query_chunks(query_text: str, n_results: int = 5, settings: Settings = DEFAULT_SETTINGS) -> list[dict]:
    """Embed a query and return the top-n most similar stored chunks.

    Returns a flat list of dicts (chunk_id, text, filename, page_number,
    topic, distance), ordered by ascending cosine distance (closest/most
    similar first).
    """
    query_vector = embed_texts([query_text], settings)[0]

    session_factory = get_session_factory(settings)
    with session_factory() as session:
        rows = session.execute(
            _QUERY_SQL,
            {"query_embedding": _vector_literal(query_vector), "n_results": n_results},
        ).mappings().all()

    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "filename": row["filename"],
            "page_number": row["page_number"],
            "topic": row["topic"],
            "distance": float(row["distance"]),
        }
        for row in rows
    ]

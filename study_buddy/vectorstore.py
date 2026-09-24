"""Step 5: persist embedded chunks in Postgres via pgvector.

Chunks live in the same Postgres database as quiz interactions (see
study_buddy.db.SlideChunk for the schema). Reads and writes here go
through SQLAlchemy's ORM/Core expression API rather than raw SQL:
pgvector's SQLAlchemy integration (the Vector column type on
SlideChunk.embedding) already knows how to convert a Python list of floats
into Postgres's `vector` wire format and back, and
SlideChunk.embedding.cosine_distance(...) compiles directly to pgvector's
`<=>` operator. Letting the ORM handle that conversion — rather than
hand-formatting embeddings into pgvector's text literal syntax and casting
them in raw SQL — means a plain Python list goes in as-is, with no
error-prone manual formatting step in between.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from study_buddy.chunking import Chunk
from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.db import SlideChunk, get_session_factory
from study_buddy.embeddings import embed_texts

_UPSERT_COLUMNS = ("filename", "page_number", "topic", "text", "extraction_method", "embedding")


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
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]

    stmt = pg_insert(SlideChunk).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["chunk_id"],
        set_={column: getattr(stmt.excluded, column) for column in _UPSERT_COLUMNS},
    )

    session_factory = get_session_factory(settings)
    with session_factory() as session:
        session.execute(stmt)
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
        distance = SlideChunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = select(SlideChunk, distance).order_by(distance).limit(n_results)
        rows = session.execute(stmt).all()

    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "topic": chunk.topic,
            "distance": float(distance_value),
        }
        for chunk, distance_value in rows
    ]

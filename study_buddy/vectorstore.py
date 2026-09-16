"""Step 5: persist embedded chunks in a local ChromaDB collection.

ChromaDB is used in its embedded, on-disk "PersistentClient" mode: no
server to run, everything lives under `settings.chroma_dir`. We pass
embeddings in ourselves (computed via study_buddy.embeddings) rather than
letting Chroma manage an embedding function, so the same MiniLM model is
unambiguously used for both storage and querying.
"""

from __future__ import annotations

from study_buddy.chunking import Chunk
from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.embeddings import embed_texts


def get_collection(settings: Settings = DEFAULT_SETTINGS):
    """Open (creating if needed) the persistent ChromaDB collection."""
    import chromadb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(chunks: list[Chunk], settings: Settings = DEFAULT_SETTINGS) -> int:
    """Embed and upsert a list of chunks into the collection.

    Upsert (rather than add) makes re-ingesting the same PDF idempotent:
    a chunk's deterministic id (filename + page + position) means re-running
    the CLI on an unchanged file updates its vectors in place instead of
    duplicating them.
    """
    if not chunks:
        return 0

    collection = get_collection(settings)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, settings)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[c.to_metadata() for c in chunks],
    )
    return len(chunks)


def query_chunks(query_text: str, n_results: int = 5, settings: Settings = DEFAULT_SETTINGS) -> dict:
    """Embed a query and return the top-n most similar stored chunks."""
    collection = get_collection(settings)
    query_vector = embed_texts([query_text], settings)[0]
    return collection.query(query_embeddings=[query_vector], n_results=n_results)

"""Step 4: turn chunk text into vectors with sentence-transformers.

The model is loaded once per process (it's ~80MB and downloads from the
Hugging Face Hub on first use, then caches locally) and reused for every
chunk and every query, so embedding a query later uses the exact same
vector space as the chunks stored in ChromaDB.
"""

from __future__ import annotations

from functools import lru_cache

from study_buddy.config import DEFAULT_SETTINGS, Settings


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embedder(settings: Settings = DEFAULT_SETTINGS):
    """Return the (cached) sentence-transformers model for `settings`."""
    return _load_model(settings.embedding_model)


def embed_texts(texts: list[str], settings: Settings = DEFAULT_SETTINGS) -> list[list[float]]:
    """Embed a batch of strings into a list of float vectors."""
    if not texts:
        return []
    model = get_embedder(settings)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()

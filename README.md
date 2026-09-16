# study-buddy

A PDF ingestion pipeline for study material: extract text from PDFs, split
it into topic-tagged chunks, embed those chunks, and store them in a local
vector database for semantic search. Also includes a small relational
schema for logging quiz question/answer interactions (e.g. from an AI
tutoring loop).

## Pipeline

1. **Accept PDFs** — a single file or a folder of files.
2. **Extract text** ([`study_buddy/extraction.py`](study_buddy/extraction.py)) — PyMuPDF does the extraction for
   every page. A page is flagged as math-heavy/low-quality (too little text,
   or a high density of LaTeX/math symbols) and re-extracted with
   **marker-pdf** if that optional dependency is installed; otherwise the
   PyMuPDF text is kept and tagged `pymupdf_lowconf` so you know it may be
   lossy for that page.
3. **Chunk semantically** ([`study_buddy/chunking.py`](study_buddy/chunking.py)) — chunks never span
   multiple pages (each page/slide is a hard boundary), and within a page,
   headings (lines whose font is noticeably larger than the page's body
   text) start new chunks. No fixed character count is used anywhere.
   Every chunk carries `filename`, `page_number`, and `topic` (the heading
   that introduced it) as metadata.
4. **Embed** ([`study_buddy/embeddings.py`](study_buddy/embeddings.py)) — `sentence-transformers/all-MiniLM-L6-v2`,
   loaded once and cached.
5. **Store** ([`study_buddy/vectorstore.py`](study_buddy/vectorstore.py)) — a local, on-disk ChromaDB collection
   (`./data/chroma` by default). Chunk IDs are deterministic, so re-ingesting
   an unchanged PDF updates its vectors in place instead of duplicating them.
6. **Track interactions** ([`study_buddy/db.py`](study_buddy/db.py)) — a SQLAlchemy model, `QuestionInteraction`,
   with fields `question_id, topic_id, question_text, user_answer, is_correct,
   response_time_ms, timestamp, model_used, tool_call_valid`. Backed by
   SQLite by default (`./data/study_buddy.db`); point `Settings.database_url`
   at a `postgresql://...` URL to use Postgres instead — no code changes.

## Setup

This machine's default Python is 3.14, which is too new for some of the ML
dependencies (torch, chromadb, onnxruntime) to have compatible wheels yet.
The project was set up with a Python 3.12 virtual environment via `uv`:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" --python .venv/bin/python
```

To also enable the marker-pdf fallback for math-heavy pages (a heavier,
optional dependency that pulls in its own OCR/layout models):

```bash
uv pip install -e ".[dev,marker]" --python .venv/bin/python
```

Activate the environment for interactive use:

```bash
source .venv/bin/activate
```

## CLI usage

```bash
# Ingest every PDF in a folder (or point at a single .pdf file)
study-buddy ingest ./my_slides/

# Sanity-check retrieval
study-buddy query "eigenvalues of a symmetric matrix"

# Log a quiz interaction
study-buddy log-interaction \
  --topic-id "Backpropagation" \
  --question-text "What rule does backpropagation apply?" \
  --user-answer "chain rule" \
  --is-correct true \
  --response-time-ms 3200 \
  --model-used "claude-sonnet-5" \
  --tool-call-valid true
```

(If you didn't `pip install -e .`, replace `study-buddy` with
`.venv/bin/python -m study_buddy.cli`.)

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Tests generate small real PDFs on the fly with PyMuPDF (no checked-in binary
fixtures) and verify:
- chunks split on heading/page boundaries and preserve metadata
  ([`tests/test_chunking.py`](tests/test_chunking.py)),
- ingested chunks are retrievable from ChromaDB by semantic query, and
  re-ingestion doesn't duplicate them ([`tests/test_ingest_pipeline.py`](tests/test_ingest_pipeline.py)),
- quiz interactions persist correctly to the database across sessions
  ([`tests/test_db.py`](tests/test_db.py)).

Each test uses an isolated `Settings` (temp ChromaDB dir + temp SQLite file)
so runs never interfere with each other or with `./data`.

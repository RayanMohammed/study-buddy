# study-buddy

A PDF ingestion pipeline for study material: extract text from PDFs, split
it into topic-tagged chunks, embed those chunks, and store them for semantic
search. Slide chunks and quiz question/answer interactions (e.g. from an AI
tutoring loop) live together in one Postgres database, using the
[pgvector](https://github.com/pgvector/pgvector) extension for similarity
search.

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
5. **Store** ([`study_buddy/vectorstore.py`](study_buddy/vectorstore.py)) — embeddings are written to the
   `slide_chunks` table ([`study_buddy/db.py`](study_buddy/db.py)) as a pgvector column. Chunk IDs are
   deterministic, and storage uses a Postgres `INSERT ... ON CONFLICT DO
   UPDATE` upsert, so re-ingesting an unchanged PDF updates its row in place
   instead of duplicating it. Retrieval orders by pgvector's cosine distance
   directly in SQL.
6. **Track interactions** ([`study_buddy/db.py`](study_buddy/db.py)) — a SQLAlchemy model, `QuestionInteraction`,
   with fields `question_id, topic_id, question_text, user_answer, is_correct,
   response_time_ms, timestamp, model_used, tool_call_valid`, in the same
   Postgres database as `slide_chunks`.
7. **Tutor** ([`study_buddy/tutor.py`](study_buddy/tutor.py), [`study_buddy/tutor_session.py`](study_buddy/tutor_session.py)) — an interactive
   agentic loop (`study-buddy tutor <topic>`) built on Groq's
   `llama-3.3-70b-versatile` via the OpenAI-compatible chat-completions API
   (`tools`/`tool_choice`, not Anthropic's `tool_use` shape). The model
   itself decides which of five tools to call next — `retrieve_context`,
   `generate_question`, `record_answer`, `get_weakest_topic`, `end_session`
   — including whether to remediate a weak topic or progress; that
   adaptive reasoning lives entirely in `tutor.py`'s `SYSTEM_PROMPT`, not in
   hardcoded Python branching. `generate_question`'s actual executor
   (currently `GroqQuestionGenerator`) sits behind a `QuestionGenerator`
   Protocol so a local model (e.g. via Ollama) can later be swapped in
   without touching the orchestrator; `get_weakest_topic` similarly sits
   behind a `WeaknessEstimator` Protocol, today backed by a plain SQL
   aggregate over `question_interactions` as a placeholder for a future
   knowledge-tracing model. Every tool call — name, arguments, timestamp,
   and the raw model response that produced it — is logged to a
   `tool_call_logs` table for later hand-evaluation of tool-selection
   accuracy. Groq calls go through a shared retry/backoff wrapper
   (`study_buddy/groq_client.py`) that respects `Retry-After` on 429s,
   since Groq's free-tier requests-per-minute cap is tighter than its
   tokens-per-minute cap and this loop makes several calls per turn.

## Setup

This machine's default Python is 3.14, which is too new for some of the ML
dependencies (torch, onnxruntime) to have compatible wheels yet. The project
was set up with a Python 3.12 virtual environment via `uv`:

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

### Database

`DATABASE_URL` must point at a Postgres instance with the `pgvector`
extension available (a `.env` file at the project root, gitignored, is
where this is expected — [`config.py`](study_buddy/config.py) loads it
automatically). This project was verified against a
[Neon](https://neon.tech) instance; Neon has pgvector pre-installed, and
`init_db()` runs `CREATE EXTENSION IF NOT EXISTS vector` plus
`CREATE TABLE IF NOT EXISTS` for all three tables on first use, so no
manual migration step is needed beyond having `DATABASE_URL` set.

To use `study-buddy tutor`, also add a Groq API key:

```
GROQ_API_KEY="gsk_..."
```

`GROQ_API_KEY` is optional at the `Settings` level — every other command
(`ingest`/`query`/`log-interaction`) works fine without it. The error only
surfaces when `study-buddy tutor` actually tries to reach Groq.

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

# Start an interactive AI tutoring session (needs GROQ_API_KEY, see above)
study-buddy tutor "Photosynthesis"
study-buddy tutor "Photosynthesis" --max-turns 10   # safety cap, default 20
```

(If you didn't `pip install -e .`, replace `study-buddy` with
`.venv/bin/python -m study_buddy.cli`.)

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Tests generate small real PDFs on the fly with PyMuPDF (no checked-in binary
fixtures). [`tests/test_chunking.py`](tests/test_chunking.py) (chunk boundaries/metadata) needs no
database at all and always runs.

[`tests/test_db.py`](tests/test_db.py) and [`tests/test_ingest_pipeline.py`](tests/test_ingest_pipeline.py) exercise `slide_chunks`
(pgvector) and `question_interactions`, so they need a real Postgres
connection — SQLite can't run either the `vector` extension or a pgvector
column. Set `TEST_DATABASE_URL` in `.env` to a **dedicated Neon branch**
(Neon → your project → Branches → Create branch), separate from the
`DATABASE_URL` branch that holds real study material:

```
TEST_DATABASE_URL="postgresql://...-test-branch.../neondb?sslmode=require"
```

The [`db_settings`](tests/conftest.py) fixture truncates `slide_chunks`,
`question_interactions`, and `tool_call_logs` on that branch before every
test that uses it, so it's safe for tests to write and wipe data there —
just never point `TEST_DATABASE_URL` at the same branch as `DATABASE_URL`.
If `TEST_DATABASE_URL` isn't set, those tests are skipped (not failed) with
a message saying so; chunking tests still run either way.

`study_buddy/tutor*.py` and `study_buddy/groq_client.py`/`question_generator.py`
are covered by `tests/test_groq_client.py`, `test_question_generator.py`,
`test_tutor_session.py`, and `test_tutor_loop.py` — all mocked (plus
`test_tutor_session.py`'s DB assertions, gated on `db_settings` like the
rest), so they need no `GROQ_API_KEY` and make no real network calls.
[`tests/test_tutor_live.py`](tests/test_tutor_live.py) is the one exception: gated on a `groq_settings`
fixture (same skip-if-unset pattern, built on top of `db_settings`) that
needs `GROQ_API_KEY` set, and makes one real call to Groq to catch actual
wire-format drift rather than assert specific tutoring behavior.

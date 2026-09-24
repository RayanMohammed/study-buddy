"""Step 6: relational schema for slide chunks and quiz interactions, via SQLAlchemy.

Both tables live in the same Postgres database (see config.Settings.database_url,
sourced from DATABASE_URL): slide_chunks holds embedded study material for
pgvector similarity search, question_interactions holds the record of a
learner (or an AI tutor) answering a question about that material.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy import DateTime
from sqlalchemy import text as sql_text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from study_buddy.config import DEFAULT_SETTINGS, Settings

# sentence-transformers/all-MiniLM-L6-v2's output size (see embeddings.py) —
# the slide_chunks.embedding column must match this exactly.
EMBEDDING_DIMENSIONS = 384


class Base(DeclarativeBase):
    pass


class SlideChunk(Base):
    """One embedded chunk of study material, ready for pgvector similarity search."""

    __tablename__ = "slide_chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)


class ToolCallLog(Base):
    """One individual tool invocation during a `study-buddy tutor` session,
    logged for later hand-evaluation of the model's tool-selection accuracy.

    Distinct from QuestionInteraction.tool_call_valid (a simple per-question
    flag on the answer-recording row) — this is a full audit trail of every
    tool call the model made, including calls like retrieve_context and
    get_weakest_topic that never touch question_interactions at all.
    """

    __tablename__ = "tool_call_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class QuestionInteraction(Base):
    """One recorded attempt at answering a question about ingested material."""

    __tablename__ = "question_interactions"

    question_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    topic_id: Mapped[str] = mapped_column(String(255), nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    user_answer: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    model_used: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_call_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)


def get_engine(settings: Settings = DEFAULT_SETTINGS):
    """Create the SQLAlchemy engine for the configured Postgres database.

    Neon (and most providers) hand out plain `postgresql://` URLs, which
    SQLAlchemy defaults to the psycopg2 driver. This project installs
    psycopg (v3) instead, so a bare `postgresql://` scheme is upgraded to
    `postgresql+psycopg://` here rather than requiring DATABASE_URL itself
    to know which driver the app happens to use.
    """
    url = settings.database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url)


def init_db(settings: Settings = DEFAULT_SETTINGS):
    """Enable pgvector and create all tables if they don't already exist."""
    engine = get_engine(settings)
    with engine.begin() as conn:
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(settings: Settings = DEFAULT_SETTINGS) -> sessionmaker[Session]:
    engine = init_db(settings)
    return sessionmaker(bind=engine)

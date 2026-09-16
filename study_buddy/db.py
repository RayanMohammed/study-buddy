"""Step 6: relational schema for quiz interactions, via SQLAlchemy.

This is deliberately separate from the ChromaDB vector store: ChromaDB
holds study material for retrieval, this database holds the record of a
learner (or an AI tutor) answering a question about that material. The
same models and engine work against SQLite (the default, zero-setup) or
Postgres — only `settings.database_url` changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from study_buddy.config import DEFAULT_SETTINGS, Settings


class Base(DeclarativeBase):
    pass


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
    """Create the SQLAlchemy engine, ensuring a local sqlite file's dir exists."""
    if settings.database_url.startswith("sqlite:///./"):
        from pathlib import Path

        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(settings.database_url)


def init_db(settings: Settings = DEFAULT_SETTINGS):
    """Create all tables if they don't already exist. Returns the engine."""
    engine = get_engine(settings)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(settings: Settings = DEFAULT_SETTINGS) -> sessionmaker[Session]:
    engine = init_db(settings)
    return sessionmaker(bind=engine)

"""Tests for the quiz-interaction relational schema."""

from __future__ import annotations

import uuid

from study_buddy.config import Settings
from study_buddy.db import QuestionInteraction, get_session_factory


def test_interaction_persists_all_fields(db_settings: Settings):
    session_factory = get_session_factory(db_settings)
    question_id = str(uuid.uuid4())

    with session_factory() as session:
        session.add(
            QuestionInteraction(
                question_id=question_id,
                topic_id="The Calvin Cycle",
                question_text="Where does the Calvin cycle take place?",
                user_answer="stroma of the chloroplast",
                is_correct=True,
                response_time_ms=4200,
                model_used="claude-sonnet-5",
                tool_call_valid=True,
            )
        )
        session.commit()

    # Fresh session against the same database: proves it was actually
    # written to disk, not just held in the first session's identity map.
    with session_factory() as session:
        row = session.get(QuestionInteraction, question_id)
        assert row is not None
        assert row.topic_id == "The Calvin Cycle"
        assert row.user_answer == "stroma of the chloroplast"
        assert row.is_correct is True
        assert row.response_time_ms == 4200
        assert row.model_used == "claude-sonnet-5"
        assert row.tool_call_valid is True
        assert row.timestamp is not None


def test_multiple_interactions_are_independently_queryable(db_settings: Settings):
    session_factory = get_session_factory(db_settings)

    with session_factory() as session:
        session.add(
            QuestionInteraction(
                question_id=str(uuid.uuid4()),
                topic_id="Light-Dependent Reactions",
                question_text="What gas is released?",
                user_answer="nitrogen",
                is_correct=False,
                response_time_ms=1500,
                model_used="claude-sonnet-5",
                tool_call_valid=False,
            )
        )
        session.add(
            QuestionInteraction(
                question_id=str(uuid.uuid4()),
                topic_id="Light-Dependent Reactions",
                question_text="What gas is released?",
                user_answer="oxygen",
                is_correct=True,
                response_time_ms=2100,
                model_used="claude-sonnet-5",
                tool_call_valid=True,
            )
        )
        session.commit()

    with session_factory() as session:
        rows = session.query(QuestionInteraction).filter_by(topic_id="Light-Dependent Reactions").all()
        assert len(rows) == 2
        assert {row.is_correct for row in rows} == {True, False}

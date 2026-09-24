"""Tests for TutorSession's five tools, dispatch/logging, and the
WeaknessEstimator seam. Uses db_settings (a real Postgres test branch) and
hand-written fakes for QuestionGenerator/WeaknessEstimator — no Groq calls.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy.exc import SQLAlchemyError

from study_buddy.chunking import Chunk
from study_buddy.config import Settings
from study_buddy.db import QuestionInteraction, ToolCallLog, get_session_factory
from study_buddy.question_generator import GeneratedQuestion
from study_buddy.tutor_session import SqlAggregateWeaknessEstimator, TutorSession
from study_buddy.vectorstore import store_chunks


class FakeQuestionGenerator:
    def __init__(self, question_text="What is 2+2?", correct_answer="4"):
        self.question_text = question_text
        self.correct_answer = correct_answer
        self.calls = []

    def generate(self, context: str, difficulty: str, topic: str) -> GeneratedQuestion:
        self.calls.append({"context": context, "difficulty": difficulty, "topic": topic})
        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            question_text=self.question_text,
            correct_answer=self.correct_answer,
            difficulty=difficulty,
            topic=topic,
        )


class FakeWeaknessEstimator:
    def __init__(self, result: dict):
        self.result = result
        self.calls = 0

    def weakest_topic(self) -> dict:
        self.calls += 1
        return self.result


def _make_session(db_settings: Settings, **kwargs) -> TutorSession:
    return TutorSession(
        initial_topic="Photosynthesis",
        settings=db_settings,
        question_generator=kwargs.get("question_generator", FakeQuestionGenerator()),
        weakness_estimator=kwargs.get("weakness_estimator"),
    )


# ---- retrieve_context ----


def test_retrieve_context_returns_chunks_via_query_chunks(db_settings: Settings):
    chunk = Chunk(
        chunk_id="c1",
        filename="bio.pdf",
        page_number=1,
        topic="The Calvin Cycle",
        text="The Calvin cycle fixes carbon dioxide into organic molecules.",
        extraction_method="pymupdf",
    )
    store_chunks([chunk], db_settings)

    session = _make_session(db_settings)
    result = session.retrieve_context(topic="carbon fixation")

    assert result["topic"] == "carbon fixation"
    assert any("Calvin cycle" in c["text"] for c in result["chunks"])
    assert session.current_topic == "carbon fixation"


# ---- generate_question ----


def test_generate_question_stores_pending_and_returns_full_payload_including_correct_answer(db_settings: Settings):
    generator = FakeQuestionGenerator(question_text="What is the Calvin cycle?", correct_answer="Carbon fixation")
    session = _make_session(db_settings, question_generator=generator)

    result = session.generate_question(context="...", difficulty="medium")

    assert result["question_text"] == "What is the Calvin cycle?"
    assert result["correct_answer"] == "Carbon fixation"
    assert result["difficulty"] == "medium"
    assert result["topic"] == "Photosynthesis"
    assert result["question_id"] in session._pending
    assert generator.calls == [{"context": "...", "difficulty": "medium", "topic": "Photosynthesis"}]


# ---- record_answer ----


def test_record_answer_writes_question_interaction_with_computed_fields(db_settings: Settings):
    session = _make_session(db_settings)
    generated = session.generate_question(context="ctx", difficulty="easy")
    question_id = generated["question_id"]

    result = session.record_answer(question_id=question_id, user_answer="4", is_correct=True)

    assert result == {"recorded": True, "question_id": question_id, "is_correct": True}
    assert question_id not in session._pending

    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        row = db_session.get(QuestionInteraction, question_id)
        assert row is not None
        assert row.topic_id == "Photosynthesis"
        assert row.question_text == "What is 2+2?"
        assert row.user_answer == "4"
        assert row.is_correct is True
        assert row.response_time_ms >= 0
        assert row.model_used == db_settings.groq_model
        assert row.tool_call_valid is True


def test_record_answer_unknown_question_id_errors_without_writing_a_row(db_settings: Settings):
    session = _make_session(db_settings)

    with pytest.raises(ValueError, match="Unknown question_id"):
        session.record_answer(question_id="does-not-exist", user_answer="x", is_correct=False)

    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        assert db_session.query(QuestionInteraction).count() == 0


def test_record_answer_falls_back_to_tool_call_valid_false_on_db_write_failure(db_settings: Settings, monkeypatch):
    session = _make_session(db_settings)
    generated = session.generate_question(context="ctx", difficulty="easy")
    question_id = generated["question_id"]

    real_session_factory = session.session_factory
    call_count = {"n": 0}

    def flaky_session_factory():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SQLAlchemyError("simulated transient DB error")
        return real_session_factory()

    monkeypatch.setattr(session, "session_factory", flaky_session_factory)

    result = session.record_answer(question_id=question_id, user_answer="4", is_correct=True)

    assert result["recorded"] is True
    assert question_id not in session._pending

    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        row = db_session.get(QuestionInteraction, question_id)
        assert row is not None
        assert row.tool_call_valid is False


def test_record_answer_leaves_pending_question_when_both_writes_fail_and_allows_retry(db_settings: Settings, monkeypatch):
    session = _make_session(db_settings)
    generated = session.generate_question(context="ctx", difficulty="easy")
    question_id = generated["question_id"]

    real_session_factory = session.session_factory
    should_fail = {"value": True}

    def sometimes_flaky_session_factory():
        if should_fail["value"]:
            raise SQLAlchemyError("simulated persistent DB error")
        return real_session_factory()

    monkeypatch.setattr(session, "session_factory", sometimes_flaky_session_factory)

    with pytest.raises(SQLAlchemyError):
        session.record_answer(question_id=question_id, user_answer="4", is_correct=True)

    # The question must still be pending — a model retry for the same
    # question_id must not see a fabricated "unknown question_id".
    assert question_id in session._pending

    should_fail["value"] = False
    result = session.record_answer(question_id=question_id, user_answer="4", is_correct=True)
    assert result["recorded"] is True
    assert question_id not in session._pending


# ---- get_weakest_topic ----


def test_get_weakest_topic_returns_lowest_accuracy_topic(db_settings: Settings):
    session = _make_session(db_settings)
    for topic, correct in [("A", True), ("A", True), ("B", False), ("B", True)]:
        generated = session.generate_question(context="ctx", difficulty="easy")
        session.current_topic = topic
        # generate_question already stored PendingQuestion with the topic at
        # call time; overwrite it here since we want fine control per-row.
        session._pending[generated["question_id"]].topic = topic
        session.record_answer(question_id=generated["question_id"], user_answer="x", is_correct=correct)

    result = session.get_weakest_topic()

    assert result["weakest_topic"] == "B"
    assert result["accuracy"] == 0.5


def test_get_weakest_topic_with_no_interactions_returns_none(db_settings: Settings):
    session = _make_session(db_settings)
    result = session.get_weakest_topic()
    assert result == {"weakest_topic": None, "message": "No recorded interactions yet."}


def test_get_weakest_topic_delegates_to_injected_weakness_estimator(db_settings: Settings):
    fake_estimator = FakeWeaknessEstimator({"weakest_topic": "Injected Topic", "accuracy": 0.1})
    session = _make_session(db_settings, weakness_estimator=fake_estimator)

    result = session.get_weakest_topic()

    assert result == {"weakest_topic": "Injected Topic", "accuracy": 0.1}
    assert fake_estimator.calls == 1


def test_sql_aggregate_weakness_estimator_matches_direct_query(db_settings: Settings):
    session = _make_session(db_settings)
    generated = session.generate_question(context="ctx", difficulty="easy")
    session.record_answer(question_id=generated["question_id"], user_answer="4", is_correct=True)

    estimator = SqlAggregateWeaknessEstimator(db_settings)
    result = estimator.weakest_topic()

    assert result["weakest_topic"] == "Photosynthesis"
    assert result["accuracy"] == 1.0


# ---- end_session ----


def test_end_session_returns_summary_and_sets_ended(db_settings: Settings):
    session = _make_session(db_settings)
    generated = session.generate_question(context="ctx", difficulty="easy")
    session.record_answer(question_id=generated["question_id"], user_answer="4", is_correct=True)

    summary = session.end_session()

    assert session.ended is True
    assert summary == {
        "session_id": session.session_id,
        "questions_answered": 1,
        "correct": 1,
        "accuracy": 1.0,
        "ungraded_questions": [],
    }


def test_end_session_with_no_answers_has_none_accuracy(db_settings: Settings):
    session = _make_session(db_settings)
    summary = session.end_session()
    assert summary["questions_answered"] == 0
    assert summary["accuracy"] is None
    assert summary["ungraded_questions"] == []


def test_end_session_reports_questions_generated_but_never_graded(db_settings: Settings):
    # Regression test for a real bug found via a live session's ToolCallLog:
    # generate_question was called and the question was shown to the learner,
    # but the model never followed up with record_answer for it. This is
    # observability only -- end_session() must surface the gap, not silently
    # drop it or force a grade on the model's behalf.
    session = _make_session(db_settings)
    graded = session.generate_question(context="ctx", difficulty="easy")
    ungraded = session.generate_question(context="ctx", difficulty="medium")
    session.record_answer(question_id=graded["question_id"], user_answer="4", is_correct=True)

    summary = session.end_session()

    assert summary["questions_answered"] == 1
    assert summary["ungraded_questions"] == [ungraded["question_id"]]


# ---- dispatch + logging ----


def test_dispatch_writes_a_tool_call_log_row_per_call(db_settings: Settings):
    session = _make_session(db_settings)
    arguments_json = json.dumps({"topic": "Photosynthesis"})

    session.dispatch("retrieve_context", arguments_json, raw_response="RAW")

    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        rows = db_session.query(ToolCallLog).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.session_id == session.session_id
        assert row.tool_name == "retrieve_context"
        assert json.loads(row.arguments) == {"topic": "Photosynthesis"}
        assert row.raw_response == "RAW"


def test_dispatch_unknown_tool_logs_and_returns_error(db_settings: Settings):
    session = _make_session(db_settings)

    result = session.dispatch("not_a_real_tool", "{}", raw_response="RAW")

    assert result == {"error": "Unknown tool 'not_a_real_tool'"}
    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        assert db_session.query(ToolCallLog).count() == 1


def test_dispatch_malformed_arguments_json_logs_and_returns_error(db_settings: Settings):
    session = _make_session(db_settings)

    result = session.dispatch("retrieve_context", "{not valid json", raw_response="RAW")

    assert "error" in result
    assert "Malformed arguments JSON" in result["error"]
    session_factory = get_session_factory(db_settings)
    with session_factory() as db_session:
        row = db_session.query(ToolCallLog).one()
        assert row.arguments == "{not valid json"


def test_dispatch_handler_exception_logs_and_returns_error_without_raising(db_settings: Settings):
    session = _make_session(db_settings)

    result = session.dispatch("record_answer", json.dumps({"question_id": "nope", "user_answer": "x", "is_correct": False}), raw_response="RAW")

    assert "error" in result
    assert "Unknown question_id" in result["error"]

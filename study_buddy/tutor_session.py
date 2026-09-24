"""Per-session state and the five tools the tutor loop's model can call.

TutorSession owns all persistence side effects (writing QuestionInteraction
and ToolCallLog rows) and the in-memory bookkeeping a tool-calling loop
needs between turns (which questions are still awaiting an answer). Which
tool to call, and when, is never decided here — that reasoning lives
entirely in tutor.py's SYSTEM_PROMPT and the model's own tool selection.
This module only executes whatever the model picked and logs it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.exc import SQLAlchemyError

from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.db import QuestionInteraction, ToolCallLog, get_session_factory
from study_buddy.question_generator import GroqQuestionGenerator, QuestionGenerator
from study_buddy.vectorstore import query_chunks

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": (
                "Semantically search the learner's ingested study material for content "
                "relevant to a topic. Call this before writing a question on a topic you "
                "haven't retrieved material for yet — questions must be grounded in what "
                "was actually taught, not your own general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or concept to search for, e.g. 'the Calvin cycle'. Natural language.",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_question",
            "description": (
                "Generate one quiz question grounded in the given context, at the requested "
                "difficulty. The result includes correct_answer for your own grading later — "
                "never reveal, paraphrase, or hint at correct_answer to the learner before "
                "they have answered and you have graded them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Retrieved study material to base the question on.",
                    },
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["context", "difficulty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_answer",
            "description": (
                "Record the learner's answer to a question you previously generated, after "
                "you have graded it yourself by comparing it to that question's correct_answer. "
                "Call this immediately after the learner responds, before deciding what to do next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_id": {
                        "type": "string",
                        "description": "The question_id returned by generate_question.",
                    },
                    "user_answer": {"type": "string", "description": "The learner's answer, verbatim."},
                    "is_correct": {
                        "type": "boolean",
                        "description": "Your grading judgment: true if the learner's answer is substantively/conceptually correct -- judge by semantic equivalence to correct_answer, not exact wording. Accept valid synonyms, alternate names, or equivalent phrasings as correct even if they differ from correct_answer's exact terms.",
                    },
                },
                "required": ["question_id", "user_answer", "is_correct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weakest_topic",
            "description": (
                "Look up the learner's all-time accuracy per topic and return the one with "
                "the lowest accuracy. Call this periodically (e.g. after recording an answer) "
                "to decide whether to remediate a weak topic or move forward."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_session",
            "description": (
                "End the tutoring session. Call this when the learner asks to stop, or when "
                "you judge the session has reached a natural stopping point. Returns a summary."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@dataclass
class PendingQuestion:
    question_text: str
    topic: str
    asked_at: datetime


class WeaknessEstimator(Protocol):
    """Seam for get_weakest_topic()'s internals — mirrors QuestionGenerator.

    SqlAggregateWeaknessEstimator below is a deliberate placeholder for the
    future Layer 4 knowledge-tracing model (Phase 3, not built here). Phase
    3 can replace the concrete estimator wholesale without changing
    TOOL_SCHEMAS, TutorSession, or tutor.py's loop at all.
    """

    def weakest_topic(self) -> dict: ...


class SqlAggregateWeaknessEstimator:
    """Phase 1/2 implementation: a plain SQL GROUP BY over all-time
    question_interactions, grouped by topic_id."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.session_factory = get_session_factory(settings)

    def weakest_topic(self) -> dict:
        # Postgres has no sum(boolean) — cast to Integer first.
        stmt = select(
            QuestionInteraction.topic_id,
            func.count().label("total"),
            func.sum(cast(QuestionInteraction.is_correct, Integer)).label("correct"),
        ).group_by(QuestionInteraction.topic_id)

        with self.session_factory() as db_session:
            rows = db_session.execute(stmt).all()

        if not rows:
            return {"weakest_topic": None, "message": "No recorded interactions yet."}

        stats = [
            {"topic_id": row.topic_id, "total": row.total, "correct": row.correct, "accuracy": row.correct / row.total}
            for row in rows
        ]
        weakest = min(stats, key=lambda s: s["accuracy"])
        return {
            "weakest_topic": weakest["topic_id"],
            "accuracy": weakest["accuracy"],
            "total_attempts": weakest["total"],
            "all_topics": stats,
        }


class TutorSession:
    def __init__(
        self,
        initial_topic: str,
        settings: Settings = DEFAULT_SETTINGS,
        question_generator: QuestionGenerator | None = None,
        weakness_estimator: WeaknessEstimator | None = None,
    ):
        self.settings = settings
        self.session_id = str(uuid.uuid4())
        self.session_factory = get_session_factory(settings)
        self.question_generator = question_generator or GroqQuestionGenerator(settings)
        self.weakness_estimator = weakness_estimator or SqlAggregateWeaknessEstimator(settings)
        # Seeded from the CLI's <topic> argument, updated by retrieve_context —
        # generate_question/record_answer have no topic argument of their own
        # (per the specified tool signatures), so this is the "topic currently
        # in focus" used to tag QuestionInteraction rows.
        self.current_topic = initial_topic
        self._pending: dict[str, PendingQuestion] = {}
        self._answers: list[bool] = []
        self.ended = False
        self.summary: dict | None = None

    # ---- tool implementations (kwarg names must match TOOL_SCHEMAS exactly) ----

    def retrieve_context(self, topic: str) -> dict:
        self.current_topic = topic
        results = query_chunks(topic, n_results=5, settings=self.settings)
        return {"topic": topic, "chunks": results}

    def generate_question(self, context: str, difficulty: str) -> dict:
        generated = self.question_generator.generate(context=context, difficulty=difficulty, topic=self.current_topic)
        self._pending[generated.question_id] = PendingQuestion(
            question_text=generated.question_text,
            topic=generated.topic,
            asked_at=datetime.now(timezone.utc),
        )
        return {
            "question_id": generated.question_id,
            "question_text": generated.question_text,
            "correct_answer": generated.correct_answer,
            "difficulty": generated.difficulty,
            "topic": generated.topic,
        }

    def record_answer(self, question_id: str, user_answer: str, is_correct: bool) -> dict:
        # .get(), not .pop() — the entry must survive until a write actually
        # succeeds. See the del() at the bottom of this method.
        pending = self._pending.get(question_id)
        if pending is None:
            raise ValueError(f"Unknown question_id {question_id!r} — no such question was generated this session.")
        response_time_ms = int((datetime.now(timezone.utc) - pending.asked_at).total_seconds() * 1000)

        def _write(tool_call_valid: bool) -> None:
            with self.session_factory() as db_session:
                db_session.add(
                    QuestionInteraction(
                        question_id=question_id,
                        topic_id=pending.topic,
                        question_text=pending.question_text,
                        user_answer=user_answer,
                        is_correct=is_correct,
                        response_time_ms=response_time_ms,
                        model_used=self.settings.groq_model,
                        tool_call_valid=tool_call_valid,
                    )
                )
                db_session.commit()

        try:
            _write(tool_call_valid=True)
        except SQLAlchemyError:
            # The primary write failed (transient DB error, dropped
            # connection, etc). Rather than silently losing the interaction,
            # retry once with the same answer/grading but
            # tool_call_valid=False, so a row still exists for analysis but
            # is flagged as not fully trustworthy. This is THE place
            # tool_call_valid=False gets decided — dispatch() has no
            # special-case knowledge of this retry. If this second write
            # also fails, it's left to propagate uncaught: execution never
            # reaches the del() below, so question_id stays in self._pending
            # and a model retry of record_answer for the same question_id
            # gets a genuine second attempt instead of a fabricated "unknown
            # question_id".
            _write(tool_call_valid=False)

        del self._pending[question_id]
        self._answers.append(is_correct)
        return {"recorded": True, "question_id": question_id, "is_correct": is_correct}

    def get_weakest_topic(self) -> dict:
        return self.weakness_estimator.weakest_topic()

    def end_session(self) -> dict:
        total, correct = len(self._answers), sum(self._answers)
        self.summary = {
            "session_id": self.session_id,
            "questions_answered": total,
            "correct": correct,
            "accuracy": (correct / total) if total else None,
            # Passive observability only -- never enforced or auto-graded.
            # Whatever is still sitting in self._pending at session end is a
            # question that was generated (and, per the .get()/del() logic
            # in record_answer above, never had a record_answer call commit
            # for it -- see that method's docstring-equivalent comment).
            # This measures how often the model skips grading despite
            # SYSTEM_PROMPT's explicit "grade every question" instruction;
            # it does not change control flow or grade anything itself.
            "ungraded_questions": list(self._pending.keys()),
        }
        self.ended = True
        return self.summary

    # ---- dispatch + logging ----

    def dispatch(self, tool_name: str, arguments_json: str, raw_response: str) -> dict:
        """Never raises, for any failure mode: malformed argument JSON, an
        unknown tool name, and an exception from the handler itself all
        funnel into the same {"error": ...} result path. This is what lets
        tutor.py's loop run with no try/except around dispatch() at all —
        a failed tool call is surfaced to the model as its result content,
        never aborts the session.
        """
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            result = {"error": f"Malformed arguments JSON: {exc}"}
        else:
            handler = _TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result = {"error": f"Unknown tool {tool_name!r}"}
            else:
                try:
                    result = handler(self, **arguments)
                except Exception as exc:
                    result = {"error": str(exc)}

        with self.session_factory() as db_session:
            db_session.add(
                ToolCallLog(
                    id=str(uuid.uuid4()),
                    session_id=self.session_id,
                    tool_name=tool_name,
                    arguments=arguments_json,
                    raw_response=raw_response,
                )
            )
            db_session.commit()
        return result


_TOOL_HANDLERS: dict[str, Callable[..., dict]] = {
    "retrieve_context": TutorSession.retrieve_context,
    "generate_question": TutorSession.generate_question,
    "record_answer": TutorSession.record_answer,
    "get_weakest_topic": TutorSession.get_weakest_topic,
    "end_session": TutorSession.end_session,
}

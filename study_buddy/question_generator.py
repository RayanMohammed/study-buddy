"""Provider-agnostic question generation for the tutor loop.

QuestionGenerator is a Protocol so the actual executor behind
generate_question can be swapped later (e.g. a local fine-tuned model via
Ollama) without changing TutorSession's dispatch logic or the orchestrator
in tutor.py at all — GroqQuestionGenerator is just today's implementation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.groq_client import call_groq, get_client


@dataclass
class GeneratedQuestion:
    question_id: str
    question_text: str
    correct_answer: str  # for the model's own later grading — never shown to the learner up front
    difficulty: str
    topic: str


class QuestionGenerator(Protocol):
    def generate(self, context: str, difficulty: str, topic: str) -> GeneratedQuestion: ...


_QUESTION_PROMPT = """Write one {difficulty} quiz question about the study material below.

Respond with ONLY a JSON object with exactly two keys:
- "question": the question text, containing no hint of the answer
- "answer": a concise, unambiguous correct answer

Study material:
\"\"\"
{context}
\"\"\"
"""


class GroqQuestionGenerator:
    """Generates one question per call via a separate, non-tool-calling
    Groq chat-completion — this is not a tool call itself, it's the model
    that generate_question's tool handler calls out to."""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS, client: OpenAI | None = None):
        self.settings = settings
        self.client = client or get_client(settings)

    def generate(self, context: str, difficulty: str, topic: str) -> GeneratedQuestion:
        response = call_groq(
            self.client,
            messages=[
                {"role": "user", "content": _QUESTION_PROMPT.format(difficulty=difficulty, context=context)}
            ],
            settings=self.settings,
            response_format={"type": "json_object"},
        )
        try:
            payload = json.loads(response.choices[0].message.content)
            question_text, correct_answer = payload["question"], payload["answer"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Groq did not return a valid question JSON payload: {exc!r}") from exc

        return GeneratedQuestion(
            question_id=str(uuid.uuid4()),
            question_text=question_text,
            correct_answer=correct_answer,
            difficulty=difficulty,
            topic=topic,
        )

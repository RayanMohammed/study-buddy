"""Tests for question_generator.py's Groq-backed implementation.

Mocked client — no network calls, no GROQ_API_KEY needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from study_buddy.config import Settings
from study_buddy.question_generator import GeneratedQuestion, GroqQuestionGenerator


def _fake_client(content: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


def test_generate_parses_json_payload_into_generated_question():
    payload = json.dumps({"question": "What is the Calvin cycle?", "answer": "Carbon fixation in the stroma."})
    client = _fake_client(payload)
    settings = Settings(database_url="sqlite:///unused.db")
    generator = GroqQuestionGenerator(settings=settings, client=client)

    result = generator.generate(context="Photosynthesis notes...", difficulty="medium", topic="Photosynthesis")

    assert isinstance(result, GeneratedQuestion)
    assert result.question_text == "What is the Calvin cycle?"
    assert result.correct_answer == "Carbon fixation in the stroma."
    assert result.difficulty == "medium"
    assert result.topic == "Photosynthesis"
    assert result.question_id  # non-empty uuid string


def test_generate_raises_value_error_on_malformed_json():
    client = _fake_client("not json at all")
    settings = Settings(database_url="sqlite:///unused.db")
    generator = GroqQuestionGenerator(settings=settings, client=client)

    with pytest.raises(ValueError, match="did not return a valid question JSON payload"):
        generator.generate(context="...", difficulty="easy", topic="X")


def test_generate_raises_value_error_on_missing_keys():
    client = _fake_client(json.dumps({"question": "only a question, no answer key"}))
    settings = Settings(database_url="sqlite:///unused.db")
    generator = GroqQuestionGenerator(settings=settings, client=client)

    with pytest.raises(ValueError, match="did not return a valid question JSON payload"):
        generator.generate(context="...", difficulty="easy", topic="X")

"""Tests for tutor.py's run_tutor_session loop mechanics.

Fully mocked: a fake OpenAI-shaped client, a fake question_generator, and
fake input_prompt/output callables — no network calls, no GROQ_API_KEY,
and no real database writes are exercised here (only loop control flow;
TutorSession's own persistence is covered by test_tutor_session.py).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from study_buddy.config import Settings
from study_buddy.tutor import run_tutor_session


def _tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def _assistant_message(*, content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        **({"tool_calls": [{"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}] for tc in (tool_calls or [])} if tool_calls else {}),
    }
    return message


def _response_with(message: MagicMock) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


class FakeSession:
    """Stands in for tutor_session.TutorSession — the loop only calls
    .dispatch(...) and reads .ended/.summary, so that's all this needs."""

    def __init__(self, *args, **kwargs):
        self.ended = False
        self.summary = None
        self.dispatch_calls = []

    def dispatch(self, tool_name, arguments_json, raw_response):
        self.dispatch_calls.append((tool_name, arguments_json))
        if tool_name == "end_session":
            return self.end_session()
        return {"ok": True}

    def end_session(self):
        self.ended = True
        self.summary = {"session_id": "s1", "questions_answered": 1, "correct": 1, "accuracy": 1.0}
        return self.summary


def _settings() -> Settings:
    return Settings(database_url="sqlite:///unused.db")


def test_loop_dispatches_tool_calls_and_feeds_results_back_as_tool_messages(monkeypatch):
    monkeypatch.setattr("study_buddy.tutor.TutorSession", FakeSession)
    client = MagicMock()
    tool_call = _tool_call("call_1", "retrieve_context", {"topic": "X"})
    client.chat.completions.create.side_effect = [
        _response_with(_assistant_message(tool_calls=[tool_call])),
        _response_with(_assistant_message(content="Here is a question.")),
    ]
    outputs = []

    run_tutor_session(
        "X",
        settings=_settings(),
        client=client,
        output=outputs.append,
        input_prompt=lambda: "the learner's answer",
        max_turns=2,
    )

    # Second call's messages must include a tool-role message with the dispatched result.
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"]) == {"ok": True}
    # max_turns=2 is intentionally tight here (this test only cares about
    # tool-call dispatch/feedback) so the safety cap fires right after the
    # plain-text turn prints — that's expected, not the behavior under test.
    assert "Here is a question." in outputs


def test_loop_prints_plain_text_and_prompts_human_then_appends_reply(monkeypatch):
    monkeypatch.setattr("study_buddy.tutor.TutorSession", FakeSession)
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response_with(_assistant_message(content="What is 2+2?")),
        _response_with(_assistant_message(tool_calls=[_tool_call("call_2", "end_session", {})])),
        _response_with(_assistant_message(content="Great job, goodbye!")),
    ]
    outputs = []

    run_tutor_session(
        "Math",
        settings=_settings(),
        client=client,
        output=outputs.append,
        input_prompt=lambda: "4",
        max_turns=5,
    )

    third_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    user_messages = [m for m in third_call_messages if m.get("role") == "user"]
    assert any(m["content"] == "4" for m in user_messages)
    # Filter out tutor.py's per-call/per-turn timing diagnostics -- this
    # test only cares about what the learner actually sees, not the latency
    # instrumentation added alongside it.
    learner_facing = [o for o in outputs if not o.startswith("  [groq call") and not o.startswith("[turn total")]
    assert learner_facing == ["What is 2+2?", "Great job, goodbye!"]


def test_loop_breaks_without_prompting_once_end_session_was_called(monkeypatch):
    monkeypatch.setattr("study_buddy.tutor.TutorSession", FakeSession)
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response_with(_assistant_message(tool_calls=[_tool_call("call_3", "end_session", {})])),
        _response_with(_assistant_message(content="Session summary remark.")),
    ]

    def _fail_prompt():
        raise AssertionError("input_prompt must not be called after end_session")

    summary = run_tutor_session(
        "Topic",
        settings=_settings(),
        client=client,
        output=lambda _: None,
        input_prompt=_fail_prompt,
        max_turns=5,
    )

    assert summary == {"session_id": "s1", "questions_answered": 1, "correct": 1, "accuracy": 1.0}
    assert client.chat.completions.create.call_count == 2


def test_loop_force_ends_at_max_turns_when_model_never_calls_end_session(monkeypatch):
    monkeypatch.setattr("study_buddy.tutor.TutorSession", FakeSession)
    client = MagicMock()
    # Always returns plain text, never a tool call, never ends — should hit the safety cap.
    client.chat.completions.create.return_value = _response_with(_assistant_message(content="Another question?"))
    outputs = []

    summary = run_tutor_session(
        "Topic",
        settings=_settings(),
        client=client,
        output=outputs.append,
        input_prompt=lambda: "an answer",
        max_turns=3,
    )

    assert client.chat.completions.create.call_count == 3
    assert any("safety cap reached" in o for o in outputs)
    assert summary is not None

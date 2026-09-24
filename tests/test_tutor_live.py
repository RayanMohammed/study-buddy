"""One real end-to-end call against Groq's actual API.

Gated on GROQ_API_KEY (via the groq_settings fixture, itself built on the
db_settings Neon test branch) — skipped, not failed, when unset. Purpose:
catch real wire-format drift (e.g. Groq changing how it serializes
tool_calls, or dropping response_format support), not to assert specific
tutoring behavior, since model output isn't deterministic.
"""

from __future__ import annotations

import json

from study_buddy.config import Settings
from study_buddy.groq_client import call_groq, get_client
from study_buddy.tutor_session import TOOL_SCHEMAS
from study_buddy.tutor import SYSTEM_PROMPT

_KNOWN_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


def test_live_groq_call_returns_valid_tool_call_shape(groq_settings: Settings):
    client = get_client(groq_settings)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "I want to study the Calvin cycle. Please retrieve context on "
                "the Calvin cycle before asking me anything."
            ),
        },
    ]

    response = call_groq(client, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", settings=groq_settings)
    message = response.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            assert tool_call.function.name in _KNOWN_TOOL_NAMES
            # Must be parseable JSON — this is exactly what tutor_session.dispatch() relies on.
            json.loads(tool_call.function.arguments or "{}")
    else:
        # Some models occasionally respond in plain text instead of calling
        # a tool for an ambiguous first turn — acceptable, just confirm the
        # response has *some* content rather than being empty/malformed.
        assert message.content

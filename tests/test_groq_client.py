"""Tests for groq_client.py: client construction and the 429 retry policy.

All mocked — no network calls, no GROQ_API_KEY needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from openai import RateLimitError

from study_buddy.config import Settings
from study_buddy.groq_client import _wait_for_retry_after, call_groq, get_client


def _rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, request=request, headers=headers)
    return RateLimitError("rate limited", response=response, body=None)


def test_get_client_raises_clear_runtime_error_when_api_key_missing():
    settings = Settings(groq_api_key=None, database_url="sqlite:///unused.db")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_client(settings)


def test_call_groq_retries_on_rate_limit_then_succeeds():
    real_response = MagicMock(name="real_response")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _rate_limit_error(retry_after="0.01"),
        _rate_limit_error(retry_after="0.01"),
        real_response,
    ]

    result = call_groq(client, messages=[{"role": "user", "content": "hi"}], settings=Settings(database_url="sqlite:///unused.db"))

    assert result is real_response
    assert client.chat.completions.create.call_count == 3


def test_call_groq_reraises_after_exhausting_retries():
    client = MagicMock()
    client.chat.completions.create.side_effect = _rate_limit_error(retry_after="0.01")

    with pytest.raises(RateLimitError):
        call_groq(client, messages=[{"role": "user", "content": "hi"}], settings=Settings(database_url="sqlite:///unused.db"))

    assert client.chat.completions.create.call_count == 5  # stop_after_attempt(5)


def test_wait_for_retry_after_uses_header_when_present():
    retry_state = MagicMock()
    retry_state.outcome.exception.return_value = _rate_limit_error(retry_after="2.5")

    assert _wait_for_retry_after(retry_state) == 2.5


def test_wait_for_retry_after_falls_back_to_backoff_when_header_absent():
    retry_state = MagicMock()
    retry_state.attempt_number = 1  # wait_exponential_jitter does real arithmetic on this
    retry_state.outcome.exception.return_value = _rate_limit_error(retry_after=None)

    wait_seconds = _wait_for_retry_after(retry_state)

    assert isinstance(wait_seconds, float)
    assert wait_seconds != 2.5  # sanity: didn't accidentally read a stale header value

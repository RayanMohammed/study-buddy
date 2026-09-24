"""Shared low-level access to Groq's OpenAI-compatible chat-completions API.

Groq is reached via the `openai` package pointed at Groq's base_url — not
the `groq` SDK — using the OpenAI tools/tool_choice wire format throughout
(a different shape than Anthropic's tool_use/input_schema format).

Every Groq call in the app must go through call_groq() so the 429
retry/backoff policy (Groq's requests-per-minute cap is tighter than its
tokens-per-minute cap, and the tutor loop makes several calls per turn)
lives in exactly one place.
"""

from __future__ import annotations

from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from study_buddy.config import DEFAULT_SETTINGS, Settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_fallback_wait = wait_exponential_jitter(initial=1, max=30)


def get_client(settings: Settings = DEFAULT_SETTINGS) -> OpenAI:
    """Build the Groq-pointed OpenAI-compatible client.

    Raises here — not at Settings-construction time — so every other CLI
    command (ingest/query/log-interaction) keeps working for users who
    haven't set GROQ_API_KEY yet.

    max_retries=0 disables the openai client's own built-in retry/backoff
    so call_groq's tenacity policy below is the single source of retry
    behavior — otherwise the two retry layers would compound in confusing
    ways (e.g. tenacity retrying an already-internally-retried call).
    """
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file at the "
            "project root (or export it) to use `study-buddy tutor`."
        )
    return OpenAI(api_key=settings.groq_api_key, base_url=GROQ_BASE_URL, max_retries=0)


def _wait_for_retry_after(retry_state):
    """Respect Groq's Retry-After header on a 429 if present; otherwise
    fall back to exponential backoff with jitter."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitError) and exc.response is not None:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return _fallback_wait(retry_state)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=_wait_for_retry_after,
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_groq(client: OpenAI, *, messages: list[dict], settings: Settings = DEFAULT_SETTINGS, **kwargs):
    """The one place every Groq chat-completion call goes through.

    kwargs is passed straight through to chat.completions.create — tools,
    tool_choice, response_format, etc — so this same function serves both
    the tool-calling orchestrator loop and question_generator's plain
    (non-tool-calling) question-generation calls.
    """
    return client.chat.completions.create(model=settings.groq_model, messages=messages, **kwargs)

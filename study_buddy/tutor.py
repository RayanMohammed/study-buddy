"""The system prompt and the agentic tool-calling loop.

This is the only file in the tutor feature with control flow, and that
control flow is deliberately limited to: loop mechanics (call the model,
dispatch whatever it picked, feed results back, repeat), a hard max_turns
safety cap (a runaway-cost circuit breaker, not a pedagogical decision),
and reacting to end_session(). Which tool to call next — including whether
to remediate a weak topic or progress — is never decided here in Python;
that reasoning lives entirely in SYSTEM_PROMPT and the model's own tool
selection, informed by tool results fed back into the conversation.
"""

from __future__ import annotations

import json
import time

import click

from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.groq_client import call_groq, get_client
from study_buddy.tutor_session import TOOL_SCHEMAS, TutorSession

SYSTEM_PROMPT = """You are an adaptive AI tutor running a one-on-one study session over previously ingested course material.

You have five tools:

- retrieve_context(topic): semantic search over the learner's ingested slides/notes. Call this whenever you need material to write a question about a topic — your own knowledge is NOT a substitute for the retrieved material, since questions must be grounded in what the learner was actually taught.
- generate_question(context, difficulty): writes one question from context you retrieved, at difficulty "easy", "medium", or "hard". The result includes a correct_answer field for YOUR use only — you must never reveal, paraphrase, or hint at correct_answer to the learner before they have submitted an answer and you have graded it. Present only the question text to the learner.
- record_answer(question_id, user_answer, is_correct): after the learner answers a question you presented, grade their answer yourself by comparing it to the correct_answer you were given, then call this tool with your verdict. Do this immediately, before deciding what to do next. Grade every single question you present to the learner — there must be no question left ungraded, even one that feels like a quick aside or a check-in rather than the "main" quiz item. Judge is_correct by conceptual/semantic equivalence to correct_answer, not exact wording — accept valid synonyms, alternate names, or equivalent phrasings as correct (e.g. "absolute loss" and "mean absolute error" name the same concept; do not mark an answer wrong just because it used different, still-correct terminology than correct_answer's exact phrasing).
- get_weakest_topic(): looks up the learner's all-time accuracy across every topic they've ever been quizzed on and returns the one with the lowest accuracy. Call this periodically — e.g. after recording an answer — to inform whether you should remediate a weak topic or move forward.
- end_session(): ends the session and returns a summary. Call this when the learner says they want to stop, or when you judge the session has reached a good stopping point.

How to run the session:
1. Retrieve context on the learner's requested topic and ask an easy-to-medium question to gauge their level.
2. Every question you present to the learner must be graded — call record_answer for each one, with no exceptions, even for a question that felt more like a quick check than the core quiz item. If a question you're about to ask isn't something you intend to grade, don't present it as an answerable quiz question in the first place. After every answer, call record_answer to grade and log it. If you already know you'll also want get_weakest_topic's result to decide what's next, call both tools in the same response rather than waiting for record_answer's result first — you don't need one's output to call the other. Never call get_weakest_topic twice in a row without a new record_answer in between — nothing about the learner's history changes between two back-to-back calls, so a repeat call wastes a turn for no new information. Use get_weakest_topic's result plus the conversation so far to decide, in your own judgment, what to do next: another question on the same topic, a different difficulty, remediating a weaker topic, or wrapping up. There is no fixed script — decide based on the evidence in front of you.
3. If accuracy on a topic is consistently high, raise the difficulty or move to new material. If it's low, stay on that topic at an easier difficulty, or explicitly pivot to whichever topic get_weakest_topic reports as weakest.
4. Keep messages to the learner concise: one question at a time, plus brief feedback (correct/incorrect and why) after grading, before moving on.
5. Never reveal a question's correct_answer before the learner has answered and you have graded it.
6. End the session gracefully with end_session when the learner wants to stop or you judge enough ground has been covered — summarize what was covered and how they did.

Always call a tool when a tool applies; only send plain text to ask the learner a question, give feedback, or make a closing remark. When you already know you need more than one tool before your next message to the learner, call them together in the same response instead of spreading them across separate turns — each turn has real latency and API cost, so avoid unnecessary round trips."""


def run_tutor_session(
    topic: str,
    settings: Settings = DEFAULT_SETTINGS,
    max_turns: int = 20,
    client=None,
    question_generator=None,
    output=click.echo,
    input_prompt=lambda: click.prompt("You", prompt_suffix="> "),
) -> dict:
    """Runs the interactive tool-calling loop until the model calls
    end_session() or max_turns (a pure cost breaker, not an orchestrator
    decision) is reached. Returns the session summary dict.

    client and question_generator are injectable (default to the real Groq
    client / GroqQuestionGenerator) so tests can drive this whole loop
    against a fake client with zero network calls or API key.
    """
    client = client or get_client(settings)
    session = TutorSession(initial_topic=topic, settings=settings, question_generator=question_generator)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"I want to study {topic}. Let's begin."},
    ]

    # turn_elapsed/turn_call_count track wall-clock time and Groq call count
    # since the last thing the learner actually saw — a single learner-visible
    # "turn" (answer submitted -> next question shown) can silently cost
    # several chained Groq calls (record_answer, get_weakest_topic, etc.),
    # each with its own reasoning-model latency. This is diagnostic output
    # only, not control flow: it never affects what the loop does.
    turn_elapsed = 0.0
    turn_call_count = 0

    for _ in range(max_turns):
        call_start = time.perf_counter()
        response = call_groq(
            client,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            parallel_tool_calls=True,
            settings=settings,
        )
        call_elapsed = time.perf_counter() - call_start
        turn_elapsed += call_elapsed
        turn_call_count += 1

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if message.tool_calls:
            tool_names = ", ".join(tool_call.function.name for tool_call in message.tool_calls)
            output(f"  [groq call: {call_elapsed:.1f}s -> tool_call(s): {tool_names}]")
            raw_response = json.dumps(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                result = session.dispatch(tool_call.function.name, tool_call.function.arguments, raw_response=raw_response)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
            # No try/except around dispatch() above — it never raises (see
            # tutor_session.py), so a failed tool call is surfaced to the
            # model as its result content on the next turn, never aborts
            # the session. Deliberate: only end_session() or the max_turns
            # cap below can end a session.
            continue  # let the model react to the tool result(s) next turn

        output(f"  [groq call: {call_elapsed:.1f}s -> text]")
        output(f"[turn total: {turn_elapsed:.1f}s across {turn_call_count} Groq call(s)]")
        turn_elapsed = 0.0
        turn_call_count = 0

        output(message.content or "")
        if session.ended:
            break
        messages.append({"role": "user", "content": input_prompt()})
    else:
        output(f"[safety cap reached after {max_turns} turns — ending session]")
        if not session.ended:
            session.end_session()

    return session.summary or {
        "session_id": session.session_id,
        "questions_answered": 0,
        "correct": 0,
        "accuracy": None,
        "ungraded_questions": [],
    }

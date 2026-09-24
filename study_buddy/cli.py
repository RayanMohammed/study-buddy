"""Command-line interface for the study-buddy ingestion pipeline.

    study-buddy ingest ./my_slides/          # extract + chunk + embed + store
    study-buddy query "eigenvalues"          # sanity-check retrieval
    study-buddy log-interaction ...          # record a quiz answer
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import click

from study_buddy.config import DEFAULT_SETTINGS
from study_buddy.db import QuestionInteraction, get_session_factory
from study_buddy.ingest import ingest_folder, ingest_pdf
from study_buddy.tutor import run_tutor_session
from study_buddy.vectorstore import query_chunks


@click.group()
def main():
    """study-buddy: ingest PDFs into a searchable, topic-tagged knowledge base."""


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def ingest(path: Path):
    """Ingest a single PDF file or every PDF in a folder."""
    if path.is_dir():
        results = ingest_folder(path, DEFAULT_SETTINGS)
        if not results:
            click.echo(f"No PDFs found in {path}")
            return
    else:
        results = [ingest_pdf(path, DEFAULT_SETTINGS)]

    total_chunks = 0
    for result in results:
        click.echo(
            f"{result.filename}: {result.page_count} page(s) -> "
            f"{result.chunk_count} chunk(s)"
        )
        total_chunks += result.chunk_count
    click.echo(f"Done. {len(results)} file(s), {total_chunks} chunk(s) stored.")


@main.command()
@click.argument("text")
@click.option("-n", "--n-results", default=5, show_default=True)
def query(text: str, n_results: int):
    """Embed TEXT and print the most similar stored chunks (for sanity-checking retrieval)."""
    results = query_chunks(text, n_results=n_results, settings=DEFAULT_SETTINGS)

    if not results:
        click.echo("No results. Have you run `study-buddy ingest` yet?")
        return

    for result in results:
        click.echo(
            f"--- {result['filename']} p.{result['page_number']} "
            f"[{result['topic']}] (distance={result['distance']:.4f})"
        )
        preview = result["text"][:200] + ("..." if len(result["text"]) > 200 else "")
        click.echo(preview)
        click.echo()


@main.command("log-interaction")
@click.option("--topic-id", required=True)
@click.option("--question-text", required=True)
@click.option("--user-answer", required=True)
@click.option("--is-correct", type=bool, required=True)
@click.option("--response-time-ms", type=int, required=True)
@click.option("--model-used", required=True)
@click.option("--tool-call-valid", type=bool, required=True)
def log_interaction(
    topic_id: str,
    question_text: str,
    user_answer: str,
    is_correct: bool,
    response_time_ms: int,
    model_used: str,
    tool_call_valid: bool,
):
    """Record a single quiz question/answer interaction."""
    session_factory = get_session_factory(DEFAULT_SETTINGS)
    with session_factory() as session:
        interaction = QuestionInteraction(
            question_id=str(uuid.uuid4()),
            topic_id=topic_id,
            question_text=question_text,
            user_answer=user_answer,
            is_correct=is_correct,
            response_time_ms=response_time_ms,
            model_used=model_used,
            tool_call_valid=tool_call_valid,
        )
        session.add(interaction)
        session.commit()
        click.echo(f"Logged interaction {interaction.question_id}")


@main.command()
@click.argument("topic")
@click.option(
    "--max-turns",
    default=20,
    show_default=True,
    help="Safety cap on total model turns before the session force-ends.",
)
def tutor(topic: str, max_turns: int):
    """Start an interactive AI tutoring session on TOPIC."""
    summary = run_tutor_session(topic, settings=DEFAULT_SETTINGS, max_turns=max_turns)
    click.echo(f"\nSession summary: {summary}")


if __name__ == "__main__":
    sys.exit(main())

"""Step 3: split extracted pages into semantically meaningful chunks.

Rather than cutting text every N characters (which can slice a formula or a
bullet list in half), we chunk on structural boundaries:

1. Every PDF page is a boundary by itself — this matches how slide decks
   work (one slide = one idea) and gives a hard upper bound on chunk size
   for dense documents too.
2. Within a page, we further split on detected section headings: a line
   is treated as a heading when its font is noticeably larger than the
   page's body text and it's short enough to plausibly be a title rather
   than a wrapped sentence.

Each chunk carries the heading that introduced it as its "topic", so a
retrieved chunk can always be traced back to what section it came from.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from study_buddy.config import DEFAULT_SETTINGS, Settings
from study_buddy.extraction import PageContent


@dataclass
class Chunk:
    """One semantically coherent unit of text, ready to embed."""

    chunk_id: str
    filename: str
    page_number: int
    topic: str
    text: str
    extraction_method: str

    def to_metadata(self) -> dict:
        """Metadata dict in the flat, scalar-only form ChromaDB requires."""
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "topic": self.topic,
            "extraction_method": self.extraction_method,
        }


def _make_chunk_id(filename: str, page_number: int, index: int) -> str:
    digest = hashlib.sha1(f"{filename}|{page_number}|{index}".encode()).hexdigest()[:12]
    return f"{filename}-p{page_number}-c{index}-{digest}"


def _first_line_topic(text: str, settings: Settings, page_number: int) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= settings.max_heading_chars:
            return stripped
    return f"Page {page_number}"


def _dominant_font_size(span_font_sizes: list[tuple[str, float]]) -> float:
    sizes = [round(size, 1) for _, size in span_font_sizes if size > 0]
    if not sizes:
        return 0.0
    return Counter(sizes).most_common(1)[0][0]


def _is_heading(line_text: str, size: float, body_size: float, settings: Settings) -> bool:
    if not line_text or len(line_text) > settings.max_heading_chars:
        return False
    if body_size <= 0:
        return False
    return size >= body_size * settings.heading_font_ratio


def chunk_page(page: PageContent, settings: Settings = DEFAULT_SETTINGS) -> list[Chunk]:
    """Split one extracted page into topic-bounded chunks."""
    # Marker-extracted (or otherwise span-less) pages have no per-line font
    # info to detect headings from, so the whole page becomes one chunk.
    if not page.span_font_sizes:
        text = page.text.strip()
        if not text:
            return []
        topic = _first_line_topic(text, settings, page.page_number)
        return [
            Chunk(
                chunk_id=_make_chunk_id(page.filename, page.page_number, 0),
                filename=page.filename,
                page_number=page.page_number,
                topic=topic,
                text=text,
                extraction_method=page.extraction_method,
            )
        ]

    body_size = _dominant_font_size(page.span_font_sizes)

    sections: list[tuple[str, list[str]]] = []  # (topic, lines)
    current_topic: str | None = None
    current_lines: list[str] = []

    for line_text, size in page.span_font_sizes:
        if _is_heading(line_text, size, body_size, settings):
            if current_lines:
                sections.append((current_topic or f"Page {page.page_number}", current_lines))
            current_topic = line_text
            current_lines = []
        else:
            current_lines.append(line_text)

    if current_lines:
        sections.append((current_topic or f"Page {page.page_number}", current_lines))

    if not sections:
        return []

    chunks: list[Chunk] = []
    for index, (topic, lines) in enumerate(sections):
        text = "\n".join(lines).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(page.filename, page.page_number, index),
                filename=page.filename,
                page_number=page.page_number,
                topic=topic,
                text=text,
                extraction_method=page.extraction_method,
            )
        )
    return chunks


def chunk_pages(pages: list[PageContent], settings: Settings = DEFAULT_SETTINGS) -> list[Chunk]:
    """Chunk every page of a document and flatten into one ordered list."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, settings))
    return chunks

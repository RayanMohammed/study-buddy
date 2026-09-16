"""Step 2: pull text out of a PDF, page by page.

Each page is extracted with PyMuPDF (fast, works for normal text). Pages
that look math-heavy or garbled — lots of LaTeX-ish symbols, or suspiciously
little text for how much ink is on the page — are re-extracted with
marker-pdf, which runs a layout+OCR model that handles formulas far better.
marker-pdf is an optional, heavy dependency; if it isn't installed we fall
back to the PyMuPDF text and flag the page so callers know it may be lossy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

from study_buddy.config import DEFAULT_SETTINGS, Settings

# Characters that show up disproportionately often in math/LaTeX source or
# in PyMuPDF's mangled rendering of formulas (glyph soup from embedded math
# fonts, stray control characters, etc.).
_MATH_SYMBOL_PATTERN = re.compile(
    r"[=+\-*/^_{}\\∑∫∏√≈≤≥≠±∞∂∇αβγδθλμπσφω]"
)


@dataclass
class PageContent:
    """Raw extracted text for a single PDF page, before chunking."""

    filename: str
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str
    extraction_method: str  # "pymupdf" or "marker"
    span_font_sizes: list[tuple[str, float]] = field(default_factory=list)
    """(line_text, dominant_font_size) pairs, used later for heading detection."""


def _symbol_density(text: str) -> float:
    if not text:
        return 0.0
    hits = len(_MATH_SYMBOL_PATTERN.findall(text))
    return hits / max(len(text), 1)


def is_math_heavy_or_low_quality(text: str, settings: Settings) -> bool:
    """Heuristic: should this page be re-extracted with marker-pdf?

    Two independent signals trigger the fallback:
    - the page has almost no extractable text (likely a scanned page, or a
      slide that's mostly a rendered equation/image PyMuPDF can't read), or
    - the extracted text is unusually dense with math/LaTeX symbols,
      suggesting formulas that PyMuPDF's plain-text extraction mangles.
    """
    stripped = text.strip()
    if len(stripped) < settings.min_chars_per_page:
        return True
    return _symbol_density(stripped) >= settings.math_symbol_density_threshold


def _extract_page_with_pymupdf(doc: fitz.Document, page_index: int) -> tuple[str, list[tuple[str, float]]]:
    page = doc[page_index]
    text = page.get_text("text")

    span_font_sizes: list[tuple[str, float]] = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text:
                continue
            # A line's "size" is its largest span's font size, so a bold
            # heading isn't diluted by a stray smaller trailing span.
            dominant_size = max(s.get("size", 0.0) for s in spans)
            span_font_sizes.append((line_text, dominant_size))

    return text, span_font_sizes


_marker_converter = None


def _get_marker_converter():
    """Lazily construct marker-pdf's PDF-to-text converter.

    Import + model loading is expensive, so we only pay for it the first
    time a page actually needs the fallback, and we cache the converter.
    """
    global _marker_converter
    if _marker_converter is not None:
        return _marker_converter

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    _marker_converter = PdfConverter(artifact_dict=create_model_dict())
    return _marker_converter


def extract_page_with_marker(pdf_path: Path, page_index: int) -> str:
    """Re-extract a single page's text using marker-pdf's OCR/layout model.

    Raises ImportError with a helpful message if marker-pdf isn't installed
    (it's an optional extra: `pip install study-buddy[marker]`).
    """
    try:
        converter = _get_marker_converter()
    except ImportError as exc:
        raise ImportError(
            "marker-pdf is not installed. Install it with "
            "`pip install study-buddy[marker]` to enable math-heavy page "
            "fallback extraction."
        ) from exc

    from marker.output import text_from_rendered

    # marker-pdf converts whole documents; we scope it to one page via
    # PyMuPDF's page-subset export so single-page fallback stays cheap.
    single_page_doc = fitz.open()
    with fitz.open(pdf_path) as src:
        single_page_doc.insert_pdf(src, from_page=page_index, to_page=page_index)
    tmp_path = pdf_path.with_name(f".{pdf_path.stem}_p{page_index}_marker_tmp.pdf")
    single_page_doc.save(tmp_path)
    single_page_doc.close()

    try:
        rendered = converter(str(tmp_path))
        text, _, _ = text_from_rendered(rendered)
    finally:
        tmp_path.unlink(missing_ok=True)

    return text


def extract_pdf_pages(
    pdf_path: Path, settings: Settings = DEFAULT_SETTINGS
) -> list[PageContent]:
    """Extract every page of a PDF, applying the marker-pdf fallback as needed."""
    pages: list[PageContent] = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            text, span_font_sizes = _extract_page_with_pymupdf(doc, page_index)
            method = "pymupdf"

            if is_math_heavy_or_low_quality(text, settings):
                try:
                    marker_text = extract_page_with_marker(pdf_path, page_index)
                    if marker_text.strip():
                        text = marker_text
                        method = "marker"
                except ImportError:
                    # marker-pdf not installed: keep the pymupdf text but
                    # note it may be lossy for this page.
                    method = "pymupdf_lowconf"

            pages.append(
                PageContent(
                    filename=pdf_path.name,
                    page_number=page_index + 1,
                    text=text,
                    extraction_method=method,
                    span_font_sizes=span_font_sizes,
                )
            )
    return pages

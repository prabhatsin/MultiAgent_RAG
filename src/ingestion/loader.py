"""Loads PDFs from a directory and extracts text per page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    source_file: str
    page_number: int  # 1-indexed
    text: str


def load_pdfs(data_dir: Path) -> list[PageText]:
    """Extracts text from every page of every PDF in data_dir.

    Pages with no extractable text (e.g. scanned images without OCR) are
    skipped rather than stored empty.
    """
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    pages: list[PageText] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(
                PageText(
                    source_file=pdf_path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )
    return pages

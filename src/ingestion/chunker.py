"""Splits page text into overlapping chunks for embedding/retrieval.

Chunking is done per-page (not across page boundaries) so every chunk keeps
an exact source_file + page_number for citations. The tradeoff: a sentence
that straddles a page break may end up split across two chunks with no
overlap between them. For most PDFs (paragraphs rarely align to page
boundaries anyway) this is an acceptable cost for precise citations.

Splitting prefers paragraph breaks, falling back to sentence breaks, falling
back to a hard character cut -- so chunks stay coherent whenever possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import PageText

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int  # index of this chunk within its page
    text: str
    doc_type: str = ""  # set by ingest.py after classifying the source file
    client_name: str = ""


def _split_long_unit(unit: str, max_len: int) -> list[str]:
    """Breaks a unit larger than max_len into sentence-sized (or hard-cut) pieces."""
    sentences = _SENTENCE_SPLIT.split(unit)
    if len(sentences) == 1:
        # No sentence boundaries found; hard-cut.
        return [unit[i : i + max_len] for i in range(0, len(unit), max_len)]
    return sentences


def _pack_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily packs text units into chunks of ~chunk_size chars with overlap."""
    chunks: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > chunk_size:
            for piece in _split_long_unit(unit, chunk_size):
                current = _append_unit(chunks, current, piece, chunk_size, overlap)
            continue

        current = _append_unit(chunks, current, unit, chunk_size, overlap)

    if current:
        chunks.append(current)
    return chunks


def _append_unit(
    chunks: list[str], current: str, unit: str, chunk_size: int, overlap: int
) -> str:
    """Appends unit to current, flushing to chunks + carrying overlap when full."""
    candidate = f"{current}\n\n{unit}" if current else unit
    if len(candidate) <= chunk_size:
        return candidate

    if current:
        chunks.append(current)
        tail = current[-overlap:] if overlap > 0 else ""
        return f"{tail}\n\n{unit}" if tail else unit

    # current is empty but unit alone still exceeds chunk_size (shouldn't
    # happen given _split_long_unit, but guard anyway).
    chunks.append(unit)
    return ""


def chunk_pages(
    pages: list[PageText], chunk_size: int, overlap: int
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in pages:
        paragraphs = _PARAGRAPH_SPLIT.split(page.text)
        texts = _pack_units(paragraphs, chunk_size, overlap)
        for i, text in enumerate(texts):
            chunk_id = f"{page.source_file}::p{page.page_number}::c{i}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=i,
                    text=text,
                )
            )
    return all_chunks

"""Filename-based document classification.

Classification is purely a filename pattern match -- no LLM call, no PDF
content inspection -- so it's fast, free, and deterministic at ingestion
time, and cheaply reusable by the Router Agent at query time (to know which
client names it's allowed to filter on) without touching the vector store.
"""

from __future__ import annotations

import re
from pathlib import Path

from src import config

# Matches "<name>-<m>-<d>.pdf", e.g. "robert-05-07.pdf", "nathan-6-2.pdf".
_TRANSCRIPT_PATTERN = re.compile(r"^(?P<name>[a-zA-Z]+)-\d{1,2}-\d{1,2}\.pdf$", re.IGNORECASE)


def classify_document(filename: str) -> tuple[str, str]:
    """Returns (doc_type, client_name). client_name is "" for policy docs."""
    match = _TRANSCRIPT_PATTERN.match(filename)
    if match:
        return "transcript", match.group("name").lower()
    return "policy", ""


def known_client_roster(data_dir: Path = config.DATA_DIR) -> set[str]:
    """Scans data_dir for transcript filenames and returns the set of client names."""
    roster: set[str] = set()
    for pdf_path in data_dir.glob("*.pdf"):
        _doc_type, client_name = classify_document(pdf_path.name)
        if client_name:
            roster.add(client_name)
    return roster

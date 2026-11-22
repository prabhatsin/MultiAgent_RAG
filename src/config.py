"""Central configuration for the Multi-Agent RAG system.

All tunables live here so ingestion and the agents stay in sync (e.g. chunk
size doesn't matter here, but embedding model/dims must match between
ingestion and query time).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_COLLECTION_NAME = "pdf_chunks"

# --- API key ---------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Copy .env.example to .env and fill in your key."
    )

# --- Models ---------------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # output_dimensionality requested from the embedding model
# Note: gemini-2.5-flash (requested in the original spec) returns 404 for
# this API key ("no longer available to new users"); the API error points
# to gemini-3.6-flash as the current equivalent.
GENERATION_MODEL = "gemini-3.6-flash"
ROUTER_MODEL = "gemini-3.6-flash"
RELEVANCE_GRADER_MODEL = "gemini-3.6-flash"
ANSWER_GRADER_MODEL = "gemini-3.6-flash"

# --- Chunking (content-aware: policy docs vs. client transcripts) -----------
POLICY_CHUNK_SIZE_CHARS = 1000
POLICY_CHUNK_OVERLAP_CHARS = 200

# Transcripts are conversational; smaller chunks preserve turn-level context.
TRANSCRIPT_CHUNK_SIZE_CHARS = 500
TRANSCRIPT_CHUNK_OVERLAP_CHARS = 100

# --- Retrieval ---------------------------------------------------------------
TOP_K = 5
MAX_SUB_QUERIES = 3
# Total retry budget, shared by Critic and Answer Grader follow-ups.
MAX_CRITIC_RETRIES = 1
MAX_CONTEXT_CHUNKS = 8  # cap on total chunks fed to the Synthesis Agent

# Chunks scoring below this cosine similarity are still passed to Synthesis
# (better a weak hint than nothing) but count as "not clearing" for the
# low_confidence signal below.
MIN_SIMILARITY_THRESHOLD = 0.5
# If fewer than this many retrieved chunks clear MIN_SIMILARITY_THRESHOLD,
# the Retriever flags state.low_confidence so Synthesis hedges more.
LOW_CONFIDENCE_MIN_CHUNKS = 2

# --- Embedding API batching ---------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 64

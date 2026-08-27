"""Thin async wrapper around the google-genai SDK.

Centralizes all Gemini API calls (embeddings + generation) so the rest of
the codebase never touches the SDK directly. Embeddings are L2-normalized
here because gemini-embedding-001 only auto-normalizes to unit length at its
native 3072-dim output -- when we request a smaller output_dimensionality
(see config.EMBEDDING_DIM), we must normalize ourselves for cosine
similarity to behave correctly.
"""

from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types

from src import config

# Silences a one-time informational warning the SDK logs about automatic
# function calling; we don't use tools/function calling here so it doesn't apply.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

_client = genai.Client(api_key=config.GEMINI_API_KEY)


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


async def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embeds a batch of texts. task_type is one of RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY."""
    if not texts:
        return []

    response = await _client.aio.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=config.EMBEDDING_DIM,
        ),
    )
    return [_normalize(e.values) for e in response.embeddings]


async def embed_texts_in_batches(
    texts: list[str], task_type: str, batch_size: int
) -> list[list[float]]:
    """Embeds many texts, respecting the API's per-request batch limits."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(
        *(embed_texts(batch, task_type) for batch in batches)
    )
    embeddings: list[list[float]] = []
    for batch_result in results:
        embeddings.extend(batch_result)
    return embeddings


async def generate(
    prompt: str,
    *,
    model: str = config.GENERATION_MODEL,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Single-turn text generation."""
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        temperature=temperature,
    )
    response = await _client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )
    return response.text or ""

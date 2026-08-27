"""Retriever Agent.

Runs one embedding + vector search per sub-query (in parallel), then merges
and deduplicates results across sub-queries. Accumulates into
state.retrieved_chunks so it can be called again (e.g. after a critic
retry) without losing earlier results.
"""

from __future__ import annotations

import asyncio

from src import config
from src.agents.state import State
from src.llm.gemini_client import embed_texts
from src.vectorstore.chroma_store import ChromaStore, RetrievedChunk


class RetrieverAgent:
    name = "Retriever Agent"

    def __init__(self, store: ChromaStore) -> None:
        self._store = store

    async def run(self, state: State) -> State:
        sub_queries = state.sub_queries or [state.query]
        where = {"client_name": state.client_name_filter} if state.client_name_filter else None

        results_per_query = await asyncio.gather(
            *(self._retrieve_one(q, where) for q in sub_queries)
        )

        merged: dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in state.retrieved_chunks
        }
        for chunks in results_per_query:
            for c in chunks:
                existing = merged.get(c.chunk_id)
                if existing is None or c.score > existing.score:
                    merged[c.chunk_id] = c

        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
        state.retrieved_chunks = ranked[: config.MAX_CONTEXT_CHUNKS]

        above_threshold = sum(
            1 for c in state.retrieved_chunks if c.score >= config.MIN_SIMILARITY_THRESHOLD
        )
        state.low_confidence = above_threshold < config.LOW_CONFIDENCE_MIN_CHUNKS

        state.log(
            agent_name=self.name,
            input_summary=f"sub_queries={sub_queries}, client_filter={state.client_name_filter!r}",
            output_summary=(
                f"retrieved {len(state.retrieved_chunks)} unique chunk(s) "
                f"from {sum(len(c) for c in results_per_query)} raw hit(s); "
                f"{above_threshold} clear the {config.MIN_SIMILARITY_THRESHOLD} threshold; "
                f"low_confidence={state.low_confidence}"
            ),
        )
        return state

    async def _retrieve_one(self, sub_query: str, where: dict | None) -> list[RetrievedChunk]:
        [embedding] = await embed_texts([sub_query], task_type="RETRIEVAL_QUERY")
        return self._store.query(embedding, top_k=config.TOP_K, where=where)

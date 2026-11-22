"""Relevance Grader Agent.

Runs between the Retriever and the Synthesizer. Cosine similarity only says
a chunk is about the same topic as the query; this agent asks an LLM whether
each chunk actually helps answer the question (Self-RAG's ISREL step), and
drops the ones that don't before they reach the Synthesizer.

All chunks are graded in a single batched call to keep latency and cost to
one LLM call per retrieval pass.
"""

from __future__ import annotations

import json

from src import config
from src.agents.state import State
from src.llm.gemini_client import generate

_SYSTEM_INSTRUCTION = """You are a relevance grading agent in a RAG system. \
You are given a user question and numbered context excerpts retrieved from a \
private document collection. Decide which excerpts are relevant.

Respond with ONLY a JSON object of this shape:
{
  "relevant_excerpts": [int, ...]
}

Rules:
- An excerpt is relevant if it contains information that helps answer the \
question, or any part of it (the question may have several parts).
- An excerpt that is merely on the same general topic but does not help \
answer the question is NOT relevant.
- When unsure, mark the excerpt as relevant: dropping useful evidence is \
worse than keeping a weak excerpt.
- List the excerpt numbers only. Use [] if none are relevant.
"""


def _format_context(state: State) -> str:
    lines = []
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        lines.append(f"[{i}] {chunk.text}")
    return "\n\n".join(lines)


class RelevanceGraderAgent:
    name = "Relevance Grader Agent"

    async def run(self, state: State) -> State:
        if not state.retrieved_chunks:
            state.log(
                agent_name=self.name,
                input_summary="no retrieved chunks to grade",
                output_summary="skipped",
            )
            return state

        prompt = (
            f"Question: {state.query}\n\n"
            f"Context excerpts:\n\n{_format_context(state)}"
        )
        raw = await generate(
            prompt=prompt,
            model=config.RELEVANCE_GRADER_MODEL,
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
        )

        total = len(state.retrieved_chunks)
        relevant = self._parse(raw, total)

        if relevant is None:
            # Unparseable grader output: keep everything rather than lose evidence.
            outcome = f"could not parse grader output; kept all {total} chunk(s)"
        elif not relevant:
            # Nothing judged relevant. Emptying retrieved_chunks would send the
            # Synthesizer down its no-retrieval (general knowledge) path, which
            # is exactly the hallucination risk we want to avoid. Keep the
            # chunks but force the hedged, low-confidence answer instead.
            state.low_confidence = True
            outcome = f"0/{total} chunk(s) relevant; kept all, forced low_confidence=True"
        else:
            state.retrieved_chunks = [state.retrieved_chunks[i - 1] for i in relevant]
            outcome = f"kept {len(relevant)}/{total} chunk(s): excerpts {relevant}"

        state.log(
            agent_name=self.name,
            input_summary=f"query + {total} retrieved chunk(s)",
            output_summary=outcome,
        )
        return state

    @staticmethod
    def _parse(raw: str, total: int) -> list[int] | None:
        """Returns sorted, de-duplicated 1-based excerpt numbers, or None on malformed output."""
        try:
            data = json.loads(raw)
            indices = data.get("relevant_excerpts")
            if not isinstance(indices, list):
                raise ValueError("malformed grader output")
            # Silently ignore out-of-range or non-integer entries.
            return sorted({i for i in indices if isinstance(i, int) and 1 <= i <= total})
        except (json.JSONDecodeError, ValueError, AttributeError):
            return None

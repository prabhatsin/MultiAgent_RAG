"""Critique / Verifier Agent.

Checks whether the synthesized answer is actually supported by the
retrieved context. If not, it can suggest a follow-up search query the
Orchestrator may use for one bounded retry.
"""

from __future__ import annotations

import json

from src.agents.state import Critique, State
from src.llm.gemini_client import generate

_SYSTEM_INSTRUCTION = """You are a verification agent. You are given a question, \
numbered context excerpts, and a generated answer. Check whether the answer's \
claims are actually supported by the context excerpts.

Respond with ONLY a JSON object of this shape:
{
  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
  "explanation": string,
  "suggested_followup_query": string | null
}

- SUPPORTED: every claim in the answer is backed by the excerpts.
- PARTIALLY_SUPPORTED: some claims are backed, others are not or go beyond \
the excerpts.
- UNSUPPORTED: the answer is not backed by the excerpts at all (including \
the case where the answer claims no information was found but relevant \
information actually IS present in the excerpts).
- If verdict is not SUPPORTED, set suggested_followup_query to a short, \
focused search query that might retrieve the missing information. \
Otherwise set it to null.
"""


def _format_context(state: State) -> str:
    lines = []
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        lines.append(f"[{i}] {chunk.text}")
    return "\n\n".join(lines)


class CriticAgent:
    name = "Critique/Verifier Agent"

    async def run(self, state: State) -> State:
        if not state.retrieved_chunks:
            state.critique = Critique(
                verdict="NOT_APPLICABLE",
                explanation="No retrieval was performed for this query.",
            )
            state.log(
                agent_name=self.name,
                input_summary="no retrieved context to verify against",
                output_summary="NOT_APPLICABLE",
            )
            return state

        context = _format_context(state)
        prompt = (
            f"Question: {state.query}\n\n"
            f"Context excerpts:\n\n{context}\n\n"
            f"Generated answer:\n{state.answer}"
        )
        raw = await generate(
            prompt=prompt,
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
        )
        critique = self._parse(raw)
        state.critique = critique

        state.log(
            agent_name=self.name,
            input_summary="question + answer + context",
            output_summary=f"verdict={critique.verdict}: {critique.explanation[:150]}",
        )
        return state

    @staticmethod
    def _parse(raw: str) -> Critique:
        try:
            data = json.loads(raw)
            verdict = data.get("verdict")
            if verdict not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}:
                raise ValueError("malformed critic verdict")
            return Critique(
                verdict=verdict,
                explanation=str(data.get("explanation", "")),
                suggested_followup_query=data.get("suggested_followup_query") or None,
            )
        except (json.JSONDecodeError, ValueError, AttributeError):
            return Critique(
                verdict="PARTIALLY_SUPPORTED",
                explanation="Could not parse critic output; treating as unverified.",
            )

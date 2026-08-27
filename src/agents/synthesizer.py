"""Synthesis / Answer Agent.

Generates a grounded answer from the retrieved chunks. When there are no
retrieved chunks (NO_RETRIEVAL plan, or retrieval found nothing), it either
answers from general knowledge (clearly labeled) or says it can't answer.
"""

from __future__ import annotations

from src import config
from src.agents.state import State
from src.llm.gemini_client import generate

_GROUNDED_SYSTEM_INSTRUCTION = """You are an answer-synthesis agent in a RAG \
system. You are given a user question and numbered context excerpts pulled \
from a private document collection.

Rules:
- Answer ONLY using information present in the context excerpts below.
- Cite every claim with the excerpt number(s) it came from, like [1] or [1,3].
- If the context does not contain enough information to answer, say so \
explicitly instead of guessing or using outside knowledge.
- Be concise and direct.
"""

_LOW_CONFIDENCE_ADDENDUM = """

IMPORTANT: Retrieval confidence is LOW for this question -- fewer than \
2 of the retrieved excerpts are strongly similar to the question, so they \
may only be tangentially related. Hedge explicitly: say clearly that the \
supporting evidence is weak/partial, avoid asserting anything the excerpts \
don't clearly support, and prefer "the documents don't clearly say" over a \
confident-sounding guess.
"""

_NO_CONTEXT_SYSTEM_INSTRUCTION = """You are an answer-synthesis agent in a RAG \
system. This question was classified as not needing document retrieval \
(e.g. small talk or general knowledge unrelated to the document collection). \
Answer directly and concisely. If it's a general-knowledge question, briefly \
note that the answer is from general knowledge, not the document collection."""


def _format_context(state: State) -> str:
    lines = []
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        lines.append(
            f"[{i}] (source: {chunk.source_file}, page {chunk.page_number})\n{chunk.text}"
        )
    return "\n\n".join(lines)


class SynthesizerAgent:
    name = "Synthesis/Answer Agent"

    async def run(self, state: State) -> State:
        if state.retrieved_chunks:
            context = _format_context(state)
            prompt = f"Context excerpts:\n\n{context}\n\nQuestion: {state.query}"
            system_instruction = _GROUNDED_SYSTEM_INSTRUCTION
            if state.low_confidence:
                system_instruction += _LOW_CONFIDENCE_ADDENDUM
        else:
            prompt = f"Question: {state.query}"
            system_instruction = _NO_CONTEXT_SYSTEM_INSTRUCTION

        answer = await generate(
            prompt=prompt,
            model=config.GENERATION_MODEL,
            system_instruction=system_instruction,
            temperature=0.2,
        )
        state.answer = answer.strip()

        state.log(
            agent_name=self.name,
            input_summary=f"query + {len(state.retrieved_chunks)} chunk(s) of context",
            output_summary=state.answer[:200],
        )
        return state

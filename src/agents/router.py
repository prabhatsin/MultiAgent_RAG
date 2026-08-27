"""Query Planner / Router Agent.

Decides whether a question needs retrieval at all, and if so whether it can
be answered from a single retrieval pass or needs to be decomposed into
several sub-questions (e.g. comparisons, multi-part questions).
"""

from __future__ import annotations

import json

from src import config
from src.agents.state import State
from src.ingestion.doc_classifier import known_client_roster
from src.llm.gemini_client import generate

_BASE_SYSTEM_INSTRUCTION = """You are a query planning agent in a RAG system over a \
private collection of PDF documents (case files, program standards, policy \
guidelines). Decide how to answer the user's question.

Respond with ONLY a JSON object of this shape:
{
  "plan": "NO_RETRIEVAL" | "SINGLE_RETRIEVAL" | "MULTI_RETRIEVAL",
  "sub_queries": [string, ...],
  "client_name": string | null
}

Rules:
- Use NO_RETRIEVAL only for greetings, chit-chat, or questions that are \
purely about general world knowledge unrelated to any document collection. \
sub_queries must be [] in this case.
- Use SINGLE_RETRIEVAL for a straightforward question answerable by one \
focused search. sub_queries must contain exactly one entry: the search \
query to use (can be a cleaned-up version of the user's question).
- Use MULTI_RETRIEVAL when the question has multiple distinct parts, asks \
to compare/combine information, or is broad enough that one search is \
unlikely to cover it. sub_queries must contain 2 to __MAX_SUB_QUERIES__ \
focused search queries, each covering one part.
- When in doubt between NO_RETRIEVAL and retrieval, prefer retrieval: this \
system's job is to answer from the document collection.
- Known clients in the document collection: __CLIENT_ROSTER__. If the \
question refers to one of these clients by name, set "client_name" to \
that exact roster value (lowercase). Otherwise set "client_name" to null. \
Do not guess a client_name that isn't in the roster.
""".replace("__MAX_SUB_QUERIES__", str(config.MAX_SUB_QUERIES))

_VALID_PLANS = {"NO_RETRIEVAL", "SINGLE_RETRIEVAL", "MULTI_RETRIEVAL"}


class RouterAgent:
    name = "Router/Planner Agent"

    def __init__(self) -> None:
        self._known_clients = known_client_roster()
        roster_str = ", ".join(sorted(self._known_clients)) or "(none)"
        self._system_instruction = _BASE_SYSTEM_INSTRUCTION.replace(
            "__CLIENT_ROSTER__", roster_str
        )

    async def run(self, state: State) -> State:
        raw = await generate(
            prompt=f"User question: {state.query}",
            model=config.ROUTER_MODEL,
            system_instruction=self._system_instruction,
            response_mime_type="application/json",
            temperature=0.0,
        )

        plan, sub_queries, client_name = self._parse(raw, state.query)
        state.plan = plan
        state.sub_queries = sub_queries
        state.client_name_filter = client_name

        state.log(
            agent_name=self.name,
            input_summary=state.query,
            output_summary=(
                f"plan={plan}, sub_queries={sub_queries}, "
                f"client_name_filter={client_name!r}"
            ),
        )
        return state

    def _parse(self, raw: str, fallback_query: str) -> tuple[str, list[str], str | None]:
        try:
            data = json.loads(raw)
            plan = data.get("plan")
            sub_queries = data.get("sub_queries", [])
            if plan not in _VALID_PLANS or not isinstance(sub_queries, list):
                raise ValueError("malformed router output")
            if plan != "NO_RETRIEVAL" and not sub_queries:
                raise ValueError("retrieval plan with no sub_queries")

            client_name = data.get("client_name")
            if isinstance(client_name, str):
                client_name = client_name.strip().lower() or None
            if client_name not in self._known_clients:
                client_name = None  # defensive: never trust an unknown/hallucinated name

            return (
                plan,
                [str(q) for q in sub_queries][: config.MAX_SUB_QUERIES],
                client_name,
            )
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Safe fallback: treat as a single retrieval on the raw question, no filter.
            return "SINGLE_RETRIEVAL", [fallback_query], None

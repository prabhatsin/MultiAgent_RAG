"""Orchestrator: sequences agents over a shared State object.

Flow:
  Router -> [Retriever] -> Synthesizer -> [Critic] -> (bounded retry loop)

The orchestrator is the only component that knows the overall sequence;
agents only ever see the shared State, never each other.
"""

from __future__ import annotations

from src import config
from src.agents.critic import CriticAgent
from src.agents.retriever import RetrieverAgent
from src.agents.router import RouterAgent
from src.agents.state import State
from src.agents.synthesizer import SynthesizerAgent
from src.vectorstore.chroma_store import ChromaStore


class Orchestrator:
    def __init__(self, use_critic: bool = True) -> None:
        self._store = ChromaStore()
        self._router = RouterAgent()
        self._retriever = RetrieverAgent(self._store)
        self._synthesizer = SynthesizerAgent()
        self._critic = CriticAgent()
        self._use_critic = use_critic

    async def answer(self, query: str) -> State:
        state = State(query=query)

        state = await self._router.run(state)

        if state.plan != "NO_RETRIEVAL":
            state = await self._retriever.run(state)

        state = await self._synthesizer.run(state)

        if not self._use_critic:
            return state

        state = await self._critic.run(state)

        while (
            state.critique is not None
            and state.critique.verdict in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}
            and state.critique.suggested_followup_query
            and state.retry_count < config.MAX_CRITIC_RETRIES
        ):
            state.retry_count += 1
            state.sub_queries = [state.critique.suggested_followup_query]
            state = await self._retriever.run(state)
            state = await self._synthesizer.run(state)
            state = await self._critic.run(state)

        return state

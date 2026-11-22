"""Orchestrator: sequences agents over a shared State object.

Flow:
  Router -> [Retriever -> Relevance Grader] -> Synthesizer
         -> [Critic -> (Answer Grader, if SUPPORTED)] -> (bounded retry loop)

The orchestrator is the only component that knows the overall sequence;
agents only ever see the shared State, never each other.
"""

from __future__ import annotations

from src import config
from src.agents.answer_grader import AnswerGraderAgent
from src.agents.critic import CriticAgent
from src.agents.relevance_grader import RelevanceGraderAgent
from src.agents.retriever import RetrieverAgent
from src.agents.router import RouterAgent
from src.agents.state import State
from src.agents.synthesizer import SynthesizerAgent
from src.vectorstore.chroma_store import ChromaStore


class Orchestrator:
    def __init__(
        self,
        use_critic: bool = True,
        use_grader: bool = True,
        use_answer_grader: bool = True,
    ) -> None:
        self._store = ChromaStore()
        self._router = RouterAgent()
        self._retriever = RetrieverAgent(self._store)
        self._grader = RelevanceGraderAgent()
        self._synthesizer = SynthesizerAgent()
        self._critic = CriticAgent()
        self._answer_grader = AnswerGraderAgent()
        self._use_critic = use_critic
        self._use_grader = use_grader
        self._use_answer_grader = use_answer_grader

    async def answer(self, query: str) -> State:
        state = State(query=query)

        state = await self._router.run(state)

        if state.plan != "NO_RETRIEVAL":
            state = await self._retrieve(state)

        state = await self._synthesizer.run(state)

        if not self._use_critic:
            return state

        state = await self._verify(state)

        while (
            (followup := self._followup_query(state)) is not None
            and state.retry_count < config.MAX_CRITIC_RETRIES
        ):
            state.retry_count += 1
            state.sub_queries = [followup]
            state = await self._retrieve(state)
            state = await self._synthesizer.run(state)
            state = await self._verify(state)

        return state

    async def _verify(self, state: State) -> State:
        """Critic first; Answer Grader only if the answer is grounded."""
        state = await self._critic.run(state)
        state.answer_grade = None  # clear any grade from a previous attempt
        if self._use_answer_grader and state.critique.verdict == "SUPPORTED":
            state = await self._answer_grader.run(state)
        return state

    @staticmethod
    def _followup_query(state: State) -> str | None:
        """The follow-up search to retry with, or None if no retry is warranted."""
        critique = state.critique
        if critique is None or critique.verdict == "NOT_APPLICABLE":
            return None
        if critique.verdict in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}:
            return critique.suggested_followup_query
        # SUPPORTED: grounded, but may still not answer the whole question.
        grade = state.answer_grade
        if grade is not None and grade.verdict != "FULLY_ANSWERED":
            return grade.suggested_followup_query
        return None

    async def _retrieve(self, state: State) -> State:
        state = await self._retriever.run(state)
        if self._use_grader:
            state = await self._grader.run(state)
        return state

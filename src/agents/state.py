"""Shared state passed between agents.

Agents communicate exclusively by reading from and writing to this object,
which the Orchestrator threads through each agent call in sequence. No
agent holds a reference to another agent or to global mutable state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.vectorstore.chroma_store import RetrievedChunk


@dataclass
class AgentEvent:
    agent_name: str
    input_summary: str
    output_summary: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Critique:
    verdict: str  # SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | NOT_APPLICABLE
    explanation: str
    suggested_followup_query: str | None = None


@dataclass
class AnswerGrade:
    verdict: str  # FULLY_ANSWERED | PARTIALLY_ANSWERED | NOT_ANSWERED
    explanation: str
    suggested_followup_query: str | None = None


@dataclass
class State:
    query: str

    plan: str = ""  # NO_RETRIEVAL | SINGLE_RETRIEVAL | MULTI_RETRIEVAL
    sub_queries: list[str] = field(default_factory=list)
    client_name_filter: str | None = None  # set by Router, consumed by Retriever

    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    low_confidence: bool = False  # set by Retriever, consumed by Synthesizer

    answer: str = ""
    critique: Critique | None = None
    answer_grade: AnswerGrade | None = None  # set only when critique is SUPPORTED

    retry_count: int = 0
    trace: list[AgentEvent] = field(default_factory=list)

    def log(self, agent_name: str, input_summary: str, output_summary: str) -> None:
        self.trace.append(
            AgentEvent(
                agent_name=agent_name,
                input_summary=input_summary,
                output_summary=output_summary,
            )
        )

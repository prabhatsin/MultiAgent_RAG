"""Answer Grader Agent.

Runs after the Critic, and only when the Critic found the answer SUPPORTED.
The Critic checks that the answer is grounded in the context (Self-RAG's
ISSUP); this agent checks the separate question of whether the answer
actually addresses the user's question in full (Self-RAG's ISUSE). A fully
grounded answer can still skip part of a multi-part question -- this agent
catches that and suggests a follow-up search for the missing part.
"""

from __future__ import annotations

import json

from src import config
from src.agents.state import AnswerGrade, State
from src.llm.gemini_client import generate

_SYSTEM_INSTRUCTION = """You are an answer grading agent. You are given a user \
question and a generated answer. Decide whether the answer actually addresses \
the question. Do NOT judge factual accuracy -- that is checked separately.

Respond with ONLY a JSON object of this shape:
{
  "verdict": "FULLY_ANSWERED" | "PARTIALLY_ANSWERED" | "NOT_ANSWERED",
  "explanation": string,
  "suggested_followup_query": string | null
}

- FULLY_ANSWERED: every part of the question is addressed.
- PARTIALLY_ANSWERED: some parts of the question are addressed, others are \
ignored or left open.
- NOT_ANSWERED: the answer does not address the question (including answers \
that only say the information could not be found).
- If verdict is not FULLY_ANSWERED, set suggested_followup_query to a short, \
focused search query for the part that is missing. Otherwise set it to null.
"""

_VALID_VERDICTS = {"FULLY_ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERED"}


class AnswerGraderAgent:
    name = "Answer Grader Agent"

    async def run(self, state: State) -> State:
        prompt = f"Question: {state.query}\n\nGenerated answer:\n{state.answer}"
        raw = await generate(
            prompt=prompt,
            model=config.ANSWER_GRADER_MODEL,
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.0,
        )
        grade = self._parse(raw)
        state.answer_grade = grade

        state.log(
            agent_name=self.name,
            input_summary="question + answer",
            output_summary=f"verdict={grade.verdict}: {grade.explanation[:150]}",
        )
        return state

    @staticmethod
    def _parse(raw: str) -> AnswerGrade:
        try:
            data = json.loads(raw)
            verdict = data.get("verdict")
            if verdict not in _VALID_VERDICTS:
                raise ValueError("malformed answer grader verdict")
            return AnswerGrade(
                verdict=verdict,
                explanation=str(data.get("explanation", "")),
                suggested_followup_query=data.get("suggested_followup_query") or None,
            )
        except (json.JSONDecodeError, ValueError, AttributeError):
            # No follow-up query, so an unparseable grade never triggers a retry.
            return AnswerGrade(
                verdict="PARTIALLY_ANSWERED",
                explanation="Could not parse answer grader output; treating as unverified.",
            )

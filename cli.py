#!/usr/bin/env python3
"""CLI entrypoint for the Multi-Agent RAG system.

Usage:
    python cli.py "What are the check-in guidelines?"
    python cli.py --no-critic "What are the check-in guidelines?"
    python cli.py --no-grader "What are the check-in guidelines?"
    python cli.py --no-answer-grader "What are the check-in guidelines?"
    python cli.py                     # interactive REPL
"""

from __future__ import annotations

import argparse
import asyncio

from src.agents.state import State
from src.orchestrator import Orchestrator

_SEP = "-" * 72


def print_trace(state: State) -> None:
    print(_SEP)
    print("AGENTS RUN")
    print(_SEP)
    for event in state.trace:
        print(f"[{event.agent_name}]")
        print(f"  in : {event.input_summary}")
        print(f"  out: {event.output_summary}")
    print()


def print_retrieved(state: State) -> None:
    print(_SEP)
    print(f"RETRIEVED CHUNKS ({len(state.retrieved_chunks)})")
    print(_SEP)
    print(f"client_name_filter: {state.client_name_filter!r}   low_confidence: {state.low_confidence}")
    if not state.retrieved_chunks:
        print("(none)")
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        tag = f"{chunk.doc_type}" + (f"/{chunk.client_name}" if chunk.client_name else "")
        print(f"[{i}] score={chunk.score:.3f}  [{tag}]  {chunk.source_file} p.{chunk.page_number}")
        preview = chunk.text.replace("\n", " ")
        preview = preview[:150] + ("..." if len(preview) > 150 else "")
        print(f"    {preview}")
    print()


def print_critique(state: State) -> None:
    if state.critique is None:
        return
    print(_SEP)
    print("CRITIQUE")
    print(_SEP)
    print(f"verdict: {state.critique.verdict}")
    print(f"explanation: {state.critique.explanation}")
    if state.answer_grade is not None:
        print(f"answer grade: {state.answer_grade.verdict}")
        print(f"answer grade explanation: {state.answer_grade.explanation}")
    if state.retry_count:
        print(f"retries used: {state.retry_count}")
    print()


def print_answer(state: State) -> None:
    print(_SEP)
    print("ANSWER")
    print(_SEP)
    print(state.answer)
    print()


async def ask(orchestrator: Orchestrator, query: str) -> None:
    state = await orchestrator.answer(query)
    print()
    print_trace(state)
    print_retrieved(state)
    print_critique(state)
    print_answer(state)


async def repl(orchestrator: Orchestrator) -> None:
    print("Multi-Agent RAG CLI. Type a question, or 'exit' to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        await ask(orchestrator, query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the ingested PDFs.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Skip the Critique/Verifier Agent for a faster response.",
    )
    parser.add_argument(
        "--no-grader",
        action="store_true",
        help="Skip the Relevance Grader Agent and pass all retrieved chunks to synthesis.",
    )
    parser.add_argument(
        "--no-answer-grader",
        action="store_true",
        help="Skip the Answer Grader Agent (whether a supported answer fully answers the question).",
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(
        use_critic=not args.no_critic,
        use_grader=not args.no_grader,
        use_answer_grader=not args.no_answer_grader,
    )

    if args.question:
        asyncio.run(ask(orchestrator, args.question))
    else:
        asyncio.run(repl(orchestrator))


if __name__ == "__main__":
    main()

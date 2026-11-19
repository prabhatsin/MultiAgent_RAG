# Advanced Upgrades

A running log of upgrades that take this project from a working multi-agent RAG
demo to a production-grade system. Each upgrade gets its own numbered file so
the progression is easy to follow (and easy to walk through in an interview).

## How to add an upgrade

1. Copy `_TEMPLATE.md` to `NN_short_name.md` (next number, e.g. `01_hybrid_search.md`).
2. Fill in the problem, the design, and — most importantly — the measured before/after.
3. Update the status in the tracker below.

## Tracker

Status: 💡 idea · 🚧 in progress · ✅ done

| #  | Upgrade                                                    | Area          | Status |
|----|------------------------------------------------------------|---------------|--------|
| 01 | Evaluation harness (golden Q&A set, faithfulness/recall metrics) | Quality  | 💡 |
| 02 | Hybrid search (BM25 + dense) with Reciprocal Rank Fusion   | Retrieval     | 💡 |
| 03 | Cross-encoder / LLM reranking of retrieved chunks          | Retrieval     | 💡 |
| 04 | Contextual chunk headers + `section_title` metadata        | Ingestion     | 💡 |
| 05 | PageIndex tree retrieval for the 63-page Standards doc     | Retrieval     | 💡 |
| 06 | Extend router filters beyond client (doc_type, date)       | Agents        | 💡 |
| 07 | Query rewriting / HyDE for vague questions                 | Retrieval     | 💡 |
| 08 | Claim-level critic with per-sentence citations             | Agents        | 💡 |
| 09 | Conversation memory (multi-turn follow-ups)                | Agents        | 💡 |
| 10 | Semantic answer cache + embedding cache                    | Performance   | 💡 |
| 11 | Tracing & observability (per-agent latency, tokens, cost)  | Ops           | 💡 |
| 12 | Incremental ingestion (hash-based, only re-embed changes)  | Ingestion     | 💡 |
| 13 | FastAPI service with streaming responses                   | Serving       | 💡 |
| 14 | Guardrails: prompt-injection detection, PII redaction      | Safety        | 💡 |
| 15 | Docker + CI (lint, tests, eval regression gate)            | Ops           | 💡 |

See `TOPIC_FIT_MAP.md` for where BM25, RRF, reranking, PageIndex and
AST autodiscovery fit (or don't) in this codebase.

Already in the base system: client-name metadata filtering, parallel sub-query retrieval.

Suggested order: start with **01** — every later upgrade is more convincing
when you can show a metric moving.

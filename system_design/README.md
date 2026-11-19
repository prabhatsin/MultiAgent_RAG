# System Design: Multi-Agent RAG over PDFs

| File | What it is |
|---|---|
| `architecture.png` / `architecture.svg` | Full system diagram: ingestion, vector store, agents, Gemini API, retry loop |
| `architecture.dot` | Graphviz source. Re-render with `dot -Tsvg architecture.dot -o architecture.svg && dot -Tpng -Gdpi=130 architecture.dot -o architecture.png` |
| This file | Mermaid diagrams (render on GitHub or in VS Code): sequence diagram, decision flow, and a proposed hybrid BM25 design |

![architecture](architecture.png)

---

## 1. Query sequence (one question, end to end)

```mermaid
sequenceDiagram
    autonumber
    actor U as User (cli.py)
    participant O as Orchestrator
    participant R as Router Agent
    participant Rt as Retriever Agent
    participant S as Synthesis Agent
    participant C as Critic Agent
    participant E as Gemini Embeddings
    participant G as Gemini LLM
    participant V as Chroma (HNSW, cosine)

    U->>O: answer(query)
    O->>R: run(State)
    R->>G: plan? (JSON, temp 0, roster in prompt)
    G-->>R: {plan, sub_queries, client_name}
    R-->>O: State.plan / sub_queries / client_name_filter

    alt plan != NO_RETRIEVAL
        O->>Rt: run(State)
        par one per sub-query (≤3)
            Rt->>E: embed(sub_query, RETRIEVAL_QUERY)
            E-->>Rt: 768-d vector (normalized)
            Rt->>V: query(vec, top_k=5, where={client_name})
            V-->>Rt: chunks + cosine distance
        end
        Rt-->>O: merged, deduped top-8 + low_confidence
    end

    O->>S: run(State)
    S->>G: grounded prompt (context → question), temp 0.2
    G-->>S: answer with [n] citations
    S-->>O: State.answer

    opt critic enabled (default)
        O->>C: run(State)
        C->>G: verify answer vs excerpts (JSON, temp 0)
        G-->>C: verdict + suggested_followup_query
        loop while verdict ≠ SUPPORTED and followup and retry_count < 1
            O->>Rt: run(sub_queries=[followup])
            O->>S: run
            O->>C: run
        end
    end
    O-->>U: State (trace, chunks, critique, answer)
```

## 2. Orchestrator decision flow

```mermaid
flowchart TD
    Q([query]) --> R{Router plan}
    R -- NO_RETRIEVAL --> S0[Synthesizer:<br/>general-knowledge mode]
    R -- SINGLE / MULTI --> RT[Retriever:<br/>1–3 sub-queries × top-5<br/>merge · dedupe · top-8]
    RT --> LC{≥2 chunks with<br/>cosine ≥ 0.5?}
    LC -- yes --> SG[Synthesizer:<br/>grounded + citations]
    LC -- no --> SH[Synthesizer:<br/>grounded + HEDGE addendum]
    S0 --> CR
    SG --> CR
    SH --> CR
    CR{Critic enabled?} -- no --> OUT([return State])
    CR -- yes --> V{Critic verdict}
    V -- SUPPORTED / NOT_APPLICABLE --> OUT
    V -- PARTIAL / UNSUPPORTED --> RL{followup query<br/>and retry_count < 1?}
    RL -- no --> OUT
    RL -- yes --> RT2[sub_queries = followup<br/>retry_count += 1] --> RT
```

## 3. Ingestion flow

```mermaid
flowchart LR
    P[/data/*.pdf/] --> L[loader.py<br/>pypdf, per page]
    L --> DC[doc_classifier.py<br/>filename regex]
    DC -->|policy| CP[chunker<br/>1000 / 200]
    DC -->|transcript + client_name| CT[chunker<br/>500 / 100]
    CP --> EM[embed batches of 64<br/>RETRIEVAL_DOCUMENT, 768-d, L2 norm]
    CT --> EM
    EM --> RS{--reset?}
    RS -- yes --> DEL[delete collection] --> ADD
    RS -- no --> ADD[(Chroma add<br/>ids: file::pN::cM<br/>+ text + metadata)]
```

Embedding runs **before** reset, so a failed API call leaves the existing index untouched.

---

## 4. Current vs. proposed retrieval (BM25 hybrid)

**Today the system uses dense-only retrieval. BM25 is not used.** Dense embeddings handle paraphrase well but miss exact tokens, and this corpus has plenty of them: standard numbers ("4.020"), dates ("5/21"), names, and acronyms. The proposed change keeps the agent contract and changes only the inside of `RetrieverAgent`.

```mermaid
flowchart LR
    SQ[sub-query] --> D[Dense: Gemini embed<br/>→ Chroma top-20]
    SQ --> B[Sparse: BM25<br/>rank_bm25 / SQLite FTS5<br/>top-20, same where-filter]
    D --> F[Reciprocal Rank Fusion<br/>score = Σ 1/(60 + rank)]
    B --> F
    F --> RR[Cross-encoder reranker<br/>optional]
    RR --> K[top-8 → Synthesizer]
```

| | Current | Proposed |
|---|---|---|
| Retrieval | Dense cosine only | Dense + BM25, fused with RRF |
| Fusion across sub-queries | Max raw cosine score (scores not comparable across queries) | RRF on ranks (scale-free) |
| Exact-match queries ("Standard 4.020", "5/21") | Weak | Strong |
| Extra infra | none | In-memory BM25 built at ingest (354 chunks → trivial) |
| Precision step | none | Optional reranker over the fused top 20–50 |

---

## 5. Key design properties (for discussion)

- **Blackboard pattern:** agents share only `State`. The `Orchestrator` alone owns the sequence, so agents are independently testable and swappable.
- **Fail toward retrieval:** if the router's output can't be parsed, the query falls back to `SINGLE_RETRIEVAL`. If the critic's output can't be parsed, the verdict becomes `PARTIALLY_SUPPORTED` and no retry happens.
- **Bounded self-correction:** at most 1 retry, so the worst case is about 9 API calls.
- **Citation-ready storage:** Chroma stores vector, text, and metadata together, so there's no side lookup.
- **Known gaps:** see `../PROJECT_WALKTHROUGH.md` §6. In short: `add()` doesn't overwrite existing chunks, a retry's new chunks can be dropped by the top-8 cut, the client filter excludes policy docs, and an empty retrieval falls back to general knowledge.

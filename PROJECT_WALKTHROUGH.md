# Multi-Agent RAG — Project Walkthrough & Interview Prep

This guide covers:

1. **How to read the codebase** (the order to open files in)
2. **How to explain it to someone who doesn't know RAG**
3. **How a request flows through the system**, step by step
4. **The core RAG concepts used**, and where each one lives in the code
5. **A component-by-component deep dive** with the design trade-offs
6. **Known weaknesses and bugs** (be ready for these, because a good reviewer will find them)
7. **Senior-engineer questions with model answers**
8. **What "production-grade" would add**
9. **A cheat sheet of numbers**

---

## 1. Reading order (start here)

The whole project is about 1,000 lines of Python, so you can read all of it in one sitting. Go in this order. It follows the data, first offline (ingestion) and then online (query time).

| # | File | Why read it now | Time |
|---|------|-----------------|------|
| 1 | `README.md` | The author's intent and design decisions | 5 min |
| 2 | `src/config.py` | Every tunable number in the system. Keep it open in a side tab | 3 min |
| 3 | `src/llm/gemini_client.py` | The only place that calls the LLM or embedding API: `embed_texts()` and `generate()` | 5 min |
| **Offline path: ingestion** ||||
| 4 | `src/ingestion/loader.py` | PDF → text, one page at a time | 2 min |
| 5 | `src/ingestion/doc_classifier.py` | Filename → `policy` or `transcript` + client name | 2 min |
| 6 | `src/ingestion/chunker.py` | Text → overlapping chunks. **Read it slowly; this is where a lot of RAG quality is decided** | 10 min |
| 7 | `src/ingestion/ingest.py` | Ties 4–6 together: load → classify → chunk → embed → store | 5 min |
| 8 | `src/vectorstore/chroma_store.py` | Vector DB wrapper: `add()` and `query()`, distance → similarity | 5 min |
| **Online path: answering a question** ||||
| 9 | `src/agents/state.py` | The shared "blackboard" every agent reads and writes | 3 min |
| 10 | `src/orchestrator.py` | **The control flow of the whole system in 57 lines.** Read it before the agents | 5 min |
| 11 | `src/agents/router.py` | Query planning: whether to retrieve, and with which sub-queries | 10 min |
| 12 | `src/agents/retriever.py` | Embed sub-queries, search, merge, dedupe, confidence signal | 10 min |
| 13 | `src/agents/synthesizer.py` | Grounded generation with citations | 5 min |
| 14 | `src/agents/critic.py` | Faithfulness check and the bounded self-correction loop | 5 min |
| 15 | `cli.py` | How results and the trace are shown to the user | 3 min |

**How to proceed after reading:**
1. Run `python -m src.ingestion.ingest --reset` and watch the per-file chunk counts.
2. Run `python cli.py` and try these four query types. Each one exercises a different branch:
   - `hi there` → `NO_RETRIEVAL`
   - `What are the check-in guidelines?` → `SINGLE_RETRIEVAL`
   - `Compare the grievance process with the check-in rules` → `MULTI_RETRIEVAL`
   - `What did Nathan talk about in his sessions?` → client metadata filter (`client_name=nathan`)
3. For each query, read the **AGENTS RUN** trace and match every line to the code that produced it.
4. Change one number in `config.py` (for example `TOP_K=2` or `MIN_SIMILARITY_THRESHOLD=0.7`) and see how the answers change. That's how the knobs start to make sense.

---

## 2. Explaining it to someone who doesn't know RAG

### The one-sentence version
> "It's a chatbot that answers questions about a specific folder of PDFs. It looks up the relevant pages first and then answers *only* from those pages, with citations."

### The analogy: an open-book exam
- A plain LLM like ChatGPT is a student taking a **closed-book** exam. It answers from memory, and when it doesn't know, it may confidently make something up (this is called "hallucination"). It has also never seen *your* private documents.
- **RAG (Retrieval-Augmented Generation)** turns this into an **open-book** exam. Before answering, the student finds the relevant pages in the book, reads them, and writes an answer based on them, citing the page.

### How it "finds the relevant pages": meaning-based search
- Every paragraph of every PDF is turned into a list of 768 numbers, called an **embedding**. Paragraphs with similar *meaning* get similar numbers, even if they use different words ("check-in" and "reporting to your officer").
- Your question is turned into numbers the same way, and the system finds the paragraphs whose numbers are closest. That's a **vector search**.

### Why "multi-agent"
Instead of one big step, four specialists pass a shared notepad down the line:
1. **Router (the planner):** "Does this question need the documents at all? Is it one question or three bundled together? Is it about a specific client?"
2. **Retriever (the librarian):** finds the most relevant paragraphs.
3. **Synthesizer (the writer):** writes the answer from those paragraphs and cites them as [1], [2].
4. **Critic (the fact-checker):** re-reads the answer against the paragraphs. If something isn't backed up, it suggests a better search and the team tries **one** more time.

### What the documents are
The data is community-corrections material: Colorado standards, check-in guidelines, the grievance and appeal process, internal programming, and the 8 Principles of Effective Intervention. These are the **policy** documents. There are also session **transcripts** for two clients, Nathan and Robert. So a typical user could be a case manager asking "What does policy say about X?" or "What did Robert discuss on 5/21?"

---

## 3. A request, end to end

```
                         OFFLINE (run once: python -m src.ingestion.ingest)
 data/*.pdf ──► loader ──► classifier ──► chunker ──► embed (RETRIEVAL_DOCUMENT) ──► Chroma (storage/)
              (per page)  (policy/       (1000/200 or         768-d, L2-normalized        HNSW, cosine
                          transcript)     500/100 chars)                                  + text + metadata

                         ONLINE (python cli.py "question")
 query ─► Router ──plan──► Retriever ──chunks──► Synthesizer ──answer──► Critic
           │ LLM, JSON       │ embed(RETRIEVAL_QUERY)  │ LLM, cites [n]     │ LLM, JSON, temp 0
           │ temp 0          │ per sub-query, parallel │ hedges if          │ SUPPORTED / PARTIAL /
           │                 │ Chroma top-5 each       │ low_confidence     │ UNSUPPORTED
           │                 │ merge+dedupe → top-8    │                    │
           │                 │                         │                    ▼
           └─NO_RETRIEVAL───────────────────────────►  │      not SUPPORTED + followup query
                                                       │      + retry_count < 1 ?
                                                       ◄──────── re-run Retriever→Synth→Critic once
```

### Walkthrough for "What did Nathan say about his job search?"

1. **`cli.py`** builds an `Orchestrator`. Its constructor opens Chroma and the Router scans `data/` for client names, giving the roster `{nathan, robert}`.
2. **`Orchestrator.answer`** creates `State(query=...)`.
3. **Router** sends one LLM call with JSON output and temperature 0. It returns something like
   `{"plan":"SINGLE_RETRIEVAL","sub_queries":["Nathan job search"],"client_name":"nathan"}`.
   `_parse` validates it. An unknown client name is discarded, and broken JSON falls back to `SINGLE_RETRIEVAL` on the raw query.
4. **Retriever** embeds each sub-query with `task_type=RETRIEVAL_QUERY` and queries Chroma with
   `where={"client_name":"nathan"}` and `n_results=5`. It then merges results, keeping the best score per `chunk_id`, sorts, keeps the top 8, and sets `low_confidence` if fewer than 2 chunks have cosine similarity ≥ 0.5.
5. **Synthesizer** formats the chunks as `[i] (source: file, page p)\n text` and puts the question *after* the context. The system prompt says: answer only from the context, cite every claim, say so if the context isn't enough. When `low_confidence` is set, a "hedge explicitly" addendum is appended.
6. **Critic** makes an LLM call with the question, numbered excerpts, and the answer, and gets back a verdict in JSON. If the verdict is not `SUPPORTED` and it comes with a follow-up query, the orchestrator replaces `sub_queries` with that query and runs Retriever → Synthesizer → Critic **once** more (`MAX_CRITIC_RETRIES=1`).
7. **`cli.py`** prints the trace, the chunks with their scores, the verdict, and the answer.

**LLM and API calls per query:** 1 router + N embeds (N ≤ 3) + 1 synth + 1 critic = **4–6 calls**. In the worst case with a retry it's **7–9**.

---

## 4. Core RAG concepts used in this project

| Concept | What it means | Where it lives |
|---|---|---|
| **Ingestion / indexing pipeline** | The offline job that prepares documents for search | `src/ingestion/ingest.py` |
| **Document parsing** | Extracting a text layer from PDFs (no OCR) | `loader.py` (pypdf) |
| **Chunking** | Splitting documents into retrievable units | `chunker.py` |
| ↳ *Recursive / structure-aware splitting* | Split on paragraphs first, then sentences, then a hard character cut | `_PARAGRAPH_SPLIT`, `_SENTENCE_SPLIT`, `_split_long_unit` |
| ↳ *Chunk overlap* | Repeat the tail of one chunk at the start of the next so context isn't cut | `_append_unit` (`current[-overlap:]`) |
| ↳ *Content-aware chunk sizing* | Policy docs use 1000/200 characters, conversational transcripts use 500/100 | `config.py`, `ingest._chunk_all_pages` |
| ↳ *Page-bounded chunks* | Chunks never cross page breaks, so every citation has an exact page | `chunk_pages` |
| **Metadata enrichment** | Store `source_file`, `page_number`, `chunk_index`, `doc_type`, `client_name` with each chunk | `ingest.py`, `doc_classifier.py` |
| **Deterministic chunk IDs** | `file::pN::cM`, which makes re-ingestion idempotent (see the caveat in §6) | `chunker.py:93` |
| **Dense embeddings** | Text → 768-d vector (`gemini-embedding-001`) | `gemini_client.embed_texts` |
| ↳ *Asymmetric embeddings* | Documents embedded as `RETRIEVAL_DOCUMENT`, queries as `RETRIEVAL_QUERY` | `ingest.py:62`, `retriever.py:63` |
| ↳ *Matryoshka / reduced dimensionality* | Ask for 768 of the model's native 3072 dimensions to save storage and compute | `config.EMBEDDING_DIM` |
| ↳ *L2 normalization* | Unit vectors so that cosine = dot product | `gemini_client._normalize` |
| ↳ *Batching* | 64 texts per embedding request, batches sent concurrently | `embed_texts_in_batches` |
| **Vector store / ANN search** | Chroma with an HNSW index in cosine space, persisted locally | `chroma_store.py` |
| **Top-K retrieval** | 5 nearest chunks per sub-query | `config.TOP_K` |
| **Metadata filtering (pre-filter)** | `where={"client_name": ...}` limits search to one client's transcripts | `retriever.py:27` |
| **Query routing / adaptive retrieval** | Decide whether to retrieve at all (`NO_RETRIEVAL`) | `router.py` |
| **Query decomposition (multi-query)** | Break a complex question into 2–3 sub-queries | `router.py` (`MULTI_RETRIEVAL`) |
| **Entity extraction for filtering** | The LLM pulls a client name out of the question, validated against a roster | `router.py` + `known_client_roster` |
| **Result fusion & dedupe** | Combine hits across sub-queries, keeping the max score per chunk | `retriever.py:33-43` |
| **Context window budgeting** | Pass at most 8 chunks to the LLM | `config.MAX_CONTEXT_CHUNKS` |
| **Similarity threshold / confidence signal** | Flag `low_confidence` when fewer than 2 chunks have similarity ≥ 0.5 | `retriever.py:45-48` |
| **Grounded generation / prompt augmentation** | "Answer ONLY from context", with context placed before the question | `synthesizer.py` |
| **Inline citations / attribution** | `[1]`, `[1,3]` mapped to source file and page | `synthesizer._format_context` |
| **Abstention / "I don't know"** | Say the context is insufficient instead of guessing | Synthesizer prompt |
| **Hallucination mitigation** | Low temperature, grounding rules, hedging, critic | throughout |
| **Self-critique / faithfulness verification (LLM-as-judge)** | Check that the answer's claims are supported by the context | `critic.py` |
| **Corrective RAG (CRAG / Self-RAG-style loop)** | When the answer is unsupported, re-retrieve with a better query and regenerate | `orchestrator.py:45-55` |
| **Structured output** | JSON mode for router and critic, validated, with safe fallbacks | `response_mime_type="application/json"` + `_parse` |
| **Observability / tracing** | A per-agent input/output event log | `State.trace`, `AgentEvent`, `cli.print_trace` |
| **Blackboard architecture** | Agents communicate only through shared state; one orchestrator owns the flow | `state.py`, `orchestrator.py` |

**Not used (expect questions about these):** hybrid or keyword search (BM25), re-ranking with a cross-encoder, HyDE, conversation memory or query rewriting from chat history, parent-document / small-to-big retrieval, semantic chunking, caching, an evaluation dataset or metrics, streaming, guardrails or PII redaction, tests.

---

## 5. Component deep dive (the "why", not only the "what")

### 5.1 `config.py`
- It's the single source of truth. That matters because the **embedding model and dimension must match between ingestion and query time**. If you change them, you must re-ingest with `--reset`.
- It raises at *import time* if `GEMINI_API_KEY` is missing. That fails fast, but it also means you can't import anything (or unit-test the chunker) without a key.

### 5.2 `gemini_client.py`
- It's a thin adapter. No other module imports the SDK, so swapping providers changes one file.
- **Normalization:** Gemini only returns unit vectors at its native 3072 dimensions. At 768 dimensions the vectors aren't unit length, so the code normalizes them. Strictly, Chroma's `cosine` space already normalizes internally, so this is *defensive*: it keeps dot-product or other stores correct too. If someone asks, say exactly that.
- `embed_texts_in_batches` uses `asyncio.gather` over **all** batches with no concurrency limit, no retry, and no backoff. That's fine for 354 chunks (6 batches) but would hit 429 rate limits on a large corpus.

### 5.3 Ingestion
- **Loader:** pypdf extracts text page by page and skips empty pages. It has no OCR and no table or layout awareness. The "Ignoring wrong pointing object" warnings in the logs are pypdf complaining about malformed PDFs.
- **Classifier:** a regex on the filename (`name-m-d.pdf`). It's deterministic, costs nothing, and the Router reuses it to build the client roster. On the other hand it's brittle: `nathan_2024-04-14.pdf` would be silently classified as a policy document.
- **Chunker:** greedy packing of paragraphs. A paragraph larger than the chunk size is split into sentences. A unit with no sentence boundary is hard-cut. The overlap is the **last N characters** of the previous chunk, so it can start mid-word.
- **Ingest:** note that it **embeds before it resets**. The expensive network step runs first, so if embedding fails, the existing index is still intact. That's a good detail to mention.
- **Actual corpus stats** (measured): 10 PDFs, 107 pages → **354 chunks**. 228 come from policy docs (199 of those from the Colorado Standards alone) and 126 from transcripts. Median chunk length is 690 characters and the maximum is 1284.

### 5.4 `chroma_store.py`
- `PersistentClient` writes to `storage/` (SQLite plus HNSW segment files).
- `hnsw:space = cosine`, so `distance = 1 − cos_sim` and the code converts back with `similarity = 1 − distance`. In theory that ranges over [−1, 1], but in practice it's mostly [0, 1].
- Embeddings, text, and metadata are stored together, so results come back ready to cite without a second lookup. That's the README's argument for choosing Chroma over FAISS.

### 5.5 Agents and orchestrator
- **The contract:** `async def run(state) -> State`. Agents never call each other. The orchestrator is the only component that knows the sequence. This makes agents easy to test and reorder. You could map it onto a LangGraph `StateGraph`, but here it's plain asyncio with no framework.
- **Router:** temperature 0 and JSON mode, with the roster injected into the system prompt. The parser validates the plan, the shape of `sub_queries`, and the `client_name` against the roster (to guard against hallucinated filters), and caps sub-queries at 3. On any failure it falls back to `SINGLE_RETRIEVAL` on the raw query, so it **fails toward retrieval**.
- **Retriever:** sub-queries run concurrently. It accumulates results with the chunks already in `state` (so a retry can add evidence), dedupes by `chunk_id` keeping the max score, keeps the top 8, and computes the confidence flag.
- **Synthesizer:** has three modes: grounded, grounded with low-confidence hedging, and no-context general knowledge. Temperature 0.2.
- **Critic:** temperature 0 and JSON. It treats "the answer says no info was found, but the info *is* in the excerpts" as `UNSUPPORTED`, which catches false abstentions. A parse failure becomes `PARTIALLY_SUPPORTED` without a follow-up query, so no retry happens.
- **Bounded loop:** at most 1 retry. A stubbornly unsupported answer is still returned, with an honest verdict, instead of looping forever.

---

## 6. Known weaknesses and bugs (know these before someone else points them out)

These come from reading the code. Mentioning them yourself is the strongest signal of real understanding.

1. **Re-ingestion does not overwrite.** The README says re-ingesting "overwrites its old chunks", but `ChromaStore.add()` calls `collection.add()`. Chroma **ignores** existing IDs with a warning (`"Add of existing embedding ID"` in `chromadb/segment/impl/vector/local_hnsw.py`). Edited PDFs keep stale text unless you use `--reset`. *Fix:* use `upsert()`, and delete `where={"source_file": f}` before re-adding so pages that now have fewer chunks don't leave orphans behind.
2. **The critic retry can be a no-op.** The retriever merges the *old* chunks with the new follow-up results and keeps the top 8 by score. If the follow-up hits score lower than the original 8, which is likely because the follow-up query is narrower, they are dropped and the synthesizer sees the same context again. *Fix:* reserve slots for new results, use reciprocal rank fusion (RRF), or use a reranker.
3. **The client filter excludes policy docs.** `where={"client_name":"nathan"}` applies to *all* sub-queries, so "Did Nathan's check-ins follow the check-in guidelines?" can never retrieve the guidelines document. *Fix:* apply the filter per sub-query (let the router emit `{query, filter}` pairs), or use `$or: [{client_name: nathan}, {doc_type: policy}]`.
4. **Only one client per query.** "Compare Nathan and Robert" can only filter on one of them, or on neither.
5. **Empty retrieval falls back to general knowledge.** If retrieval returns 0 chunks (empty store, or a filter with no matches), the synthesizer uses `_NO_CONTEXT_SYSTEM_INSTRUCTION`, which says the question "was classified as not needing retrieval", and answers from world knowledge. For a question about a private client, that's a hallucination path. *Fix:* branch on `state.plan`, not on `bool(retrieved_chunks)`, and abstain when retrieval was planned but found nothing.
6. **Chunks exceed their size limit.** The overlap tail plus the next unit isn't re-checked against `chunk_size`, and long sentences pass through whole. **35 of 354 chunks are over their limit** (maximum 1284 characters against a limit of 1000).
7. **The similarity threshold is uncalibrated.** A threshold of 0.5 is a guess. Gemini cosine scores tend to cluster, so the threshold should be set from a labeled evaluation set. Also, the "fewer than 2" in `_LOW_CONFIDENCE_ADDENDUM` is hardcoded instead of read from `config.LOW_CONFIDENCE_MIN_CHUNKS`.
8. **Synchronous Chroma call inside async code.** `self._store.query` blocks the event loop, so only the embedding calls are truly concurrent. *Fix:* `asyncio.to_thread`.
9. **No resilience.** There are no timeouts, retries, backoff, or concurrency limits on Gemini calls. One failed batch aborts the whole ingestion.
10. **The critic grades its own family.** It uses the same model as the synthesizer, so it shares the same blind spots (self-preference bias). It also sees only chunk text, not source or page, so it can't check that citation numbers point to the right excerpt.
11. **Stateless REPL.** A follow-up question like "what about Robert?" has no memory of the previous turn.
12. **Privacy and security.** Client transcripts, which are sensitive and possibly criminal-justice PII, are sent to an external API. There's no prompt-injection defense for text inside the documents, and no authentication or per-user access control on which transcripts a user may see.
13. **Brittle identity.** Chunk IDs use the filename, so renaming a file creates duplicates. Doc type comes from the filename too.
14. **No evaluation, no tests.** There's no way to show that a change (chunk size, `TOP_K`, prompt) improved quality.

---

## 7. Questions a senior engineer would ask, with model answers

### A. Architecture & agents
**Q1. Why multi-agent? Isn't this a pipeline with LLM calls?**
Honestly, yes. It's a deterministic, orchestrator-driven pipeline in which some steps use an LLM to make decisions (routing, verification). "Agents" here means single-responsibility units with a uniform `run(State)` contract, not autonomous tool-using loops. The benefits: each step can be tested and swapped on its own, the trace shows exactly which step failed, and control flow is explicit and bounded. A fully autonomous agent would make cost and latency unpredictable.

**Q2. Why no LangChain or LangGraph?**
There are four steps and one loop, so a framework would add abstraction and dependency churn for little gain. The `State` + `Orchestrator` design is essentially a hand-rolled LangGraph `StateGraph`, so migrating would be mechanical if we later need checkpointing, branching, or human-in-the-loop.

**Q3. What's the latency and cost per query? How would you reduce it?**
4–6 LLM or embedding calls, or 7–9 with a retry, all sequential except the embeddings. Ways to reduce it: use a smaller and faster model for the router, or a rules or embedding classifier; skip the critic for high-confidence retrievals; stream the synthesizer output; cache query embeddings and answers; run the critic asynchronously and flag answers afterward instead of blocking.

**Q4. Why is the retry bounded at 1?**
Each retry costs about 3 calls, and the gains diminish quickly. Unbounded self-correction loops can oscillate. Returning an answer with an honest `UNSUPPORTED` verdict is better than hanging.

**Q5. What happens when the Router returns garbage?**
`_parse` validates the plan enum, the list type, a non-empty `sub_queries` for retrieval plans, and the roster membership of `client_name`. On any failure it falls back to `SINGLE_RETRIEVAL` on the raw query with no filter. It **fails open toward retrieval**, which is the safe direction for a document QA system.

**Q6. Is mutating a shared State safe?**
Yes, here, because agents run sequentially. If agents were parallelized (for example running the critic alongside something else), you would want immutable state updates or reducers, as in LangGraph.

### B. Ingestion & chunking
**Q7. Why 1000/200 for policy docs and 500/100 for transcripts?**
Policy text is dense, section-structured prose, and a larger chunk keeps a whole rule together with its conditions. Transcripts are short conversational turns, and large chunks would mix several topics and dilute the embedding. Overlap (about 20%) keeps a sentence at a boundary visible in both neighbors. These sizes were chosen by judgment, not tuned. I'd tune them with a retrieval evaluation (recall@k against chunk size).

**Q8. Why characters and not tokens?**
It's simpler, and it doesn't depend on a tokenizer. Roughly 4 characters make one token, so 1000 characters is about 250 tokens, well under the embedding model's input limit. In production I'd measure in tokens so the context budget is exact.

**Q9. Why chunk per page? What do you lose?**
You get exact page citations. You lose continuity for content that spans a page break: there's no overlap across the break, and a sentence can be cut in two. An alternative is to chunk across pages and store a `page_start`/`page_end` range, or a character-offset-to-page map.

**Q10. Fixed-size vs semantic vs structure-aware chunking?**
This is recursive, structure-aware splitting (paragraph → sentence → hard cut). Semantic chunking (splitting where the embedding similarity between adjacent sentences drops) costs embeddings at ingest time. For policy docs, heading-aware chunking (keeping section numbers and titles) would probably help most. For transcripts, chunking by speaker turn would.

**Q11. How do you handle tables, scanned PDFs, headers and footers?**
They're not handled. pypdf reads only the text layer. Scanned pages are skipped silently, tables flatten into jumbled text, and repeated headers and footers end up in chunks (which adds noise and makes near-duplicate embeddings). Fixes: an OCR fallback (Tesseract or Document AI), a layout-aware parser (Unstructured, Docling, LlamaParse), and removing lines that repeat on most pages.

**Q12. Is ingestion idempotent?**
It's *meant* to be, through deterministic IDs, but `add()` ignores existing IDs, so it isn't really (see §6.1). The correct version uses `upsert`, deletes chunks by `source_file` before re-adding, and ideally keeps a content hash per file to skip unchanged files (incremental indexing).

**Q13. Why is the doc type classified by filename instead of by content or an LLM?**
It's free, deterministic, and runs in microseconds, and the Router reuses it for the roster. The trade-off is brittleness. In production, the doc type would come from the source system's metadata (a case-management system), not from filename conventions.

### C. Embeddings
**Q14. What are asymmetric embeddings and why use them?**
Queries are short questions and documents are long passages, so they're phrased differently. With `task_type=RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`, the model maps them into the space so that a question lands near the passages that answer it, rather than near other questions. Using the wrong task type quietly lowers recall.

**Q15. Why 768 dimensions instead of 3072?**
`gemini-embedding-001` is Matryoshka-trained: the first N dimensions are a usable lower-resolution embedding. 768 dimensions take 4× less storage and memory and give faster ANN search, with a small quality loss. At 354 chunks storage doesn't matter. The choice anticipates scale.

**Q16. Why normalize, and is it necessary with Chroma cosine?**
Truncated Matryoshka vectors aren't unit length. Cosine similarity is invariant to scale, and Chroma's cosine space normalizes anyway, so for *this* store it's redundant. It protects correctness if you switch to an inner-product index (FAISS `IndexFlatIP`, pgvector `<#>`), where unnormalized vectors would rank wrongly.

**Q17. What happens if you change the embedding model?**
Every vector becomes incompatible, so you need a full re-index. In production you'd version the collection (`pdf_chunks_v2`), backfill it, run evaluation, and cut over atomically, or run both in parallel for A/B testing.

### D. Retrieval
**Q18. Why only dense retrieval? When does it fail?**
Dense retrieval is good at paraphrase but weak at exact tokens: statute numbers ("Standard 4.020"), names, dates ("5/21"), and acronyms. Hybrid search (BM25 + dense, fused with RRF) is the standard fix, and this corpus, full of numbered standards and dated transcripts, would benefit from it.

**Q19. How are results from multiple sub-queries merged? Is max-score right?**
They're deduped by `chunk_id`, keeping the maximum cosine score, then sorted globally. The weakness is that cosine scores aren't comparable across queries, so one sub-query with generally higher scores can take all 8 slots and starve the others. That defeats the point of decomposition. Better options are RRF (rank-based), round-robin per sub-query, or a cross-encoder reranker over the union.

**Q20. Why TOP_K=5 and MAX_CONTEXT_CHUNKS=8?**
5 per sub-query × up to 3 sub-queries gives 15 candidates, trimmed to 8 to bound prompt size and limit the "lost in the middle" effect. These are heuristics. Retrieve more (k=20–50), rerank, and keep 5–8: that's the usual production pattern.

**Q21. How was 0.5 chosen as the threshold? What does low_confidence actually do?**
It wasn't calibrated. It only *adds a hedging instruction* to the prompt and never blocks an answer. Calibration means plotting the score distributions of relevant and irrelevant chunks on a labeled set and choosing the threshold for your precision and recall target. Cosine values are specific to the model, so the threshold doesn't carry over between embedding models.

**Q22. Pre-filtering vs post-filtering in Chroma?**
Chroma applies the `where` filter during the search, so you get up to k results from the filtered subset, which avoids the case where post-filtering leaves 0 results. With very selective filters on large HNSW indexes, recall can still drop. Systems like Qdrant and Weaviate handle filtered HNSW explicitly.

**Q23. Why does the client filter exist, and what's wrong with it?**
It stops Robert's transcripts from answering a question about Nathan, which is both a precision gain and a basic privacy guard. But it's applied globally, so it also removes all policy docs, and it supports only one client (§6.3–4). It's also **not** a security boundary: it depends on the LLM extracting the name. Real access control must be enforced server-side from the user's identity.

**Q24. Why is there no reranker?**
For simplicity and latency. A cross-encoder reranker (Cohere Rerank, bge-reranker, or a Vertex ranking API) on the top 20–50 is usually the single largest precision gain in a RAG pipeline. It would also fix the score-comparability problem in Q19.

### E. Generation & grounding
**Q25. How do you prevent hallucination?**
In layers: (1) routing defaults toward retrieval; (2) a strict grounded prompt ("ONLY the excerpts", "say so if insufficient"); (3) low temperature (0.2); (4) mandatory citations, so claims can be audited; (5) low-confidence hedging; (6) a post-hoc critic with one corrective retry. None of these is a guarantee. The critic is itself an LLM.

**Q26. Why is the context placed before the question?**
For long prompts, putting the question at the end improves instruction following, because the model reads the evidence with the question still ahead of it. Order chunks deliberately too. Here they're sorted by score, so the best chunk comes first.

**Q27. How do you verify citations are correct?**
Currently nothing checks that `[3]` actually supports the claim it's attached to. The critic sees the excerpts by number but grades only overall support. Options: parse the citations and check each claim against its excerpt with an NLI model or an LLM; check that cited numbers are within range; give the critic the source and page as well.

**Q28. What if a document contains "Ignore previous instructions…"?**
It would be injected straight into the synthesizer prompt. Mitigations: delimit the context clearly and tell the model the excerpts are untrusted data; sanitize at ingestion; keep generated output from triggering tools or actions (this system has none, which limits the impact).

### F. The Critic / self-correction
**Q29. Is LLM-as-judge reliable here?**
It's moderately reliable. The same model judging its own output has correlated errors and a known bias toward its own text. Improvements: use a different (or stronger) model family as the judge; run claim-level NLI checks; calibrate the critic against human labels; set temperature 0 (already done).

**Q30. Does the retry actually improve answers?**
Not always. Because of the merge-then-truncate behavior (§6.2), the new evidence may never reach the synthesizer. I'd measure it: the share of retries where the verdict improves, and the share where the context actually changed.

**Q31. Why does the critic return a follow-up query instead of the orchestrator re-planning?**
The critic knows *what is missing* (the specific unsupported claim), so its query is targeted. Re-running the router would mostly regenerate the same sub-queries.

### G. Evaluation & observability
**Q32. How do you know it works? What metrics would you track?**
There's no evaluation in the repo yet, and that's the biggest gap. I'd build a golden set of 50–200 questions, each with gold chunks and gold answers, covering policy, per-client, comparison, and out-of-scope questions. Retrieval metrics: recall@k, MRR, nDCG. Generation metrics: faithfulness/groundedness, answer relevance, citation precision, abstention correctness. Tools: RAGAS, TruLens, or an LLM judge calibrated against humans. Run it in CI on every change to a prompt, the chunker, or config.

**Q33. What do you log in production?**
`State.trace` is a start: agent inputs and outputs. Add latency per agent, token counts and cost, retrieved chunk IDs and scores, router plan distribution, verdict distribution, retry rate, and user feedback, exported to OpenTelemetry or Langfuse. Log at a level that's careful with PII, given the transcripts.

### H. Scaling & operations
**Q34. How would this scale to 10M chunks or 1,000 QPS?**
Ingestion becomes a queue-based worker pipeline with rate-limited, retrying embedding calls and incremental indexing. The vector DB moves to a managed or distributed one (Qdrant, Weaviate, pgvector, Vertex Vector Search), possibly with quantization or IVF-PQ. Query serving runs behind a stateless API (FastAPI) with async Chroma or DB calls, connection pooling, embedding and answer caches, and streaming responses. The router moves to a cheap model, and the critic moves off the critical path.

**Q35. Why Chroma over FAISS or pgvector?**
Chroma persists locally and stores vectors, documents, and metadata together with filtering built in, which is ideal for a local prototype. FAISS is only an index: you'd build persistence and metadata yourself. pgvector is attractive when you already run Postgres and want SQL joins and access control in one place. The `ChromaStore` wrapper keeps a swap contained.

**Q36. What about rate limits and API failures?**
They aren't handled yet. I'd add `tenacity`-style exponential backoff with jitter on 429/5xx errors, an `asyncio.Semaphore` to limit batch concurrency, per-call timeouts, and graceful degradation (skip the critic if it fails; return the answer with an "unverified" flag).

### I. Security, privacy, domain
**Q37. These are client transcripts. What are the privacy concerns?**
The data is likely sensitive, and possibly regulated, criminal-justice data. Concerns: sending raw text to a third-party API (check the data-processing terms; use an enterprise endpoint or Vertex with no training on data); redacting PII before embedding where possible; per-user, per-client access control enforced in the vector query (not by the LLM); audit logs; retention policy; encryption at rest for `storage/`.

**Q38. What should happen for a question outside the corpus?**
The router should choose `NO_RETRIEVAL` only for chit-chat, and the synthesizer labels general-knowledge answers. For a domain question the corpus can't answer, the grounded prompt should abstain. The gap is the empty-retrieval branch (§6.5).

### J. Code-level probes
**Q39. Walk me through `_append_unit`. Can a chunk exceed `chunk_size`?**
Yes. After a flush it returns `tail + "\n\n" + unit` without checking the length, and a single sentence longer than `chunk_size` passes through whole. In the measured corpus, 35 of 354 chunks are oversized, with a maximum of 1284 characters.

**Q40. Why is `similarity = 1 - distance`?**
Chroma's cosine distance is `1 − cos θ`. Note that with `l2` or `ip` space this formula would be wrong, which is a coupling between `chroma_store.py` and the collection metadata.

**Q41. Is the retriever really parallel?**
Partially. The embedding calls are awaited concurrently through `gather`, but `store.query` is a synchronous call that blocks the event loop. It's harmless locally and a problem in a server.

**Q42. How would you test this?**
- Unit tests: chunker (boundaries, overlap, sizes), classifier regex, router and critic `_parse` with malformed JSON, the merge and dedupe logic.
- Contract tests: agents with a fake `generate`/`embed`.
- Integration test: a tiny fixture PDF with ingest → query.
- Evaluation suite (Q32) for quality.

Making `config` not raise at import time is a prerequisite.

---

## 8. What "production-grade" would add (roadmap answer)

1. **Evaluation harness first**, so every later change is measured.
2. **Hybrid retrieval + reranker**, the largest quality gain.
3. **Fix fusion** (RRF / per-sub-query quotas) and **per-sub-query filters** (`client` OR `policy`).
4. **Ingestion hardening**: `upsert` + delete-by-source, content hashes, OCR/layout parser, header/footer stripping, token-based chunking, section metadata.
5. **Resilience**: retries/backoff, semaphores, timeouts, async DB calls.
6. **Conversation memory** with query rewriting ("what about Robert?" → full standalone question).
7. **Citation verification** at claim level, and a different model as the judge.
8. **Serving**: FastAPI, streaming, caching, auth + server-side access control on client data.
9. **Observability**: OpenTelemetry/Langfuse traces, cost dashboards, feedback capture.
10. **Privacy**: PII redaction, data-residency/DPA review, audit logging.

---

## 9. Cheat sheet

| Item | Value |
|---|---|
| Corpus | 10 PDFs (5 policy, 5 transcripts: nathan ×3, robert ×2), 107 pages |
| Chunks | 354 (228 policy, 126 transcript), median 690 chars, max 1284 |
| Policy chunking | 1000 chars / 200 overlap |
| Transcript chunking | 500 chars / 100 overlap |
| Embedding model | `gemini-embedding-001`, 768-d, L2-normalized, asymmetric task types |
| Embedding batch | 64 |
| Generation / router model | `gemini-3.6-flash` |
| Temperatures | Router 0.0, Critic 0.0, Synthesizer 0.2 |
| Vector store | Chroma `PersistentClient`, HNSW, cosine, `storage/` |
| TOP_K per sub-query | 5 |
| Max sub-queries | 3 |
| Max context chunks | 8 |
| Similarity threshold | 0.5; low confidence if < 2 chunks clear it |
| Critic retries | 1 |
| Plans | `NO_RETRIEVAL`, `SINGLE_RETRIEVAL`, `MULTI_RETRIEVAL` |
| Verdicts | `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `NOT_APPLICABLE` |
| Calls per query | 4–6 normal, up to 9 with retry |

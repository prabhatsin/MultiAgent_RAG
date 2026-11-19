# Topic Fit Map

Where the techniques I've been reading about fit in *this* codebase, and where
they don't. I checked each one against the actual corpus and code.

**Corpus facts that drive these decisions** (`data/`):
- 1 long structured doc: *2022 Colorado Community Corrections Standards*, 63 pages
  (its PDF outline has only 2 entries, so it has no usable built-in TOC)
- 4 short policy docs, 1–3 pages each
- 5 client transcripts (nathan ×3, robert ×2), 5–10 pages each, conversational

| Topic            | Fit        | Where it lives                                                     |
|------------------|------------|--------------------------------------------------------------------|
| BM25             | ✅ Strong  | new `src/vectorstore/bm25_index.py`, called from `RetrieverAgent`  |
| RRF              | ✅ Strong  | replaces the merge in `src/agents/retriever.py:33-43`              |
| Reranking        | ✅ Strong  | after fusion, before the `MAX_CONTEXT_CHUNKS` cut                  |
| PageIndex        | 🟡 Partial | only for the 63-page Standards doc; ingestion + a new retrieval route |
| AST autodiscovery| ❌ No fit  | the corpus has no code; see below                                  |

---

## 1. BM25 (keyword / sparse retrieval) ✅

**Why it fits:** right now retrieval is dense-only (`ChromaStore.query`). Embeddings
are weak on exact tokens, and this corpus is full of them: standard/section
numbers, acronyms (UA, LSI-R, etc.), dates in transcripts, drug names, and
client names. A question like "what does standard 3-040 say" is exactly where
BM25 beats cosine similarity.

**Where:**
- **Build:** `src/ingestion/ingest.py` after `_chunk_all_pages` (line 56). Tokenize
  `texts` and persist the index next to Chroma (e.g. `storage/bm25.pkl`). Another
  option: build it lazily at startup from `collection.get()`. The corpus is only a
  few hundred chunks, so in-memory is fine.
- **Query:** `RetrieverAgent._retrieve_one` (`retriever.py:62`) runs a BM25 search
  alongside the dense search for each sub-query.
- **Gotcha:** BM25 has to apply the same `client_name` filter that Chroma gets through
  `where` (`retriever.py:27`). Otherwise a "robert" question pulls in nathan's transcript.
- **Lib:** `rank_bm25` (pure Python, no heavy deps).

## 2. RRF (Reciprocal Rank Fusion) ✅

**Why it fits:** once BM25 exists you have to fuse two ranked lists, and their
scores can't be compared (BM25 is unbounded, cosine is 0–1). RRF uses only rank:
`score = Σ 1 / (60 + rank)`. You also get a second fit for free. The retriever
already merges results across multiple sub-queries (MULTI_RETRIEVAL) by taking
the max cosine score (`retriever.py:36-42`). RRF is the principled way to fuse
**sub-queries × retrievers** in a single step.

**Where:** replace the merge/sort block in `src/agents/retriever.py:33-43` with a
`rrf_fuse(ranked_lists)` helper (e.g. `src/retrieval/fusion.py`).

**Gotchas:**
- `low_confidence` (`retriever.py:45-48`) compares `score` against
  `MIN_SIMILARITY_THRESHOLD = 0.5`, which is a *cosine* scale. RRF scores are ~0.01–0.03.
  Add a `dense_score` field to `RetrievedChunk` (`chroma_store.py:18`) and keep the
  confidence check on that, or move it to reranker scores (next section).
- The critic retry re-runs the retriever and merges in the previous
  `state.retrieved_chunks` with their old scores (`retriever.py:33`). Decide
  whether earlier chunks join the fusion as their own ranked list.

## 3. Reranking ✅

**Why it fits:** the pipeline sends the top 8 chunks to the Synthesizer, and the
Critic retries when the answer is unsupported. Better precision in those 8 chunks
means fewer critic retries, which is a measurable win (tokens and latency saved).

**Where:**
- Widen recall first: `config.TOP_K` 5 → ~20 per list.
- Then fuse (RRF) → **rerank the top ~25 against the original `state.query`** (not
  the sub-query) → keep `MAX_CONTEXT_CHUNKS = 8`. All of this is inside
  `RetrieverAgent.run`, or in a `src/retrieval/reranker.py` that it calls. It's a
  pipeline step, not a decision-maker, so it doesn't need to be its own agent in
  `orchestrator.py`.
- Use the reranker score for `low_confidence`. It's a far better relevance signal
  than raw cosine.

**Options:**
- Cross-encoder (`BAAI/bge-reranker-base`, `ms-marco-MiniLM`) through
  `sentence-transformers`: best quality and runs locally, but adds torch.
- LLM rerank with Gemini (one JSON-scored call): no new deps, but adds an API call.
- Worth comparing both in the eval harness. That comparison is a good interview story.

## 4. PageIndex (reasoning-based, vectorless tree retrieval) 🟡

**What it is:** build a hierarchical table-of-contents tree of a long document
(sections → subsections, each with a summary). At query time an LLM *navigates*
the tree to the right section instead of doing similarity search.

**Where it fits:** only the **63-page Colorado Standards** doc. It's long, formally
structured, and questions about it are usually "what does section X require",
where chunk-level similarity loses the section context.

**Where it does NOT fit:** transcripts (conversational, no hierarchy) and the
1–3 page policies (the whole doc fits in context anyway). Don't force it there.

**Where:**
- **Ingestion:** new `src/ingestion/tree_builder.py` → `storage/trees/<doc>.json`.
  The PDF outline is nearly empty, so the tree must come from heading regex and/or
  an LLM pass over the pages.
- **Query:** the router (`router.py`) gains a route, e.g. a `doc_type` hint or a
  `TREE_NAVIGATION` plan, and a `TreeNavigatorAgent` walks the tree and returns
  section text as `RetrievedChunk`s, so Synthesizer and Critic stay unchanged.
  `orchestrator.py` picks the route.

**Cheaper step first (same root problem):** `chunker.py` splits per page, so a
mid-section chunk doesn't know which section it belongs to. Two smaller fixes:
- detect headings and store `section_title` in metadata (`ingest.py:71-79`)
- prepend the heading to the chunk text before embedding ("contextual chunk headers")

Do this first, then try full PageIndex and compare both in the eval harness.

## 5. AST autodiscovery ❌

**What it is:** parsing *source code* with Python's `ast` module, either to chunk
code by function/class for code-RAG or to auto-register plugins/tools.

**Why it doesn't fit:**
- The corpus is PDFs. There's no code to parse.
- Auto-registering agents conflicts with a deliberate design choice:
  `orchestrator.py` is *intentionally* the only place that knows the agent
  sequence (see its docstring). Hiding that behind discovery makes the
  system harder to reason about without adding capability.

**When it would fit:** if you later add a second corpus that is code (even "ask
questions about this repo itself"). Then AST-based chunking would go in a new
loader next to `src/ingestion/loader.py`. That's a separate project, not an
upgrade to this one.

---

## Suggested build order

1. **Eval harness** first. Without it you can't show that any of the above helped.
2. **BM25 + RRF** together as one upgrade (hybrid search).
3. **Reranking** on top of the fused list, and move `low_confidence` onto it.
4. **Contextual chunk headers / section metadata.**
5. **PageIndex** for the Standards doc, as an A/B experiment against step 4.

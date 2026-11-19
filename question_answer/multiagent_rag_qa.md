# Multi-Agent RAG: Q&A Notes

Questions and answers about this project's architecture: multi-agent RAG, shared state, what the "agents" really are, and the Router.

**Contents**
1. [What is multi-agent RAG, and is this project multi-agent RAG?](#q1)
2. [What exactly is the shared state, and how is it implemented? Is Redis used?](#q2)
3. [Are the Router, Critic, Synthesizer… separate agent loops?](#q3)
4. [Explain "Router classifies queries as no-, single- or multi-retrieval and generates sub-queries"](#q4)
5. [How is it decided whether a query needs no, single or multi retrieval?](#q5)

---

<a id="q1"></a>
## Q1. What is multi-agent RAG? Is this project multi-agent RAG? (It's on my resume.)

### What is multi-agent RAG?

**Plain RAG** follows a fixed recipe: embed the question, fetch the top-k chunks, have the LLM write an answer. It runs the same way every time and makes no decisions.

**Agentic RAG** puts an LLM in charge of decisions inside the pipeline. It decides whether to retrieve, what to search for, and whether the answer is good enough or it should search again.

**Multi-agent RAG** splits that work across **several specialized agents**. Each has its own role, prompt and sometimes its own tools or data source. They coordinate through an orchestrator or supervisor, or by passing messages. The typical traits are:
1. **Role specialization**: planner, retriever, writer, verifier, or one agent per data source.
2. **LLM-driven decisions that change control flow**, not just LLM calls in a fixed order.
3. **Coordination**: a supervisor or orchestrator, shared state, or messages between agents.
4. Often **tool use and iteration**: agents choose tools and loop until they're satisfied.

There's no strict standard definition. In industry the term covers everything from "a pipeline with an LLM router" up to fully autonomous agent teams.

### Is this project multi-agent RAG?

**Yes, as an orchestrated multi-agent RAG system, and you can defend that. It is not the autonomous kind.** (Q3 below revises this under a stricter definition of "agent".)

| Agent | Makes an LLM decision? | Does it change the flow? |
|---|---|---|
| Router | ✅ Whether to retrieve, single vs multi search, sub-queries, client filter | ✅ Can skip retrieval entirely |
| Critic | ✅ Judges whether the answer is supported and writes a follow-up query | ✅ Triggers the corrective retry |
| Synthesizer | Generates text only | ❌ |
| Retriever | ❌ No LLM; plain code that embeds and searches | ❌ |

**What supports the label:** four role-specialized components, a shared state they all work on, two agents whose LLM decisions really change what happens next, and a self-correction loop. That's the same pattern as published approaches like Corrective RAG (CRAG) and Self-RAG.

**What a sharp interviewer will push on:**
- The **sequence is hardcoded** in `orchestrator.py`. No LLM chooses which agent runs next.
- The **"Retriever Agent" isn't an agent.** It's a function with no LLM and no choices.
- The agents have **no tool calling** and never talk to each other directly.
- Iteration is limited to **one retry**, so the agents have very little autonomy.

### Resume advice

Suggested wording:
> *Built a multi-agent RAG system (Router, Retriever, Synthesizer, Critic) over policy documents and client transcripts. LLM-based query routing and decomposition with metadata filtering; grounded, cited generation; and a Critic agent that verifies faithfulness and triggers corrective re-retrieval. Implemented in plain asyncio with no framework.*

**If they ask "Is this really multi-agent?":**
> "It's an orchestrated multi-agent design. The agents are specialized roles that share one state object. I chose a deterministic orchestrator on purpose so cost and latency stay bounded. The Router and Critic make LLM decisions that change the control flow. The Retriever is deliberately plain code, because retrieval doesn't need LLM judgment. The next step toward more autonomy would be a supervisor LLM that chooses which agent runs next."

**Upgrades that would make the label clearly true:**
1. **One retriever agent per data source**: a Policy agent and a Transcript agent, with the Router dispatching to one or both. This would also fix the bug where filtering by client hides all the policy documents.
2. **Give the Retriever tools to choose from**: vector search, BM25 keyword search, or a lookup by date or client. The LLM picks which to use.
3. **A supervisor loop**: an LLM decides the next step ("retrieve more", "answer", "verify") until it's confident or hits a step budget.

---

<a id="q2"></a>
## Q2. What exactly is the shared state, and how is it implemented? Is Redis used?

### What it is

The "shared state" is **one plain Python object in memory**: the `State` dataclass in `src/agents/state.py`. There's **no Redis, no database and no message queue**. It's just a Python object passed from agent to agent within a single process.

```python
@dataclass
class State:
    query: str
    plan: str = ""
    sub_queries: list[str] = field(default_factory=list)
    client_name_filter: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    low_confidence: bool = False
    answer: str = ""
    critique: Critique | None = None
    retry_count: int = 0
    trace: list[AgentEvent] = field(default_factory=list)
```

Think of it as a **notepad that's handed down the line**. Each agent reads what earlier agents wrote, adds its own part, and passes it on. This is the "blackboard" pattern.

### How it works mechanically

`src/orchestrator.py:30-57`:

```python
state = State(query=query)                 # 1. created fresh for each question
state = await self._router.run(state)      # 2. passed in, modified, returned
state = await self._retriever.run(state)
state = await self._synthesizer.run(state)
state = await self._critic.run(state)
return state                               # 3. handed to cli.py to print, then discarded
```

- **Python passes objects by reference.** Each agent receives the *same object* and changes its fields directly (`state.plan = ...`). It then returns that same object. The `return state` is a convention that keeps the interface uniform, not a copy.
- **Lifetime:** one question. A new `State` is created for every question and is thrown away once the answer is printed. Nothing is saved, and nothing carries over between questions. That's why the REPL has no conversation memory.
- **Scope:** one process. The agents run one after another, so only one agent touches the state at any moment. No locks or other synchronization are needed.

### Who writes and reads each field

| Field | Written by | Read by | Purpose |
|---|---|---|---|
| `query` | Orchestrator (at creation) | Router, Synthesizer, Critic | The user's question |
| `plan` | Router | Orchestrator | Whether to skip retrieval (`NO_RETRIEVAL`) |
| `sub_queries` | Router; Orchestrator on retry | Retriever | What to search for |
| `client_name_filter` | Router | Retriever | Metadata filter (`where`) |
| `retrieved_chunks` | Retriever | Synthesizer, Critic, CLI | Evidence, with source, page and score |
| `low_confidence` | Retriever | Synthesizer | Tells the Synthesizer to hedge |
| `answer` | Synthesizer | Critic, CLI | The answer text |
| `critique` | Critic | Orchestrator, CLI | Verdict and follow-up query, which drive the retry |
| `retry_count` | Orchestrator | Orchestrator, CLI | Limits retries to 1 |
| `trace` | Every agent, via `state.log()` | CLI | Log of each agent's inputs and outputs, for observability |

The key point: **agents never call each other.** The Router doesn't know the Retriever exists. It writes `sub_queries`, and the Retriever later reads `sub_queries`. The state is the only channel between them, which keeps the agents decoupled and easy to test on their own.

### How the state changes for one question

Example question: "What did Nathan say about his job?"

```
After creation:    {query: "What did Nathan say about his job?", everything else empty}
After Router:      + plan=SINGLE_RETRIEVAL, sub_queries=["Nathan job"], client_name_filter="nathan"
After Retriever:   + retrieved_chunks=[8 chunks with scores], low_confidence=False
After Synthesizer: + answer="Nathan said ... [1][3]"
After Critic:      + critique={verdict: PARTIALLY_SUPPORTED, followup: "Nathan employment status"}
Retry:             retry_count=1, sub_queries=["Nathan employment status"]
                   → Retriever merges new chunks into the existing retrieved_chunks
                   → Synthesizer overwrites answer → Critic overwrites critique
trace:             grows by one entry at every step (5 entries, or 8 with a retry)
```

### Why there's no Redis

| Redis would be needed if… | This project |
|---|---|
| Agents ran in **different processes or machines** | All in one Python process |
| State had to **survive a crash or restart** (checkpoint and resume) | A query takes seconds; if it fails, you rerun it |
| State had to **outlive the request** (chat memory, sessions) | No memory between questions |
| **Many workers or servers** needed to see the same state | Single-user CLI |
| Agents ran **in parallel** and needed coordinated writes | Strictly sequential |

Adding Redis here would only add network latency and serialization work, and gain nothing.

### When you would move it out of memory
- **Conversation memory:** store the previous turns in Redis, keyed by `session_id`, so "What about Robert?" can be rewritten into a full question.
- **Running as a web API with many workers:** each request still gets its own in-memory `State`. Only session memory goes into Redis.
- **Long or human-in-the-loop workflows:** save the state after each agent (in Redis or Postgres) so a run can pause and resume. LangGraph's "checkpointer" does exactly this, and its `State` + nodes model is almost the same as this design.
- **Agents as separate services:** serialize the state to JSON and pass it in a message queue (Kafka, SQS), or store it centrally under a `request_id`.

### Weak spots an interviewer might poke at
1. **It's mutable and has no checks.** Any agent could overwrite any field. The "who writes what" table is a convention, not something the code enforces. A stricter design has each agent return only the fields it changes, and the orchestrator merges them (LangGraph's reducer pattern), or uses frozen dataclasses.
2. **It isn't safe for parallel agents.** If two agents ran at the same time with `asyncio.gather`, both could write to `retrieved_chunks` and clash. Today that can't happen because the agents run one after another.
3. **The trace disappears.** `trace` is only printed to the terminal. In production you'd send it to a tracing tool (Langfuse, OpenTelemetry) under a `request_id`.

**One-line interview answer:**
> "The shared state is an in-process Python dataclass, created fresh for each request. It's passed by reference through agents that run in sequence, and each agent reads and writes its own fields. It's the blackboard pattern, so agents are decoupled and never call each other. There's no Redis because everything is single-process, sequential and lives for one request. I'd add Redis or a checkpointer only for session memory, distributed agents or resumable workflows."

---

<a id="q3"></a>
## Q3. An agent is usually a loop + tools + schema + a harness around it. So are the Router, Critic and Synthesizer separate loops?

### Short answer

**No. None of them is a loop.** By that definition (a loop, plus tools, plus a schema, plus a harness around it), **none of the four is a real agent.** Each one is a Python class with a `run()` method that makes **at most one LLM call** and then returns. The LLM never chooses an action, never calls a tool, and never loops.

### What each "agent" actually is

| "Agent" | LLM calls | Loop inside? | Tools? | Schema? | What it really is |
|---|---|---|---|---|---|
| Router | 1 | ❌ | ❌ | ✅ JSON output schema, written in the prompt and checked by `_parse` | One classification/extraction call with structured output |
| Retriever | **0** | ❌ | ❌ | ❌ | Ordinary code: embed, then `chroma.query`, then merge |
| Synthesizer | 1 | ❌ | ❌ | ❌ (free text) | One generation call with the chunks in the prompt |
| Critic | 1 | ❌ | ❌ | ✅ JSON output schema | One LLM-as-judge call with structured output |

Mapping the four parts of an agent onto this code:
- **Loop:** there's exactly one, and it's in the **orchestrator** (`orchestrator.py:45-55`): a hardcoded `while` that runs at most once. Python decides whether to loop, by checking the critic's verdict. The LLM doesn't decide.
- **Tools:** there's no function calling anywhere. The vector search is called *by code*. The LLM never sees a list of tools and never asks to use one.
- **Schema:** the Router and Critic use **output** schemas (JSON mode). That's structured output, not a tool schema.
- **Harness:** the `Orchestrator`, `State` and `gemini_client` are the harness, but they wrap fixed steps, not agent loops.

So the system is **one fixed workflow with LLM-powered steps**. In Anthropic's "Building Effective Agents" terms it's a **workflow, not an agent**: the code follows predefined paths, whereas an agent directs its own process and tool use. It combines three standard workflow patterns:
1. **Routing**: the Router picks a branch.
2. **Prompt chaining**: Router, then Synthesizer, then Critic, each feeding the next.
3. **Evaluator-optimizer**: the Critic evaluates and triggers one redo.

The word "agent" here follows the CrewAI-style habit of calling "an LLM with a role prompt" an agent. It doesn't mean an autonomous loop.

### What a real agent would look like in this project

Take the Retriever. As a real agent it would be a loop in which the LLM chooses tools until it decides it has enough evidence:

```python
tools = [vector_search(query, client_name=None, doc_type=None),
         keyword_search(query),            # BM25
         get_page(source_file, page_number),
         list_documents()]

messages = [system_prompt, user_question]
for step in range(MAX_STEPS):                       # loop (bounded)
    resp = llm(messages, tools=tools)               # the LLM decides
    if resp.tool_calls:
        for call in resp.tool_calls:
            result = run_tool(call)                 # harness runs the tool
            messages.append(tool_result(call, result))
    else:
        return resp.final_evidence                  # the LLM decides it's done
```

Now it fits the definition: a loop, tools, a schema for each tool, and a harness. It could decide on its own to "search Nathan's transcripts, *and* the check-in policy, then open page 4 of the policy". The current fixed pipeline can't do that, which is the root cause of the client-filter bug.

A real **multi-agent** version would be several of these loops, such as a Policy agent and a Transcript agent, with a supervisor agent calling them as tools.

### What this means for the resume (revising Q1)

By the strict loop + tools + schema + harness definition, which is the one a strong interviewer will use, **this project isn't multi-agent; it's an agentic RAG workflow.** Two options:

1. **Change the wording to match what it is:**
   > *Built an agentic RAG pipeline with LLM-based query routing and decomposition, metadata-filtered retrieval, grounded cited generation, and an LLM critic that verifies faithfulness and triggers corrective re-retrieval (routing + evaluator-optimizer patterns, plain asyncio).*
2. **Keep "multi-agent" and make it true:** turn the Retriever into a tool-using loop, as sketched above, preferably as Policy and Transcript sub-agents under a supervisor. That's roughly a day of work on this codebase.

**If you keep the current wording and get asked "are these really agents?":**
> "Strictly, no. They're LLM-powered steps in a fixed workflow: routing, chaining and an evaluator-optimizer loop. Only the orchestrator loops, and it's bounded. I chose a workflow because cost and latency stay predictable for a narrow Q&A task. Turning the Retriever into a tool-using agent would let it combine policy and transcript searches on its own."

---

<a id="q4"></a>
## Q4. "Router classifies queries as no-, single- or multi-retrieval and generates sub-queries": what does this mean? What are sub-queries?

### What the line means

When a question arrives, the **Router** makes one LLM call that answers two things before any search happens:

1. **Do we need to search the documents at all, and how many searches do we need?** That's the classification: no, single or multi retrieval.
2. **What exactly should we search for?** Those search strings are the **sub-queries**.

It's the planning step. The Router does no searching itself; it only decides how the search should be done.

### The three classes, with examples from the documents

#### 1. `NO_RETRIEVAL`: no need to search
For greetings or general-knowledge questions that have nothing to do with the PDFs.
```
User:   "hi, how are you?"
Router: {"plan": "NO_RETRIEVAL", "sub_queries": [], "client_name": null}
```
→ The Retriever is skipped, and the Synthesizer answers directly, saying that the answer comes from general knowledge.
**Why it exists:** searching for "hi" would return 8 random chunks and could confuse the answer. It also saves an embedding call and a search.

#### 2. `SINGLE_RETRIEVAL`: one search is enough
For a question about one topic.
```
User:   "um so what are the rules for checking in?"
Router: {"plan": "SINGLE_RETRIEVAL",
         "sub_queries": ["check-in rules and requirements"],
         "client_name": null}
```
→ One search. The sub-query is a **cleaned-up version** of the question with the filler ("um so", "what are the") removed, so its embedding is focused on the topic.

#### 3. `MULTI_RETRIEVAL`: the question has several parts
For comparisons, multi-part questions or broad questions.
```
User:   "How does the grievance process compare to the appeal process, and what are the check-in rules?"
Router: {"plan": "MULTI_RETRIEVAL",
         "sub_queries": ["grievance filing process",
                         "appeal process steps",
                         "check-in rules"],
         "client_name": null}
```
→ Three separate searches run in parallel (capped at `MAX_SUB_QUERIES = 3`), and the results are merged.

### What sub-queries are, and why they're needed

A **sub-query** is a short, focused search string covering **one piece** of the user's question. This splitting is called **query decomposition**.

**Why not search the whole question at once?** Because of how embeddings work. The whole question is turned into **one** vector, which is roughly a blend of all its topics:

```
"grievance vs appeal, and check-in rules"  →  one vector sitting BETWEEN
                                              grievance, appeal and check-in
```

That blended vector lands somewhere in the middle and isn't close to any single topic. The top 5 results might all be grievance chunks and leave out check-in completely, or they might be mediocre matches for all three topics.

With sub-queries, each one gets its **own** vector that lands right on its topic:
```
"grievance filing process" → top 5 grievance chunks
"appeal process steps"     → top 5 appeal chunks
"check-in rules"           → top 5 check-in chunks
                            → merge, remove duplicates, keep the best 8 → Synthesizer
```
So each part of the question gets its own evidence. That's the whole point.

### The third output: `client_name`

The Router also checks whether the question is about a known client. The roster is built from filenames, which here gives `nathan` and `robert`.
```
User:   "What did Nathan say about his job?"
Router: {"plan": "SINGLE_RETRIEVAL", "sub_queries": ["Nathan job employment"], "client_name": "nathan"}
```
→ The Retriever searches **only Nathan's chunks** (`where={"client_name": "nathan"}`), so Robert's transcripts can't leak into the answer.
If the LLM returns a name that isn't in the roster, the code discards it (`router.py:97`).

### Where it lives in the code
- **Prompt and rules:** `src/agents/router.py:17-45`
- **The LLM call** (JSON output, temperature 0, so it's repeatable): `router.py:61-67`
- **Validation and fallback:** `router.py:84-107`. If the JSON is broken, the plan is invalid, or a retrieval plan comes with no sub-queries, the Router falls back to `SINGLE_RETRIEVAL` using the user's raw question. It **fails toward searching**, the safe choice for a document Q&A system.
- **Parallel searches:** `retriever.py:29-31` (`asyncio.gather`, one search per sub-query).
- **Merge:** `retriever.py:33-43`. Duplicates are removed by chunk ID, the highest score is kept, everything is sorted, and the top 8 are kept.

### Follow-up questions to be ready for

**"Why use an LLM for routing instead of rules?"**
Rules like "if the question contains 'compare', use multi" break easily. The LLM understands intent, and it also rewrites the queries and extracts the client name in the same call. The cost is one extra LLM call, about 0.5–1 s, on every question.

**"What if the Router gets it wrong?"**
- Says NO_RETRIEVAL when it should search: the answer isn't grounded. That's why the prompt says "when in doubt, prefer retrieval".
- Says SINGLE when it should be MULTI: the blended-vector problem above, so some parts of the question are covered weakly.
- Sends a bad JSON reply: the fallback runs a single search on the raw question.

**"Is there a weakness in how sub-query results are merged?"** (Say this yourself; it shows depth.)
Yes. Results are ranked by raw similarity score across all sub-queries. If one sub-query's results score high overall, they can take all 8 slots and push out the other parts, which defeats the point of splitting. Fixes: reserve a quota per sub-query, or merge by rank with reciprocal rank fusion (RRF) instead of by score.

**"Why a maximum of 3 sub-queries?"**
Each sub-query costs an embedding call and a search, and returns up to 5 chunks. Only 8 chunks fit in the final context, so more than 3 sub-queries would mostly produce chunks that get thrown away.

### A clearer version of the resume line

> *Built an LLM query router that decides whether a question needs document retrieval and decomposes multi-part questions into focused sub-queries (query decomposition), run as parallel vector searches and merged with deduplication; also extracts client entities for metadata-filtered retrieval.*

---

<a id="q5"></a>
## Q5. How exactly is it decided whether a query needs no or single retrieval? The question's content may or may not be in the docs, so how is that decided?

### The key point: the Router does not look at the documents

The Router **never checks whether the answer is in the documents.** It doesn't search Chroma or look at any chunk before deciding. It decides from **only two inputs**:

1. **The question text.**
2. **Its system prompt**, which contains a set of rules, a one-line description of the collection, and the client names.

So the decision is an **LLM's judgment about what kind of question this is**, not a check of what the documents contain. It's a guess about intent.

### What the LLM actually sees

Everything the Router knows about the documents is in this prompt (`router.py:17-45`):

```
You are a query planning agent in a RAG system over a private collection of PDF
documents (case files, program standards, policy guidelines).
                ↑ this one line is ALL it knows about the content

- Use NO_RETRIEVAL only for greetings, chit-chat, or questions that are purely
  about general world knowledge unrelated to any document collection.
- Use SINGLE_RETRIEVAL for a straightforward question answerable by one focused search.
- Use MULTI_RETRIEVAL when the question has multiple distinct parts, asks to
  compare/combine information, or is broad enough that one search won't cover it.
- When in doubt between NO_RETRIEVAL and retrieval, prefer retrieval.
- Known clients in the document collection: nathan, robert.
```

The user's message is just `User question: <question>`. It's called with temperature 0, so the same question gives the same decision every time.

### How each decision is actually made

**NO_RETRIEVAL vs retrieval: "does this sound like it could be about corrections/policy/case files?"**
The LLM compares the question with the one-line description and the rules:
- "hi there" → chit-chat → `NO_RETRIEVAL`
- "What is the capital of France?" → clearly unrelated → `NO_RETRIEVAL`
- "What are the check-in rules?" → sounds like policy → retrieval
- Anything unclear → retrieval, because of the "when in doubt" rule

**SINGLE vs MULTI: based only on the shape of the question**, not the content:
- One topic → `SINGLE`
- "and", "compare", "vs", several questions in one, or a very broad question → `MULTI`

None of these decisions involve reading the documents.

### Where it can go wrong

Since the Router guesses from the question alone, two failures are possible:

**1. A false NO_RETRIEVAL: it sounds general, but the documents cover it.**
> "What is risk-need-responsivity?"

That sounds like a general criminology question, so the LLM *might* choose `NO_RETRIEVAL`. But it's exactly what `8 Principles of Effective Intervention.pdf` covers. The answer would come from general knowledge without citing the documents. This is the more dangerous failure, which is why the prompt keeps NO_RETRIEVAL narrow and says "when in doubt, retrieve."

**2. Retrieval for something the documents don't contain.**
> "What is Colorado's sales tax rate?"

It sounds Colorado-related, so the Router chooses retrieval. That's handled safely afterward (next section).

### The real "is it in the documents?" check happens after retrieval

Vector search **always returns its top-k results, even when nothing is relevant.** It returns the *nearest* chunks, not *matching* ones. So the system checks relevance afterward, using similarity scores:

```
Router: "retrieve"  (a guess, made without looking)
   ↓
Retriever: top-8 chunks + cosine scores
   ↓
fewer than 2 chunks with similarity ≥ 0.5?  →  low_confidence = True   ← evidence-based check
   ↓
Synthesizer: "answer ONLY from the excerpts; say so if they're not enough"
            (+ "hedge explicitly" when low_confidence is set)
   ↓
Critic: is the answer actually supported by the excerpts?
```

So there are **two layers**:

| Layer | Question it answers | Based on |
|---|---|---|
| Router | "Should I *try* searching? How many searches?" | Question text + prompt rules (a guess) |
| Retriever threshold + Synthesizer + Critic | "Did the documents *actually* have it?" | Similarity scores and the retrieved text (evidence) |

### All four outcomes

| | Answer **is** in the docs | Answer **is not** in the docs |
|---|---|---|
| **Router says retrieve** | ✅ Normal path: grounded, cited answer | ✅ Low scores → hedge, "the documents don't say" |
| **Router says NO_RETRIEVAL** | ❌ **Missed:** general-knowledge answer, docs ignored | ✅ Correct: answered from general knowledge and labeled as such |

Only one cell is a real failure, and the "prefer retrieval" rule pushes decisions away from it. Wrongly retrieving costs one search; wrongly skipping costs a grounded answer.

### How production systems do this better

1. **Retrieve first, then decide.** Always run a cheap search and route on the top similarity score: "if the best match is below X, treat the question as out of scope." That uses real evidence instead of a guess.
2. **Give the Router a corpus summary**, such as a list of document titles and topics, so it knows "risk-need-responsivity" is covered.
3. **Semantic router.** Compare the question's embedding to labeled example questions for each route. No LLM call, about 10 ms, and you can tune it.
4. **Trained classifier.** Log real queries, label the right route, and train a small model. It's cheap, fast and measurable.
5. **Measure it.** Build a labeled set of questions with the correct route and track the Router's accuracy, especially the **false NO_RETRIEVAL rate**.

**One-line interview answer:**
> "The Router doesn't know what's in the documents. It's an LLM classifier working from the question and a short description of the collection, biased toward retrieving when unsure. Whether the documents actually contain the answer is decided afterward, from evidence: similarity thresholds flag weak retrieval, the Synthesizer is told to abstain, and the Critic verifies the answer. The weak spot is a false NO_RETRIEVAL. I'd fix it by routing after a cheap probe search, or by giving the Router a summary of the collection's topics."

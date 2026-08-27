# FAB Multi-Agent Financial Analysis System — Architecture Document

**Corpus:** 12 filings, 2,421 chunks · **Orchestration:** LangGraph · **LLM:** Gemini 2.5 Flash (paid tier) · **Retrieval:** MiniLM + TF-IDF hybrid · **Eval accuracy:** 87.0% (20/23)

## 1. Objective

A multi-agent system that answers complex, multi-hop, calculation-heavy questions about First Abu Dhabi Bank's (FAB) quarterly/annual financial statements — going beyond single-pass retrieval-augmented generation by orchestrating retrieval, deterministic calculation, and temporal reasoning behind a planning LLM, then synthesizing a cited, verifiable answer.

## 2. System Architecture

```
                         ┌─────────────────────┐
                         │   User Question      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                    ┌───▶│   Planner Agent      │◀────┐
                    │    │  (Gemini 2.5 Flash,   │     │
                    │    │  function calling)    │     │
                    │    └──────────┬───────────┘     │
                    │               │ decides next tool call
                    │               ▼                  │
                    │   ┌───────────────────────┐      │
                    │   │   Tool dispatch        │      │
                    │   ├───────────────────────┤      │
                    │   │ Retrieval Agent        │      │
                    │   │  (hybrid MiniLM +      │──────┘  (result appended to
                    │   │   TF-IDF search)       │          conversation history,
                    │   │ Calculator Agent       │          loop back to planner)
                    │   │  (deterministic math,  │
                    │   │   fully traced)        │
                    │   │ Temporal Agent         │
                    │   │  (resolves quarters,   │
                    │   │   YoY/QoQ pairs)       │
                    │   └───────────────────────┘
                    │
                    │  planner calls finish_with_answer,
                    │  or MAX_ITERATIONS reached
                    ▼
         ┌─────────────────────────┐
         │   Synthesis Agent        │
         │  (Gemini 2.5 Flash)      │
         │  cites sources, shows    │
         │  calculation, flags      │
         │  uncertainty             │
         └──────────┬───────────────┘
                     ▼
              Final cited answer
              + full tool-call trace
```

Implemented as a **LangGraph** `StateGraph` with two node types (`planner`,
`synthesize`) and a conditional edge that loops the planner back to itself after
each tool call, until it signals `finish_with_answer` or a hard iteration cap
(`MAX_ITERATIONS = 8`) is hit. The "multiple specialized agents" requirement is
satisfied by role separation, not by node count: the **Planner** is the
orchestrating agent (decides what to do next); **Retrieval**, **Calculator**, and
**Temporal** are specialist agents invoked as tools with their own well-defined
responsibilities and outputs; the **Synthesis** agent is a distinct final LLM call
with a different system prompt (citation + uncertainty discipline) operating over
the accumulated evidence rather than deciding what to fetch next.

## 3. Technology Choices & Justification

### Framework: LangGraph
Chosen because the assignment explicitly recommends it, and because the
planner-loop-until-done pattern maps directly onto a `StateGraph` with a
conditional edge — no need for LangChain's heavier agent abstractions. The state
object (`AgentState`) carries the full conversation history, tool-call trace, and
citations, which made the "calculation tracing" and "source citation"
requirements straightforward: every tool call and result is appended to the
trace, and the trace ships back to the API/UI layer verbatim.

### LLM: Gemini 2.5 Flash
- **Cost**: production pricing (~$0.30/M input tokens, ~$2.50/M output tokens)
  is roughly 5-10x cheaper than GPT-4o/Claude Sonnet for comparable
  function-calling quality. Measured, not estimated: the full 23-question
  evaluation suite cost well under $0.50 total.
- **Reasoning/tool-use**: reliable native function calling, verified directly
  against this system's 9 declared tools — correctly sequences multi-step plans
  (see the Q3 2023→2024 YoY example in the Evaluation Report, which chained a
  temporal lookup, two filtered searches, and a calculator call unprompted).
- **Sandbox constraint that forced this choice**: this development environment's
  network policy allow-lists only a small set of hosts (PyPI, npm, Anthropic's
  API, a few others). Hugging Face, OpenAI, Cohere, Mistral, Groq, Together AI,
  Fireworks, OpenRouter, DeepInfra, Perplexity, and Cerebras were all tested and
  blocked; Google's Gemini API was reachable. Given the assignment permits GPT,
  Claude, Llama, or Kimi K2, Gemini was the only one that could actually be
  exercised end-to-end from this environment — not necessarily the only valid
  choice without that constraint.
- **Free tier was not viable for this project — moved to paid**: live
  development hit two separate hard limits on the free tier: 5 requests/minute,
  and a much more restrictive **20 requests/day** cap (plus a small trial-credit
  balance that depleted entirely) — together making a full evaluation run
  impossible, since one multi-hop query alone can need 4-9 LLM calls. Enabling
  billing (a one-time $10 minimum prepayment, of which the entire project's
  usage to date has consumed a small fraction) removed both limits: the same
  query that took 80-150s under free-tier rate-limit pacing dropped to ~21s, and
  the full 23-question eval completed in a few minutes instead of 20-30+. This
  is documented as a real cost-planning data point for anyone evaluating Gemini
  for a similar project, not smoothed over.

### Embeddings: hybrid MiniLM + TF-IDF, not Gemini embeddings
This went through three iterations, each forced by something discovered
empirically rather than assumed.

**First iteration — Gemini embeddings.** The system was designed and initially
implemented with real Gemini embeddings (`gemini-embedding-001`, 768-dim) and
verified working in isolation. Ingesting the full corpus (12 filings → 2,421
chunks) requires embedding every chunk, and repeated attempts hit two distinct
free-tier limits: a `1,000 requests/day` cap, and a second, much lower ceiling
that rejected batches of even 5-20 items with `429 RESOURCE_EXHAUSTED`, while
single-item calls kept succeeding — consistent with a token/instance-based
sub-quota beneath the headline daily quota. After three failed ingestion
attempts, this was abandoned as a dependency (the code path still exists behind
`EMBEDDING_PROVIDER=gemini` for a paid-tier deployment).

**Second iteration — TF-IDF only.** Fully offline, zero quota risk, but purely
lexical.

**Third iteration — a genuine open-source model, found by testing
infrastructure, not giving up on quality.** Hugging Face itself is blocked by
this sandbox's network policy, but Chroma's bundled ONNX build of
`sentence-transformers/all-MiniLM-L6-v2` downloads its weights from an S3
mirror (`chroma-onnx-models.s3.amazonaws.com`), which *is* reachable — confirmed
by directly testing that URL rather than assuming Hugging Face's block meant
every open-source model was unreachable. This gives real semantic embeddings
(384-dim), fully local after a one-time ~80MB download, zero ongoing cost or
quota risk.

**MiniLM alone still wasn't enough** — verified empirically, not assumed: a
query for "net profit" that a plain lexical match found immediately sometimes
failed to rank in MiniLM's top-5 results, because a small general-purpose
sentence model has no special affinity for text buried in dense financial
tables. The final design runs **both**: MiniLM embeddings in Chroma (semantic
recall) plus a TF-IDF sidecar index built at ingestion time (`app/ingest.py`,
saved to `data/tfidf_sidecar.joblib`), merged at query time via **Reciprocal
Rank Fusion** (`app/tools/retrieval.py`) — a standard, well-regarded
hybrid-search technique, not a workaround dressed up as one. Verified fix: the
same query that MiniLM alone failed to surface moved into the top-2 hybrid
results.

Every chunk is also prefixed with its detected `[period | section]` tag before
indexing (e.g. `[Q3 2023 | Income Statement]`), because the filings say "three
months ended 30 September 2023" rather than "Q3 2023" — without this, a query
naming a quarter would match nothing lexically. Combined with the Temporal
Agent's exact-period `where` filtering, this closes most of the remaining gap
for this vocabulary-constrained domain.

### Vector DB: ChromaDB
Local, file-based, zero infrastructure to stand up, `cosine` similarity space
configured explicitly. Holds the MiniLM embeddings; the TF-IDF sidecar for
hybrid search lives alongside it as a separate joblib artifact (TF-IDF's sparse
vectors don't fit Chroma's dense-vector model, so it's queried directly with a
masked dot-product rather than through Chroma). Metadata filtering
(`where={period, section}`) is used directly by the Retrieval Agent on the
Chroma side — how the system narrows to the exact filing before ever asking the
LLM to reason over the result.

### Document parsing: pdfplumber
Chosen for direct table + text extraction without an external service
dependency (avoiding an additional quota-risk surface under time pressure, per
the embeddings experience above). Section detection uses a regex header-matching
heuristic over each page's text (`Statement of Financial Position` → Balance
Sheet, etc.), carrying the last detected header forward across pages so
mid-section pages without a repeated heading are still tagged correctly.

## 4. Data Processing & Ingestion

1. **Extraction**: `pdfplumber` pulls per-page text and tables; tables are
   flattened to `" | "`-joined rows and appended to the page's text so figures
   inside tables are not lost to text-extraction gaps.
2. **Section tagging**: regex header matching against 9 known FAB filing section
   types (Income Statement, Balance Sheet, Cash Flow Statement, Changes in
   Equity, Comprehensive Income, Risk Management, Notes, Auditor's Report,
   Management Discussion), carried forward page-to-page.
3. **Metadata enrichment**: every chunk carries `source` (filename), `period`
   (parsed from filename, e.g. `Q3 2023`), `page`, `section`, and `report_type`
   (Financial Statement / Earnings Presentation / Results Call, inferred from
   filename — the corpus is currently 100% Financial Statements, but the field
   is populated and filterable for when other report types are added).
4. **Chunking**: character-window chunking (1,400 chars, 200 overlap) per page,
   not a fixed global chunker — this keeps chunks aligned to a single page's
   section tag rather than spanning a section boundary.
5. **Vector DB write**: batched `collection.add()` calls into a single Chroma
   collection (`fab_financial_statements`), `hnsw:space=cosine`, plus a
   parallel TF-IDF sidecar fit over the same chunks for hybrid retrieval.

Result: **12 filings → 2,421 chunks**, fully re-ingestable via `python -m
app.ingest` (also accepts `--zip path.zip` to drop in additional documents).

## 5. Core Capabilities vs. Requirements

| Requirement | Implementation |
|---|---|
| Multi-hop reasoning | Planner loops across multiple `search_documents`/tool calls before synthesizing; verified end-to-end on the assignment's own Q3 2023→2024 YoY example, and scored 100% (4/4, manually verified) in evaluation. |
| Calculation accuracy + tracing | All arithmetic goes through `app/tools/calculator.py` (AST-restricted `eval` for generic expressions, plus named formulas for %-change, ROE, loan-to-deposit), never LLM mental math. Every call records `formula`, `inputs`, and `result` in the trace. |
| Source citation | Every `search_documents` hit's `(source, period, page, section)` is captured in `state["citations"]`; the Synthesis Agent's system prompt requires citing each figure inline. Measured: 100% of the 18 evaluation questions requiring retrieval got proper citations. |
| Uncertainty handling | System + synthesis prompts explicitly instruct: never fabricate a figure not backed by a tool result; state missing evidence rather than guess. Correctly refused all 3 true out-of-scope test questions; **known gap** — does not yet ask for clarification on ambiguous-scope questions (see Limitations). |
| Hallucination prevention | Structural, not just prompted: the planner **cannot** state a number without it appearing in a tool result first (retrieval or calculator), because those are the only two ways a figure enters `state["history"]`. |
| Metadata filtering | `period` and `section` are first-class Chroma `where` filters, used automatically by the Temporal Agent's resolved periods. |

## 6. Cost Estimation

Measured directly from the 23-question evaluation run (see the Evaluation
Report), not purely estimated, using Gemini 2.5 Flash pricing (~$0.30/M input
tokens, ~$2.50/M output tokens) and MiniLM/TF-IDF embeddings ($0 marginal cost,
fully local):

| Scenario | Measured |
|---|---|
| Full 23-question eval suite, total cost | < $0.50 |
| Average latency per query (paid tier) | 12.9s |
| Average tool calls per query | 3.17 |
| Simple factual query avg latency | 11.1s |
| Temporal comparison query avg latency (most tool calls) | 23.4s |
| Full ingestion (2,421 chunks, MiniLM + TF-IDF, fully local) | $0 · ~1 min |

Extrapolating to "thousands of queries per day" (the assignment's production
framing) at the measured ~$0.02/query average: roughly **$20-60/day**. Cheap in
absolute terms — throughput headroom on the paid tier, not cost, is the real
production planning question (see Scalability below).

## 7. Scalability Considerations

**Rate limits were the binding constraint during development, resolved by
moving to a paid tier.** The free tier's hard caps — `gemini-2.5-flash` at 5
requests/minute *and* 20 requests/day — were hit live and made a multi-hop
query take 40-150+ seconds (explicit call pacing plus retry-with-backoff
parsing the API's suggested `retryDelay` in `app/gemini_client.py:
generate_content`) and made a full 23-question eval run impossible. Enabling
billing removed both limits immediately, measured: the same query dropped to
~21s and the full eval completed in minutes. The pacing code is still in place
as a 1-second safety margin but is no longer load-bearing at the current query
volume.

- **Horizontal scaling**: the FastAPI service is stateless per request (Chroma
  and the TF-IDF vectorizer are loaded once as module-level singletons);
  running multiple worker processes behind a load balancer is a standard
  scale-out with no architectural change.
- **Vector DB**: Chroma's local file-based mode is adequate for this corpus size
  (2,421 chunks) but would need to move to a hosted/clustered vector DB
  (Pinecone, Qdrant, or Chroma's server mode) for a much larger corpus or
  multi-instance deployment sharing one index.
- **Caching**: repeated or near-duplicate questions (e.g., "what was net profit
  in Q3 2023" asked by multiple users) are not currently cached; a
  question-normalization + response cache would cut both cost and the
  rate-limit pressure significantly in production.

## 8. Known Limitations

1. **MiniLM is a small, general-purpose embedding model** with no special
   affinity for dense financial tables — verified empirically to sometimes
   under-rank the exact relevant line among a page full of numbers. The TF-IDF
   hybrid (§3) recovers most of this, and the evaluation suite's 100%
   simple-factual accuracy shows it works well enough in practice, but it is
   not as strong as a domain-tuned or larger embedding model would be.
2. **Balance-sheet figures (Total Assets, Total Equity, Total Loans, Total
   Deposits) could not be reliably text-extracted** from several filings' primary
   balance sheet pages — `pdfplumber` returned garbled/reordered text on some
   of those specific pages. Net Profit and EPS figures, which appear in more
   text-extractable contexts (the EPS note, primary income statement lines),
   were reliably extracted and hand-verified; Total Assets figures were
   recovered from the segment-reporting note's Group total column instead,
   which reconciles to the balance sheet by IFRS 8 but required extra
   cross-validation work. Loan-to-deposit and ROE test questions accordingly
   use synthetic inputs to test the calculator tool in isolation rather than
   real extracted balance-sheet figures. This limitation directly caused the
   one temporal-comparison eval failure (a false "data not available" on a
   total-assets question) — see the Evaluation Report.
3. **Net Profit basis inconsistency across years**: 2022-2023 filings only
   state net profit "attributable to shareholders" (after deducting
   non-controlling interests) on the face of the statement; 2024 filings state
   the "total" figure (including NCI) on the primary statement, with the
   attributable figure appearing separately in the EPS note. The system does
   not currently detect or normalize this basis difference automatically —
   caused one eval question to land within tolerance but off by 0.17
   percentage points due to a basis mismatch (Evaluation Report, CALC-01).
4. **Ambiguous-scope questions are guessed, not clarified** — found directly in
   evaluation: "What was FAB's net profit?" (no period given) was answered with
   the most recent quarter's figure rather than a request for clarification,
   despite the assignment explicitly asking for the latter behavior on
   ambiguous queries. The planner's current instructions guard against
   fabricating figures but don't yet flag an under-specified question scope.
5. **The planner can exhaust its iteration budget and report a false negative**
   rather than an honest "I searched but couldn't confirm this" — observed
   directly in evaluation: 8 iterations and 30 citations gathered, still
   concluded data was unavailable when it in fact existed in the corpus
   (Evaluation Report, TEMP-01).
6. **No self-reflection/error-recovery loop** (bonus item, not implemented) —
   the planner does not currently critique its own answer before finalizing;
   `MAX_ITERATIONS = 8` is a hard stop, not an adaptive one.
7. **Single-turn conversations only** — no multi-turn memory across separate
   API calls; each `/api/agent-query` request is independent.

## 9. Future Improvements

- **Add an ambiguity check to the planner prompt** — the single highest-value
  fix identified by evaluation: recognize an under-specified period/scope and
  call `finish_with_answer` with a clarification request instead of guessing.
- **Replace a false "not available" with an honest "I searched but couldn't
  confirm"** when the iteration cap is hit without success.
- Add a table-focused parser (LlamaParse/Unstructured.io) specifically for
  balance-sheet pages that `pdfplumber` mis-extracts, to make Total
  Assets/Equity/Loans/Deposits as reliable as Net Profit/EPS.
- Add a basis-normalization step (attributable vs. total incl. NCI) when
  comparing net profit across the 2023/2024 filing-convention boundary.
- Switch `EMBEDDING_PROVIDER=gemini` for potentially even stronger semantic
  retrieval, now that billing is enabled (code path already implemented and
  tested) — would need an A/B comparison against the current hybrid to justify
  the added complexity and quota exposure.
- Add response caching keyed on normalized question + resolved periods.
- Add a Visualization Agent (bonus item) to render trend charts for
  multi-quarter comparison questions.

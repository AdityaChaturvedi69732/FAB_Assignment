# Evaluation Report — FAB Multi-Agent Financial Analysis System

## Summary

23 test questions were run end-to-end through the live multi-agent system (`python -m eval.run_eval`), spanning simple factual, calculation-heavy, temporal comparison, multi-hop, and out-of-scope categories, per the assignment's evaluation framework requirement.

| Metric | Value |
|---|---|
| Total questions | 23 |
| Auto-gradable (exact numeric/refusal check) | 16 |
| Auto-gradable accuracy | **87.5%** (14/16) |
| Manually reviewed (multi-hop / temporal, qualitative) | 7 |
| Manual-review accuracy | **85.7%** (6/7) |
| **Combined accuracy** | **87.0%** (20/23) |
| Citation rate on questions requiring retrieval | **100%** (18/18) |
| Average latency | 12.9s/query |
| Average tool calls per query | 3.17 |

Ground truth figures were hand-verified against the raw extracted PDF text before the eval ran (see `eval/extract_facts.py` output and `eval/test_suite.json`'s notes field) — not generated or guessed by an LLM.

## Results by category

| Category | Count | Accuracy | Avg latency |
|---|---|---|---|
| Simple factual | 6 | 100% (6/6) | 11.1s |
| Calculation | 6 | 83.3% (5/6) | 11.7s |
| Out-of-scope / ambiguous | 4 | 75% (3/4) | 4.5s |
| Multi-hop | 4 | 100% (4/4, manual) | 17.7s |
| Temporal comparison | 3 | 67% (2/3, manual) | 23.4s |

## A correction made mid-evaluation (documented, not hidden)

The first eval run reported only 62.5% auto-gradable accuracy. Investigating the failures found that **most of them were bugs in the grading script, not the system**:

1. **Unit-scale bug**: the ground truth is expressed in AED million, but the agent sometimes correctly answers in AED thousand (e.g. "AED 5,123,259 thousand" for a 5,120.263-million ground truth — the same figure, stated in a different unit present in the source filing). The original `numeric_match()` only compared raw digits, so a numerically-correct answer in a different unit scored as wrong. Fixed by checking the extracted number at 1x, 1000x, and 0.001x scale against the expected value.
2. **A genuine ground-truth labeling error** on my part: SF-04 asked for "basic EPS for Q3 2024" and I had recorded the ground truth as `1.10`, which is actually the **nine-month cumulative** EPS figure on the same filing line — the correct discrete three-month figure is `0.38`. The system's answer (`0.38`) was right; my test suite was wrong. Corrected in `eval/test_suite.json` with a note explaining the mistake.

After both fixes (re-scored from the same saved model outputs — **no new LLM calls were made**, so this is not re-running until a passing result appears), auto-gradable accuracy rose to 87.5%. This is reported transparently here rather than only showing the corrected number, because catching your own evaluation bugs is part of the evaluation, not separate from it.

## Failure analysis

### CALC-04 — retrieval gave up rather than fabricating (acceptable, but incomplete)
*"What was the percentage growth in FAB's full-year net profit from 2022 to 2023?"*
The agent responded: *"I am sorry, but I cannot answer your question as the net profit for FAB for the full years 2022 and 2023 is not available in the provided tool results."* This is **not a hallucination failure** (the system correctly refused to guess), but it is a **retrieval failure** — both FY2022 and FY2023 net profit figures are in the corpus (verified manually: FY2022 = AED 13,411.198M in `FAB-FS-Q4-2022-English.pdf` p.73; FY2023 = AED 16,405.493M in `FAB-FS-Q4-2023-English.pdf` p.72) and other queries in this same eval run successfully retrieved adjacent figures from the same documents. The planner's search phrasing for this particular question didn't surface them within its iteration budget. This is the calculation-category question's one failure (5/6 = 83.3%).

### TEMP-01 — false "data not available" after hitting the iteration cap
*"How did FAB's total assets change between the end of 2023 and the end of 2024?"*
The agent claimed FY2024 data "is not yet available," which is incorrect — `FAB-FS-Q4-2024-English.pdf` is in the corpus and was used correctly by other questions in the same run (e.g. TEMP-02, TEMP-03). The trace shows it ran the full `MAX_ITERATIONS = 8` search/tool loop and gathered 30 citations without landing on the specific "Total assets" figure for that period, then the synthesis step incorrectly reported non-availability instead of "I searched extensively but could not confidently locate this figure." This connects directly to a limitation already called out in the Architecture Document: **Total Assets figures live in a segment-reporting note, not a cleanly-labeled balance sheet line**, and are consequently harder for hybrid retrieval to surface reliably than Net Profit or EPS figures, which appear in more consistently-worded contexts (the EPS note, the primary income statement).

### AMBIG-01 — guesses a period instead of asking for clarification
*"What was FAB's net profit?"* (no period specified)
Expected behavior: ask which quarter/year, per the assignment's explicit requirement that ambiguous queries should prompt for clarification rather than guess. Actual: the system picked the most recent period (FY2024) and answered directly. This is a real, unaddressed gap — the current system prompt tells the planner not to fabricate *figures*, but doesn't instruct it to recognize an *ambiguous scope* and ask before proceeding. A straightforward fix (not yet implemented, given time constraints): add an explicit planner instruction to call `finish_with_answer` with a clarification request when a question names no period and multiple periods could apply.

### CALC-01 — correct within tolerance, but surfaced a real basis inconsistency
*"YoY change in Net Profit between Q3 2023 and Q3 2024?"* → answered 5.03%, ground truth 4.86% (both within the 5% calculation tolerance, so scored correct). The 0.17-point gap is because the agent's Q3 2023 figure came from the "attributable to shareholders" basis (4,254.798M, from Q3 2023's own filing) while its Q3 2024 figure was the "total including non-controlling interests" basis (4,469M, from Q3 2024's own filing) — each individually correct, but not on a matched basis. This is exactly the corpus-level basis inconsistency documented as Known Limitation #4 in the Architecture Document (2022–2023 filings only state the attributable figure; 2024 filings state the total figure). The system has no basis-normalization step, so it can silently produce a slightly-off comparison when the two periods it's comparing come from filings on different sides of that reporting-convention change.

## What worked well

- **Simple factual retrieval: 100% (6/6)**, all exactly matching hand-verified ground truth once the eval script's unit bug was fixed.
- **Multi-hop reasoning: 100% (4/4, manually verified)** — the system correctly chained temporal resolution → multiple filtered searches → calculator calls → cited synthesis, including on the assignment's own flagship example query type (YoY comparison with explanation). MULTIHOP-02 in particular chained a 3-year, 2-step growth calculation correctly.
- **Out-of-scope refusal: the 3 true out-of-scope questions were all correctly refused** (a different company, a future prediction, an unrelated topic) — the failure in this bucket (AMBIG-01) is a different behavior (ambiguity handling, not refusal) that happened to be grouped in the same category.
- **Citations: 100% of retrieval-based answers cited a real source/period/page** — the two calculator-only questions and three out-of-scope refusals correctly had none, since no retrieval was needed for either.
- **Calculation tracing**: every calculator invocation in every trace shows the exact formula and inputs used (visible in `eval/results.json`), satisfying the "calculation tracing" requirement literally, not just in spirit.

## Approaches tried (and why the final one was chosen)

| Approach | Outcome |
|---|---|
| TF-IDF only (scikit-learn) | Worked, but purely lexical - no semantic matching. Kept as the hybrid's lexical signal. |
| Gemini embeddings (`gemini-embedding-001`) | Best theoretical quality, but the free tier's embedding quota was exhausted mid-ingestion three times (see Architecture Document §3) - not viable to depend on for this project's timeline. |
| MiniLM only (`all-MiniLM-L6-v2`, local ONNX) | Real semantic embeddings, zero quota risk - but alone it sometimes failed to rank the exact "Net profit for the period" line highly among dense financial tables (a small general-purpose model has no special affinity for tabular filing text). |
| **MiniLM + TF-IDF hybrid (reciprocal rank fusion)** — **used in the final system** | Combines MiniLM's semantic recall with TF-IDF's lexical precision on domain-specific figures/line-items. Verified empirically: the same query that MiniLM alone failed to rank correctly moved into the top-2 results after adding the TF-IDF signal. |
| Gemini 2.5 Flash on the free tier | Two separate hard limits hit live during development: 5 requests/minute, and (more severely) a **20 requests/day** cap plus a small depleted trial-credit balance - together made it impossible to run a full 23-question eval, since a single multi-hop query alone can need 4-9 LLM calls. |
| Gemini 2.5 Flash on a paid tier (final) | Removed both limits; the same query that took 80-150s under free-tier rate-limit pacing completed in ~21s once billing was enabled, and the full eval suite completed in a few minutes instead of 20-30+. |

## Cost and latency (measured, not estimated)

- Average latency: **12.9s/query** across all 23 questions (paid tier, minimal pacing).
- Average tool calls per query: **3.17**.
- Simple factual queries average 11.1s; temporal comparison queries (which chain the most tool calls) average 23.4s.
- At Gemini 2.5 Flash's published pricing (~$0.30/$2.50 per M tokens), running all 23 questions cost well under $0.50 total — consistent with the Architecture Document's per-query cost estimate of $0.01-0.03 for a multi-hop query.

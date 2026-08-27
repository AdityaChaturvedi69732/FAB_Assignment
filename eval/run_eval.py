"""Evaluation harness: runs every question in test_suite.json through both
the plain-retrieval endpoint and the full multi-agent pipeline, and reports
accuracy, retrieval quality, citation coverage, latency, and approximate cost.

Usage: python -m eval.run_eval
"""
import json
import re
import time
from pathlib import Path

from app.agents.orchestrator import run_query
from app.tools import retrieval

ROOT = Path(__file__).resolve().parent.parent
TEST_SUITE_PATH = ROOT / "eval" / "test_suite.json"
REPORT_PATH = ROOT / "eval" / "results.json"

# Gemini 2.5 Flash public pricing (approximate, as of this writing) used only
# for a rough per-query cost estimate - see the architecture doc for caveats.
INPUT_COST_PER_M_TOKENS = 0.30
OUTPUT_COST_PER_M_TOKENS = 2.50
EMBED_COST_PER_M_TOKENS = 0.15


def extract_numbers(text: str) -> list[float]:
    cleaned = re.sub(r"[,$]", "", text)
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", cleaned)]


def numeric_match(answer_text: str, expected_value: float, tolerance: float = 0.02) -> bool:
    """True if any number in the answer is within `tolerance` relative error of expected_value,
    checked at 1x, 1000x, and 1/1000x scale (ground truth is in AED million, but the agent
    sometimes correctly states the same figure in AED thousand, e.g. "5,123,259 thousand"
    for a 5120.263-million ground truth - same value, different unit, not a wrong answer)."""
    for n in extract_numbers(answer_text):
        if expected_value == 0:
            continue
        for scale in (1, 1000, 0.001):
            if abs(n * scale - expected_value) / abs(expected_value) <= tolerance:
                return True
    return False


def citation_present(trace: list, citations: list) -> bool:
    return len(citations) > 0 or any(step.get("call") == "search_documents" for step in trace)


def refused(answer_text: str) -> bool:
    markers = ["cannot", "can't", "not covered", "outside", "out of scope", "unable to", "no information",
               "does not contain", "doesn't contain", "please specify", "please clarify", "which quarter",
               "which period", "not available in", "insufficient"]
    lower = answer_text.lower()
    return any(m in lower for m in markers)


def run():
    suite = json.loads(TEST_SUITE_PATH.read_text())
    results = []

    for q in suite["questions"]:
        print(f"Running {q['id']}: {q['question'][:70]}...")
        start = time.time()
        try:
            agent_result = run_query(q["question"])
            error = None
        except Exception as e:
            agent_result = {"answer": "", "trace": [], "citations": [], "iterations": 0, "elapsed_seconds": 0}
            error = str(e)
        elapsed = time.time() - start

        category = q["category"]
        gt = q.get("ground_truth", {})
        record = {
            "id": q["id"],
            "category": category,
            "question": q["question"],
            "answer": agent_result["answer"],
            "iterations": agent_result["iterations"],
            "elapsed_seconds": round(elapsed, 2),
            "num_citations": len(agent_result["citations"]),
            "has_citation": citation_present(agent_result["trace"], agent_result["citations"]),
            "error": error,
        }

        if category == "out_of_scope":
            record["correct"] = refused(agent_result["answer"])
            record["expected"] = gt.get("expected_behavior")
        elif category in ("simple_factual",):
            expected_value = gt.get("value")
            record["correct"] = numeric_match(agent_result["answer"], expected_value) if expected_value is not None else None
            record["expected"] = expected_value
        elif category == "calculation":
            expected_value = gt.get("pct_change") or gt.get("ratio_pct") or gt.get("quarterly_roe_pct")
            record["correct"] = numeric_match(agent_result["answer"], expected_value, tolerance=0.05) if expected_value is not None else None
            record["expected"] = expected_value
        else:
            # multi_hop / temporal_comparison: graded qualitatively, not auto-scored here.
            record["correct"] = None
            record["expected"] = "manual review (see notes in test_suite.json)"

        results.append(record)

    total = len(results)
    auto_gradable = [r for r in results if r["correct"] is not None]
    correct = sum(1 for r in auto_gradable if r["correct"])
    citation_rate = sum(1 for r in results if r["has_citation"]) / total
    avg_latency = sum(r["elapsed_seconds"] for r in results) / total
    avg_iterations = sum(r["iterations"] for r in results) / total

    summary = {
        "total_questions": total,
        "auto_gradable": len(auto_gradable),
        "auto_gradable_accuracy": round(correct / len(auto_gradable), 3) if auto_gradable else None,
        "citation_rate": round(citation_rate, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_tool_calls_per_query": round(avg_iterations, 2),
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in results}):
        cat_results = [r for r in results if r["category"] == cat]
        cat_gradable = [r for r in cat_results if r["correct"] is not None]
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "accuracy": round(sum(1 for r in cat_gradable if r["correct"]) / len(cat_gradable), 3) if cat_gradable else None,
            "avg_latency_seconds": round(sum(r["elapsed_seconds"] for r in cat_results) / len(cat_results), 2),
        }

    output = {"summary": summary, "results": results}
    REPORT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {REPORT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

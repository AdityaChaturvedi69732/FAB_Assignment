"""Temporal reasoning tool: resolves quarter/year references and relative
expressions ("last 6 quarters", "YoY", "QoQ") into explicit periods that
exist in the ingested corpus.
"""
import re

_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _period_key(period: str):
    m = re.match(r"Q([1-4]) (\d{4})", period)
    if not m:
        return (0, 0)
    return (int(m.group(2)), int(m.group(1)))


def all_periods_sorted(available_periods: list[str]) -> list[str]:
    return sorted(set(available_periods), key=_period_key)


def extract_periods_from_text(text: str) -> list[str]:
    """Find explicit 'Q3 2023' / '2023 Q3' style mentions in free text."""
    found = []
    for m in re.finditer(r"\bQ([1-4])\b[^\d]{0,6}?(20\d\d)", text, re.IGNORECASE):
        found.append(f"Q{m.group(1)} {m.group(2)}")
    for m in re.finditer(r"\b(20\d\d)\b[^\d]{0,6}?Q([1-4])\b", text, re.IGNORECASE):
        found.append(f"Q{m.group(2)} {m.group(1)}")
    seen = []
    for p in found:
        if p not in seen:
            seen.append(p)
    return seen


def last_n_quarters(available_periods: list[str], n: int, before: str | None = None) -> dict:
    ordered = all_periods_sorted(available_periods)
    if before:
        try:
            idx = ordered.index(before)
            ordered = ordered[:idx + 1]
        except ValueError:
            pass
    result = ordered[-n:]
    return {
        "tool": "temporal.last_n_quarters",
        "inputs": {"n": n, "before": before},
        "result": result,
    }


def year_over_year_pair(period: str) -> dict:
    """Given 'Q3 2024', return the matching prior-year quarter 'Q3 2023'."""
    m = re.match(r"(Q[1-4]) (\d{4})", period)
    if not m:
        raise ValueError(f"Unrecognized period format: {period}")
    quarter, year = m.group(1), int(m.group(2))
    prior = f"{quarter} {year - 1}"
    return {
        "tool": "temporal.year_over_year_pair",
        "inputs": {"period": period},
        "result": {"current": period, "prior_year": prior},
    }


def quarter_over_quarter_pair(period: str, available_periods: list[str]) -> dict:
    """Given a period, return the immediately preceding quarter present in the corpus."""
    ordered = all_periods_sorted(available_periods)
    if period not in ordered:
        raise ValueError(f"{period} not found among available periods")
    idx = ordered.index(period)
    prior = ordered[idx - 1] if idx > 0 else None
    return {
        "tool": "temporal.quarter_over_quarter_pair",
        "inputs": {"period": period},
        "result": {"current": period, "prior_quarter": prior},
    }

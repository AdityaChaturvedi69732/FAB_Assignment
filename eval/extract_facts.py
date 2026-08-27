"""Extract candidate ground-truth figures from the raw PDF text.

This does NOT auto-fill the eval test suite. It prints, per document, the
exact line(s) containing key financial line-items (Net profit, Total assets,
Total equity, Customer deposits, Loans and advances, EPS) so a human can
read the real filing text and hand-verify the numbers before writing them
into test_suite.json as ground truth. Financial figures used to grade
"calculation accuracy" must be verified against the source text, not
regex-guessed blindly.

Usage: python -m eval.extract_facts
"""
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "documents"

LINE_ITEMS = [
    "net profit for the period",
    "net profit for the year",
    "total assets",
    "total equity",
    "total shareholders",
    "customer deposits",
    "due to banks",
    "loans and advances",
    "earnings per share",
]


def guess_period(filename: str) -> str:
    m = re.search(r"(Q[1-4]).{0,4}(20\d\d)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    m = re.search(r"(20\d\d).{0,4}(Q[1-4])", filename, re.IGNORECASE)
    if m:
        return f"{m.group(2).upper()} {m.group(1)}"
    return filename


def main():
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        period = guess_period(pdf_path.name)
        print(f"\n{'=' * 90}\n{pdf_path.name}  ({period})\n{'=' * 90}")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    lower = line.lower()
                    for item in LINE_ITEMS:
                        if item in lower and re.search(r"\d", line):
                            print(f"[p.{page_num}] {line.strip()}")
                            break


if __name__ == "__main__":
    main()

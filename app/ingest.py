"""Ingest FAB financial documents into a local Chroma vector DB.

Extracts text + tables per page, tags each page with its financial-statement
section (Income Statement, Balance Sheet, Cash Flow, Notes, Risk, MD&A, ...),
chunks it, embeds with Gemini, and stores everything with rich metadata
(source, period, page, section, report_type) for filtered retrieval.

Usage:
    python -m app.ingest                 # ingest everything in documents/
    python -m app.ingest --zip path.zip  # extract a zip of PDFs into documents/ first, then ingest
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

import chromadb
import joblib
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

from app.embeddings import get_embedder

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "documents"
DB_DIR = ROOT / "data" / "chroma"
TFIDF_SIDECAR_PATH = ROOT / "data" / "tfidf_sidecar.joblib"
COLLECTION_NAME = "fab_financial_statements"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200

# Ordered so more specific headers are checked before generic ones.
SECTION_PATTERNS = [
    ("Income Statement", r"statement of profit or loss|statement of income"),
    ("Comprehensive Income", r"statement of comprehensive income"),
    ("Balance Sheet", r"statement of financial position"),
    ("Cash Flow Statement", r"statement of cash flows"),
    ("Changes in Equity", r"statement of changes in equity"),
    ("Risk Management", r"risk management|credit risk|market risk|liquidity risk"),
    ("Notes", r"notes to the (condensed )?consolidated"),
    ("Auditor's Report", r"independent auditor"),
    ("Management Discussion", r"directors[' ]+report|chairman|group chief executive|management discussion"),
]


def extract_zip(zip_path: Path) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".pdf"):
                continue
            name = Path(member).name
            with zf.open(member) as src, open(DOCS_DIR / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Extracted PDFs from {zip_path} into {DOCS_DIR}")


def detect_section(text: str, current: str) -> str:
    lower = text[:2000].lower()
    for name, pattern in SECTION_PATTERNS:
        if re.search(pattern, lower):
            return name
    return current


def extract_pages(pdf_path: Path):
    """Yield (page_num, text, section) for every non-empty page, carrying the
    last detected section header forward across pages until a new one appears."""
    section = "General"
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                rows = [" | ".join(cell or "" for cell in row) for row in table]
                table_text += "\n" + "\n".join(rows)
            full_text = (text + "\n" + table_text).strip()
            if not full_text:
                continue
            section = detect_section(text, section)
            yield page_num, full_text, section


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def guess_period(filename: str) -> str:
    m = re.search(r"(Q[1-4]).{0,4}(20\d\d)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    m = re.search(r"(20\d\d).{0,4}(Q[1-4])", filename, re.IGNORECASE)
    if m:
        return f"{m.group(2).upper()} {m.group(1)}"
    return filename


def guess_report_type(filename: str) -> str:
    lower = filename.lower()
    if "presentation" in lower or "earnings" in lower:
        return "Earnings Presentation"
    if "call" in lower or "transcript" in lower:
        return "Results Call"
    return "Financial Statement"


def build_index():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCS_DIR}")
        return

    ids, docs, metas = [], [], []
    for pdf_path in pdf_files:
        period = guess_period(pdf_path.name)
        report_type = guess_report_type(pdf_path.name)
        print(f"Processing {pdf_path.name} ({period}, {report_type})...")
        for page_num, page_text, section in extract_pages(pdf_path):
            for i, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{pdf_path.stem}_p{page_num}_c{i}")
                # Prepend period/section so those tokens help lexical matches too;
                # filings say "three months ended September 30" rather than "Q3".
                docs.append(f"[{period} | {section}] {chunk}")
                metas.append({
                    "source": pdf_path.name,
                    "period": period,
                    "page": page_num,
                    "section": section,
                    "report_type": report_type,
                })

    embedder = get_embedder()
    print(f"Embedding + indexing {len(docs)} chunks from {len(pdf_files)} documents with '{embedder.name}' embeddings...")

    write_batch = 100
    if embedder.name in ("gemini", "minilm"):
        # Embed and write to Chroma in the same batch, so a quota/network
        # failure partway through still leaves everything embedded so far
        # persisted and queryable (a re-run then only needs the remainder).
        # Gemini needs small batches to respect its free-tier rate limit;
        # MiniLM runs fully locally so a much larger batch is fine.
        embed_batch = 90 if embedder.name == "gemini" else 200
        num_batches = (len(docs) + embed_batch - 1) // embed_batch
        for batch_num, i in enumerate(range(0, len(docs), embed_batch), start=1):
            batch_embeddings = embedder.embed_documents(docs[i:i + embed_batch])
            collection.add(
                ids=ids[i:i + embed_batch],
                documents=docs[i:i + embed_batch],
                metadatas=metas[i:i + embed_batch],
                embeddings=batch_embeddings,
            )
            print(f"  indexed batch {batch_num}/{num_batches} ({collection.count()} chunks so far)")
    else:
        # TF-IDF must be fit on the whole corpus at once for a consistent
        # vector space, so embed everything up front, then write in batches.
        embeddings = embedder.embed_documents(docs)
        for i in range(0, len(docs), write_batch):
            collection.add(
                ids=ids[i:i + write_batch],
                documents=docs[i:i + write_batch],
                metadatas=metas[i:i + write_batch],
                embeddings=embeddings[i:i + write_batch],
            )

    if embedder.name != "tfidf":
        # Build a TF-IDF sidecar index regardless of the primary (semantic)
        # embedder, for hybrid retrieval: dense embeddings alone missed exact
        # figures/line-items buried in dense financial tables (verified
        # empirically - MiniLM alone failed to surface the right "Net profit
        # for the period" line for a query that literal TF-IDF matched
        # immediately). Combining both via reciprocal rank fusion at query
        # time (see app/tools/retrieval.py) gets semantic recall AND lexical
        # precision on domain vocabulary/numbers.
        print("Building TF-IDF sidecar index for hybrid retrieval...")
        tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        matrix = tfidf.fit_transform(docs)
        joblib.dump({"vectorizer": tfidf, "matrix": matrix, "ids": ids, "docs": docs, "metas": metas}, TFIDF_SIDECAR_PATH)

    print(f"Done. Indexed {collection.count()} chunks from {len(pdf_files)} PDFs into {DB_DIR} (provider={embedder.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, help="Path to a zip file of PDFs to extract into documents/ before ingesting")
    args = parser.parse_args()

    if args.zip:
        extract_zip(Path(args.zip))

    build_index()

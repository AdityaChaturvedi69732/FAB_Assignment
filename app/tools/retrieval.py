"""Vector search tool over the ingested FAB financial documents.

Hybrid retrieval: combines the primary semantic embedder (MiniLM by default)
with a TF-IDF sidecar index via Reciprocal Rank Fusion. This was added after
finding empirically that MiniLM alone (a small general-purpose model with no
special affinity for dense financial tables) sometimes failed to surface the
exact "Net profit for the period" line that a plain lexical match found
immediately - combining both signals recovers that precision without losing
MiniLM's ability to match paraphrased queries.
"""
from pathlib import Path

import chromadb
import joblib
import numpy as np

from app.embeddings import get_query_embedder
from app.ingest import COLLECTION_NAME, DB_DIR, TFIDF_SIDECAR_PATH

RRF_K = 60
CANDIDATE_POOL = 20

_client = None
_collection = None
_available_periods = None
_query_embedder = None
_sidecar = None


def _embed_query(text: str) -> list[float]:
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = get_query_embedder()
    return _query_embedder.embed_query(text)


def get_collection():
    global _client, _collection
    if _collection is None:
        if not DB_DIR.exists():
            raise RuntimeError("Vector DB not built yet. Run `python -m app.ingest` first.")
        _client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = _client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_sidecar():
    global _sidecar
    if _sidecar is None and TFIDF_SIDECAR_PATH.exists():
        _sidecar = joblib.load(TFIDF_SIDECAR_PATH)
    return _sidecar


def available_periods() -> list[str]:
    global _available_periods
    if _available_periods is None:
        collection = get_collection()
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        _available_periods = sorted({m["period"] for m in all_meta})
    return _available_periods


def _build_where(period: str | None, section: str | None):
    conditions = []
    if period:
        conditions.append({"period": period})
    if section:
        conditions.append({"section": section})
    if len(conditions) == 1:
        return conditions[0]
    if len(conditions) > 1:
        return {"$and": conditions}
    return None


def _semantic_candidates(query: str, where, k: int) -> list[tuple[str, dict]]:
    """Returns [(id, {text, source, period, page, section, report_type}), ...] ranked by semantic similarity."""
    collection = get_collection()
    query_embedding = _embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k, where=where, include=["documents", "metadatas"])
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    out = []
    for id_, d, m in zip(ids, docs, metas):
        text = d.split("] ", 1)[-1] if d.startswith("[") else d
        out.append((id_, {**m, "text": text}))
    return out


def _lexical_candidates(query: str, period: str | None, section: str | None, k: int) -> list[tuple[str, dict]]:
    sidecar = _get_sidecar()
    if sidecar is None:
        return []
    vectorizer = sidecar["vectorizer"]
    matrix = sidecar["matrix"]
    ids = sidecar["ids"]
    docs = sidecar["docs"]
    metas = sidecar["metas"]

    mask = np.array([
        (period is None or m["period"] == period) and (section is None or m["section"] == section)
        for m in metas
    ])
    if not mask.any():
        return []

    query_vec = vectorizer.transform([query])
    sims = (matrix[mask] @ query_vec.T).toarray().ravel()
    top_local_idx = np.argsort(-sims)[:k]
    global_idx = np.where(mask)[0][top_local_idx]

    out = []
    for gi in global_idx:
        if sims[np.where(np.where(mask)[0] == gi)[0][0]] <= 0:
            continue
        text = docs[gi].split("] ", 1)[-1] if docs[gi].startswith("[") else docs[gi]
        out.append((ids[gi], {**metas[gi], "text": text}))
    return out


def search(query: str, top_k: int = 5, period: str | None = None, section: str | None = None) -> dict:
    """Hybrid semantic + lexical search over the corpus, optionally filtered by period and/or section."""
    where = _build_where(period, section)

    semantic = _semantic_candidates(query, where, CANDIDATE_POOL)
    lexical = _lexical_candidates(query, period, section, CANDIDATE_POOL)

    scores: dict[str, float] = {}
    info: dict[str, dict] = {}
    for rank, (id_, meta) in enumerate(semantic):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        info[id_] = meta
    for rank, (id_, meta) in enumerate(lexical):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        info.setdefault(id_, meta)

    ranked_ids = sorted(scores, key=lambda i: -scores[i])[:top_k]

    hits = []
    for id_ in ranked_ids:
        m = info[id_]
        hits.append({
            "text": m["text"],
            "source": m["source"],
            "period": m["period"],
            "page": m["page"],
            "section": m["section"],
            "report_type": m["report_type"],
            "relevance": round(scores[id_], 4),
        })

    return {"tool": "retrieval.search", "query": query, "filters": {"period": period, "section": section}, "hits": hits}

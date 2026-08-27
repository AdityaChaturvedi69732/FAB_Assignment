"""Embedding backend abstraction.

Three interchangeable backends, controlled by the EMBEDDING_PROVIDER env var:

  - "minilm" (default): a real open-source neural embedding model
    (sentence-transformers/all-MiniLM-L6-v2, 384-dim, via ONNX Runtime).
    Fully local after a one-time ~80MB download, zero ongoing cost or quota
    risk, genuine semantic similarity (handles paraphrasing, unlike TF-IDF).
    Hugging Face itself is blocked by this sandbox's network policy, but
    Chroma's bundled ONNX build of this model downloads from an S3 mirror
    (chroma-onnx-models.s3.amazonaws.com), which is reachable - confirmed
    empirically. This is the best available option in this environment and
    the recommended default even outside it (good quality/cost trade-off).

  - "gemini": real embeddings via the Gemini API (gemini-embedding-001,
    truncated to 768-dim). Best quality of the three, but the free tier's
    embedding quota turned out to be low enough that ingesting the full
    12-document corpus (~2,400 chunks) exhausted it, even with aggressive
    batching (confirmed empirically: batches as small as 5 items started
    getting 429 RESOURCE_EXHAUSTED after normal development-time usage).
    Kept as an option for when running on a paid tier.

  - "tfidf": scikit-learn TF-IDF. Fully offline, zero dependencies beyond
    scikit-learn, but purely lexical - kept only as a last-resort fallback
    if even the ONNX model download is unavailable in some environment.

The provider used to build the current index is recorded in
data/embedding_provider.txt so query time automatically loads the matching
backend regardless of what EMBEDDING_PROVIDER is set to later.
"""
import os
from pathlib import Path

import joblib
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from sklearn.feature_extraction.text import TfidfVectorizer

from app.gemini_client import embed_query as _gemini_embed_query
from app.gemini_client import embed_texts as _gemini_embed_texts

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.joblib"
PROVIDER_MARKER_PATH = DATA_DIR / "embedding_provider.txt"

DEFAULT_PROVIDER = "minilm"


def active_provider() -> str:
    if PROVIDER_MARKER_PATH.exists():
        return PROVIDER_MARKER_PATH.read_text().strip()
    return os.environ.get("EMBEDDING_PROVIDER", DEFAULT_PROVIDER)


def _set_active_provider(name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROVIDER_MARKER_PATH.write_text(name)


class GeminiEmbedder:
    name = "gemini"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _gemini_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return _gemini_embed_query(text)


class MiniLMEmbedder:
    """Open-source sentence-transformers/all-MiniLM-L6-v2 via Chroma's ONNX build."""

    name = "minilm"

    def __init__(self):
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._fn([text])[0])


class LocalTfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self._vectorizer: TfidfVectorizer | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        matrix = self._vectorizer.fit_transform(texts)
        joblib.dump(self._vectorizer, VECTORIZER_PATH)
        return matrix.toarray().tolist()

    def _load(self):
        if self._vectorizer is None:
            if not VECTORIZER_PATH.exists():
                raise RuntimeError("TF-IDF vectorizer not found. Run `python -m app.ingest` first.")
            self._vectorizer = joblib.load(VECTORIZER_PATH)
        return self._vectorizer

    def embed_query(self, text: str) -> list[float]:
        return self._load().transform([text]).toarray()[0].tolist()


_BACKENDS = {"gemini": GeminiEmbedder, "minilm": MiniLMEmbedder, "tfidf": LocalTfidfEmbedder}


def get_embedder(provider: str | None = None):
    provider = provider or os.environ.get("EMBEDDING_PROVIDER", DEFAULT_PROVIDER)
    cls = _BACKENDS.get(provider, MiniLMEmbedder)
    _set_active_provider(provider if provider in _BACKENDS else DEFAULT_PROVIDER)
    return cls()


def get_query_embedder():
    """Load the embedder matching whichever provider built the current index."""
    provider = active_provider()
    cls = _BACKENDS.get(provider, MiniLMEmbedder)
    return cls()

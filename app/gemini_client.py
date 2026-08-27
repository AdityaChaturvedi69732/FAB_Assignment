"""Shared Gemini client and thin wrappers used by ingestion and agents."""
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent

GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # truncate from the model's native 3072 dims to keep the vector DB lean

# On the free tier gemini-2.5-flash was capped at 5 requests/minute (and a
# separate 20/day cap), confirmed via live 429s. Billing is now enabled on
# this project, which raises both limits by orders of magnitude, so pacing
# is a light safety margin rather than the load-bearing constraint it was.
_MIN_SECONDS_BETWEEN_GENERATE_CALLS = 1.0
_last_generate_call_time = 0.0


def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GEMINI_API_KEY not set (env var or .env file)")


_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=_load_api_key())
    return _client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 90) -> list[list[float]]:
    """Embed a list of texts, batching requests and retrying on transient errors.

    Small batches + a pause between calls keep us under the free-tier
    embedding rate limit (large batches triggered 429 RESOURCE_EXHAUSTED
    almost immediately even though a single call succeeds fine).
    """
    client = get_client()
    out: list[list[float]] = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i:i + batch_size]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                out.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(min(5 * (attempt + 1), 30))
        if batch_num % 10 == 0 or batch_num == num_batches:
            print(f"  embedded batch {batch_num}/{num_batches}")
        time.sleep(1.2)
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def _parse_retry_delay(exc: Exception) -> float | None:
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


def generate_content(contents, config: types.GenerateContentConfig):
    """Rate-limited, retrying wrapper around generate_content for the free tier's
    5 requests/minute cap on gemini-2.5-flash."""
    global _last_generate_call_time
    client = get_client()

    for attempt in range(6):
        elapsed = time.time() - _last_generate_call_time
        if elapsed < _MIN_SECONDS_BETWEEN_GENERATE_CALLS:
            time.sleep(_MIN_SECONDS_BETWEEN_GENERATE_CALLS - elapsed)
        try:
            resp = client.models.generate_content(model=GENERATION_MODEL, contents=contents, config=config)
            _last_generate_call_time = time.time()
            return resp
        except Exception as e:
            _last_generate_call_time = time.time()
            if attempt == 5:
                raise
            delay = _parse_retry_delay(e) or (5 * (attempt + 1))
            time.sleep(delay + 1)

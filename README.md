# FAB Multi-Agent Financial Analysis System

A multi-agent system that answers complex, multi-hop, calculation-heavy questions about First Abu Dhabi Bank's (FAB) quarterly financial statements. A LangGraph-orchestrated Planner Agent (Gemini 2.5 Flash) dispatches to Retrieval, Calculator, and Temporal agents, then a Synthesis Agent produces a cited, verifiable answer.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) for the full design writeup and evaluation results (87% accuracy across 23 test questions).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key (used for the agent's LLM reasoning; embeddings are fully local and need no key):

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey). **Note:** the free tier caps out at 5 requests/minute and 20 requests/day for `gemini-2.5-flash` — enough for light manual testing, but not for the full eval suite. Enabling billing (a one-time $10 minimum) removes both caps; the whole project's usage to date has cost well under $1.

## Add documents

Drop PDFs into `documents/`, or ingest a zip of PDFs directly:

```bash
python -m app.ingest --zip /path/to/documents.zip
```

## Build the vector index

```bash
python -m app.ingest
```

This chunks and section-tags each PDF, embeds with a local open-source model (`all-MiniLM-L6-v2` via ONNX, downloaded once on first run), and builds a TF-IDF sidecar index for hybrid retrieval. Fully offline — no API calls, no cost. Re-run any time you add or change documents in `documents/`.

## Run the app

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 — the chat UI has a toggle between **Agent (multi-hop)** mode (full pipeline) and **Retrieval only** mode (baseline, for comparison).

## Run the evaluation suite

```bash
python -m eval.run_eval
```

Runs all 23 test questions in `eval/test_suite.json` through the live agent and writes `eval/results.json` with accuracy, citation rate, latency, and per-category breakdowns.

## Project structure

```
app/
  agents/orchestrator.py   # LangGraph multi-agent graph (planner + synthesis)
  tools/
    calculator.py          # traced financial arithmetic
    temporal.py             # quarter/year resolution, YoY/QoQ pairing
    retrieval.py             # hybrid MiniLM + TF-IDF search
  embeddings.py             # pluggable embedding backends (minilm / gemini / tfidf)
  gemini_client.py          # rate-limited Gemini wrapper
  ingest.py                 # PDF -> chunks -> vector DB
  main.py                   # FastAPI app
  static/index.html         # chat UI
eval/
  test_suite.json           # 23 hand-verified test questions
  run_eval.py                # evaluation harness
  extract_facts.py           # ground-truth extraction helper
docs/
  ARCHITECTURE.md            # design doc
  EVALUATION_REPORT.md       # eval results + failure analysis
```

## Embedding provider

Controlled by `EMBEDDING_PROVIDER` (defaults to `minilm`):

- `minilm` (default) — local, open-source, zero cost/quota risk.
- `gemini` — real Gemini embeddings, better quality, needs a paid tier for a corpus this size (the free tier's embedding quota is easily exhausted).
- `tfidf` — pure lexical fallback, zero dependencies beyond scikit-learn.

The provider used to build the current index is recorded in `data/embedding_provider.txt` so queries automatically use the matching backend.

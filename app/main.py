"""FastAPI backend for the FAB multi-agent financial analysis system."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.orchestrator import run_query
from app.tools import retrieval

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="FAB Documenta")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    period: str
    page: int
    section: str
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Plain retrieval only (no agent reasoning) - useful as a baseline to
    compare against the multi-agent endpoint in the evaluation report."""
    try:
        result = retrieval.search(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    hits = result["hits"]
    sources = [
        Source(source=h["source"], period=h["period"], page=h["page"], section=h["section"], excerpt=h["text"][:500])
        for h in hits
    ]
    if not hits:
        answer = "No relevant passages found in the ingested documents."
    else:
        top = hits[0]
        answer = f"Top match: {top['source']} ({top['period']}, {top['section']}, page {top['page']}):\n\n{top['text'][:500]}"

    return QueryResponse(answer=answer, sources=sources)


class AgentQueryRequest(BaseModel):
    question: str


class AgentQueryResponse(BaseModel):
    answer: str
    trace: list
    citations: list
    iterations: int
    elapsed_seconds: float


@app.post("/api/agent-query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest):
    """Full multi-agent pipeline: planner -> retrieval/calculator/temporal agents -> synthesis."""
    try:
        result = run_query(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AgentQueryResponse(**result)


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")

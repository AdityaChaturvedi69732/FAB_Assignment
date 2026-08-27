"""Multi-agent orchestration graph (LangGraph) for FAB financial analysis.

Agents:
  - Planner Agent: decides the next step (retrieve, calculate, resolve
    temporal references, or finish) using Gemini function calling.
  - Retrieval Agent: executes vector search against the ingested filings.
  - Calculator Agent: executes deterministic, traceable financial math.
  - Temporal Agent: resolves quarter/year references and comparison pairs.
  - Synthesis Agent: once the planner signals it has enough evidence,
    produces the final answer, citing sources and stating uncertainty
    when the gathered evidence doesn't support a confident answer.

The full sequence of tool calls and their results is kept in the trace so
every number in the final answer can be traced back to a retrieved passage
or a calculator invocation (calculation tracing + source citation
requirements).
"""
import time
from typing import TypedDict

from google.genai import types

from app.gemini_client import generate_content
from app.tools import calculator, retrieval, temporal

MAX_ITERATIONS = 8

SYSTEM_INSTRUCTION = """You are the planning agent in a multi-agent financial analysis system for \
First Abu Dhabi Bank (FAB) quarterly/annual financial statements.

Rules you must follow:
- Only state financial figures that come from a search_documents result or a calculator tool result. \
Never invent or estimate a number from memory.
- For any question requiring a calculation (percentage change, ratio, ROE, growth rate, \
loan-to-deposit ratio, comparisons across periods), you MUST call the appropriate calculator tool \
rather than computing it yourself in text.
- For questions naming a relative time period ("last 6 quarters", "year-over-year", \
"quarter-over-quarter"), call the temporal tools to resolve exact periods before searching.
- Retrieve evidence for BOTH sides of any comparison before calculating.
- If, after retrieving, the documents do not contain the information needed, or the question is \
out of scope (not about FAB's financials), call finish_with_answer and say so explicitly rather \
than guessing.
- Call finish_with_answer once you have gathered enough evidence (or determined the question \
cannot be answered from the corpus).
"""

SYNTHESIS_INSTRUCTION = """You are the synthesis agent. Using ONLY the tool results in the \
conversation so far, write the final answer to the user's question.

Requirements:
- Cite every figure with its source document, quarter, and page number, e.g. (FAB-FS-Q3-2023-English.pdf, Q3 2023, p.5).
- Show the calculation explicitly when a calculator tool was used (formula and inputs -> result).
- If evidence is missing or insufficient, say exactly what is missing rather than guessing.
- Do not state any number that did not come from a tool result.
- Be concise and direct.
"""


def _tool_declarations() -> types.Tool:
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Semantic search over FAB's ingested financial filings. Returns cited passages.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="What to search for"),
                    "period": types.Schema(type="STRING", description="Optional exact period filter, e.g. 'Q3 2023'"),
                    "section": types.Schema(type="STRING", description="Optional section filter, e.g. 'Income Statement', 'Balance Sheet', 'Risk Management'"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description="Evaluate a numeric arithmetic expression (+ - * / **).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"expression": types.Schema(type="STRING")},
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate_percentage_change",
            description="Compute percentage change between an old and a new value.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "old_value": types.Schema(type="NUMBER"),
                    "new_value": types.Schema(type="NUMBER"),
                    "label": types.Schema(type="STRING"),
                },
                required=["old_value", "new_value"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate_roe",
            description="Compute Return on Equity = net_income / shareholder_equity * 100.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "net_income": types.Schema(type="NUMBER"),
                    "shareholder_equity": types.Schema(type="NUMBER"),
                },
                required=["net_income", "shareholder_equity"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate_loan_to_deposit",
            description="Compute loan-to-deposit ratio = total_loans / total_deposits * 100.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "total_loans": types.Schema(type="NUMBER"),
                    "total_deposits": types.Schema(type="NUMBER"),
                },
                required=["total_loans", "total_deposits"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_last_n_quarters",
            description="Resolve 'the last N quarters' into explicit periods present in the corpus.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "n": types.Schema(type="INTEGER"),
                    "before": types.Schema(type="STRING", description="Optional period to count backwards from, e.g. 'Q4 2024'"),
                },
                required=["n"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_year_over_year_pair",
            description="Given a period like 'Q3 2024', return it paired with the same quarter one year earlier.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"period": types.Schema(type="STRING")},
                required=["period"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_quarter_over_quarter_pair",
            description="Given a period, return it paired with the immediately preceding quarter in the corpus.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"period": types.Schema(type="STRING")},
                required=["period"],
            ),
        ),
        types.FunctionDeclaration(
            name="finish_with_answer",
            description="Call once enough evidence has been gathered to answer, or if the question cannot be answered from the corpus.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"reason": types.Schema(type="STRING")},
            ),
        ),
    ])


class AgentState(TypedDict):
    question: str
    history: list
    trace: list
    citations: list
    iterations: int
    done: bool
    final_answer: str


def _execute_tool(name: str, args: dict) -> dict:
    """Dispatch a planner-requested tool call to the right specialist agent."""
    if name == "search_documents":
        return retrieval.search(
            query=args["query"],
            period=args.get("period"),
            section=args.get("section"),
        )
    if name == "calculate":
        return calculator.evaluate(args["expression"])
    if name == "calculate_percentage_change":
        return calculator.percentage_change(args["old_value"], args["new_value"], args.get("label", ""))
    if name == "calculate_roe":
        return calculator.roe(args["net_income"], args["shareholder_equity"])
    if name == "calculate_loan_to_deposit":
        return calculator.loan_to_deposit(args["total_loans"], args["total_deposits"])
    if name == "get_last_n_quarters":
        return temporal.last_n_quarters(retrieval.available_periods(), int(args["n"]), args.get("before"))
    if name == "get_year_over_year_pair":
        return temporal.year_over_year_pair(args["period"])
    if name == "get_quarter_over_quarter_pair":
        return temporal.quarter_over_quarter_pair(args["period"], retrieval.available_periods())
    raise ValueError(f"Unknown tool: {name}")


def planner_node(state: AgentState) -> AgentState:
    resp = generate_content(
        contents=state["history"],
        config=types.GenerateContentConfig(
            tools=[_tool_declarations()],
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0,
        ),
    )
    part = resp.candidates[0].content.parts[0]
    state["history"] = state["history"] + [types.Content(role="model", parts=[part])]

    if part.function_call is None:
        # Model answered directly without calling finish_with_answer; treat as done.
        state["done"] = True
        return state

    name = part.function_call.name
    args = dict(part.function_call.args or {})

    if name == "finish_with_answer":
        state["done"] = True
        return state

    try:
        result = _execute_tool(name, args)
        error = None
    except Exception as e:
        result = {"error": str(e)}
        error = str(e)

    state["trace"].append({"call": name, "args": args, "result": result, "error": error})

    if name == "search_documents" and not error:
        for hit in result["hits"]:
            state["citations"].append({
                "source": hit["source"], "period": hit["period"], "page": hit["page"], "section": hit["section"],
            })

    fn_response_part = types.Part.from_function_response(name=name, response={"result": result})
    state["history"] = state["history"] + [types.Content(role="user", parts=[fn_response_part])]
    state["iterations"] += 1
    return state


def synthesize_node(state: AgentState) -> AgentState:
    resp = generate_content(
        contents=state["history"] + [types.Content(
            role="user",
            parts=[types.Part(text=f"Now answer the original question: {state['question']}")],
        )],
        config=types.GenerateContentConfig(system_instruction=SYNTHESIS_INSTRUCTION, temperature=0),
    )
    state["final_answer"] = resp.text or ""
    return state


def route(state: AgentState) -> str:
    if state["done"] or state["iterations"] >= MAX_ITERATIONS:
        return "synthesize"
    return "planner"


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route, {"planner": "planner", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(question: str) -> dict:
    start = time.time()
    initial_state: AgentState = {
        "question": question,
        "history": [types.Content(role="user", parts=[types.Part(text=question)])],
        "trace": [],
        "citations": [],
        "iterations": 0,
        "done": False,
        "final_answer": "",
    }
    result = get_app().invoke(initial_state, config={"recursion_limit": MAX_ITERATIONS * 2 + 4})
    elapsed = time.time() - start
    return {
        "answer": result["final_answer"],
        "trace": result["trace"],
        "citations": result["citations"],
        "iterations": result["iterations"],
        "elapsed_seconds": round(elapsed, 2),
    }

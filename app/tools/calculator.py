"""Deterministic calculator tool for financial arithmetic.

Every function returns a trace dict (formula, inputs, result) so calculations
are verifiable rather than left to LLM arithmetic, per the assignment's
"Calculation Accuracy" and "calculation tracing" requirements.
"""
import ast
import operator

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


def evaluate(expression: str) -> dict:
    """Safely evaluate a numeric arithmetic expression (+ - * / ** only)."""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return {"tool": "calculator.evaluate", "expression": expression, "result": result}


def percentage_change(old_value: float, new_value: float, label: str = "") -> dict:
    if old_value == 0:
        raise ValueError("old_value is zero; percentage change is undefined")
    result = (new_value - old_value) / old_value * 100
    return {
        "tool": "calculator.percentage_change",
        "label": label,
        "formula": "(new_value - old_value) / old_value * 100",
        "inputs": {"old_value": old_value, "new_value": new_value},
        "result": round(result, 2),
    }


def ratio(numerator: float, denominator: float, label: str = "") -> dict:
    if denominator == 0:
        raise ValueError("denominator is zero; ratio is undefined")
    result = numerator / denominator
    return {
        "tool": "calculator.ratio",
        "label": label,
        "formula": "numerator / denominator",
        "inputs": {"numerator": numerator, "denominator": denominator},
        "result": round(result, 4),
    }


def roe(net_income: float, shareholder_equity: float) -> dict:
    if shareholder_equity == 0:
        raise ValueError("shareholder_equity is zero; ROE is undefined")
    result = net_income / shareholder_equity * 100
    return {
        "tool": "calculator.roe",
        "formula": "net_income / shareholder_equity * 100",
        "inputs": {"net_income": net_income, "shareholder_equity": shareholder_equity},
        "result": round(result, 2),
    }


def loan_to_deposit(total_loans: float, total_deposits: float) -> dict:
    if total_deposits == 0:
        raise ValueError("total_deposits is zero; loan-to-deposit ratio is undefined")
    result = total_loans / total_deposits * 100
    return {
        "tool": "calculator.loan_to_deposit",
        "formula": "total_loans / total_deposits * 100",
        "inputs": {"total_loans": total_loans, "total_deposits": total_deposits},
        "result": round(result, 2),
    }

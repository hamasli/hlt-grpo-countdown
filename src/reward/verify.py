"""Countdown checker and reward functions.

The answer must be inside <answer>...</answer>. It is parsed with `ast`
(never eval), may use only + - * / and brackets, must use each given
number exactly once, and must equal the target.
"""
import ast
import re
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Matches reasoning followed by answer, allowing optional opening <think> for prefilled completions
FORMAT_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
MAX_LEN = 200  # longer answers are rejected (protects the parser)


def get_text(completion):
    """TRL passes either a plain string or a chat list like [{'role': 'assistant', 'content': ...}]."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def extract_answer(text):
    """Return the text of the LAST <answer> block, or None."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    # Remove markdown code formatting if present
    answer = answer.replace("`", "").strip()

    if "=" in answer:
        parts = answer.split("=")
        left, right = parts[0].strip(), parts[1].strip()
        ops = {"+", "-", "*", "/", "×", "÷"}
        left_has_op = any(op in left for op in ops)
        right_has_op = any(op in right for op in ops)
        # If target is on the left (e.g. "72 = 50 + 25 - 3"), pick the right side with operators
        if right_has_op and not left_has_op:
            answer = right
        else:
            answer = left

    return answer.replace("×", "*").replace("÷", "/").strip()


def safe_eval(expr):
    """Evaluate expr with exact fractions. Return (value, numbers_used) or None if not allowed."""
    if not expr or len(expr) > MAX_LEN:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    used = []

    def walk(node):
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        raise ValueError(f"not allowed: {type(node).__name__}")

    try:
        value = walk(tree.body)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None
    return value, used


def check_equation(expr, numbers, target):
    """True if expr uses each number exactly once and equals target."""
    result = safe_eval(expr)
    if result is None:
        return False
    value, used = result
    return sorted(used) == sorted(numbers) and value == target


def is_correct(text, numbers, target):
    expr = extract_answer(text)
    return expr is not None and check_equation(expr, numbers, target)


def has_format(text):
    return FORMAT_RE.fullmatch(text) is not None


# ---- reward functions (TRL GRPOTrainer signature) ----
# TRL passes dataset columns (numbers, target) as keyword arguments.

def correctness_reward(completions, numbers, target, **kwargs):
    return [
        1.0 if is_correct(get_text(c), n, t) else 0.0
        for c, n, t in zip(completions, numbers, target)
    ]


def format_reward(completions, **kwargs):
    return [0.1 if has_format(get_text(c)) else 0.0 for c in completions]

"""Tests for the Countdown checker. Run with: python -m pytest -v"""
import random

from src.data.generate import generate_split
from src.reward.verify import (correctness_reward, format_reward,
                               has_format, is_correct)

NUMS, TARGET = [3, 25, 50], 72


def ans(expr):
    return f"<think>...</think><answer>{expr}</answer>"


# ---- correct answers ----
def test_correct():
    assert is_correct(ans("50 + 25 - 3"), NUMS, TARGET)

def test_order_does_not_matter():
    assert is_correct(ans("25 - 3 + 50"), NUMS, TARGET)

def test_equals_sign_allowed():
    assert is_correct(ans("50 + 25 - 3 = 72"), NUMS, TARGET)

def test_target_on_left_allowed():
    assert is_correct(ans("72 = 50 + 25 - 3"), NUMS, TARGET)

def test_markdown_backticks_allowed():
    assert is_correct(ans("`50 + 25 - 3`"), NUMS, TARGET)

def test_unicode_ops():
    assert is_correct(ans("(34 ÷ 17) × (82 - 50)"), [34, 82, 50, 17], 64)

def test_fraction_inside_is_ok():
    assert is_correct(ans("(3 / 2) * 4"), [2, 3, 4], 6)

def test_last_answer_is_used():
    text = "<answer>1 + 1</answer> wait <answer>50 + 25 - 3</answer>"
    assert is_correct(text, NUMS, TARGET)


# ---- wrong answers ----
def test_wrong_value():
    assert not is_correct(ans("50 + 25 + 3"), NUMS, TARGET)

def test_missing_number():
    assert not is_correct(ans("50 + 22"), NUMS, TARGET)

def test_reused_number():
    assert not is_correct(ans("50 + 25 - 3 - 3 + 3"), NUMS, TARGET)

def test_extra_number():
    assert not is_correct(ans("(50 + 25 - 3) / 1"), NUMS, TARGET)

def test_negative_sign_rejected():
    assert not is_correct(ans("-3 + 50 + 25"), NUMS, TARGET)

def test_decimal_rejected():
    assert not is_correct(ans("50 + 25 - 3.0"), NUMS, TARGET)

def test_power_rejected():
    assert not is_correct(ans("50 + 25 - 3 ** 1"), NUMS, TARGET)

def test_code_injection_rejected():
    assert not is_correct(ans("__import__('os').system('dir')"), NUMS, TARGET)

def test_division_by_zero():
    assert not is_correct(ans("50 / (25 - 25) + 3"), [50, 25, 25, 3], 72)

def test_no_answer_tag():
    assert not is_correct("the answer is 50 + 25 - 3", NUMS, TARGET)

def test_too_long_rejected():
    assert not is_correct(ans("(" * 150 + "50 + 25 - 3" + ")" * 150), NUMS, TARGET)


# ---- format ----
def test_format_ok():
    assert has_format(ans("50 + 25 - 3"))

def test_format_prefilled_think():
    # Prompt prefill puts <think> in the prompt; completion starts after <think>
    assert has_format("reasoning step by step </think><answer>50 + 25 - 3</answer>")

def test_format_missing_think():
    assert not has_format("<answer>50 + 25 - 3</answer>")

def test_format_extra_text_after():
    assert not has_format(ans("50 + 25 - 3") + " hello")


# ---- reward functions (TRL style) ----
def test_rewards_plain_text():
    comps = [ans("50 + 25 - 3"), "nothing"]
    assert correctness_reward(comps, [NUMS] * 2, [TARGET] * 2) == [1.0, 0.0]
    assert format_reward(comps) == [0.1, 0.0]

def test_rewards_chat_format():
    comps = [[{"role": "assistant", "content": ans("50 + 25 - 3")}]]
    assert correctness_reward(comps, [NUMS], [TARGET]) == [1.0]
    assert format_reward(comps) == [0.1]

def test_rewards_prefilled_completion():
    comps = ["step by step reasoning </think><answer>50 + 25 - 3</answer>"]
    assert correctness_reward(comps, [NUMS], [TARGET]) == [1.0]
    assert format_reward(comps) == [0.1]


# ---- generator + verifier agree ----
def test_generator_solutions_pass_verifier():
    cfg = {"num_min": 1, "num_max": 100, "t_min": 10, "t_max": 100}
    puzzles = generate_split(200, [3, 4], random.Random(0), set(), cfg)
    for p in puzzles:
        assert is_correct(ans(p["solution"]), p["numbers"], p["target"]), p

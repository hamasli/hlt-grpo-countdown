"""
test_verify.py - unit tests for the Countdown verifier (src/reward/verify.py)
============================================================================

WHAT
    Calls the functions of verify.py with answers whose correct result we
    already know, and checks that the verifier gives that result.

WHY
    The verifier is our reward during GRPO training. If it made mistakes,
    the model would be rewarded for wrong answers and learn the wrong thing.

RUN
    python -m pytest -v
"""
import random

from src.data.generate import generate_split
from src.reward.verify import (correctness_reward, format_reward,
                               has_format, is_correct)

# Example puzzles used by many tests
NUMS, TARGET = [3, 25, 50], 72        # 3-number puzzle (stage 2 level)
NUMS2, TARGET2 = [2, 9], 18           # 2-number puzzle (stage 1 level)


def ans(expr):
    """Wrap an equation in the expected output format."""
    return f"<think>...</think><answer>{expr}</answer>"


# ---- correct answers: must be ACCEPTED ----
def test_correct():
    assert is_correct(ans("50 + 25 - 3"), NUMS, TARGET)

def test_order_does_not_matter():
    assert is_correct(ans("25 - 3 + 50"), NUMS, TARGET)

def test_equals_sign_allowed():
    assert is_correct(ans("50 + 25 - 3 = 72"), NUMS, TARGET)

def test_unicode_ops():
    assert is_correct(ans("(34 ÷ 17) × (82 - 50)"), [34, 82, 50, 17], 64)

def test_fraction_inside_is_ok():
    assert is_correct(ans("(3 / 2) * 4"), [2, 3, 4], 6)

def test_last_answer_is_used():
    text = "<answer>1 + 1</answer> wait <answer>50 + 25 - 3</answer>"
    assert is_correct(text, NUMS, TARGET)


# ---- NEW: 2-number puzzles (stage 1 of the curriculum) ----
def test_two_numbers_correct():
    # 2 * 9 = 18 uses both numbers once
    assert is_correct(ans("2 * 9"), NUMS2, TARGET2)

def test_two_numbers_wrong_value():
    # 2 + 9 = 11, not 18
    assert not is_correct(ans("2 + 9"), NUMS2, TARGET2)

def test_two_numbers_reused():
    # 9 + 9 = 18 is the right value, but 9 is used twice and 2 is missing
    assert not is_correct(ans("9 + 9"), NUMS2, TARGET2)

def test_two_numbers_only_target():
    # writing just the target uses no given number
    assert not is_correct(ans("18"), NUMS2, TARGET2)


# ---- wrong answers: must be REJECTED ----
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


# ---- NEW: edge cases ----
def test_first_correct_last_wrong():
    # only the LAST answer counts: here it is wrong, so the whole answer is wrong
    text = "<answer>50 + 25 - 3</answer> no wait <answer>50 + 25 + 3</answer>"
    assert not is_correct(text, NUMS, TARGET)

def test_empty_answer():
    assert not is_correct(ans(""), NUMS, TARGET)


# ---- format ----
def test_format_ok():
    assert has_format(ans("50 + 25 - 3"))

def test_format_missing_think():
    assert not has_format("<answer>50 + 25 - 3</answer>")

def test_format_extra_text_after():
    assert not has_format(ans("50 + 25 - 3") + " hello")

def test_format_extra_text_before():
    # NEW: text before <think> also breaks the format
    assert not has_format("Sure! " + ans("50 + 25 - 3"))


# ---- reward functions, called the way TRL calls them ----
def test_rewards_plain_text():
    comps = [ans("50 + 25 - 3"), "nothing"]
    assert correctness_reward(comps, [NUMS] * 2, [TARGET] * 2) == [1.0, 0.0]
    assert format_reward(comps) == [0.1, 0.0]

def test_rewards_chat_format():
    comps = [[{"role": "assistant", "content": ans("50 + 25 - 3")}]]
    assert correctness_reward(comps, [NUMS], [TARGET]) == [1.0]
    assert format_reward(comps) == [0.1]


# ---- generator + verifier agree ----
def test_generator_solutions_pass_verifier():
    # every solution made by the generator must pass the verifier,
    # for the training levels (2, 3) and the out-of-domain level (4)
    cfg = {"num_min": 1, "num_max": 100, "t_min": 10, "t_max": 100}
    puzzles = generate_split(200, [2, 3, 4], random.Random(0), set(), cfg)
    for p in puzzles:
        assert is_correct(ans(p["solution"]), p["numbers"], p["target"]), p
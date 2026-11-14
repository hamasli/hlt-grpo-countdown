"""
trick_baselines.py - how many puzzles can simple tricks solve, without any model?
=================================================================================

WHAT
    Tries simple "no-thinking" strategies on the test puzzles and measures how
    many they solve, using the same verifier as the model evaluation.

WHY
    1. To prove the task is not trivial (tricks should solve few puzzles).
    2. To set the bar the model must beat: a model that solves fewer puzzles
       than random guessing is not reasoning.

THE TRICKS
    target_in_numbers : write one of the given numbers (works only if the target
                        is one of them; our generator removes those puzzles, so ~0%)
    sum_all           : add all the numbers
    product_all       : multiply all the numbers
    random_guess@1    : one random valid equation (each number once)
    random_guess@4    : 4 random equations, solved if ANY is correct.
                        Same budget as GRPO, which samples 4 answers per puzzle.

RUN
    python -m src.eval.trick_baselines                              (test set: 2 and 3 numbers)
    python -m src.eval.trick_baselines --data data/test_ood.jsonl   (4-number OOD test)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.generate import random_expression
from src.reward.verify import check_equation


def trick_sum(nums):
    return " + ".join(map(str, nums))


def trick_product(nums):
    return " * ".join(map(str, nums))


def make_random_guess(rng, tries):
    """Return a trick that writes `tries` random equations and keeps the first
    correct one (like pass@k for random guessing). No correct one -> a sum."""
    def guess(nums, target):
        for _ in range(tries):
            r = random_expression(nums, rng)
            if r is not None and r[0] == target:
                return r[1]
        return trick_sum(nums)
    return guess


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description="Trick baselines on Countdown")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    puzzles = load_jsonl(args.data)
    rng = random.Random(args.seed)
    tricks = {
        "sum_all": lambda n, t: trick_sum(n),
        "product_all": lambda n, t: trick_product(n),
        "random_guess@1": make_random_guess(rng, 1),
        "random_guess@4": make_random_guess(rng, 4),
    }

    results = {}
    # target_in_numbers: try every given number as the answer
    solved, total = defaultdict(int), defaultdict(int)
    for p in puzzles:
        total[p["n_numbers"]] += 1
        solved[p["n_numbers"]] += any(check_equation(str(x), p["numbers"], p["target"])
                                      for x in p["numbers"])
    results["target_in_numbers"] = (solved, total)

    for name, trick in tricks.items():
        solved, total = defaultdict(int), defaultdict(int)
        for p in puzzles:
            total[p["n_numbers"]] += 1
            solved[p["n_numbers"]] += check_equation(trick(p["numbers"], p["target"]),
                                                     p["numbers"], p["target"])
        results[name] = (solved, total)

    # percentages per level + all
    table = {}
    for name, (solved, total) in results.items():
        row = {f"{lv}_numbers": round(100 * solved[lv] / total[lv], 2) for lv in sorted(total)}
        row["all"] = round(100 * sum(solved.values()) / sum(total.values()), 2)
        table[name] = row

    cols = list(next(iter(table.values())))
    print(f"{'trick':<20}" + "".join(f"{c:>12}" for c in cols))
    for name, row in table.items():
        print(f"{name:<20}" + "".join(f"{row[c]:>11.2f}%" for c in cols))

    # e.g. results/trick_baselines.json or results/trick_baselines_test_ood.json
    stem = Path(args.data).stem
    out = Path("results") / ("trick_baselines.json" if stem == "test" else f"trick_baselines_{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
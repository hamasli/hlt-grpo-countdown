"""How many test puzzles do simple tricks solve? (should be low)

Run: python -m src.eval.trick_baselines
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.generate import random_expression
from src.reward.verify import check_equation


def trick_in_numbers(nums):
    return str(nums[0])  # just write one number (works only if target is in the list)


def trick_sum(nums):
    return " + ".join(map(str, nums))


def trick_product(nums):
    return " * ".join(map(str, nums))


def make_random_guess(rng, tries):
    """Random equation; solved if any of `tries` guesses is correct (like pass@k)."""
    def guess(nums, target):
        for _ in range(tries):
            r = random_expression(nums, rng)
            if r is not None and r[0] == target:
                return r[1]
        return trick_sum(nums)  # no valid guess -> fallback
    return guess


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--out", default="results/trick_baselines.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    puzzles = load_jsonl(args.data)
    rng = random.Random(args.seed)
    tricks = {
        "target_in_numbers": lambda n, t: trick_in_numbers(n),
        "sum_all": lambda n, t: trick_sum(n),
        "product_all": lambda n, t: trick_product(n),
        "random_guess@1": make_random_guess(rng, 1),
        "random_guess@8": make_random_guess(rng, 8),
    }

    results = {}
    for name, trick in tricks.items():
        solved, total = defaultdict(int), defaultdict(int)
        for p in puzzles:
            level = p["n_numbers"]
            total[level] += 1
            # for target_in_numbers: try every number
            if name == "target_in_numbers":
                ok = any(check_equation(str(x), p["numbers"], p["target"]) for x in p["numbers"])
            else:
                ok = check_equation(trick(p["numbers"], p["target"]), p["numbers"], p["target"])
            solved[level] += ok
        results[name] = {f"{lv}_numbers": round(100 * solved[lv] / total[lv], 2) for lv in sorted(total)}
        results[name]["all"] = round(100 * sum(solved.values()) / sum(total.values()), 2)

    print(f"{'trick':<20}" + "".join(f"{k:>12}" for k in next(iter(results.values()))))
    for name, r in results.items():
        print(f"{name:<20}" + "".join(f"{v:>11.2f}%" for v in r.values()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
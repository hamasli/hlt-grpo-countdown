"""Countdown puzzle generator.

Builds solvable puzzles: pick random numbers, combine them with a random
equation, and use the result as the target.
"""
import argparse
import json
import random
from pathlib import Path

OPS = ["+", "-", "*", "/"]


def combine(a, b, op):
    """Combine two (value, expr) items with op. Return None if the step is not allowed."""
    (va, ea), (vb, eb) = a, b
    if op in ("-", "/") and va < vb:  # bigger number first -> positive results
        (va, ea), (vb, eb) = (vb, eb), (va, ea)
    if op == "+":
        return va + vb, f"({ea} + {eb})"
    if op == "-":
        if va == vb:  # result 0 is useless
            return None
        return va - vb, f"({ea} - {eb})"
    if op == "*":
        return va * vb, f"({ea} * {eb})"
    if op == "/":
        if va % vb != 0:  # only exact division
            return None
        return va // vb, f"({ea} / {eb})"


def random_expression(numbers, rng):
    """Combine all numbers (each used once) into one random equation."""
    pool = [(n, str(n)) for n in numbers]
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        new = combine(pool[i], pool[j], rng.choice(OPS))
        if new is None:
            return None
        pool = [p for k, p in enumerate(pool) if k not in (i, j)] + [new]
    value, expr = pool[0]
    return value, expr[1:-1]  # remove outer brackets


def make_puzzle(n_numbers, rng, num_min, num_max, t_min, t_max, max_tries=1000):
    """Try random numbers + equations until the target is in range."""
    for _ in range(max_tries):
        numbers = [rng.randint(num_min, num_max) for _ in range(n_numbers)]
        result = random_expression(numbers, rng)
        if result is None:
            continue
        target, solution = result
        if t_min <= target <= t_max and target not in numbers:
            return {"numbers": numbers, "target": target,
                    "n_numbers": n_numbers, "solution": solution}
    return None


def generate_split(size_per_level, levels, rng, seen, cfg):
    """Make one split with size_per_level puzzles for each level. `seen` blocks duplicates."""
    puzzles = []
    for n in levels:
        count = 0
        while count < size_per_level:
            p = make_puzzle(n, rng, **cfg)
            if p is None:
                continue
            key = (n, tuple(sorted(p["numbers"])), p["target"])
            if key in seen:
                continue
            seen.add(key)
            puzzles.append(p)
            count += 1
    rng.shuffle(puzzles)
    return [{"id": i, **p} for i, p in enumerate(puzzles)]


def save_jsonl(puzzles, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in puzzles:
            f.write(json.dumps(p) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Countdown puzzles")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--train", type=int, default=5000, help="per level")
    parser.add_argument("--val", type=int, default=100, help="per level")
    parser.add_argument("--test", type=int, default=500, help="per level")
    args = parser.parse_args()

    cfg = {"num_min": 1, "num_max": 100, "t_min": 10, "t_max": 100}
    rng = random.Random(args.seed)
    seen = set()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # test first: the test set stays the same even if we change the train size
    for name, size in [("test", args.test), ("val", args.val), ("train", args.train)]:
        puzzles = generate_split(size, args.levels, rng, seen, cfg)
        save_jsonl(puzzles, out / f"{name}.jsonl")
        print(f"{name}: {len(puzzles)} puzzles -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
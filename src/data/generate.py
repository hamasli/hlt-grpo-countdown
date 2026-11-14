"""
generate.py - Countdown puzzle generator
=========================================

WHAT
    Creates the Countdown puzzles for our project and saves them as .jsonl files
    (one puzzle per line) in data/.

WHY
    We build our own data so that (1) every puzzle is guaranteed to have a
    solution, (2) we control the difficulty (how many numbers), and (3) the
    dataset is reproducible: the same seed always gives the same puzzles.

HOW A PUZZLE IS MADE
    1. Pick N random numbers between 1 and 100.
    2. Combine them with a random equation (+ - * /), using each number once.
    3. The result of that equation becomes the target (kept only if 10-100).
    -> a solution exists by construction (the equation we just built).

FILES PRODUCED (new setting, Nov 2026)
    data/train.jsonl     5,000 x 2-number + 5,000 x 3-number   (GRPO training)
    data/val.jsonl         100 x 2-number +   100 x 3-number   (checks during training)
    data/test.jsonl        500 x 2-number +   500 x 3-number   (final results)
    data/test_ood.jsonl    500 x 4-number                      (out-of-domain test:
                                                                harder than anything
                                                                seen in training)

    The test files are created FIRST, so changing the train size later never
    changes the test puzzles. A shared `seen` set makes sure no puzzle appears
    in two files.

RUN
    python -m src.data.generate
"""
import argparse
import json
import random
from pathlib import Path

OPS = ["+", "-", "*", "/"]


def combine(a, b, op):
    """Combine two (value, text) items with one operation.

    Returns the new (value, text) item, or None if the step is not allowed.
    Rules keep every intermediate result a positive whole number:
      - for - and /, the bigger number goes first (no negative results)
      - a result of 0 is rejected (useless for the next steps)
      - division is allowed only when it has no remainder
    """
    (va, ea), (vb, eb) = a, b
    if op in ("-", "/") and va < vb:
        (va, ea), (vb, eb) = (vb, eb), (va, ea)
    if op == "+":
        return va + vb, f"({ea} + {eb})"
    if op == "-":
        if va == vb:
            return None
        return va - vb, f"({ea} - {eb})"
    if op == "*":
        return va * vb, f"({ea} * {eb})"
    if op == "/":
        if va % vb != 0:
            return None
        return va // vb, f"({ea} / {eb})"


def random_expression(numbers, rng):
    """Build one random equation that uses every number exactly once.

    Repeats: pick 2 items from the pool, combine them, put the result back,
    until one item is left. Returns (value, equation_text) or None.
    """
    pool = [(n, str(n)) for n in numbers]
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        new = combine(pool[i], pool[j], rng.choice(OPS))
        if new is None:
            return None
        pool = [p for k, p in enumerate(pool) if k not in (i, j)] + [new]
    value, expr = pool[0]
    return value, expr[1:-1]  # remove the outer brackets


def make_puzzle(n_numbers, rng, num_min, num_max, t_min, t_max, max_tries=1000):
    """Try random numbers + equations until the target is in [t_min, t_max]
    and is not simply one of the given numbers (that would be trivial)."""
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
    """Make one split: `size_per_level` puzzles for EACH level in `levels`.

    `seen` holds every puzzle already created (in any split), so there are
    no duplicates and no overlap between train / val / test.
    The `solution` field is only for checking the generator; the model never sees it.
    """
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
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3],
                        help="numbers per puzzle used for train/val/test")
    parser.add_argument("--ood_level", type=int, default=4,
                        help="harder level used only for the out-of-domain test (0 = none)")
    parser.add_argument("--train", type=int, default=5000, help="per level")
    parser.add_argument("--val", type=int, default=100, help="per level")
    parser.add_argument("--test", type=int, default=500, help="per level")
    parser.add_argument("--ood_test", type=int, default=500)
    args = parser.parse_args()

    cfg = {"num_min": 1, "num_max": 100, "t_min": 10, "t_max": 100}
    rng = random.Random(args.seed)
    seen = set()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Order matters: test sets first, so they never change if train size changes.
    splits = [("test", args.levels, args.test)]
    if args.ood_level:
        splits.append(("test_ood", [args.ood_level], args.ood_test))
    splits += [("val", args.levels, args.val), ("train", args.levels, args.train)]

    for name, levels, size in splits:
        puzzles = generate_split(size, levels, rng, seen, cfg)
        save_jsonl(puzzles, out / f"{name}.jsonl")
        print(f"{name:9s}: {len(puzzles):6d} puzzles, levels {levels} -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
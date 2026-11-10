"""Why do the models fail? Sort every answer into one error category.

Reads all results/*_outputs.jsonl files (no GPU needed).
Run: python -m src.eval.error_analysis
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.reward.verify import extract_answer, safe_eval

CATEGORIES = ["correct", "no_answer", "invalid", "wrong_numbers", "wrong_value"]


def classify(text, numbers, target):
    """Return (category, details). Checks run in order; the first failing check wins."""
    expr = extract_answer(text)
    if expr is None:
        return "no_answer", {}
    result = safe_eval(expr)
    if result is None:
        return "invalid", {"expr": expr}
    value, used = result
    given, got = Counter(numbers), Counter(used)
    if given != got:
        return "wrong_numbers", {
            "expr": expr,
            "reused": any(got[n] > given[n] for n in given),   # given number used too many times
            "missing": any(got[n] < given[n] for n in given),  # given number not used
            "invented": any(n not in given for n in got),      # number that was not given
        }
    if value != target:
        return "wrong_value", {"expr": expr, "distance": abs(float(value - target))}
    return "correct", {"expr": expr}


def analyse(records):
    """Percentages per category + sub-types + mean distance."""
    counts, subtypes, distances = Counter(), Counter(), []
    for r in records:
        cat, d = classify(r["output"], r["numbers"], r["target"])
        counts[cat] += 1
        if cat == "wrong_numbers":
            subtypes.update(k for k in ("reused", "missing", "invented") if d[k])
        if cat == "wrong_value":
            distances.append(d["distance"])
    n = len(records)
    out = {c: round(100 * counts[c] / n, 1) for c in CATEGORIES}
    wn = counts["wrong_numbers"]
    out["wrong_numbers_detail"] = {k: round(100 * subtypes[k] / wn, 1) if wn else 0.0
                                   for k in ("reused", "missing", "invented")}
    out["wrong_value_mean_distance"] = round(sum(distances) / len(distances), 1) if distances else None
    out["n"] = n
    return out


def examples(records, k=1):
    """First k examples for each category."""
    ex = defaultdict(list)
    for r in records:
        cat, d = classify(r["output"], r["numbers"], r["target"])
        if len(ex[cat]) < k:
            ex[cat].append({"numbers": r["numbers"], "target": r["target"],
                            "answer": d.get("expr", r["output"][-80:])})
    return ex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default="results/error_analysis.json")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.results_dir}/*_outputs.jsonl"))
    if not files:
        print("No *_outputs.jsonl files found. Run evaluate.py first.")
        return

    report = {}
    header = f"{'run':<34}{'level':<11}" + "".join(f"{c:>15}" for c in CATEGORIES)
    print(header + "\n" + "-" * len(header))
    for f in files:
        tag = Path(f).name.replace("_outputs.jsonl", "")
        with open(f, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        levels = defaultdict(list)
        for r in records:
            levels[f"{r['n_numbers']}_numbers"].append(r)
        levels["all"] = records

        report[tag] = {lv: analyse(rs) for lv, rs in sorted(levels.items())}
        report[tag]["examples"] = examples(records)
        for lv in sorted(levels):
            a = report[tag][lv]
            print(f"{tag:<34}{lv:<11}" + "".join(f"{a[c]:>14}%" for c in CATEGORIES))

    print("\nDetails (all levels):")
    for tag, rep in report.items():
        a = rep["all"]
        wd = a["wrong_numbers_detail"]
        print(f"  {tag}: wrong_numbers -> reused {wd['reused']}%, missing {wd['missing']}%, "
              f"invented {wd['invented']}% | wrong_value mean distance = {a['wrong_value_mean_distance']}")

    print("\nExamples:")
    for tag, rep in report.items():
        print(f"  {tag}")
        for cat, exs in rep["examples"].items():
            for e in exs:
                print(f"    [{cat}] numbers={e['numbers']} target={e['target']} -> {e['answer']!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

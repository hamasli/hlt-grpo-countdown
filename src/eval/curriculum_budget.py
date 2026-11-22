"""Compare a Stage 2 adapter's 256- and 512-token answers on the same OOD puzzles."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.curriculum import validate_curriculum
from src.eval.curriculum import (
    adapter_identity,
    evaluate_split,
    extract_curriculum_answer,
    score_curriculum_response,
    summarize,
)
from src.eval.reward_grid import load_jsonl
from src.reward.verify import safe_eval


def failure_reason(row):
    if row["correct"]:
        return "correct"
    expression = extract_curriculum_answer(row["output"])
    if expression is None:
        if "<answer>" in row["output"] or row["hit_token_cap"]:
            return "incomplete_answer"
        return "missing_answer"
    evaluated = safe_eval(expression)
    if evaluated is None:
        return "invalid_expression"
    _, numbers_used = evaluated
    return "wrong_numbers" if sorted(numbers_used) != sorted(row["numbers"]) else "wrong_value"


def match_responses(puzzles, records, cap):
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != {row["id"] for row in puzzles}:
        raise ValueError("Responses do not cover the same unique puzzle IDs.")
    ordered = []
    for puzzle in puzzles:
        row = by_id[puzzle["id"]]
        if any(row[field] != puzzle[field] for field in ("numbers", "target")):
            raise ValueError(f"Puzzle contents differ for ID {puzzle['id']}.")
        if (row["correct"], row["format_ok"]) != score_curriculum_response(
            row["output"], puzzle["numbers"], puzzle["target"]
        ) or row["hit_token_cap"] != (row["tokens"] >= cap):
            raise ValueError(f"Saved verifier result or cap flag differs for ID {puzzle['id']}.")
        ordered.append(row)
    return ordered


def analyze(short, long):
    pairs = []
    counts = Counter()
    capped_wrong = capped_recovered = 0
    for before, after in zip(short, long, strict=True):
        if before["correct"]:
            transition = "still_correct" if after["correct"] else "regressed"
        else:
            transition = "recovered" if after["correct"] else "still_wrong"
        counts[transition] += 1
        if not before["correct"] and before["hit_token_cap"]:
            capped_wrong += 1
            capped_recovered += int(after["correct"])
        pairs.append({
            "id": before["id"], "transition": transition,
            "failure_256": failure_reason(before), "failure_512": failure_reason(after),
            "hit_256_cap": before["hit_token_cap"], "hit_512_cap": after["hit_token_cap"],
            "tokens_256": before["tokens"], "tokens_512": after["tokens"],
        })
    short_metrics, long_metrics = summarize(short), summarize(long)
    return pairs, {
        "at_256": short_metrics, "at_512": long_metrics,
        "strict_solve_change_pp": round(long_metrics["strict_solve_rate_pct"] - short_metrics["strict_solve_rate_pct"], 8),
        "transitions": {key: counts[key] for key in ("still_correct", "recovered", "regressed", "still_wrong")},
        "capped_wrong_at_256": capped_wrong, "recovered_capped_failures": capped_recovered,
        "failure_counts_256": dict(Counter(failure_reason(row) for row in short if not row["correct"])),
        "failure_counts_512": dict(Counter(failure_reason(row) for row in long if not row["correct"])),
    }


def run(data_dir, reference, adapter, out_dir, limit):
    manifest = validate_curriculum(data_dir)
    reference_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    split = manifest["splits"]["test_ood"]
    if (
        reference_summary["benchmark_commit"] != manifest["source_commit"]
        or reference_summary["split_sha256"]["test_ood"] != split["sha256"]
        or reference_summary["protocol"] != protocol
        or protocol["evaluation_max_new_tokens"] != 256
        or reference_summary["decoding"] != "greedy"
        or reference_summary["smoke_test"]
    ):
        raise ValueError("Expected a full, matching Objective 8 evaluation at 256 tokens.")
    revision = reference_summary["model_revision"]
    identity = adapter_identity(adapter, protocol["base_model"], revision)
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if identity[key] != reference_summary[key]:
            raise ValueError(f"The adapter differs from Objective 8 ({key}).")

    puzzles = load_jsonl(data_dir / split["file"])
    short = match_responses(puzzles, load_jsonl(reference / "stage2/test_ood.jsonl"), 256)
    if summarize(short) != reference_summary["metrics"]["stage2"]["4_numbers"]:
        raise ValueError("The 256-token responses differ from the Objective 8 summary.")
    puzzles, short = puzzles[:limit], short[:limit]
    if out_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {out_dir}")

    import peft
    import torch
    import transformers
    from src.eval.evaluate import load_model

    torch.manual_seed(reference_summary["seed"])
    tokenizer, model, device = load_model(protocol["base_model"], adapter, revision=revision)
    # Group rows as in Objective 8. Comparing the full set uses identical batches.
    long = evaluate_split(
        tokenizer, model, device, puzzles, {**protocol, "evaluation_max_new_tokens": 512},
        reference_summary["batch_size"], "OOD at 512 tokens",
    )
    long = match_responses(puzzles, long, 512)
    pairs, report = analyze(short, long)
    environment = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    objective_8_environment = {key: reference_summary[key] for key in environment}
    report.update({
        **identity, "benchmark_commit": manifest["source_commit"], "ood_sha256": split["sha256"],
        "protocol": protocol, "decoding": "greedy", "batch_size": reference_summary["batch_size"],
        "smoke_test": limit is not None, "environment_256": objective_8_environment,
        "environment_512": environment, "same_recorded_environment": objective_8_environment == environment,
    })
    out_dir.mkdir(parents=True)
    for name, rows in (("test_ood_512.jsonl", long), ("paired.jsonl", pairs)):
        (out_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Strict solve change: {report['strict_solve_change_pp']:+.2f} pp. {report['transitions']}")
    print(f"Capped failures recovered: {report['recovered_capped_failures']}/{report['capped_wrong_at_256']}.")
    if not report["same_recorded_environment"]:
        print("The 256- and 512-token evaluations used different recorded environments.")
    print(f"Saved Objective 9 results to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Extracted Objective 8 evaluation directory")
    parser.add_argument("--adapter", type=Path, required=True, help="The same Stage 2 final LoRA directory")
    parser.add_argument("--data-dir", type=Path, default=Path("data/curriculum"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/countdown-objective9"))
    parser.add_argument("--limit", type=int, help="Evaluate a small subset as a smoke test")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 500:
        parser.error("--limit must be between 1 and 500.")
    run(args.data_dir, args.reference, args.adapter, args.out_dir, args.limit)


if __name__ == "__main__":
    main()


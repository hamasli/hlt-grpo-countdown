"""Measure the GRPO reward signal on fresh base-model validation completions."""

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from src.eval.reward_grid import (
    RewardConfig,
    group_advantages,
    load_jsonl,
    sample_responses,
    score_groups,
    summarize,
)
from src.prompts import PREFILL, TEMPLATES

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SETTINGS = dict(samples=4, temperature=0.8, max_tokens=512, batch_size=1, seed=42)
POLICIES = {
    "correctness_only": RewardConfig(1, 0),
    "additive_0.01": RewardConfig(1, 0.01),
    "additive_0.1": RewardConfig(1, 0.1),
    "gated_0.1": RewardConfig(1, 0.1, True),
}


def select_puzzles(puzzles, per_size, seed):
    rng = random.Random(seed)
    selected = []
    for size in (3, 4):
        pool = [puzzle for puzzle in puzzles if len(puzzle["numbers"]) == size]
        if not 1 <= per_size <= len(pool):
            raise ValueError(f"Expected 1 to {len(pool)} puzzles for {size} inputs, got {per_size}.")
        rng.shuffle(pool)
        selected.extend(pool[:per_size])
    return selected


def summarize_pilot(records):
    groups = score_groups(records)
    if any(size not in (3, 4) or len(group) != SETTINGS["samples"] for size, group in groups):
        raise ValueError("Expected groups of four completions for three- or four-input puzzles.")

    summary = {}
    for size in sorted({size for size, _ in groups}):
        rows = [(n, group) for n, group in groups if n == size]
        all_wrong = [group for _, group in rows if not any(correct for correct, _ in group)]
        format_counts = Counter(sum(formatted for _, formatted in group) for group in all_wrong)
        policies = {}
        for name, config in POLICIES.items():
            metrics = summarize(rows, config)["all"]
            positive = [
                advantage
                for group in all_wrong
                for advantage in group_advantages([config.score(*pair) for pair in group])
                if advantage > 0
            ]
            policies[name] = {
                "reward": asdict(config),
                "groups_with_format_only_signal": metrics["all_wrong_groups_with_format_signal"],
                "format_only_signal_rate_pct": metrics["all_wrong_format_signal_rate"],
                "mean_positive_advantage_in_all_wrong_groups": mean(positive) if positive else None,
                "wrong_bonus_rate_pct": metrics["wrong_bonus_rate"],
                "wrong_positive_advantage_rate_pct": metrics["wrong_positive_advantage_rate"],
            }

        summary[str(size)] = {
            "groups": len(rows),
            "completions": len(rows) * SETTINGS["samples"],
            "groups_with_a_solution": len(rows) - len(all_wrong),
            "all_wrong_groups": len(all_wrong),
            "all_wrong_groups_by_formatted_count": {
                str(count): format_counts[count] for count in range(SETTINGS["samples"] + 1)
            },
            "policies": policies,
        }
    return summary


def print_summary(summary):
    for size, metrics in summary.items():
        groups = metrics["groups"]
        solved = metrics["groups_with_a_solution"]
        wrong = metrics["all_wrong_groups"]
        print(f"\n{size} inputs: {solved}/{groups} groups have a solution. {wrong} groups are all wrong.")
        for name, policy in metrics["policies"].items():
            count = policy["groups_with_format_only_signal"]
            pct = policy["format_only_signal_rate_pct"]
            adv = policy["mean_positive_advantage_in_all_wrong_groups"]
            rate_text = f"{pct:.1f}%" if pct is not None else "n/a"
            adv_text = f"{adv:.4f}" if adv is not None else "n/a"
            print(f"  {name:17} {count}/{wrong} format-only ({rate_text}), mean positive advantage {adv_text}")


def run_pilot(data_path, output_dir, *, per_size=25, seed=42):
    import torch
    import transformers

    from src.eval.evaluate import load_model

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    responses_path = output_dir / "responses.jsonl"
    report_path = output_dir / "summary.json"
    if responses_path.exists() or report_path.exists():
        raise FileExistsError(f"Choose a new output directory to preserve the run in {output_dir}.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run the validation pilot in an environment with a CUDA GPU.")

    settings = SETTINGS | {"seed": seed}
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    selected = select_puzzles(load_jsonl(data_path), per_size, seed)
    print(f"Sampling {len(selected)} groups of {settings['samples']} responses from {MODEL}.", flush=True)
    tokenizer, model, _ = load_model(MODEL)
    records = sample_responses(model, tokenizer, selected, **settings)
    summary = summarize_pilot(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump({
            "model": MODEL,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "gpu": torch.cuda.get_device_name(model.device),
            "dtype": str(model.dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "data": data_path.name,
            "data_sha256": data_sha256,
            "settings": settings | {"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
            "prompt_template": TEMPLATES["grpo"],
            "assistant_prefill": PREFILL["grpo"],
            "responses": responses_path.name,
            "by_input_count": summary,
        }, stream, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print_summary(run_pilot(args.data, args.out_dir, per_size=args.per_size, seed=args.seed))
    print(f"Saved responses and summary in {args.out_dir}.")


if __name__ == "__main__":
    main()

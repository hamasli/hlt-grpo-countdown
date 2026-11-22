"""Reproduce the Stage 2 Countdown evaluation with matched base and LoRA runs."""

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.data.curriculum import validate_curriculum
from src.prompts import build_prompt
from src.reward.verify import ANSWER_RE, check_equation, has_format

# Reference counts from TODO.md at the curriculum source commit.
REFERENCE_SOLVES = {"2_numbers": 487, "3_numbers": 347, "in_domain": 834, "4_numbers": 82}
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
NOTATION = (
    ("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("×", "*"), ("÷", "/"),
    ("\\(", ""), ("\\)", ""), ("\\[", ""), ("\\]", ""),
)


def extract_curriculum_answer(text):
    """Use the same answer normalization as the reference curriculum run."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip().replace("`", "").strip()
    for before, after in NOTATION:
        answer = answer.replace(before, after)
    if "=" in answer:
        left, right, *_ = answer.split("=")
        ops = "+-*/"
        answer = right if any(op in right for op in ops) and not any(op in left for op in ops) else left
    return answer.strip()


def score_curriculum_response(text, numbers, target):
    return check_equation(extract_curriculum_answer(text), numbers, target), has_format(text)


def evaluate_split(tokenizer, model, device, puzzles, protocol, batch_size, label):
    from src.eval.evaluate import generate

    cap = protocol["evaluation_max_new_tokens"]
    method = protocol["prompt_method"]
    records = []
    for start in range(0, len(puzzles), batch_size):
        batch = puzzles[start:start + batch_size]
        prompts = [build_prompt(tokenizer, row["numbers"], row["target"], method) for row in batch]
        outputs, lengths = generate(tokenizer, model, device, prompts, cap)
        for row, suffix, length in zip(batch, outputs, lengths, strict=True):
            output = protocol["assistant_prefill"] + suffix
            correct, formatted = score_curriculum_response(output, row["numbers"], row["target"])
            records.append({
                "id": row["id"], "numbers": row["numbers"], "target": row["target"],
                "n_numbers": len(row["numbers"]), "output": output, "tokens": length,
                "hit_token_cap": length >= cap,
                "correct": correct, "format_ok": formatted,
            })
        if (start // batch_size + 1) % 5 == 0 or len(records) == len(puzzles):
            print(f"{label}: {len(records)}/{len(puzzles)} puzzles", flush=True)
    return records


def summarize(records):
    return {
        "puzzles": len(records),
        "solves": sum(row["correct"] for row in records),
        "strict_solve_rate_pct": 100 * mean(row["correct"] for row in records),
        "format_rate_pct": 100 * mean(row["format_ok"] for row in records),
        "formatted_wrong_rate_pct": 100 * mean(row["format_ok"] and not row["correct"] for row in records),
        "mean_tokens": mean(row["tokens"] for row in records),
        "token_cap_rate_pct": 100 * mean(row["hit_token_cap"] for row in records),
    }


def adapter_identity(adapter, model_name, revision=None):
    """Verify LoRA files and resolve the identity used by downstream evaluations."""
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 adapter file is missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != model_name or config.get("peft_type") != "LORA":
        raise ValueError(f"Expected a Stage 2 LoRA adapter for {model_name}: {adapter}")
    recorded_revision = config.get("revision")
    revision_file = adapter / "base_model_revision.txt"
    if revision_file.is_file():
        local_revision = revision_file.read_text(encoding="utf-8").strip()
        if not local_revision or (recorded_revision and recorded_revision != local_revision):
            raise ValueError("Conflicting or missing base model revision in the adapter metadata.")
        recorded_revision = local_revision
    if revision and recorded_revision and revision != recorded_revision:
        raise ValueError("The requested model revision differs from the adapter metadata.")
    revision_source = "argument" if revision else "adapter" if recorded_revision else "project_pin"
    revision = revision or recorded_revision or MODEL_REVISION
    with weights_path.open("rb") as stream:
        weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "adapter": str(adapter.resolve()), "adapter_sha256": weights_hash,
        "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_revision": revision, "model_revision_source": revision_source,
    }


def run(data_dir, adapter, out_dir, batch_size, limit_per_size, revision):
    manifest = validate_curriculum(data_dir)
    protocol = manifest["protocol"]
    model_name = protocol["base_model"]

    identity = adapter_identity(adapter, model_name, revision)
    revision = identity["model_revision"]

    import peft
    import torch
    import transformers
    from src.eval.evaluate import load_jsonl, load_model

    # The reference evaluator disables optional torchao integration for standard LoRA.
    peft.import_utils._torchao_available = False
    peft.import_utils.is_torchao_available = lambda: False
    torch.manual_seed(0)
    puzzles = {}
    for split in ("test", "test_ood"):
        rows = load_jsonl(data_dir / manifest["splits"][split]["file"])
        if limit_per_size is not None:
            counts = Counter()
            rows_subset = []
            for row in rows:
                size = len(row["numbers"])
                if counts[size] < limit_per_size:
                    rows_subset.append(row)
                    counts[size] += 1
            rows = rows_subset
        puzzles[split] = rows

    out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "benchmark_commit": manifest["source_commit"],
        "split_sha256": {split: manifest["splits"][split]["sha256"] for split in puzzles},
        "protocol": protocol, **identity,
        "decoding": "greedy", "batch_size": batch_size, "seed": 0,
        "smoke_test": limit_per_size is not None, "limit_per_size": limit_per_size,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "metrics": {},
    }
    for arm, adapter_path in (("base", None), ("stage2", adapter)):
        tokenizer, model, device = load_model(model_name, adapter_path, revision=revision)
        report["device"] = torch.cuda.get_device_name(0) if device == "cuda" else device
        directory = out_dir / arm
        directory.mkdir()
        results = {}
        for split, rows in puzzles.items():
            results[split] = evaluate_split(tokenizer, model, device, rows, protocol, batch_size, f"{arm}/{split}")
            with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in results[split]:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["metrics"][arm] = {
            "2_numbers": summarize([row for row in results["test"] if row["n_numbers"] == 2]),
            "3_numbers": summarize([row for row in results["test"] if row["n_numbers"] == 3]),
            "in_domain": summarize(results["test"]),
            "4_numbers": summarize(results["test_ood"]),
        }
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    report["stage2_minus_base_pp"] = {
        level: round(metrics["strict_solve_rate_pct"] - report["metrics"]["base"][level]["strict_solve_rate_pct"], 8)
        for level, metrics in report["metrics"]["stage2"].items()
    }
    if limit_per_size is None:
        report["reference_comparison"] = {
            level: {"reported_solves": count, "observed_solves": report["metrics"]["stage2"][level]["solves"],
                    "solve_count_difference": report["metrics"]["stage2"][level]["solves"] - count}
            for level, count in REFERENCE_SOLVES.items()
        }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for level in REFERENCE_SOLVES:
        base, trained = (report["metrics"][arm][level] for arm in ("base", "stage2"))
        print(f"{level}: base {base['solves']}/{base['puzzles']} ({base['strict_solve_rate_pct']:.2f}%)"
              f" · Stage 2 {trained['solves']}/{trained['puzzles']} ({trained['strict_solve_rate_pct']:.2f}%)"
              f" · {report['stage2_minus_base_pp'][level]:+.2f} pp")
    print(f"Saved completions and summary to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="Stage 2 final LoRA directory")
    parser.add_argument("--data-dir", type=Path, default=Path("data/curriculum"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/countdown-objective8/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-size", type=int, help="Smoke test on this many puzzles per input count")
    parser.add_argument("--revision", help="Exact base model revision used during training, if known")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_per_size is not None and args.limit_per_size < 1):
        parser.error("Batch size and puzzle limit must be positive.")
    run(args.data_dir, args.adapter, args.out_dir, args.batch_size, args.limit_per_size, args.revision)


if __name__ == "__main__":
    main()

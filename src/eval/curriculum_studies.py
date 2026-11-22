"""Run the budget-matched curriculum and reward comparisons."""

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from src.data.curriculum import validate_curriculum
from src.eval.curriculum import MODEL_REVISION, adapter_identity, evaluate_split, score_curriculum_response, summarize
from src.eval.curriculum_budget import match_responses
from src.eval.reward_grid import load_jsonl, summarize as summarize_groups
from src.prompts import build_prompt
from src.train.curriculum import ADD, NUM_GENERATIONS, REWARDS, TEMPERATURE, write_json

CURRICULUM = "curriculum"
DIRECT = "three_number_only"
CONTROL = "correctness_only"


def launch_stage(data_dir, stage, size, steps, reward, seed, smoke, initial=None):
    command = [
        sys.executable, "-u", "-m", "src.train.curriculum", "--data-dir", str(data_dir),
        "--output", str(stage), "--size", str(size), "--steps", str(steps), "--reward", reward, "--seed", str(seed),
    ]
    if initial:
        command.extend(["--initial", str(initial)])
    if smoke:
        command.append("--smoke")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    stage.mkdir(parents=True, exist_ok=True)
    log_path = stage / "training.log"
    label = f"{stage.parent.name}/{stage.name}"
    print(f"Training or reusing {label}. Log at {log_path}.", flush=True)
    with log_path.open("ab") as log, subprocess.Popen(
        command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT,
    ) as process:
        try:
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"{label} is still running. Log size: {log_path.stat().st_size // 1024} KiB.", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            raise
    if code:
        raise RuntimeError(f"{label} exited with code {code}. See {log_path}.")
    return stage / "final"


def write_records(path, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluation_records(adapter, puzzles, manifest, batch_size):
    import peft
    import torch
    import transformers
    from src.eval.evaluate import load_model

    protocol = manifest["protocol"]
    identity = adapter_identity(adapter, protocol["base_model"], MODEL_REVISION)
    meta = {
        "model_revision": MODEL_REVISION, "adapter_sha256": identity["adapter_sha256"],
        "adapter_config_sha256": identity["adapter_config_sha256"],
        "splits": {key: manifest["splits"][key]["sha256"] for key in puzzles},
        "counts": {key: len(rows) for key, rows in puzzles.items()}, "protocol": protocol,
        "decoding": "greedy", "seed": 0, "batch_size": batch_size,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    folder = adapter.parent / "evaluation"
    summary = folder / "protocol.json"
    cap = protocol["evaluation_max_new_tokens"]
    if summary.is_file():
        if json.loads(summary.read_text(encoding="utf-8")) != meta:
            raise ValueError(f"Evaluation settings differ for {folder}. Use the original settings or a new study directory.")
        return {key: match_responses(rows, load_jsonl(folder / f"{key}.jsonl"), cap) for key, rows in puzzles.items()}

    torch.manual_seed(0)
    tokenizer, model, device = load_model(protocol["base_model"], adapter, revision=MODEL_REVISION)
    records = {}
    folder.mkdir(exist_ok=True)
    try:
        for key, rows in puzzles.items():
            records[key] = evaluate_split(tokenizer, model, device, rows, protocol, batch_size, f"{adapter.parent.name}/{key}")
            write_records(folder / f"{key}.jsonl", records[key])
        write_json(summary, meta)
    finally:
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return records


def validation_pilot(adapter, data_dir, manifest, seed, smoke):
    import torch
    from src.eval.evaluate import load_model

    protocol = manifest["protocol"]
    split = manifest["splits"]["val"]
    rows = load_jsonl(data_dir / split["file"])
    rng = random.Random(seed)
    rng.shuffle(rows)
    sample_count = 1 if smoke else 12
    selected = [row for size in (2, 3) for row in [p for p in rows if len(p["numbers"]) == size][:sample_count]]

    tokenizer, model, device = load_model(protocol["base_model"], adapter, revision=MODEL_REVISION)
    torch.manual_seed(seed)
    records, groups = [], []
    with torch.inference_mode():
        for row in selected:
            numbers, target = row["numbers"], row["target"]
            prompt = build_prompt(tokenizer, numbers, target, protocol["prompt_method"])
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            sequence = model.generate(
                **inputs, do_sample=True, num_return_sequences=NUM_GENERATIONS, temperature=TEMPERATURE,
                top_p=1.0, top_k=0, max_new_tokens=protocol["evaluation_max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
            suffixes = tokenizer.batch_decode(sequence[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs = [protocol["assistant_prefill"] + suffix for suffix in suffixes]
            groups.append((len(numbers), [score_curriculum_response(output, numbers, target) for output in outputs]))
            records.append({"id": row["id"], "numbers": numbers, "target": target, "responses": outputs})
    folder = adapter.parent / "reward_signal_pilot"
    folder.mkdir(exist_ok=True)
    write_records(folder / "responses.jsonl", records)
    identity = adapter_identity(adapter, protocol["base_model"], MODEL_REVISION)
    write_json(folder / "summary.json", {
        **identity, "seed": seed, "validation_sha256": split["sha256"], "smoke_test": smoke,
        "samples": NUM_GENERATIONS, "temperature": TEMPERATURE, "token_cap": protocol["evaluation_max_new_tokens"],
        "by_reward": {key: summarize_groups(groups, config) for key, config in REWARDS.items()},
    })
    print(f"Saved the Objective 11 validation pilot to {folder}.")


def metrics(rows):
    result = summarize(rows)
    result["mean_additive_reward"] = sum(REWARDS[ADD].score(r["correct"], r["format_ok"]) for r in rows) / len(rows)
    result["mean_gated_reward"] = sum(REWARDS["gated"].score(r["correct"], r["format_ok"]) for r in rows) / len(rows)
    result["mean_wrong_format_bonus"] = REWARDS[ADD].format_bonus * result["formatted_wrong_rate_pct"] / 100
    return result


def levels(records):
    test = records["test"]
    return {
        "2_numbers": [row for row in test if row["n_numbers"] == 2],
        "3_numbers": [row for row in test if row["n_numbers"] == 3],
        "in_domain": test, "4_numbers_ood": records["test_ood"],
    }


def compare(treatments, control):
    levels_by_arm = {arm: levels(rows) for arm, rows in treatments.items()}
    by_arm = {arm: {level: metrics(rows) for level, rows in partitions.items()} for arm, partitions in levels_by_arm.items()}
    paired = {}
    keys = ("strict_solve_rate_pct", "format_rate_pct", "formatted_wrong_rate_pct", "mean_additive_reward", "mean_gated_reward", "mean_wrong_format_bonus")
    for arm, partitions in levels_by_arm.items():
        if arm == control:
            continue
        paired[arm] = {}
        for level, rows in partitions.items():
            reference = levels_by_arm[control][level]
            pairs = list(zip(reference, rows, strict=True))
            if any(any(a[k] != b[k] for k in ("id", "numbers", "target")) for a, b in pairs):
                raise ValueError(f"Unpaired puzzle rows for {arm}/{level}.")
            delta = {key: round(by_arm[arm][level][key] - by_arm[control][level][key], 8) for key in keys}
            paired[arm][level] = {
                "delta": delta, "gained_solves": sum(not a["correct"] and b["correct"] for a, b in pairs),
                "lost_solves": sum(a["correct"] and not b["correct"] for a, b in pairs),
            }
            if control == CONTROL and arm == ADD:
                paired[arm][level]["meets_reward_hacking_criterion"] = (
                    delta["strict_solve_rate_pct"] <= 0 and delta["formatted_wrong_rate_pct"] > 0
                    and delta["mean_additive_reward"] > 0 and delta["mean_gated_reward"] <= 0
                )
    return {"metrics": by_arm, "paired_vs_control": paired}


def run(objective, data_dir, out_dir, seeds, smoke, batch_size, pilot_only):
    from filelock import FileLock

    manifest = validate_curriculum(data_dir)
    root = out_dir.resolve() / ("smoke" if smoke else "full")
    budgets = (2, 2) if smoke else (100, 200)
    control = DIRECT if objective == 10 else CONTROL
    arms = (CURRICULUM, DIRECT) if objective == 10 else tuple(REWARDS)
    puzzles = {key: load_jsonl(data_dir / manifest["splits"][key]["file"]) for key in ("test", "test_ood")}
    if smoke:
        puzzles = {key: [next(row for row in rows if len(row["numbers"]) == size)
                         for size in sorted({len(row["numbers"]) for row in rows})]
                   for key, rows in puzzles.items()}

    results = {}
    for seed in seeds:
        seed_dir = root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(seed_dir / ".study.lock", timeout=0):
            curriculum_stage1 = launch_stage(data_dir, seed_dir / "stage1-two", 2, budgets[0], ADD, seed, smoke)
            if pilot_only:
                validation_pilot(curriculum_stage1, data_dir, manifest, seed, smoke)
                continue
            results[seed] = {}
            for arm in arms:
                initial = curriculum_stage1
                if arm == DIRECT:
                    initial = launch_stage(data_dir, seed_dir / "stage1-three", 3, budgets[0], ADD, seed, smoke)
                reward = ADD if objective == 10 else arm
                name = f"stage2-{arm}" if arm == DIRECT else f"stage2-curriculum-{reward}"
                adapter = launch_stage(data_dir, seed_dir / name, 3, budgets[1], reward, seed, smoke, initial)
                results[seed][arm] = evaluation_records(adapter, puzzles, manifest, batch_size)

    if pilot_only:
        return
    pooled = {arm: {split: [row for seed in seeds for row in results[seed][arm][split]]
                    for split in puzzles} for arm in arms}
    report = {
        "objective": objective, "benchmark_commit": manifest["source_commit"],
        "control": control, "seeds": seeds, "smoke_test": smoke, "model_revision": MODEL_REVISION,
        "split_sha256": {split: manifest["splits"][split]["sha256"] for split in puzzles},
        "stage_updates": list(budgets), "evaluation_max_new_tokens": manifest["protocol"]["evaluation_max_new_tokens"],
        "primary_level": "3_numbers", "paired_by_seed": {str(seed): compare(rows, control) for seed, rows in results.items()},
        "pooled": compare(pooled, control),
    }
    folder = root / f"objective{objective}"
    folder.mkdir(exist_ok=True)
    path = folder / ("seeds-" + "-".join(map(str, seeds)) + ".json")
    write_json(path, report)
    for scope, comparison in [*report["paired_by_seed"].items(), ("pooled", report["pooled"])]:
        for arm, by_level in comparison["paired_vs_control"].items():
            delta = by_level["3_numbers"]["delta"]["strict_solve_rate_pct"]
            print(f"{scope} · {arm} vs {control}: {delta:+.2f} pp on three-number puzzles.")
    print(f"Saved the paired Objective {objective} report to {path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", type=int, choices=(10, 11), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/curriculum"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/countdown-curriculum-studies"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--smoke", action="store_true", help="Use separate two-update stages and one test puzzle per size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pilot-only", action="store_true", help="Run the Objective 11 validation pilot")
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Give a positive batch size and distinct nonnegative seeds.")
    if args.pilot_only and args.objective != 11:
        parser.error("The validation pilot is part of Objective 11.")
    run(args.objective, args.data_dir.resolve(), args.out_dir, sorted(args.seeds), args.smoke,
        args.batch_size, args.pilot_only)


if __name__ == "__main__":
    main()


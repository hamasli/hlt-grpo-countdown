"""Run a paired GRPO reward comparison on a single CUDA GPU."""

import argparse
import gc
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from src.eval.reward_grid import RewardConfig, load_jsonl, score_response
from src.eval.reward_signal import MODEL, SETTINGS
from src.prompts import PREFILL, TEMPLATES, build_prompt
from src.reward.verify import get_text

# Pin the base revision used by the local Objective 3 pilot.
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ROLLOUT_SIZE = SETTINGS["samples"]
ARMS = {
    "correctness_only": RewardConfig(1, 0),
    "additive": RewardConfig(1, 0.1),
    "gated": RewardConfig(1, 0.1, True),
}
LORA = dict(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


class RolloutLog:
    """Score each batch once and record the same metrics in every arm."""

    def __init__(self, config):
        self.config = config
        self.pending = []
        self.puzzle_order = []

    def reward(self, completions, numbers, target, puzzle_id, **kwargs):
        rewards = []
        for suffix, inputs, value, puzzle in zip(completions, numbers, target, puzzle_id, strict=True):
            correct, formatted = score_response(PREFILL["grpo"] + get_text(suffix), inputs, value)
            reward = self.config.score(correct, formatted)
            rewards.append(reward)
            self.pending.append((puzzle, len(inputs), correct, formatted, reward))
        return rewards

    def finish_step(self, step):
        puzzle_ids = [row[0] for row in self.pending]
        if len(puzzle_ids) != ROLLOUT_SIZE or len(set(puzzle_ids)) != 1:
            raise RuntimeError(f"Step {step} expected four completions of one puzzle, got {puzzle_ids}.")
        self.puzzle_order.append(puzzle_ids[0])
        by_size = {}
        for name, size in (("all", None), ("3", 3), ("4", 4)):
            rows = [row for row in self.pending if size is None or row[1] == size]
            count = len(rows)
            if not count:
                by_size[name] = {"completions": 0}
                continue
            correct = sum(row[2] for row in rows)
            formatted = sum(row[3] for row in rows)
            formatted_wrong = sum(not row[2] and row[3] for row in rows)
            by_size[name] = {
                "completions": count,
                "correct": correct,
                "formatted": formatted,
                "formatted_wrong": formatted_wrong,
                "strict_solve_rate_pct": 100 * correct / count,
                "format_rate_pct": 100 * formatted / count,
                "formatted_wrong_rate_pct": 100 * formatted_wrong / count,
                "mean_reward": sum(row[4] for row in rows) / count,
            }
        self.pending.clear()
        return {"step": step, "puzzle_id": puzzle_ids[0], "by_input_count": by_size}


def verify_reward_wiring():
    suffixes = [
        "Use all inputs.</think><answer>50 + 25 - 3</answer>",
        "Add everything.</think><answer>50 + 25 + 3</answer>",
    ]
    for policy, expected in {
        "correctness_only": [1.0, 0.0],
        "additive": [1.1, 0.1],
        "gated": [1.1, 0.0],
    }.items():
        reward = RolloutLog(ARMS[policy]).reward
        actual = reward(suffixes, [[3, 25, 50]] * 2, [72] * 2, [1] * 2)
        if actual != expected:
            raise RuntimeError(f"{policy} prefill/reward check failed: {actual} != {expected}")


def adapter_hash(model):
    digest = hashlib.sha256()
    params = [(name, parameter) for name, parameter in model.named_parameters()
              if parameter.requires_grad]
    if not params:
        raise RuntimeError("The model has no trainable adapter weights.")
    for name, parameter in params:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_arm(dataset, tokenizer, arm_dir, config, seed, max_steps, expected_initial_hash):
    import torch
    from peft import LoraConfig
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(seed)
    rollouts = RolloutLog(config)

    class SaveRolloutMetrics(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            record = rollouts.finish_step(state.global_step)
            with (arm_dir / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    args = GRPOConfig(
        output_dir=str(arm_dir),
        model_init_kwargs={"revision": MODEL_REVISION, "dtype": torch.float16},
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        beta=0.04,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ROLLOUT_SIZE,
        num_generations=ROLLOUT_SIZE,
        num_iterations=1,
        max_completion_length=SETTINGS["max_tokens"],
        temperature=SETTINGS["temperature"],
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        scale_rewards="group",
        multi_objective_aggregation="sum_then_normalize",
        seed=seed,
        data_seed=seed,
        max_steps=max_steps,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        logging_steps=min(10, max_steps),
        save_strategy="no",
        report_to="none",
    )
    if args.world_size != 1:
        raise RuntimeError("Run this comparison in a single process on one GPU.")

    trainer = GRPOTrainer(
        model=MODEL,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=LoraConfig(**LORA),
        reward_funcs=rollouts.reward,
        callbacks=[SaveRolloutMetrics()],
    )
    initial_hash = adapter_hash(trainer.model)
    if expected_initial_hash is not None and initial_hash != expected_initial_hash:
        raise RuntimeError("Paired initial LoRA weights do not match.")
    write_json(arm_dir / "run.json", {
        "seed": seed, "reward": asdict(config), "initial_adapter_sha256": initial_hash,
        "training_args": args.to_dict(),
    })

    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    trainer.train()
    elapsed = perf_counter() - start
    if trainer.state.global_step != max_steps or len(rollouts.puzzle_order) != max_steps:
        raise RuntimeError("Training did not log the requested number of paired steps.")
    peak_memory = torch.cuda.max_memory_allocated()
    trainer.save_model(str(arm_dir / "adapter"))
    write_json(arm_dir / "trainer_log.json", trainer.state.log_history)
    write_json(arm_dir / "training_summary.json", {
        "steps": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak_memory,
    })
    return initial_hash, rollouts.puzzle_order


def run_comparison(train_data, out_dir, seeds, max_steps, include_gated):
    import datasets
    import peft
    import torch
    import transformers
    import trl
    from transformers import AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Make exactly one CUDA GPU visible (for example, CUDA_VISIBLE_DEVICES=0).")
    verify_reward_wiring()
    puzzles = load_jsonl(train_data)
    counts = Counter(len(row["numbers"]) for row in puzzles)
    if set(counts) != {3, 4} or len({row["id"] for row in puzzles}) != len(puzzles):
        raise ValueError("Training data must contain three- and four-number puzzles with unique IDs.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = datasets.Dataset.from_list([
        {
            "prompt": build_prompt(tokenizer, row["numbers"], row["target"], "grpo"),
            "numbers": row["numbers"],
            "target": row["target"],
            "puzzle_id": row["id"],
        }
        for row in puzzles
    ])
    arms = ["correctness_only", "additive"]
    if include_gated:
        arms.append("gated")
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "experiment.json", {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "dependencies": {package.__name__: package.__version__
                         for package in (datasets, peft, torch, transformers, trl)},
        "train_data": str(train_data),
        "train_sha256": hashlib.sha256(train_data.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "puzzles_by_input_count": dict(counts),
        "prompt": TEMPLATES["grpo"],
        "prefill": PREFILL["grpo"],
        "lora": LORA,
        "arms": {name: asdict(ARMS[name]) for name in arms},
        "seeds": seeds,
        "max_steps": max_steps,
    })
    for seed in seeds:
        seed_dir = out_dir / f"seed-{seed}"
        initial_hash = expected_order = None
        for name in arms:
            print(f"Training {name}, seed {seed}, {max_steps} steps.", flush=True)
            initial_hash, order = train_arm(
                dataset, tokenizer, seed_dir / name, ARMS[name], seed, max_steps, initial_hash
            )
            if expected_order is not None and order != expected_order:
                raise RuntimeError(f"Paired puzzle order differs for seed {seed}, arm {name}.")
            expected_order = order
            gc.collect()
            torch.cuda.empty_cache()
        write_json(seed_dir / "pairing.json", {
            "arms": arms,
            "initial_adapter_sha256": initial_hash,
            "puzzle_ids_in_order": expected_order,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True, help="A new directory for this run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-steps", type=int, default=250, help="Optimizer updates per arm")
    parser.add_argument("--include-gated", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed <= 2**32 - 1 for seed in args.seeds
    ):
        parser.error("Use a positive step count and distinct seeds between 0 and 2**32 - 1.")
    if args.out_dir.exists():
        parser.error("Choose a new output directory.")
    run_comparison(args.train_data, args.out_dir, args.seeds, args.max_steps, args.include_gated)


if __name__ == "__main__":
    main()

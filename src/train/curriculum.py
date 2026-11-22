"""Train one resumable stage for the matched curriculum studies."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.data.curriculum import validate_curriculum
from src.eval.curriculum import MODEL_REVISION, adapter_identity, score_curriculum_response
from src.eval.reward_grid import RewardConfig, load_jsonl
from src.prompts import build_prompt
from src.reward.verify import get_text

ADD = "additive"
REWARDS = {
    "correctness_only": RewardConfig(1, 0),
    ADD: RewardConfig(1, 0.1),
    "gated": RewardConfig(1, 0.1, True),
}
LORA = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
NUM_GENERATIONS = 4
TEMPERATURE = 1.0
CONFIG = dict(
    num_generations=NUM_GENERATIONS, max_completion_length=256, temperature=TEMPERATURE,
    top_p=1.0, top_k=0, repetition_penalty=1.0, beta=0.0, epsilon=0.2,
    loss_type="grpo", scale_rewards="group", learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup", weight_decay=0.0, max_grad_norm=1.0,
    fp16=True, bf16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5, save_steps=25, save_total_limit=2, log_completions=False,
    report_to="none",
)


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_checkpoint(output, max_steps):
    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt")
    for step in range(max_steps, 0, -1):
        path = output / f"checkpoint-{step}"
        if all((path / name).is_file() and (path / name).stat().st_size for name in required):
            return path
    return None


def train_stage(data_dir, output, size, steps, reward_name, seed, initial, smoke):
    import datasets
    import peft
    import torch
    import transformers
    import trl

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one CUDA GPU per training worker.")
    peft.import_utils._torchao_available = False
    peft.import_utils.is_torchao_available = lambda: False
    manifest = validate_curriculum(data_dir)
    protocol = manifest["protocol"]
    model_name = protocol["base_model"]
    train_file = manifest["splits"]["train"]
    reward = REWARDS[reward_name]
    initial_sha = adapter_identity(initial, model_name, MODEL_REVISION)["adapter_sha256"] if initial else None
    settings = {
        **CONFIG, "per_device_train_batch_size": 4 if smoke else 8,
        "gradient_accumulation_steps": 1 if smoke else 4,
        "logging_steps": 1 if smoke else CONFIG["logging_steps"], "warmup_steps": max(1, int(0.05 * steps)),
    }
    versions = {m.__name__: m.__version__ for m in (datasets, peft, torch, transformers, trl)}
    plan = {
        "benchmark_commit": manifest["source_commit"], "train_sha256": train_file["sha256"],
        "model_revision": MODEL_REVISION, "protocol": protocol, "size": size, "steps": steps,
        "reward": asdict(reward), "reward_name": reward_name, "seed": seed, "smoke_test": smoke,
        "lora": LORA, "trainer": settings, "versions": versions, "initial_adapter_sha256": initial_sha,
    }
    plan_path = output / "plan.json"
    complete = output / "completed.json"
    final = output / "final"
    output.mkdir(parents=True, exist_ok=True)
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError(f"Training settings changed for {output}. Choose a new study directory.")
    else:
        if any(output.glob("checkpoint-*")) or final.exists() or complete.exists():
            raise ValueError(f"Training files in {output} have no matching plan.")
        write_json(plan_path, plan)
    if complete.is_file():
        identity = adapter_identity(final, model_name, MODEL_REVISION)
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved["steps"] != steps or saved["adapter_sha256"] != identity["adapter_sha256"]:
            raise ValueError(f"Completed adapter no longer matches its saved record: {final}")
        print(f"Reusing {final}.", flush=True)
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISION, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    method = protocol["prompt_method"]
    prefill = protocol["assistant_prefill"]
    rows = [
        {"prompt": build_prompt(tokenizer, row["numbers"], row["target"], method),
         "numbers": row["numbers"], "target": row["target"]}
        for row in load_jsonl(data_dir / train_file["file"]) if len(row["numbers"]) == size
    ]
    dataset = datasets.Dataset.from_list(rows).shuffle(seed=seed)

    def countdown_reward(completions, numbers, target, **kwargs):
        return [
            reward.score(*score_curriculum_response(prefill + get_text(completion), inputs, result))
            for completion, inputs, result in zip(completions, numbers, target, strict=True)
        ]

    transformers.set_seed(seed)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, revision=MODEL_REVISION, dtype=torch.float16, attn_implementation="sdpa"
    )
    if initial is None:
        model = peft.get_peft_model(base, peft.LoraConfig(**LORA))
    else:
        model = peft.PeftModel.from_pretrained(base, initial, is_trainable=True)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainer = trl.GRPOTrainer(
        model=model, reward_funcs=[countdown_reward], train_dataset=dataset, processing_class=tokenizer,
        args=trl.GRPOConfig(output_dir=str(output), seed=seed, max_steps=steps, **settings),
    )
    checkpoint = complete_checkpoint(output, steps)
    print(f"Training {output.name} on {size}-number puzzles, seed {seed}, {steps} updates.", flush=True)
    if checkpoint:
        print(f"Resuming from {checkpoint}.", flush=True)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if trainer.state.global_step != steps:
        raise RuntimeError(f"Expected {steps} updates, got {trainer.state.global_step}.")
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    (final / "base_model_revision.txt").write_text(MODEL_REVISION + "\n", encoding="utf-8")
    identity = adapter_identity(final, model_name, MODEL_REVISION)
    write_json(output / "metrics.json", trainer.state.log_history)
    write_json(complete, {"steps": steps, "adapter_sha256": identity["adapter_sha256"]})
    print(f"Saved {final}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(2, 3), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--reward", choices=REWARDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_stage(args.data_dir, args.output, args.size, args.steps, args.reward, args.seed, args.initial, args.smoke)


if __name__ == "__main__":
    main()


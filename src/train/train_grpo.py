"""Train a language model on Countdown puzzles using GRPO (Group Relative Policy Optimization).

Designed for Kaggle Tesla T4 (16 GB) with LoRA.

Examples:
  # Dry-run / small test on 0.5B model:
  python -m src.train.train_grpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --max_steps 50 --num_generations 4

  # Full training run on 1.5B model:
  python -m src.train.train_grpo --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct --num_train_epochs 1 --output_dir checkpoints/qwen1.5b_grpo
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.prompts import build_prompt
from src.reward.verify import correctness_reward, format_reward


def load_dataset_from_jsonl(path, tok, method="grpo", limit=None):
    """Load puzzles from JSONL and format prompts."""
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]

    records = []
    for r in rows:
        prompt = build_prompt(tok, r["numbers"], r["target"], method=method)
        records.append({
            "prompt": prompt,
            "numbers": r["numbers"],
            "target": r["target"],
            "n_numbers": r["n_numbers"],
        })
    return Dataset.from_list(records)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training on Countdown")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model from Hugging Face")
    parser.add_argument("--train_data", default="data/train.jsonl")
    parser.add_argument("--val_data", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/grpo_lora")
    parser.add_argument("--limit_train", type=int, default=None, help="Limit number of train examples")
    parser.add_argument("--limit_val", type=int, default=100, help="Limit number of validation examples")

    # Hyperparameters optimized for 16GB GPU (Tesla T4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per prompt (G). 4 is recommended for 16GB VRAM.")
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence weight")

    # LoRA settings
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA (required for 1.5B on Kaggle 16GB)")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging and checkpoints
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting GRPO Training ===")
    print(f"Model: {args.model_name_or_path} | Device: {device} | LoRA: {args.use_lora}")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading datasets...")
    train_dataset = load_dataset_from_jsonl(args.train_data, tok, method="grpo", limit=args.limit_train)
    val_dataset = load_dataset_from_jsonl(args.val_data, tok, method="grpo", limit=args.limit_val)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # LoRA configuration
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # GRPO training configuration
    # Note: T4 GPUs lack native bfloat16, so fp16 is used
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=(device == "cuda"),
        bf16=False,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
    )

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )

    print("Beginning GRPO training...")
    trainer.train()

    print(f"Saving final model checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()

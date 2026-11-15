"""
src/train/grpo_train.py  —  GRPO + LoRA training on Countdown puzzles
=====================================================================

WHAT
  Trains Qwen2.5-Instruct with GRPO (Shao et al., 2024) on our Countdown training puzzles.
  Only a small LoRA adapter is trained (Hu et al., 2021); the base model stays frozen.
  Reward = our rule-based verifier (1.0 correct equation + 0.1 correct format).
  No reward model, no critic, no KL penalty (beta = 0), as in DeepSeek-R1-Zero / Gandhi et al.

WHY
  Baselines (1.5B CoT, greedy): 26.4% on 2 numbers, 3.0% on 3 numbers.
  pass@4 showed ~50% (2 numbers) and ~25% (3 numbers) of puzzles get MIXED rewards
  in a group of 4 answers -> GRPO has a learning signal to work with.
  Full fine-tuning of 1.5B does not fit in a 16 GB T4 -> LoRA (professor's advice).

HOW (one training step)
  1. take 8 puzzles from the training set
  2. the model writes 4 answers per puzzle (sampling, temperature 1.0)  -> 32 answers
  3. the verifier scores every answer (correct? + format?)
  4. advantage of each answer = (reward - group mean) / group std
     (group = the 4 answers to the SAME puzzle)
  5. one LoRA update: above-average answers become more likely, below-average less likely
  -> 100 steps = 800 puzzles = 3,200 scored answers

CURRICULUM
  Stage 1: --levels 2, 100 steps                                            -> checkpoints/stage1
  Stage 2: --levels 3, 200 steps, --init_adapter checkpoints/stage1/final  -> checkpoints/stage2
  Stage 2 continues the Stage 1 LoRA weights, with a fresh optimizer.

RUN (from the repo root)
  smoke  : python -m src.train.grpo_train --model Qwen/Qwen2.5-0.5B-Instruct --max_steps 5 --output_dir checkpoints/smoke
  timing : python -m src.train.grpo_train --max_steps 10 --output_dir checkpoints/timing
  stage 1: python -m src.train.grpo_train --levels 2 --max_steps 100 --output_dir checkpoints/stage1
  stage 2: python -m src.train.grpo_train --levels 3 --max_steps 200 --init_adapter checkpoints/stage1/final --output_dir checkpoints/stage2
  resume : add --resume to the same command (continues from the last checkpoint-XX)

OUTPUT (inside --output_dir)
  checkpoint-25, checkpoint-50, ...  LoRA + optimizer state every 25 steps (resume + choose best on val)
  final/                             final LoRA adapter  -> evaluate.py --method grpo --adapter <this>
  log_history.json                   all logged metrics (every 5 steps) -> training curves
  run_info.json                      settings, total time, seconds/step, peak GPU memory, versions

WHAT TO WATCH IN THE LOGS
  reward                              mean total reward per answer          -> should go UP
  rewards/correctness_reward/mean     training accuracy (fraction correct)  -> should go UP
  rewards/format_reward_prefill/mean  0.1 x format rate                     -> should stay near 0.1
  frac_reward_zero_std                groups where all 4 answers got the same reward (no signal)
  completions/mean_length             answer length in tokens (one of our metrics)
  completions/clipped_ratio           answers cut at 256 tokens             -> should stay low
"""

# ---------------------------------------------------------------------------
# Kaggle gives 2 x T4. A plain `python` run of TRL uses only one GPU anyway
# (multi-GPU needs `accelerate launch`), so we pin GPU 0 BEFORE torch is imported.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path

import torch
import peft
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

# Our own frozen pieces: the SAME prompt (v3 + "<think>" prefill) and the SAME verifier
# used for the baselines -> training and evaluation see exactly the same task.
from src.prompts import PREFILL, build_prompt
from src.reward.verify import correctness_reward, format_reward

# LoRA goes on all 7 linear layers of each Qwen block (attention + MLP).
# Common choice in LoRA/QLoRA recipes: more capacity than attention-only, still ~1% of the weights.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# 1. Command-line arguments (defaults = the agreed Stage 1 plan)
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GRPO + LoRA training on Countdown")
    # what to train
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--levels", type=int, nargs="+", default=[2],
                   help="puzzle sizes to train on (Stage 1 = 2, Stage 2 = 3)")
    p.add_argument("--init_adapter", default=None,
                   help="start from an existing LoRA (Stage 2 starts from Stage 1's final/)")
    # how long
    p.add_argument("--max_steps", type=int, default=100)
    # GRPO batch: per_device_batch and grad_accum count ANSWERS, not puzzles (TRL convention).
    # 8 answers x 4 accumulation = 32 answers per step = 8 puzzles x 4 answers.
    p.add_argument("--num_generations", type=int, default=4, help="G = answers per puzzle")
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    # generation (same sampling as our pass@4 analysis)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    # optimisation
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty weight (0 = no reference model)")
    p.add_argument("--loss_type", default="grpo", choices=["grpo", "dapo", "dr_grpo", "bnpo"],
                   help="grpo = exact DeepSeekMath loss: average over the tokens of each answer, "
                        "then average over the answers")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # saving / logging
    p.add_argument("--output_dir", default="checkpoints/stage1")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in output_dir")
    p.add_argument("--push_to_hub", action="store_true", help="also upload checkpoints to Hugging Face")
    p.add_argument("--hub_model_id", default=None, help="hamas/countdown-grpo-stage1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_completions", action="store_true",
                   help="print sample answers in the log (big output! off by default)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Training data: one row per puzzle.
#    "prompt" is the full chat-formatted tgitext ending with "<think>" (prefill).
#    "numbers" and "target" are extra columns: TRL passes them to the reward
#    functions as keyword arguments, so the verifier knows what to check.
# ---------------------------------------------------------------------------
def load_train_dataset(path, levels, tok, seed):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            puzzle = json.loads(line)
            if puzzle["n_numbers"] not in levels:          # curriculum: keep only this stage's level
                continue
            rows.append({
                "prompt": build_prompt(tok, puzzle["numbers"], puzzle["target"], "grpo"),
                "numbers": puzzle["numbers"],
                "target": puzzle["target"],
            })
    if not rows:
        raise SystemExit(f"No puzzles with n_numbers in {levels} found in {path}")
    # Shuffle once with a fixed seed -> reproducible order of puzzles across runs.
    return Dataset.from_list(rows).shuffle(seed=seed)


# ---------------------------------------------------------------------------
# 3. Rewards
#    correctness_reward: 1.0 if the last <answer> is a valid equation hitting the target.
#                        It only looks at <answer>, so the missing "<think>" does not matter.
#    format_reward_prefill: the prompt already ENDS with "<think>", so the model's output
#                        starts AFTER it. We glue "<think>" back on before the format check,
#                        exactly like evaluate.py does. Without this, format reward = 0 always.
#    TRL adds the rewards together: max total = 1.0 + 0.1 = 1.1.
# ---------------------------------------------------------------------------
def format_reward_prefill(completions, **kwargs):
    full_texts = [PREFILL["grpo"] + c for c in completions]
    return format_reward(full_texts, **kwargs)


# ---------------------------------------------------------------------------
# 4. Model + LoRA
# ---------------------------------------------------------------------------
def load_model(args):
    # fp16, because the T4 has no bf16 support. The base weights stay frozen in fp16 (~3 GB for 1.5B).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa")

    if args.init_adapter:
        # Stage 2: continue training the Stage 1 LoRA (is_trainable=True, otherwise it is frozen).
        print(f"Loading existing LoRA from {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        # Stage 1: a new LoRA. B starts at zero -> at step 0 the model is exactly the base model.
        lora_cfg = LoraConfig(
            r=args.lora_r,                 # rank: size of the small update matrices
            lora_alpha=args.lora_alpha,    # scale = alpha / r = 2 (common choice: alpha = 2r)
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # The trainable LoRA weights must be fp32: fp16 mixed precision cannot update fp16 weights
    # ("Attempting to unscale FP16 gradients" error). LoRA is tiny, so fp32 costs almost nothing.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.print_trainable_parameters()     # expect ~18.5M trainable (~1.2%) for 1.5B
    return model


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")

    # Sanity check: the answers of one step must split into whole groups of G.
    answers_per_step = args.per_device_batch * args.grad_accum
    if answers_per_step % args.num_generations != 0:
        raise SystemExit("per_device_batch * grad_accum must be divisible by num_generations")
    puzzles_per_step = answers_per_step // args.num_generations

    # Tokenizer: the same chat template as the baselines.
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = load_train_dataset(args.data, args.levels, tok, args.seed)
    model = load_model(args)

    print(f"Levels {args.levels} | {len(train_ds)} puzzles available | "
          f"{puzzles_per_step} puzzles x {args.num_generations} answers = {answers_per_step} answers/step | "
          f"{args.max_steps} steps -> {args.max_steps * puzzles_per_step} puzzles used")

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,                   # we count STEPS, not epochs (we use < 1 epoch)
        seed=args.seed,

        # --- GRPO batch (in answers) ---
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,

        # --- generation: pure sampling, same as our pass@4 analysis ---
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,                                  # no nucleus cut
        top_k=0,                                    # no top-k cut (0 = disabled)
        repetition_penalty=1.0,                     # no penalty

        # --- GRPO loss ---
        beta=args.beta,                             # 0 -> no KL term, no reference model in memory
        epsilon=0.2,                                # PPO-style clip range for the probability ratio
        loss_type=args.loss_type,                   # "dapo" = same GRPO advantages, but loss averaged per
                                                    # token (TRL default), avoids the length bias of "grpo"

        # --- optimiser ---
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",   # short warm-up, then a constant lr
        warmup_steps=max(1, int(0.05 * args.max_steps)),
        weight_decay=0.0,
        max_grad_norm=1.0,                          # gradient clipping for stability

        # --- memory on a 16 GB T4 ---
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,                # recompute activations -> less memory, a bit slower
        gradient_checkpointing_kwargs={"use_reentrant": False},   # needed with frozen base + LoRA

        # --- logging & saving ---
        logging_steps=args.logging_steps,
        log_completions=True,                       # prints sample answers -> we can read what it writes
        save_strategy="steps",
        save_steps=args.save_steps,                 # keep ALL checkpoints: we pick the best on val later
        report_to="none",                           # no wandb; logs stay in log_history.json

        # --- optional upload (needs HF_TOKEN from Kaggle Secrets) ---
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",                  # uploads the latest checkpoint -> resume after a crash
        hub_private_repo=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward_prefill],
        args=config,
        train_dataset=train_ds,
        processing_class=tok,
    )

    # Resume: continue from the newest checkpoint-XX in output_dir (if any).
    last_ckpt = None
    if args.resume and os.path.isdir(args.output_dir):
        last_ckpt = get_last_checkpoint(args.output_dir)
        print(f"Resuming from {last_ckpt}" if last_ckpt else "No checkpoint found, starting fresh")

    # ---- train ----
    torch.cuda.reset_peak_memory_stats()
    start_step = int(last_ckpt.rsplit("-", 1)[-1]) if last_ckpt else 0
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    elapsed = time.time() - t0
    steps_done = trainer.state.global_step - start_step

    # ---- save the final adapter + logs ----
    out = Path(args.output_dir)
    final_dir = out / "final"
    trainer.save_model(str(final_dir))              # saves only the LoRA adapter (~70 MB), not the base model
    tok.save_pretrained(str(final_dir))

    with open(out / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    run_info = {
        "args": vars(args),
        "puzzles_per_step": puzzles_per_step,
        "answers_per_step": answers_per_step,
        "steps_done": steps_done,
        "train_time_s": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(steps_done, 1), 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__},
    }
    with open(out / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone: {steps_done} steps in {elapsed/60:.1f} min "
          f"({run_info['seconds_per_step']} s/step), peak GPU memory {run_info['peak_gpu_memory_gb']} GB")
    print(f"Final adapter: {final_dir}")


if __name__ == "__main__":
    main()
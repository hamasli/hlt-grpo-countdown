"""
src/eval/ask.py  —  ask the trained model your own Countdown puzzles (interactive, works on CPU)

WHAT
  Loads Qwen2.5-1.5B-Instruct + our GRPO LoRA (Stage 2, step 375) once, then in a loop:
  you type the numbers and the target -> the model shows its reasoning and answer,
  and our verifier says if it's correct.
WHY
  To see how the trained model actually reasons on puzzles we choose.
  Same prompt ("<think>" prefill), greedy decoding, 256-token limit and verifier as in training/evaluation.
RUN (repo root)
  python -m src.eval.ask
"""
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import PREFILL, build_prompt
from src.reward.verify import correctness_reward

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/stage2/best"
MAX_NEW_TOKENS = 256

# ---- load once: base model (downloaded the first time, ~3 GB) + our trained adapter ----
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)   # CPU needs float32
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()             # LoRA baked in -> faster
model.eval()
print(f"Ready: {BASE} + {ADAPTER} on {device}. Type 'q' to quit.\n")

while True:
    # ---- 1. read the puzzle ----
    line = input("Numbers (e.g. 23 25 82): ").strip()
    if line.lower() == "q":
        break
    t_line = input("Target (e.g. 80): ").strip()
    if t_line.lower() == "q":
        break
    try:
        numbers, target = [int(x) for x in line.replace(",", " ").split()], int(t_line)
    except ValueError:
        print("Please type whole numbers only.\n")
        continue

    # ---- 2. the model answers (greedy, same prompt as training) ----
    prompt = build_prompt(tok, numbers, target, "grpo")       # ends with "<think>"
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = PREFILL["grpo"] + tok.decode(new_tokens, skip_special_tokens=True)

    # ---- 3. show it + check it with our verifier ----
    correct = correctness_reward([answer], numbers=[numbers], target=[target])[0] == 1.0
    print("\n" + answer.strip())
    print(f"\n-> {'CORRECT' if correct else 'WRONG'} | {len(new_tokens)} tokens"
          f"{' (CLIPPED)' if len(new_tokens) >= MAX_NEW_TOKENS else ''} | {time.time() - t0:.1f} s\n")
"""Evaluate a model on Countdown puzzles.

Examples:
  python -m src.eval.evaluate --method direct --limit 5            (laptop, CPU)
  python -m src.eval.evaluate --method cot --model Qwen/Qwen2.5-1.5B-Instruct
  python -m src.eval.evaluate --method grpo --adapter checkpoints/lora   (after training)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import PREFILL, build_prompt
from src.reward.verify import has_format, is_correct


def load_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def load_model(name, adapter=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32  # T4 has no bf16
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")  # left pad for generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    if adapter:  # LoRA weights from GRPO training
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model, device


@torch.no_grad()
def generate(tok, model, device, prompts, max_new_tokens):
    """Greedy decoding for a batch. Returns texts and number of generated tokens."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    n_tokens = (new != tok.pad_token_id).sum(dim=1).tolist()
    texts = tok.batch_decode(new, skip_special_tokens=True)
    return texts, n_tokens


def format_ok(text, method):
    if method == "direct":
        return "<answer>" in text and "</answer>" in text
    return has_format(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA folder (for the GRPO model)")
    p.add_argument("--method", choices=["direct", "cot", "grpo"], required=True)
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--limit", type=int, default=None, help="use only the first N puzzles")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    max_new = args.max_new_tokens or (64 if args.method == "direct" else 512)

    puzzles = load_jsonl(args.data, args.limit)
    tok, model, device = load_model(args.model, args.adapter)
    print(f"model={args.model} method={args.method} device={device} puzzles={len(puzzles)}")

    records, start = [], time.time()
    for i in range(0, len(puzzles), args.batch_size):
        batch = puzzles[i:i + args.batch_size]
        prompts = [build_prompt(tok, q["numbers"], q["target"], args.method) for q in batch]
        texts, n_tokens = generate(tok, model, device, prompts, max_new)
        for q, text, n in zip(batch, texts, n_tokens):
            text = PREFILL[args.method] + text  # add the prefill back so the checker sees the full answer
            records.append({"id": q["id"], "n_numbers": q["n_numbers"], "numbers": q["numbers"],
                            "target": q["target"], "output": text, "tokens": n,
                            "correct": is_correct(text, q["numbers"], q["target"]),
                            "format_ok": format_ok(text, args.method)})
        print(f"  {len(records)}/{len(puzzles)} done")
    elapsed = time.time() - start

    # summary per level + all
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['n_numbers']}_numbers"].append(r)
    groups["all"] = records
    summary = {"model": args.model, "adapter": args.adapter, "method": args.method,
               "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "seconds": round(elapsed, 1),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
    for name, rs in groups.items():
        summary[name] = {"n": len(rs),
                         "accuracy": round(100 * sum(r["correct"] for r in rs) / len(rs), 2),
                         "format_rate": round(100 * sum(r["format_ok"] for r in rs) / len(rs), 2),
                         "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs), 1)}

    tag = f"{args.model.split('/')[-1]}_{args.method}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
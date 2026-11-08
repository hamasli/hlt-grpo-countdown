"""Evaluate a model on Countdown puzzles.

Greedy (1 answer per puzzle):
  python -m src.eval.evaluate --method direct --limit 5
  python -m src.eval.evaluate --method cot --model Qwen/Qwen2.5-1.5B-Instruct
Sampling (k answers per puzzle, like GRPO):
  python -m src.eval.evaluate --method cot --num_samples 8 --temperature 1.0
After training:
  python -m src.eval.evaluate --method grpo --adapter checkpoints/lora
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
def generate(tok, model, device, prompts, max_new_tokens, num_samples=1, temperature=1.0):
    """Returns a flat list of texts and token counts.
    With num_samples=k the list is [p0_s0..p0_sk-1, p1_s0..] (k answers per prompt, in order)."""
    if num_samples == 1:   # greedy: always the most likely token
        mode = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:                  # sampling: same settings as GRPO in TRL
        mode = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, num_return_sequences=num_samples)
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.pad_token_id, **mode)
    new = out[:, enc["input_ids"].shape[1]:]
    n_tokens = (new != tok.pad_token_id).sum(dim=1).tolist()
    texts = tok.batch_decode(new, skip_special_tokens=True)
    return texts, n_tokens


def format_ok(text, method):
    if method == "direct":
        return "<answer>" in text and "</answer>" in text
    return has_format(text)


def pct(x):
    return round(100 * x, 2)


def summarize_greedy(rs):
    return {"n": len(rs),
            "accuracy": pct(sum(r["correct"] for r in rs) / len(rs)),
            "format_rate": pct(sum(r["format_ok"] for r in rs) / len(rs)),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / len(rs), 1)}


def summarize_samples(rs):
    """pass@1, pass@k and how many groups give GRPO a learning signal."""
    k = len(rs[0]["correct"])
    all_correct = [c for r in rs for c in r["correct"]]
    all_format = [f for r in rs for f in r["format_ok"]]
    all_tokens = [t for r in rs for t in r["tokens"]]
    # a group is useful when its rewards are NOT all equal (otherwise advantage = 0)
    useful_correct = [0 < sum(r["correct"]) < k for r in rs]
    useful_total = [len({1.0 * c + 0.1 * f for c, f in zip(r["correct"], r["format_ok"])}) > 1
                    for r in rs]
    return {"n": len(rs), "k": k,
            "pass@1": pct(sum(all_correct) / len(all_correct)),
            f"pass@{k}": pct(sum(any(r["correct"]) for r in rs) / len(rs)),
            "useful_groups_correctness": pct(sum(useful_correct) / len(rs)),
            "useful_groups_total_reward": pct(sum(useful_total) / len(rs)),
            "format_rate": pct(sum(all_format) / len(all_format)),
            "mean_tokens": round(sum(all_tokens) / len(all_tokens), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA folder (for the GRPO model)")
    p.add_argument("--method", choices=["direct", "cot", "grpo"], required=True)
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--limit", type=int, default=None, help="use only the first N puzzles")
    p.add_argument("--batch_size", type=int, default=16, help="puzzles per batch")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=1, help="1 = greedy, k > 1 = sampling")
    p.add_argument("--temperature", type=float, default=1.0, help="only used when sampling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    max_new = args.max_new_tokens or (64 if args.method == "direct" else 512)
    k = args.num_samples
    torch.manual_seed(args.seed)

    puzzles = load_jsonl(args.data, args.limit)
    tok, model, device = load_model(args.model, args.adapter)
    print(f"model={args.model} method={args.method} device={device} "
          f"puzzles={len(puzzles)} samples_per_puzzle={k}")

    records, start = [], time.time()
    for i in range(0, len(puzzles), args.batch_size):
        batch = puzzles[i:i + args.batch_size]
        prompts = [build_prompt(tok, q["numbers"], q["target"], args.method) for q in batch]
        texts, n_tokens = generate(tok, model, device, prompts, max_new, k, args.temperature)
        texts = [PREFILL[args.method] + t for t in texts]  # add the prefill back for the checker
        for j, q in enumerate(batch):
            ts, ns = texts[j * k:(j + 1) * k], n_tokens[j * k:(j + 1) * k]
            correct = [is_correct(t, q["numbers"], q["target"]) for t in ts]
            fmt = [format_ok(t, args.method) for t in ts]
            base = {"id": q["id"], "n_numbers": q["n_numbers"], "numbers": q["numbers"], "target": q["target"]}
            if k == 1:
                records.append({**base, "output": ts[0], "tokens": ns[0],
                                "correct": correct[0], "format_ok": fmt[0]})
            else:
                records.append({**base, "outputs": ts, "tokens": ns,
                                "correct": correct, "format_ok": fmt})
        print(f"  {len(records)}/{len(puzzles)} done")
    elapsed = time.time() - start

    groups = defaultdict(list)
    for r in records:
        groups[f"{r['n_numbers']}_numbers"].append(r)
    groups["all"] = records
    summarize = summarize_greedy if k == 1 else summarize_samples
    summary = {"model": args.model, "adapter": args.adapter, "method": args.method,
               "num_samples": k, "temperature": args.temperature if k > 1 else None,
               "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "seconds": round(elapsed, 1),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
    for name, rs in groups.items():
        summary[name] = summarize(rs)

    # greedy -> *_outputs.jsonl (read by error_analysis), sampling -> *_samples.jsonl
    tag = f"{args.model.split('/')[-1]}_{args.method}" + (f"_k{k}" if k > 1 else "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}_{'outputs' if k == 1 else 'samples'}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(out / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
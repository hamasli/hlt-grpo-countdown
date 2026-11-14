"""
evaluate.py - test a model on Countdown puzzles and measure how good it is
==========================================================================

WHAT
    Loads puzzles and a model, asks the model to solve each puzzle, checks every
    answer with our verifier (src/reward/verify.py) and saves the results.
    It produces ALL our results: the baselines now, and the final comparison
    (zero-shot vs CoT vs GRPO) after training.

STEPS
    1. load the puzzles          (default: data/test.jsonl)
    2. load the model            (+ LoRA adapter with --adapter, after training)
    3. build the prompts         (src/prompts.py: zero_shot / cot / grpo)
    4. generate the answers      (GPU, in batches)
    5. check each answer         (verify.py: correct? right format?)
    6. compute the metrics       (by level: 2 numbers, 3 numbers, ..., and "all")
    7. save the results          (results/<run>_outputs.jsonl or _samples.jsonl
                                  + results/<run>_summary.json)

TWO MODES
    greedy   (--num_samples 1, default): 1 answer per puzzle, the most likely one
             -> accuracy. Used for baselines and the final comparison.
    sampling (--num_samples 4 --temperature 1.0): 4 different answers per puzzle,
             the same settings GRPO uses in training
             -> pass@4 and "useful groups" = the learning signal GRPO will get.

METRICS
    accuracy      % of puzzles answered correctly (greedy)
    format_rate   % of answers in the right format
    mean_tokens   average answer length in tokens
    clipped_rate  % of answers cut off at the max length (if high, the limit is too short)
    pass@1        % of all sampled answers that are correct
    pass@k        % of puzzles with at least 1 correct answer out of k
    useful_groups_correctness   % of puzzles whose k answers are mixed (some right,
                                some wrong): only these teach GRPO to solve

EXAMPLES
    python -m src.eval.evaluate --method zero_shot --limit 5                 (laptop test)
    python -m src.eval.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --method cot
    python -m src.eval.evaluate --method cot --num_samples 4 --temperature 1.0
    python -m src.eval.evaluate --method cot --data data/test_ood.jsonl      (4-number OOD test)
    python -m src.eval.evaluate --method grpo --adapter checkpoints/stage2   (after training)
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

# Max answer length per method. CoT/GRPO use 256, the SAME limit as training,
# so the evaluation measures the model under the conditions it was trained in.
DEFAULT_MAX_NEW = {"zero_shot": 64, "cot": 256, "grpo": 256}


def load_jsonl(path, limit=None):
    """Read one puzzle per line; keep only the first `limit` puzzles if given."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def load_model(name, adapter=None):
    """Load tokenizer + model. GPU -> fp16 (the T4 has no bf16), CPU -> fp32.
    With `adapter`, the trained LoRA weights are added on top of the base model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")  # left padding for batch generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model, device


@torch.no_grad()
def generate(tok, model, device, prompts, max_new_tokens, num_samples=1, temperature=1.0):
    """Generate answers for a batch of prompts.

    Returns a flat list of texts and of token counts. With num_samples = k the
    order is [prompt0 answer0..k-1, prompt1 answer0..k-1, ...].
    """
    if num_samples == 1:   # greedy: always the most likely next token (reproducible)
        mode = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:                  # sampling: same settings as GRPO training in TRL
        mode = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, num_return_sequences=num_samples)
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.pad_token_id, **mode)
    new = out[:, enc["input_ids"].shape[1]:]                     # keep only the new tokens
    n_tokens = (new != tok.pad_token_id).sum(dim=1).tolist()     # answer length
    texts = tok.batch_decode(new, skip_special_tokens=True)
    return texts, n_tokens


def format_ok(text, method):
    """zero_shot only needs answer tags; cot/grpo need <think>...</think><answer>...</answer>."""
    if method == "zero_shot":
        return "<answer>" in text and "</answer>" in text
    return has_format(text)


def pct(x):
    return round(100 * x, 2)


def summarize_greedy(rs, max_new):
    """Metrics for greedy runs (1 answer per puzzle)."""
    n = len(rs)
    return {"n": n,
            "accuracy": pct(sum(r["correct"] for r in rs) / n),
            "format_rate": pct(sum(r["format_ok"] for r in rs) / n),
            "mean_tokens": round(sum(r["tokens"] for r in rs) / n, 1),
            "clipped_rate": pct(sum(r["tokens"] >= max_new for r in rs) / n)}


def summarize_samples(rs, max_new):
    """Metrics for sampling runs (k answers per puzzle): the GRPO learning signal.

    A group (the k answers of one puzzle) is "useful" when its rewards are NOT all
    equal: if all are equal, the GRPO advantage is 0 and nothing is learned.
    """
    k = len(rs[0]["correct"])
    all_correct = [c for r in rs for c in r["correct"]]
    all_format = [f for r in rs for f in r["format_ok"]]
    all_tokens = [t for r in rs for t in r["tokens"]]
    useful_correct = [0 < sum(r["correct"]) < k for r in rs]
    useful_total = [len({1.0 * c + 0.1 * f for c, f in zip(r["correct"], r["format_ok"])}) > 1
                    for r in rs]
    return {"n": len(rs), "k": k,
            "pass@1": pct(sum(all_correct) / len(all_correct)),
            f"pass@{k}": pct(sum(any(r["correct"]) for r in rs) / len(rs)),
            "useful_groups_correctness": pct(sum(useful_correct) / len(rs)),
            "useful_groups_total_reward": pct(sum(useful_total) / len(rs)),
            "format_rate": pct(sum(all_format) / len(all_format)),
            "mean_tokens": round(sum(all_tokens) / len(all_tokens), 1),
            "clipped_rate": pct(sum(t >= max_new for t in all_tokens) / len(all_tokens))}


def main():
    p = argparse.ArgumentParser(description="Evaluate a model on Countdown puzzles")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA folder (for the GRPO model)")
    p.add_argument("--method", choices=["zero_shot", "cot", "grpo"], required=True)
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--limit", type=int, default=None, help="use only the first N puzzles")
    p.add_argument("--batch_size", type=int, default=16, help="puzzles per batch")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="default: 64 for zero_shot, 256 for cot/grpo")
    p.add_argument("--num_samples", type=int, default=1, help="1 = greedy, k > 1 = sampling")
    p.add_argument("--temperature", type=float, default=1.0, help="only used when sampling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    max_new = args.max_new_tokens or DEFAULT_MAX_NEW[args.method]
    k = args.num_samples
    torch.manual_seed(args.seed)

    puzzles = load_jsonl(args.data, args.limit)
    tok, model, device = load_model(args.model, args.adapter)
    print(f"model={args.model} method={args.method} data={args.data} device={device} "
          f"puzzles={len(puzzles)} samples_per_puzzle={k} max_new_tokens={max_new}")

    # ---- generate and check, batch by batch ----
    records, start = [], time.time()
    for i in range(0, len(puzzles), args.batch_size):
        batch = puzzles[i:i + args.batch_size]
        prompts = [build_prompt(tok, q["numbers"], q["target"], args.method) for q in batch]
        texts, n_tokens = generate(tok, model, device, prompts, max_new, k, args.temperature)
        texts = [PREFILL[args.method] + t for t in texts]   # add the prefill back for the checker
        for j, q in enumerate(batch):
            ts, ns = texts[j * k:(j + 1) * k], n_tokens[j * k:(j + 1) * k]
            correct = [is_correct(t, q["numbers"], q["target"]) for t in ts]
            fmt = [format_ok(t, args.method) for t in ts]
            base = {"id": q["id"], "n_numbers": q["n_numbers"],
                    "numbers": q["numbers"], "target": q["target"]}
            if k == 1:
                records.append({**base, "output": ts[0], "tokens": ns[0],
                                "correct": correct[0], "format_ok": fmt[0]})
            else:
                records.append({**base, "outputs": ts, "tokens": ns,
                                "correct": correct, "format_ok": fmt})
        print(f"  {len(records)}/{len(puzzles)} done")
    elapsed = time.time() - start

    # ---- metrics by level (2_numbers, 3_numbers, ...) and overall ----
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['n_numbers']}_numbers"].append(r)
    groups["all"] = records
    summarize = summarize_greedy if k == 1 else summarize_samples
    summary = {"model": args.model, "adapter": args.adapter, "method": args.method,
               "data": args.data, "num_samples": k,
               "temperature": args.temperature if k > 1 else None,
               "max_new_tokens": max_new,
               "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "seconds": round(elapsed, 1),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
    for name, rs in sorted(groups.items()):
        summary[name] = summarize(rs, max_new)

    # ---- save: file names say model, method, data set and k ----
    # e.g. Qwen2.5-1.5B-Instruct_cot_summary.json, ..._cot_test_ood_summary.json, ..._cot_k4_summary.json
    data_tag = "" if Path(args.data).stem == "test" else f"_{Path(args.data).stem}"
    tag = f"{args.model.split('/')[-1]}_{args.method}{data_tag}" + (f"_k{k}" if k > 1 else "")
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
    
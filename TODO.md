# Project TODO List & Next Steps

This document outlines the remaining tasks, analysis ideas, and optional code snippets for our HLT 2026/27 Group 2 project (*"Reasoning with Verifiable Rewards: Teaching a small language model to solve Countdown with GRPO"*).

---

## 1. Actionable TODO List

- [ ] **1. Poster Visualization (High Priority)**
  - Run the plotting cell in Kaggle to generate high-resolution (300 DPI) publication-ready figures:
    - `poster_figure1_accuracy_comparison.png` (Baseline vs. GRPO across levels)
    - `poster_figure2_compute_scaling.png` (Reasoning token length & truncation vs. problem complexity)
    - `poster_figure3_training_dynamics.png` (Curriculum reward curves across Stage 1 and Stage 2)
  - Download the figures and insert them into the poster / slide deck.

- [ ] **2. Test-Time Compute Budget Scaling Experiment (OOD with 512 Tokens)**
  - *Observation:* On the 4-number Out-of-Domain test set, 51.6% of outputs reached the 256-token limit (`clipped_rate: 51.6%`).
  - *Experiment:* Run evaluation on `data/test_ood.jsonl` with `--max_new_tokens 512`.
  - *Hypothesis:* Testing whether giving the model more thinking tokens allows it to complete reasoning paths, boosting accuracy above 16.4%.

- [ ] **3. Qualitative Reasoning & Error Analysis**
  - Extract 2–3 concrete examples from `results/*outputs.jsonl` where the model uses `<think>` to explore alternatives, backtracks upon an invalid path, and finds the target equation.
  - Categorize the failure cases on 3 and 4 numbers:
    - Arithmetic error (e.g., calculation mistake inside `<answer>`)
    - Constraint error (reused or missing numbers)
    - Truncation error (cut off mid-sentence due to token budget)

- [ ] **4. Live Interactive Demo (Optional for Poster Presentation)**
  - Keep a lightweight CLI countdown solver script ready on a laptop during the poster session to allow professors/peers to input custom numbers and targets live.

- [ ] **5. Final Report & Presentation Assembly**
  - Integrate comparison tables, charts, and theoretical explanations of GRPO vs PPO, verifiable rewards, and compute scaling.

---

## 2. Experimental Results Summary (Achieved)

### Qwen2.5-1.5B Baseline vs. Stage 2 GRPO

| Level | Method | Accuracy (%) | Format Rate (%) | Mean Tokens | Clipped Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2 Numbers (In-Domain)** | Baseline (CoT) | 3.2% | 0.4% | ~53 | - |
| **2 Numbers (In-Domain)** | **GRPO (Ours)** | **97.4%** | **99.0%** | **35.3** | **1.0%** |
| **3 Numbers (In-Domain)** | Baseline (CoT) | 1.8% | 0.0% | ~46 | - |
| **3 Numbers (In-Domain)** | **GRPO (Ours)** | **69.4%** | **86.6%** | **110.1** | **13.6%** |
| **All In-Domain (2 & 3 Numbers)** | Baseline (CoT) | 1.5% | 0.2% | ~50 | - |
| **All In-Domain (2 & 3 Numbers)** | **GRPO (Ours)** | **83.4%** | **92.8%** | **72.7** | **7.3%** |
| **4 Numbers (Out-of-Domain)** | Baseline (CoT) | 1.2% | 0.0% | ~55 | - |
| **4 Numbers (Out-of-Domain)** | **GRPO (Ours)** | **16.4%** | **48.2%** | **188.9** | **51.6%** |

---

## 3. Reference Scripts for Upcoming Experiments

### Script A: 512 Tokens Budget Evaluation on OOD Set
```python
import sys, subprocess
from pathlib import Path

found = list(Path("/kaggle/input").rglob("stage2/final/adapter_config.json"))
if found:
    adapter_dir = str(found[0].parent)
    subprocess.run([
        sys.executable, "-m", "src.eval.evaluate",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--method", "grpo",
        "--adapter", adapter_dir,
        "--data", "data/test_ood.jsonl",
        "--max_new_tokens", "512"
    ], check=True)
```

### Script B: Extract Qualitative Traces from outputs.jsonl
```python
import json
from pathlib import Path

output_files = list(Path(".").rglob("*grpo*outputs.jsonl"))
if output_files:
    with open(output_files[0], "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    correct_3nums = [s for s in samples if s.get("n_numbers") == 3 and s.get("correct")]
    for idx, s in enumerate(correct_3nums[:3], 1):
        print(f"--- Example {idx} | Numbers: {s['numbers']} -> Target: {s['target']} ---")
        print(s['text'])
        print("-" * 50)
```

### Script C: Interactive Live Countdown Solver
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.prompts import build_prompt, PREFILL

def solve_countdown(numbers, target, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()

    prompt = build_prompt(tok, numbers, target, "grpo")
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(PREFILL["grpo"] + generated)
```

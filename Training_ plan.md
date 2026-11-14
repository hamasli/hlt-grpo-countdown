# Training Plan: GRPO on Countdown (HLT 2026/27, Group 2)

This document explains **every decision** for the GRPO training: **what** we chose, **what it means** and **why**.

---

## 0. Where we are (results before training)

| Result | Value |
|---|---|
| Dataset | 3 and 4 numbers (1–100), targets 10–100; train 10k / val 200 / test 1k; seed 42 |
| Checker | `verify.py`: safe `ast` parsing, each number exactly once; 23/23 tests pass |
| Trick baselines | simple tricks ≤ 5%; random equation @1 = 11.0% / 2.4%, @8 = 43.6% / 11.2% (3 / 4 numbers) |
| Baselines (greedy, test) | all models 1–2% accuracy; CoT gives no gain; 1.5B format 100% |
| Error analysis | main error = **wrong value** (56–95%); 0.5B also breaks the rules (~30%) |
| pass@8 (sampling) | 0.5B: 2.6% · **1.5B: 17.5%** (3 numbers 26.8%, 4 numbers 8.2%) |

**Conclusion:** the models can't solve the task before training. 1.5B has enough learning signal for GRPO; 0.5B doesn't.

---

## 1. Model

| Setting | Value | Meaning | Why |
|---|---|---|---|
| Model | **Qwen2.5-1.5B-Instruct** | the main model we train | 17.5% learning signal (0.5B only 2.6%) |
| Test model | Qwen2.5-0.5B-Instruct | used only for a 10-step code test | small and fast, so bugs show up quickly |
| Training method | **LoRA** | freeze the model; train small extra matrices | full fine-tuning needs ~25 GB > 16 GB (the professor's point) |
| LoRA rank `r` | **16** | the middle size of the small matrices (A: 16×in, B: out×16) | standard balance of capacity vs memory |
| LoRA alpha | **32** | strength: the LoRA output × (alpha / r) = × 2 | common rule alpha = 2r |
| LoRA targets | **q, k, v, o, gate, up, down** | all 7 big matrices in each of the 28 layers | the best quality; the model adapts everywhere |
| Trained parameters | ~18.5M (~1.2%) | 196 big matrices, each with its own A + B pair | small, so it fits in memory |
| Precision | **fp16** model, fp32 LoRA | 16-bit numbers for the frozen model | the T4 has no bf16; half the memory |

**How LoRA works:** `output = W·x + (alpha/r)·B·(A·x)`. A squeezes the input to 16 numbers, B expands it back. B starts at zero, so at step 0 the model is exactly the original.

---

## 2. Data and curriculum

| Decision | Value | Why |
|---|---|---|
| Training data | `train.jsonl` only | val and test stay unseen, so the results are honest |
| **Stage 1** | **3-number** puzzles only | the most learning signal (26.8%) |
| **Stage 2** | **3 + 4 numbers mixed**, starting from the Stage 1 LoRA | harder puzzles once the model is stronger; keeping 3-number puzzles avoids forgetting |
| Puzzles per step | 4 (× 8 answers = 32) | fits memory and time |
| Puzzles used | ~600 per stage, ~1,200 total | GRPO learns from steps, not data size; less than 1 epoch is normal |
| Prompt | **the same CoT prompt + `<think>` prefill** as the baseline | a fair comparison: only the training differs |
| Data format for TRL | columns `prompt`, `numbers`, `target` | TRL passes `numbers` and `target` to the reward functions |
| Validation | `val.jsonl` (200) | progress checks at checkpoints |
| Test | `test.jsonl` (1,000) | final results only |

**Why not use all 10,000?** 1 epoch = 2,500 steps ≈ 30–40 GPU hours, more than Kaggle allows. The training cost depends on steps, not dataset size; unused puzzles cost nothing.

**Prefill detail:** the prompt ends with `<think>`, so the answer starts after it. Before the format check we add `<think>` back, the same as `evaluate.py`.

---

## 3. GRPO settings

### One training step
```
1. Take 4 new puzzles
2. Model writes 8 answers per puzzle (sampling) → 32 answers = 4 groups of 8
3. verify.py → reward per answer: 0 / 0.1 (format) / 1.0 (correct) / 1.1 (both)
4. GRPOTrainer → advantage inside each group:
      advantage = (reward − group mean) / group std
      above average → more likely | below → less likely | all equal → no signal
5. Update ONLY the LoRA matrices
6. Repeat
```

| Setting | Value | Meaning | Why |
|---|---|---|---|
| `num_generations` (G) | **8** | answers per puzzle | G=4 → only ~16% useful groups; papers use 16–64; 8 is a low-budget balance |
| `per_device_train_batch_size` | 8 | answers processed at once (1 puzzle) | memory |
| `gradient_accumulation_steps` | 4 | sum 4 small batches, then update once | 4 puzzles per step with low memory |
| `learning_rate` | **1e-5** | size of each update | safe for LoRA; 3e-5 if the reward stays flat |
| `beta` (KL) | **0.0** | no penalty for moving away from the original model | no reference model, so less memory and time; LoRA already limits drift; recent work (DAPO) also drops it |
| Rewards | **1.0 correctness + 0.1 format** | rule-based (DeepSeek-R1 style) | the format guides but can't "win" alone |
| Clipping ε | 0.2 (default) | max ~20% change per update | a safety belt from PPO |
| Temperature | **1.0** (top-k off, top-p 1.0) | plain random sampling | the 8 answers must differ; greedy would give 8 identical answers → no learning |
| `gradient_checkpointing` | True | recompute instead of storing | saves memory |
| Seed | 42 | | reproducible |

**Decoding methods (Lecture 15):** greedy, random, temperature, top-k, top-p, beam search. We use **random sampling (τ=1)** for training and **greedy** for evaluation.

---

## 4. Answer length and compute cost

| Decision | Value | Why |
|---|---|---|
| `max_completion_length` | **256** | about 4× the current average (~51–66 tokens); cut-off answers get reward 0, which teaches the model to finish |
| Watch | clipped ratio | if > 20%, raise to 384 (R1 showed answers get longer) |
| Time per step | ~40–60 s (confirmed by the timing test) | generation is ~60–70% of the time |
| Steps | **~150 per stage, ~300 total** (final number after the timing test) | budget-limited; stop early if the reward is flat |
| Generation engine | standard Hugging Face (no vLLM) | simpler and safer on a T4 |

### Memory check (T4, 16 GB)
| | Full fine-tuning | **LoRA (ours)** |
|---|---|---|
| Weights | 6.2 GB (fp32) | 3.1 GB (fp16) |
| Gradients | 6.2 GB | ~0.07 GB |
| Adam | 12.4 GB | ~0.15 GB |
| Activations + logits + generation | extra | ~3–5 GB |
| **Total** | **> 25 GB ❌** | **~7–9 GB ✅** |

### GPU budget (Kaggle: 30 h/week, 12 h/session)
| Task | Time |
|---|---|
| Smoke test (0.5B, 10 steps) | ~0.2 h |
| Timing test (1.5B, 10 steps) | ~0.3 h |
| Stage 1 (150 steps) | ~2.5 h |
| Stage 2 (150 steps) | ~2.5 h |
| Evaluations | ~1.5 h |
| **Total** | **~7 h** |

---

## 5. Running on Kaggle (safe and resumable)

| Protection | How |
|---|---|
| **Background run** | Save Version → **Save & Run All (Commit)**: runs on Kaggle's servers; the browser can be closed |
| **Checkpoints** | every **25 steps**; keep the last 2 (`save_total_limit=2`) |
| **Backup** | push checkpoints to **Hugging Face Hub** (private), `hub_strategy="checkpoint"` → `last-checkpoint` |
| **Resume** | `--resume`: download `last-checkpoint`, continue from that step (lose at most 24 steps) |
| **Stage 2** | `--init_adapter`: load only the Stage 1 LoRA weights; new optimizer; step counter from 0 |
| **Secret** | `HF_TOKEN` in Kaggle Secrets (never in the code) |
| **Versions** | fixed library versions, for reproducibility |

A checkpoint contains: LoRA weights (~70 MB), optimizer state (~150 MB), trainer state (step + logs), random state.

### Notebook cells
1. Check the GPU
2. Clone branch `hamas`
3. Install libraries (fixed versions)
4. Log in to Hugging Face (from Secrets)
5. Generate the data (seed 42)
6. Train (with `--resume` if needed)
7. Quick val check (200 puzzles)

### Before training (to do)
- [ ] Hugging Face account
- [ ] Write token → Kaggle Secrets as `HF_TOKEN`

---

## 6. Monitoring during training

Logged every 5 steps; saved in `trainer_state.json`; `log_completions=True` shows real answers.

| Metric | Healthy sign |
|---|---|
| reward (total) | slowly rises |
| correctness reward | **rises: the main goal** |
| format reward | quickly reaches ~0.1 |
| frac_reward_zero_std (groups with no signal) | goes down |
| mean_length | stable or slowly rising |
| clipped_ratio | low (< 10–20%) |
| loss | **not meaningful in GRPO**; don't judge by it |
| grad_norm | no sudden spikes |

### Warning signs
| Sign | Action |
|---|---|
| reward flat after ~50 steps | learning rate 3e-5, or easier puzzles |
| no-signal groups > 90% | stay longer on 3-number puzzles |
| clipped ratio > 20% | length 384 |
| NaN | lower the learning rate, resume from the checkpoint |
| nonsense or repeated text | stop; beta = 0.04 |
| reward ↑ but val ↔ | possible reward hacking; read the answers |

**Honest check:** greedy evaluation on **val** at the checkpoints (e.g. steps 75 and 150).

### Plots
1. Reward vs step (total, correctness, format)
2. Answer length vs step
3. % groups with no signal vs step
4. Val accuracy per checkpoint
5. Final bar chart: direct vs CoT vs GRPO

---

## 7. Final evaluation

### Comparison
| Method | Model | Prompt | Training |
|---|---|---|---|
| Direct | 1.5B | equation only | none |
| CoT | 1.5B | think + answer | none |
| **GRPO** | 1.5B + LoRA | **the same as CoT** | GRPO ~300 steps |

Reference lines: trick baselines (random @1, @8) and the 0.5B results.

**Fair test:** the same 1,000 test puzzles · greedy · the same `verify.py` · max 512 tokens.

### Metrics
| Metric | From |
|---|---|
| **Accuracy** (3 / 4 / all) | proposal |
| **Answer length** (tokens) | proposal |
| **Training cost** (GPU, hours, memory) | proposal |
| Format rate | extra |
| Error analysis | extra: which errors GRPO fixed |
| pass@8 (first 300) | extra: did the model's ability grow? |
| Reasoning examples | extra: verification and backtracking (Gandhi et al.) |

**Significance:** McNemar's test on paired results (the same puzzles); p < 0.05 = a real difference.

### Expectations (hypotheses)
| Metric | Expectation |
|---|---|
| Format | → ~100% |
| Wrong numbers | ⬇ a lot |
| Accuracy, 3 numbers | ⬆ clearly above CoT (maybe 10–30%) |
| Accuracy, 4 numbers | ⬆ smaller gain |
| Length | stable or slightly longer |
| Reasoning | some verification may appear |

### Success levels
| Level | Result |
|---|---|
| Minimum ✅ | GRPO significantly better than CoT (p < 0.05) |
| Good ⭐ | GRPO > random @1 (11% / 2.4%) |
| Great 🚀 | GRPO approaches random @8 on 3 numbers (43.6%) with 1 answer |

If GRPO doesn't improve, that's still a valid result, explained with our signal, length and model-size analyses.

---

## 8. Timeline

| Date | Task |
|---|---|
| Nov 9 | `grpo_train.py` + smoke test + timing test |
| Nov 10 | Slides for round 1 |
| Nov 11–12 | **Presentation round 1** |
| Nov 13–18 | Stage 1 + Stage 2 training |
| Nov 19–22 | Final evaluation + analysis + plots |
| Nov 24–26 | **Presentation round 2** |
| Dec 3 | Poster, code, report |

---

## 9. References and their role

| Paper | Role in the project |
|---|---|
| Shao et al. (2024), DeepSeekMath | the **GRPO** algorithm (group advantage, no critic) |
| DeepSeek-AI (2025), DeepSeek-R1 | **reward design** (accuracy + format), `<think>/<answer>`, length growth |
| Gandhi et al. (2025), Cognitive Behaviors | **Countdown + Qwen**; RL amplifies existing habits (verification, backtracking) |
| Wei et al. (2022), Chain-of-Thought | **CoT baseline**; CoT gains appear mainly in large models |
| Kojima et al. (2022), Zero-shot reasoners | zero-shot CoT prompt (to add) |
| Hu et al. (2021), LoRA | memory-efficient training |

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Step** | one update of the model (4 puzzles × 8 answers) |
| **Epoch** | one full pass over the training data |
| **Group** | the 8 answers to the same puzzle |
| **Advantage** | how much better an answer is than its group average |
| **LoRA r** | size (capacity) of the small matrices |
| **LoRA alpha** | strength of the LoRA update |
| **KL / beta** | penalty for moving away from the original model |
| **Temperature** | randomness of sampling (1.0 = the model's real distribution) |
| **Greedy** | always pick the most likely token (deterministic) |
| **pass@k** | % of puzzles with ≥ 1 correct answer out of k |
| **Checkpoint** | a saved snapshot of the training (LoRA + optimizer + step) |
| **Curriculum** | train on easy puzzles first, then harder ones |
| **Gradient accumulation** | sum several small batches before one update |
| **Gradient checkpointing** | recompute instead of storing, to save memory |
| **Clipped ratio** | % of answers cut off at the length limit |
| **Reward hacking** | the model gets reward without really solving the task |
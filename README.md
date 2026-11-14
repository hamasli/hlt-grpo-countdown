# Reasoning with Verifiable Rewards: GRPO on Countdown

HLT 2026/27 project, University of Pisa (Group 2).

**Goal:** make a small language model better at reasoning with reinforcement learning (GRPO),
using a reward that a program checks automatically (a *verifiable reward*).

---

## The task: Countdown
Given some numbers and a target, write an equation that uses **each number exactly once**,
with + − × ÷ and brackets, to reach the target.

Example: numbers `2, 5, 9`, target `19` → `2 * 5 + 9`

Countdown is hard to solve but easy to check: a short program verifies every answer exactly,
so we need no human labels and no judge model.

## Methods compared
| Method | Prompt | Training |
|---|---|---|
| Zero-shot | give only the equation | none |
| Chain-of-thought (CoT) | think inside `<think>`, then answer inside `<answer>` | none |
| **GRPO + LoRA** | same prompt as CoT | reinforcement learning *(in progress)* |

**Reward:** 1.0 for a correct equation + 0.1 for the right format.
**Metrics:** accuracy (by level), answer length (tokens), training cost (GPU time and memory).

## Model
- `Qwen2.5-1.5B-Instruct` + **LoRA**: the main model
  (full fine-tuning needs more than 25 GB; the Kaggle T4 has 16 GB)
- `Qwen2.5-0.5B-Instruct`: only to test the pipeline quickly

---

## Project structure
```
data/                      generated puzzles (train / val / test / test_ood)
docs/                      training plan and progress logs
notebooks/                 Kaggle notebooks
results/                   metrics, model outputs, analyses, charts
src/data/generate.py       puzzle generator
src/reward/verify.py       automated verifier + reward functions
src/prompts.py             prompts for zero_shot / cot / grpo
src/eval/evaluate.py       runs a model on the puzzles and scores it
src/eval/trick_baselines.py      simple tricks and random guessing (no model)
src/eval/error_analysis.py       why answers fail
src/eval/plot_learning_signal.py chart of the GRPO learning signal
src/train/                 GRPO training (next step)
tests/verify_test.py       32 unit tests for the verifier
```

## Dataset
Each puzzle is made by picking random numbers (1–100), combining them with a random equation,
and using the result as the target (10–100), so every puzzle has a solution. Seed 42.

| File | Puzzles | Levels | Used for |
|---|---|---|---|
| `train.jsonl` | 10,000 | 5,000 × 2 numbers + 5,000 × 3 numbers | GRPO training |
| `val.jsonl` | 200 | 100 + 100 | checks during training, prompt checks |
| `test.jsonl` | 1,000 | 500 + 500 | final results |
| `test_ood.jsonl` | 500 | 4 numbers | out-of-domain test |

## Verifier
Parses the answer with Python `ast` (Abstract Syntax Tree; it never runs code), allows only numbers,
+ − × ÷ and brackets, requires each given number exactly once, uses exact fractions, and accepts
common math notation (×, ÷, LaTeX `\times`, `\div`). 32 unit tests pass.

---

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
GPU runs are done on Kaggle (Tesla T4, 16 GB).

## How to run
```bash
python -m src.data.generate                                  # make the data
python -m pytest -v                                          # test the verifier
python -m src.eval.trick_baselines                           # tricks and random guessing
python -m src.eval.evaluate --method cot                     # one model baseline (GPU)
python -m src.eval.evaluate --method cot --data data/test_ood.jsonl
python -m src.eval.evaluate --method cot --num_samples 4     # learning signal (pass@4)
python -m src.eval.error_analysis                            # why answers fail
python -m src.eval.plot_learning_signal                      # learning-signal chart
```

---

## Results so far (before training)

### Trick baselines
| Trick | 2 numbers | 3 numbers | 4 numbers (OOD) |
|---|---|---|---|
| Sum of all numbers | 36.0% | 5.4% | 1.0% |
| One random equation | 52.4% | 12.2% | 2.4% |
| Best of 4 random equations | 78.6% | 27.6% | 6.8% |

### Model baselines (greedy, test set)
| Run | 2 numbers | 3 numbers | 4 numbers (OOD) |
|---|---|---|---|
| 0.5B zero-shot | 0.0% | 0.0% | 0.2% |
| 0.5B CoT | 7.6% | 1.6% | 0.2% |
| 1.5B zero-shot | 0.2% | 0.0% | 0.0% |
| **1.5B CoT** | **26.4%** | **3.0%** | **0.4%** |

All models are below random guessing. Main errors: inventing extra numbers (2 numbers)
and the wrong combination (3 numbers).

### Learning signal for GRPO (CoT, 4 sampled answers per puzzle)
% of puzzles where the 4 answers are partly correct (the only puzzles GRPO learns from):

| Model | 2 numbers | 3 numbers | All |
|---|---|---|---|
| Qwen2.5-0.5B | 3.4% | 1.4% | 2.4% |
| **Qwen2.5-1.5B** | **50.4%** | **25.4%** | **37.9%** |

→ We train the 1.5B model.

### Prompt sensitivity
| Version | Prompt | Result |
|---|---|---|
| v1 | numeric example `(1 + 2) / 3` | models copied its shape and invented numbers (0% on 2 numbers) |
| v2 | no example | zero-shot answers lost the answer tags |
| **v3 (frozen)** | placeholder `<answer> equation </answer>` + "Do not use any other numbers" | used for all results |

---

## Training plan
- **Stage 1:** 2-number puzzles → **Stage 2:** 3-number puzzles → *optional Stage 3:* 4-number puzzles
- Each step: 8 puzzles × 4 sampled answers, max 256 tokens, learning rate 1e-5, no KL penalty, LoRA, fp16
- Checkpoints saved during training, so a stopped run can resume

Details: `docs/TRAINING_PLAN.md` · daily log: `docs/PROGRESS_2026-11-14.md`

## Status
- [x] Data generator and verifier (32 tests)
- [x] Trick baselines, model baselines, error analysis
- [x] Learning-signal analysis and model choice
- [ ] GRPO training script (`src/train/grpo_train.py`)
- [ ] Stage 1 and Stage 2 training
- [ ] Final evaluation: zero-shot vs CoT vs GRPO

## References
- Shao et al. (2024). DeepSeekMath. arXiv:2402.03300 (GRPO)
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948
- Gandhi et al. (2025). Cognitive Behaviors that Enable Self-Improving Reasoners. arXiv:2503.01307
- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS
- Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners. arXiv:2205.11916
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
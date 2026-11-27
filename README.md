# Reasoning with Verifiable Rewards: GRPO on Countdown

HLT 2026/27 project, University of Pisa (Group 2).

**Goal:** make a small language model better at reasoning with reinforcement learning (GRPO),
using a reward that a program checks automatically (a *verifiable reward*).

> **Branches**
> - `main` (this branch): **v1**, natural dataset. Results below.
> - `version2`: **v2**, operator-balanced dataset (more × and ÷). These are the **final results** used in the poster and report.
>
> Model weights are not stored on GitHub.

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
| **GRPO + LoRA** | same prompt as CoT | reinforcement learning (2-stage curriculum) |

**Reward:** 1.0 for a correct equation + 0.1 for the right format.
**Metrics:** accuracy (by level), answer length (tokens), cut-off rate (hit the 256-token limit), training cost (GPU time and memory).

## Model
- `Qwen2.5-1.5B-Instruct` + **LoRA**: the main model
  (full fine-tuning needs more than 25 GB; the Kaggle T4 has 16 GB)
- `Qwen2.5-0.5B-Instruct`: only to test the pipeline quickly

---

## Project structure
```
data/                      generated puzzles (train / val / test / test_ood)
docs/                      training plan and progress logs
notebooks/                 Kaggle notebooks (evaluation and training)
results/                   metrics, model outputs, analyses, charts
report/                    result summaries (RESULTS.md)
src/data/generate.py       puzzle generator
src/reward/verify.py       automated verifier + reward functions
src/prompts.py             prompts for zero_shot / cot / grpo
src/train/grpo_train.py    GRPO + LoRA training (TRL)
src/eval/evaluate.py       runs a model on the puzzles and scores it
src/eval/trick_baselines.py      simple tricks and random guessing (no model)
src/eval/error_analysis.py       why answers fail
src/eval/summarize_stage.py      validation accuracy per checkpoint, picks the best
src/eval/plot_learning_signal.py chart of the GRPO learning signal
tests/verify_test.py       32 unit tests for the verifier
```

## Dataset (v1, natural)
Each puzzle is made by picking random numbers (1–100), combining them with a random equation,
and using the result as the target (10–100), so every puzzle has a solution. Seed 42.

| File | Puzzles | Levels | Used for |
|---|---|---|---|
| `train.jsonl` | 10,000 | 5,000 × 2 numbers + 5,000 × 3 numbers | GRPO training |
| `val.jsonl` | 200 | 100 + 100 | checkpoint selection, prompt checks |
| `test.jsonl` | 1,000 | 500 + 500 | final results |
| `test_ood.jsonl` | 500 | 4 numbers | out-of-domain test (never trained) |

**Known bias:** only 3.9% / 14.3% / 26.8% of 2 / 3 / 4-number puzzles use × or ÷.
Most v1 puzzles can be solved with + and − only. This is why we built the balanced v2 dataset (branch `res`).

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
GRPO training runs on Kaggle: see `notebooks/` and `src/train/grpo_train.py`.

---

## Training setup
| Setting | Value |
|---|---|
| LoRA | r = 16, α = 32, dropout 0.05, on q, k, v, o, gate, up, down (≈ 18.5 M trainable params, 1.2%) |
| GRPO | 8 puzzles × 4 sampled answers per step, β = 0 (no KL), ε = 0.2 |
| Optimiser | learning rate 1e-5, constant with 5% warm-up, grad clip 1.0 |
| Sampling | temperature 1.0, max 256 new tokens |
| Precision | fp16, gradient checkpointing |
| Curriculum | **Stage 1:** 2 numbers, 100 steps → **Stage 2:** 3 numbers, 400 steps (from the Stage 1 LoRA) |
| Model selection | checkpoint every 25 steps, best on validation → **step 375** |
| Cost (T4) | Stage 1: 24 min (14.5 s/step) · Stage 2: 2.72 h (24.45 s/step) |

---

## Results (v1, test set, greedy decoding)

### Main results
| Method | 2 numbers | 3 numbers | In-domain (avg) | 4 numbers (OOD) |
|---|---|---|---|---|
| 1.5B zero-shot | 0.2% | 0.0% | 0.1% | 0.0% |
| 1.5B CoT | 26.4% | 3.0% | 14.7% | 0.4% |
| Best of 4 random equations | 78.6% | 27.6% | – | 6.8% |
| **1.5B GRPO (step 375)** | **95.6%** | **64.0%** | **79.8%** | **12.8%** |

GRPO improves in-domain accuracy by about 65 points and beats random guessing on every level.
**But** v1 is dominated by + and −, so this mostly measures +/− search.
On the balanced v2 test (branch `res`), GRPO gets 92.2% / 30.8% / 11.4%,
and only 3.3% on 3-number puzzles that need × or ÷.

### Trick baselines (no model)
| Trick | 2 numbers | 3 numbers | 4 numbers (OOD) |
|---|---|---|---|
| Sum of all numbers | 36.0% | 5.4% | 1.0% |
| One random equation | 52.4% | 12.2% | 2.4% |
| Best of 4 random equations | 78.6% | 27.6% | 6.8% |

### Model baselines before training (greedy)
| Run | 2 numbers | 3 numbers | 4 numbers (OOD) |
|---|---|---|---|
| 0.5B zero-shot | 0.0% | 0.0% | 0.2% |
| 0.5B CoT | 7.6% | 1.6% | 0.2% |
| 1.5B zero-shot | 0.2% | 0.0% | 0.0% |
| **1.5B CoT** | **26.4%** | **3.0%** | **0.4%** |

### Learning signal for GRPO (CoT, 4 sampled answers per puzzle)
% of puzzles where the 4 answers are partly correct (the only puzzles GRPO learns from):

| Model | 2 numbers | 3 numbers | All |
|---|---|---|---|
| Qwen2.5-0.5B | 3.4% | 1.4% | 2.4% |
| **Qwen2.5-1.5B** | **50.4%** | **25.4%** | **37.9%** |

→ We train the 1.5B model.

### Validation during Stage 2
- 3 numbers: 24% → 33% → 46% → 54% → **60% (best, step 375)**
- 2 numbers dipped to 65% around steps 75–175, then recovered.

### Answer length and cut-off (2 / 3 / 4 numbers)
| Method | Mean tokens | Cut off at 256 tokens |
|---|---|---|
| CoT | 43 / 76 / 81 | 0.4% / 4.6% / 2.6% |
| GRPO | 29 / 52 / 88 | 0.8% / 2.2% / 8.4% |

### Error analysis (2 + 3 numbers, % of answers)
| Method | Correct | No answer | Invalid | Wrong numbers | Wrong value | Mean distance to target |
|---|---|---|---|---|---|---|
| CoT | 14.7 | 2.6 | 5.9 | 38.0 | 38.8 | 1,269 |
| GRPO | 79.8 | 0.7 | 0.0 | 4.3 | 15.2 | 21.6 |

After GRPO, wrong answers are much closer to the target.

### Prompt sensitivity
| Version | Prompt | Result |
|---|---|---|
| v1 | numeric example `(1 + 2) / 3` | models copied its shape and invented numbers (0% on 2 numbers) |
| v2 | no example | zero-shot answers lost the answer tags |
| **v3 (frozen)** | placeholder `<answer> equation </answer>` + "Do not use any other numbers" | used for all results |

---

## Status
- [x] Data generator and verifier (32 tests)
- [x] Trick baselines, model baselines, error analysis
- [x] Learning-signal analysis and model choice
- [x] GRPO training script (`src/train/grpo_train.py`)
- [x] Stage 1 and Stage 2 training (v1)
- [x] Final evaluation: zero-shot vs CoT vs GRPO (v1)
- [x] Balanced dataset and full re-run (v2, branch `res`)
- [x] Poster
- [ ] Report
- [ ] Web demo (Gradio)

## References
- Shao et al. (2024). DeepSeekMath. arXiv:2402.03300 (GRPO)
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948
- Yue et al. (2025). Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? arXiv:2504.13837
- Liu et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO). arXiv:2503.20783
- Gandhi et al. (2025). Cognitive Behaviors that Enable Self-Improving Reasoners. arXiv:2503.01307
- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS
- Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners. arXiv:2205.11916
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685

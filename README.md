# Reasoning with Verifiable Rewards: GRPO on Countdown

HLT 2026/27 project, University of Pisa (Group 2).
Goal: make a small language model better at reasoning with reinforcement learning (GRPO),
using a task whose answers can be checked by a program.

## Task: Countdown
Given numbers and a target, write an equation that equals the target.
Use + − × ÷ and brackets; each number exactly once.

Example: numbers `[3, 25, 50]`, target `72` → `50 + 25 - 3`

## Methods compared
1. **Direct (zero-shot):** the model gives only the equation
2. **Chain-of-thought (CoT):** the model reasons in `<think>`, then answers in `<answer>`
3. **GRPO:** the model is trained with RL, using the CoT prompt *(in progress)*

Rewards: **1.0** for a correct equation, **+0.1** for the right format.
Metrics: accuracy, answer length (tokens), cost (GPU time and memory).

## Models
- `Qwen2.5-0.5B-Instruct`: to build and test the pipeline
- `Qwen2.5-1.5B-Instruct`: main model
- Training with **LoRA** (full fine-tuning of 1.5B needs ~24 GB, more than Kaggle's 16 GB)

## Project structure
```
configs/             settings (model, LoRA, GRPO)
data/                generated puzzles (train / val / test)
notebooks/           Kaggle notebooks
results/             metrics, model outputs, analyses
src/data/            generate.py: puzzle generator
src/reward/          verify.py: checker + reward functions
src/prompts.py       prompts for direct / CoT / GRPO
src/eval/            evaluate.py, trick_baselines.py, error_analysis.py
src/train/           GRPO training (coming next)
tests/               tests for the checker
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```
GPU runs are done on Kaggle (Tesla T4, 16 GB).

## How to run
```bash
python -m src.data.generate                      # make data/train, val, test
python -m pytest -v                              # test the checker
python -m src.eval.trick_baselines               # simple tricks
python -m src.eval.evaluate --method cot         # model baseline (GPU recommended)
python -m src.eval.error_analysis                # why answers fail
```

## Progress

### ✅ Step 0: Setup
Repo, Python 3.11 environment, Kaggle GPU.

### ✅ Step 1: Dataset and checker
- Puzzles with 3 and 4 numbers (1–100), targets 10–100, every puzzle solvable
- Splits: train 10,000 / val 200 / test 1,000 (half 3-number, half 4-number), seed 42
- Checker: safe parsing with `ast` (no `eval`), each number exactly once, exact fractions
- 23/23 tests pass

**Trick baselines (test set):**

| Trick | 3 numbers | 4 numbers |
|---|---|---|
| Target in numbers | 0.0% | 0.0% |
| Sum of all | 5.2% | 1.0% |
| Product of all | 0.0% | 0.0% |
| Random equation @1 | 11.0% | 2.4% |
| Random equation @8 | 43.6% | 11.2% |

### ✅ Step 2: Model baselines (test set, 1,000 puzzles, greedy decoding)

| Model | Method | Acc 3 nums | Acc 4 nums | Acc all | Format | Tokens | Time (T4) |
|---|---|---|---|---|---|---|---|
| 0.5B | direct | 3.2% | 0.2% | 1.7% | 87.0% | 19.5 | 29 s |
| 0.5B | CoT | 2.2% | 0.6% | 1.4% | 0.2% | 49.5 | 137 s |
| 1.5B | direct | 2.0% | 0.4% | 1.2% | 99.9% | 20.2 | 52 s |
| 1.5B | CoT | 1.8% | 1.2% | 1.5% | 100% | 50.6 | 110 s |

**Error analysis (all levels):**

| Run | Correct | No answer | Invalid | Wrong numbers | Wrong value |
|---|---|---|---|---|---|
| 0.5B direct | 1.7% | 13.0% | 0% | 29.3% | 56.0% |
| 0.5B CoT | 1.4% | 2.1% | 0% | 31.3% | 65.2% |
| 1.5B direct | 1.2% | 0.1% | 0% | 13.2% | 85.5% |
| 1.5B CoT | 1.5% | 0% | 0% | 3.8% | 94.7% |

**Findings:**
1. Before training, all models solve only 1–2% of puzzles, below one random equation (6.7%).
2. CoT gives no accuracy gain at this scale (consistent with Wei et al., 2022).
3. The 1.5B model follows the format perfectly but is not more accurate.
4. The main error is *wrong value*: the models write valid equations but cannot find the right combination.
5. CoT helps the 1.5B model respect the rules (wrong numbers on 4-number puzzles: 25.6% → 4.2%).
6. Most correct answers come from cancelling tricks (×1, ×k ÷k), not real search.

### ⏳ Next
- pass@8 with sampling (how much learning signal GRPO gets)
- Presentation round 1 (Nov 11–12)
- GRPO training with LoRA: 0.5B first, then 1.5B

## References
- Shao et al. (2024). DeepSeekMath. arXiv:2402.03300 (GRPO)
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948
- Gandhi et al. (2025). Cognitive Behaviors that Enable Self-Improving Reasoners. arXiv:2503.01307
- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
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

Each GRPO arm gives **+1.0** for a correct equation. The arms differ in how they award the format bonus.
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
src/train/           reward_comparison.py: paired GRPO training
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

## Objective 4 · Controlled GRPO training

On Kaggle, enable Internet and select a GPU accelerator. A **T4 ×2** session
runs one seed per GPU in the notebook's full mode. The single-GPU commands
below train the arms sequentially. Import [the experiment notebook](notebooks/countdown-reward-hacking.ipynb)
from the updated `feat/countdown-reward-hacking` branch. Its setup cell clones or
fast-forwards the checkout. Run individual cells for the objective you need.

In Objective 4, run the dependency setup cell and then the smoke test cell.
After the smoke test succeeds, run the full training cell. Full training uses
seeds 42 and 43 with 250 updates per arm. Choose a fresh output directory in
the training cell for a repeat run.

| Arm | Correctness reward | Format bonus |
|---|---:|---|
| Correctness only | +1.0 | 0 |
| Additive | +1.0 | +0.1 whenever the response passes the format checker |
| Correctness-gated | +1.0 | +0.1 only for correct responses that pass the format checker |

To run the same workflow manually from the repository root in Kaggle, install
the trainer dependencies alongside Kaggle's CUDA PyTorch:

```python
!python -m pip install "transformers==4.57.6" "trl==0.29.0" "peft==0.18.1" "datasets==5.0.1" "accelerate==1.15.0"
```

Run the smoke command first. After it succeeds, run the full command in a separate cell.

```python
!CUDA_VISIBLE_DEVICES=0 python -m src.train.reward_comparison --seeds 42 --max-steps 2 --include-gated --out-dir /kaggle/working/countdown-objective4-smoke-three-arms
```

```python
!CUDA_VISIBLE_DEVICES=0 python -m src.train.reward_comparison --seeds 42 43 --max-steps 250 --include-gated --out-dir /kaggle/working/countdown-objective4-full-three-arms
```

Every arm starts from the same pinned Qwen2.5-1.5B-Instruct revision. Runs with the
same seed use matching initial LoRA weights and puzzle order, both verified by
the runner. The CoT prompt, `<think>` prefill, optimizer, and sampling settings
are shared. Each update samples four responses to one training puzzle at
temperature 0.8 with a 512-token cap.

- `experiment.json` records the run configuration and training data hash.
- `seed-*/pairing.json` records the verified initialization hash and puzzle order.
- Each arm saves an `adapter/`, a `rollouts.jsonl` record per update, the TRL log,
  and training time and GPU memory usage. The rollout log includes counts and
  rates for strict solves, format compliance and formatted wrong answers. It
  splits the counts by three and four inputs so rates can be aggregated with
  the right denominators.

Compare these common metrics across arms. The held-out evaluation belongs to
Objective 5.

## Objective 5 · Held-out reward hacking evaluation

Keep the verified `countdown-objective4-parallel.zip` outside Git. On Kaggle,
attach it as an input, enable T4 ×2 and Internet, and run the Git setup and
`load-verifier` cells in the updated [experiment notebook](notebooks/countdown-reward-hacking.ipynb).
Then go to Objective 5. Set `OBJECTIVE_4_ARCHIVE` to the ZIP's path and run
the evaluation cell. The default
`OBJECTIVE_5_GPUS = (0, 1)` runs seed 42 on GPU 0 and seed 43 on GPU 1.
Use `(0,)` on a single GPU. You can skip the earlier experiment cells.

To run from the repository root with the tested CUDA PyTorch environment:

```bash
python -m pip install "transformers==4.57.6" "peft==0.18.1"
python -m src.eval.reward_hacking --archive /path/to/countdown-objective4-parallel.zip --out-dir results/countdown-objective5 --gpus 0 1
```

Omit `--gpus 0 1` to evaluate both seeds sequentially on one GPU. The evaluator
verifies the experiment metadata and uses one greedy completion per puzzle on
`data/test.jsonl` for each of the six adapters. It uses the same prompt, prefill,
verifier, and token budget across all arms. Use a fresh output directory for a
new pair. The optional `--limit-per-size` flag runs a smoke test.

Progress is labeled by seed and arm. Each seed writes three JSONL files and a
`manifest.json` in its own directory. The evaluator writes `summary.json` after
both seeds complete and their settings and puzzle lists match. To recover an
interrupted run, move the incomplete seed directory out of the output directory
if it exists. Run the same evaluation command with `--seed 42` or `--seed 43`
in place of `--gpus 0 1` for the missing seed. Then merge the two outputs on CPU:

```bash
python -m src.eval.reward_hacking --out-dir results/countdown-objective5 --merge-only
```

The report gives strict solve rate, format compliance, and formatted-but-wrong rate
by puzzle size and for the full set. It compares each arm with its own seed's
control in addition to reporting pooled results. A shared additive score and
a shared correctness-gated score show whether a reward gain depends on bonuses
paid on wrong answers.

## Objective 6 - Reward hacking analysis

The [analysis report](results/countdown-objective6/README.md) links six figures
covering the reward grid, training rollouts, held-out behavior, reward components,
paired puzzle outcomes, and sensitivity to the scoring bonus. The plots use the
saved results and run on CPU.

Regenerate the report and plots from the repository root:

```bash
python -m pip install matplotlib
python -m src.eval.reward_hacking_figures
```

The bonus sweeps rescore saved completions. Training used a +0.1 format bonus
in the additive and gated arms.

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
- Run the three-arm GRPO comparison with Qwen2.5-1.5B-Instruct

## References
- Shao et al. (2024). DeepSeekMath. arXiv:2402.03300 (GRPO)
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948
- Gandhi et al. (2025). Cognitive Behaviors that Enable Self-Improving Reasoners. arXiv:2503.01307
- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. NeurIPS
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
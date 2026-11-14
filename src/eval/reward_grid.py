"""Rescore fixed Countdown completions over a correctness/format reward grid."""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

from src.prompts import PREFILL, build_prompt
from src.reward.verify import get_text, has_format, is_correct


@dataclass(frozen=True)
class RewardConfig:
    correctness: float = 1.0
    format_bonus: float = 0.1
    gate_format_on_correctness: bool = False

    def __post_init__(self):
        if not isfinite(self.correctness) or self.correctness <= 0:
            raise ValueError("Correctness reward must be positive and finite.")
        if not isfinite(self.format_bonus) or self.format_bonus < 0:
            raise ValueError("Format bonus must be nonnegative and finite.")

    def earns_format_bonus(self, correct, formatted):
        return (
            self.format_bonus > 0
            and formatted
            and (correct or not self.gate_format_on_correctness)
        )

    def score(self, correct, formatted):
        return self.correctness * correct + self.format_bonus * self.earns_format_bonus(
            correct, formatted
        )


def reward_grid(correctness_weights=(0.5, 1.0, 2.0), format_bonuses=(0, 0.01, 0.1, 0.5, 1.0)):
    """Include one correctness-only control per correctness weight."""
    return [
        RewardConfig(correctness, bonus, gated)
        for correctness, bonus in product(correctness_weights, format_bonuses)
        for gated in ((False, True) if bonus else (False,))
    ]


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_response(text, numbers, target):
    """Return the two binary decisions, independent of reward weights."""
    return is_correct(text, numbers, target), has_format(text)


def make_reward_callback(config, *, assistant_prefill=PREFILL["grpo"]):
    """Build a TRL reward function for generated suffixes."""
    def countdown_reward(completions, numbers, target, **kwargs):
        return [
            config.score(*score_response(assistant_prefill + get_text(completion), inputs, value))
            for completion, inputs, value in zip(completions, numbers, target, strict=True)
        ]

    return countdown_reward


def score_groups(records, *, assistant_prefill=""):
    """Read one group per record. Legacy evaluation outputs are singleton groups."""
    groups = []
    for row in records:
        numbers = row["numbers"]
        responses = row["responses"] if "responses" in row else [row["output"]]
        if not responses or not isinstance(responses, list):
            raise ValueError("Each record must contain at least one response in a list.")
        groups.append((
            len(numbers),
            [score_response(assistant_prefill + get_text(text), numbers, row["target"])
             for text in responses],
        ))
    if not groups:
        raise ValueError("No completion records found.")
    return groups


def group_advantages(rewards):
    """Group centering and sample standard deviation scaling as in TRL GRPO."""
    average = mean(rewards)
    scale = stdev(rewards) + 1e-4
    return [(reward - average) / scale for reward in rewards]


def summarize(groups, config):
    """Report completion rates and within-prompt diagnostics separately."""
    by_size = defaultdict(list)
    for size, group in groups:
        by_size[size].append(group)

    def metrics(groups_at_size):
        completions = [pair for group in groups_at_size for pair in group]
        rollout_groups = [group for group in groups_at_size if len(group) > 1]
        all_wrong = mixed = format_only = wrong_ties = wrong_outranks = 0
        positive_wrong_advantages = []
        for group in rollout_groups:
            rewards = [config.score(*pair) for pair in group]
            correct_rewards = [reward for (correct, _), reward in zip(group, rewards) if correct]
            wrong_rewards = [reward for (correct, _), reward in zip(group, rewards) if not correct]
            if not correct_rewards:
                all_wrong += 1
                format_only += min(rewards) < max(rewards)
            elif wrong_rewards:
                mixed += 1
                wrong_ties += max(wrong_rewards) == max(correct_rewards)
                wrong_outranks += max(wrong_rewards) > max(correct_rewards)
            positive_wrong_advantages.extend(
                advantage
                for (correct, _), advantage in zip(group, group_advantages(rewards))
                if not correct and advantage > 0
            )

        group_sizes = {len(group) for group in groups_at_size}
        return {
            "groups": len(groups_at_size),
            "completions": len(completions),
            "group_sizes": sorted(group_sizes),
            "solve_rate": 100 * mean(correct for correct, _ in completions),
            "pass_at_k": (100 * mean(any(correct for correct, _ in group) for group in groups_at_size)
                          if len(group_sizes) == 1 else None),
            "format_rate": 100 * mean(formatted for _, formatted in completions),
            "formatted_wrong_rate": 100 * mean(not correct and formatted
                                               for correct, formatted in completions),
            "wrong_bonus_rate": 100 * mean(not correct and config.earns_format_bonus(correct, formatted)
                                           for correct, formatted in completions),
            "mean_reward": mean(config.score(*pair) for pair in completions),
            "all_wrong_rollout_groups": all_wrong,
            "all_wrong_groups_with_format_signal": format_only,
            "all_wrong_format_signal_rate": 100 * format_only / all_wrong if all_wrong else None,
            "mixed_rollout_groups": mixed,
            "wrong_ties_best_correct_rate": 100 * wrong_ties / mixed if mixed else None,
            "wrong_outranks_best_correct_rate": 100 * wrong_outranks / mixed if mixed else None,
            "wrong_positive_advantage_rate": (
                100 * len(positive_wrong_advantages) / sum(map(len, rollout_groups))
                if rollout_groups else None
            ),
            "mean_positive_wrong_advantage": (
                mean(positive_wrong_advantages) if positive_wrong_advantages else None
            ),
        }

    return {
        "all": metrics([group for _, group in groups]),
        **{str(size): metrics(rows) for size, rows in sorted(by_size.items())},
    }


def evaluate_grid(records, configs, *, assistant_prefill=""):
    groups = score_groups(records, assistant_prefill=assistant_prefill)
    return [{"config": asdict(config), "metrics": summarize(groups, config)} for config in configs]


def sample_responses(model, tokenizer, puzzles, *, samples, temperature, max_tokens, batch_size, seed):
    """Sample raw suffixes and store the reconstructed full responses."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(puzzles), batch_size):
            batch = puzzles[start:start + batch_size]
            prompts = [build_prompt(tokenizer, row["numbers"], row["target"], "grpo") for row in batch]
            inputs = tokenizer(
                prompts, padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            output = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=samples,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            suffixes = tokenizer.batch_decode(
                output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            for index, row in enumerate(batch):
                start_index = index * samples
                records.append({
                    key: row[key] for key in ("id", "numbers", "target", "n_numbers")
                } | {
                    "responses": [
                        PREFILL["grpo"] + suffix for suffix in suffixes[start_index:start_index + samples]
                    ],
                })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with one response or a list of responses per puzzle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--format-bonus", nargs="+", type=float, default=[0, 0.01, 0.1, 0.5, 1.0])
    parser.add_argument("--suffixes", action="store_true",
                        help="Input contains only generated suffixes from prompts ending in <think>")
    args = parser.parse_args()
    configs = reward_grid(args.correctness, args.format_bonus)
    results = evaluate_grid(
        load_jsonl(args.input), configs, assistant_prefill=PREFILL["grpo"] if args.suffixes else ""
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "input": str(args.input),
        "input_contains_suffixes": args.suffixes,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(results)} reward configurations. Results: {args.output}")


if __name__ == "__main__":
    main()

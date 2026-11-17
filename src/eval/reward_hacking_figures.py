"""Build the Objective 6 figures from the saved Countdown experiments."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from src.eval.reward_grid import RewardConfig, group_advantages, load_jsonl, reward_grid, score_groups
from src.eval.reward_hacking import ARMS, SEEDS, summarize as summarize_held_out, summarize_comparison
from src.reward.verify import extract_answer

CONTROL = "correctness_only"
TREATMENTS = ("additive", "gated")
LABELS = {CONTROL: "Correctness only", "additive": "Additive", "gated": "Gated"}
COLORS = {CONTROL: "#465870", "additive": "#98442A", "gated": "#23675E"}
SPLITS = ("all", "3", "4")
SPLIT_NAMES = {"all": "All puzzles", "3": "3 numbers", "4": "4 numbers"}
SCOPES = (*map(str, SEEDS), "pooled")
SCOPE_NAMES = {"42": "Seed 42", "43": "Seed 43", "pooled": "Pooled"}
METRICS = (
    ("strict_solve_rate_pct", "Strict solves"),
    ("format_rate_pct", "Format compliance"),
    ("formatted_wrong_rate_pct", "Formatted wrong"),
)
BONUS = ARMS["additive"].format_bonus
WINDOW = 25
STEPS = 250


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, destination, number, name, title, note):
    fig.suptitle(title, x=0.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.017, note, fontsize=9, color="#475569", va="bottom")
    fig.tight_layout(rect=(0.025, 0.06, 0.985, 0.94))
    fig.savefig(destination / f"{number:02d}-{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def pilot_grid(records, destination):
    groups = score_groups(records)
    if {len(group) for _, group in groups} != {4}:
        raise ValueError("Objective 6 requires the saved four-response pilot groups.")

    configs = reward_grid()
    weights = sorted({config.correctness for config in configs})
    bonuses = sorted({config.format_bonus for config in configs})
    diagnostics = {}
    grid_rows = []
    advantages = {}
    for size, gated in ((size, gated) for size in (3, 4) for gated in (False, True)):
        selected = [group for n, group in groups if n == size]
        all_wrong = [group for group in selected if not any(correct for correct, _ in group)]
        mixed = [group for group in selected if 0 < sum(correct for correct, _ in group) < len(group)]
        grids = np.empty((4, len(weights), len(bonuses)))
        for i, weight in enumerate(weights):
            for j, bonus in enumerate(bonuses):
                config = RewardConfig(weight, bonus, gated)
                rewards = [[config.score(*outcome) for outcome in group] for group in all_wrong]
                positive = [a for scores in rewards for a in group_advantages(scores) if a > 0]
                margins = [
                    max(config.score(*outcome) for outcome in group if not outcome[0])
                    - max(config.score(*outcome) for outcome in group if outcome[0])
                    for group in mixed
                ]
                grids[:, i, j] = (
                    100 * mean(max(scores) > min(scores) for scores in rewards) if rewards else np.nan,
                    mean(positive) if positive else np.nan,
                    100 * mean(gap == 0 for gap in margins) if margins else np.nan,
                    100 * mean(gap > 0 for gap in margins) if margins else np.nan,
                )

                grid_rows.append([
                    size, "gated" if gated else "additive", weight, bonus, len(all_wrong), len(mixed),
                    *(None if np.isnan(value) else value for value in grids[:, i, j]),
                ])
        weight_index = weights.index(1.0)
        diagnostics[size, gated] = {
            "all_wrong": len(all_wrong),
            "signal_groups": round(len(all_wrong) * grids[0, weight_index, bonuses.index(BONUS)] / 100),
            "low_advantage": grids[1, weight_index, bonuses.index(0.01)],
            "tested_advantage": grids[1, weight_index, bonuses.index(BONUS)],
        }
        if not gated:
            advantages[size] = grids[1, weight_index, 1:]

    with (destination / "pilot-reward-grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "puzzle_size", "scorer", "correctness_reward", "format_bonus", "all_wrong_groups",
            "mixed_groups", "all_wrong_groups_with_signal_pct", "mean_positive_advantage_in_all_wrong_groups",
            "mixed_groups_wrong_ties_best_correct_pct", "mixed_groups_wrong_outranks_best_correct_pct",
        ])
        writer.writerows(grid_rows)

    fig, (signal, advantage) = plt.subplots(1, 2, figsize=(11.8, 5.1))
    signal.set_title("How often did format create a reward signal?", loc="left", fontsize=12)
    for row, (size, color) in enumerate(((3, COLORS["gated"]), (4, COLORS["additive"]))):
        pilot = diagnostics[size, False]
        count, total = pilot["signal_groups"], pilot["all_wrong"]
        rate = 100 * count / total
        signal.barh(row, rate, color=color, height=0.45)
        signal.text(rate + 2, row, f"{count} of {total}", va="center", fontsize=11)
    signal.set_yticks([0, 1], ["3 numbers", "4 numbers"])
    signal.invert_yaxis()
    signal.set_xlim(0, 112)
    signal.set_xticks(range(0, 101, 25))
    signal.set_xlabel("All-wrong groups with signal (%), additive +0.1")

    advantage.set_title("How much did a larger bonus help?", loc="left", fontsize=12)
    for size, color in ((3, COLORS["gated"]), (4, COLORS["additive"])):
        values = advantages[size]
        advantage.plot(range(len(values)), values, color=color, marker="o", label=f"{size} numbers")
        for j in range(2):
            advantage.annotate(f"{values[j]:.3f}", (j, values[j]), xytext=(0, 8 if size == 4 else -16),
                               textcoords="offset points", ha="center", color=color)
    advantage.set_ylim(0.60, 0.67)
    advantage.set_xticks(range(len(bonuses) - 1), [f"+{bonus:g}" for bonus in bonuses[1:]])
    advantage.set_xlabel("Additive format bonus")
    advantage.set_ylabel("Mean positive normalized advantage")
    advantage.legend(loc="lower right", frameon=False)
    signal.grid(axis="x", alpha=0.18)
    advantage.grid(axis="y", alpha=0.18)
    save(fig, destination, 1, "pilot-reward-grid", "Format alone can reward wrong answers",
         "Four sampled responses per puzzle. All responses shown here are wrong. Correctness-only and gated gave these groups zero signal.")
    return diagnostics


def training_curves(root, destination):
    fig, axes = plt.subplots(len(SEEDS), 4, figsize=(15, 7), sharex=True, sharey="col")
    periods = {}
    for i, seed in enumerate(SEEDS):
        puzzle_order = None
        for arm in ARMS:
            path = root / f"run-{seed}" / f"seed-{seed}" / arm / "rollouts.jsonl"
            rows = load_jsonl(path)
            if [row["step"] for row in rows] != list(range(1, STEPS + 1)):
                raise ValueError(f"Expected {STEPS} consecutive training steps in {path}.")
            ids = [row["puzzle_id"] for row in rows]
            if puzzle_order is not None and ids != puzzle_order:
                raise ValueError(f"Training puzzle order differs across arms for seed {seed}.")
            puzzle_order = ids

            values = []
            for start in range(0, STEPS, WINDOW):
                batch = [row["by_input_count"]["all"] for row in rows[start:start + WINDOW]]
                n, correct, formatted, wrong = (
                    sum(row[key] for row in batch) for key in ("completions", "correct", "formatted", "formatted_wrong")
                )
                if n != 4 * WINDOW:
                    raise ValueError(f"Expected four rollouts per update in {path}.")
                values.append((100 * correct / n, 100 * formatted / n, 100 * wrong / n,
                               (correct + BONUS * formatted) / n))
            periods[seed, arm] = (values[0], values[-1])
            for col, ax in enumerate(axes[i]):
                ax.plot(range(WINDOW, STEPS + 1, WINDOW), np.array(values)[:, col],
                        marker="o", markersize=3, lw=1.7, label=LABELS[arm], color=COLORS[arm])
        axes[i, 0].set_ylabel(f"Seed {seed} · rate (%)")
        for ax in axes[i, 1:3]:
            ax.set_ylim(-2, 102)
        for ax in axes[i]:
            ax.grid(axis="y", alpha=0.25)
    for ax, title in zip(axes[0], (*[label for _, label in METRICS], "Common additive reward"), strict=True):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Training step at window end")
    axes[0, -1].legend(fontsize=8, loc="best")
    save(fig, destination, 2, "training-behavior", "Behavior over the 250 GRPO updates",
         f"Non-overlapping {WINDOW}-step windows, 100 sampled completions per point. Reward uses +1 correctness and +{BONUS:g} format for every arm.")
    return periods


def held_out_effects(report, destination):
    fig, axes = plt.subplots(len(SPLITS), len(METRICS), figsize=(13, 9), sharex="col")
    offsets = {"additive": 0.12, "gated": -0.12}
    for row, split in enumerate(SPLITS):
        for col, (key, label) in enumerate(METRICS):
            ax = axes[row, col]
            ax.axhspan(1.69, 2.31, color="#eff3f7", zorder=-1)
            ax.axvline(0, color="#64748b", lw=1)
            for i, scope in enumerate(SCOPES):
                deltas = report[scope]["deltas_vs_correctness_only"]
                for arm in TREATMENTS:
                    ax.scatter(deltas[arm][split][key], i + offsets[arm], color=COLORS[arm],
                               marker="D" if scope == "pooled" else "o", s=38, zorder=3)
            ax.set_yticks(range(len(SCOPES)), [SCOPE_NAMES[s] for s in SCOPES] if col == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.22)
            if row == 0:
                ax.set_title(label)
            if row == len(SPLITS) - 1:
                ax.set_xlabel("Change vs. control (pp)")
        axes[row, 0].set_ylabel(SPLIT_NAMES[split], fontweight="bold")
    axes[0, -1].legend(handles=[Line2D([], [], color=COLORS[arm], marker="o", ls="", label=LABELS[arm])
                                for arm in TREATMENTS], loc="best", fontsize=8)
    save(fig, destination, 3, "held-out-effects", "Held-out behavior relative to correctness only",
         "Pooled markers combine both training seeds. Each seed was evaluated on the same 1,000 held-out puzzles.")


def decompose(metrics):
    solve = metrics["strict_solve_rate_pct"] / 100
    formatted = metrics["format_rate_pct"] / 100
    wrong = metrics["formatted_wrong_rate_pct"] / 100
    return np.array([solve, BONUS * (formatted - wrong), BONUS * wrong])


def reward_components(report, destination):
    keys = [(scope, split) for scope in SCOPES for split in SPLITS]
    effects = np.array([
        decompose(report[scope]["metrics"]["additive"][split]) - decompose(report[scope]["metrics"][CONTROL][split])
        for scope, split in keys
    ])
    fig, (ax, totals) = plt.subplots(1, 2, sharey=True, figsize=(13, 7))
    ax.set_title("Change by reward component")
    totals.set_title("Total change under each scorer")
    y = np.arange(len(keys))
    for col, (label, color) in enumerate(zip(
        ("Correctness", "Format on correct", "Format on wrong"), ("#465870", "#23675E", "#98442A"), strict=True
    )):
        ax.barh(y + (col - 1) * 0.22, effects[:, col], height=0.2, color=color, label=label)
    highlighted = keys.index(("43", "4"))
    for gated, offset in ((False, -0.12), (True, 0.12)):
        name = "gated" if gated else "additive"
        value = effects[:, :2].sum(axis=1) if gated else effects.sum(axis=1)
        totals.barh(y + offset, value, height=0.22, color=COLORS[name], label=LABELS[name])
        point = value[highlighted]
        totals.annotate(f"{point:+.4f}", (point, highlighted + offset),
                        xytext=(5 if point >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if point >= 0 else "right", va="center",
                        color=COLORS[name], fontsize=10, fontweight="bold")
    for panel in (ax, totals):
        panel.axhspan(highlighted - 0.48, highlighted + 0.48, color="#f5eadf", zorder=-1)
        panel.axvline(0, color="#64748b", lw=1)
        panel.grid(axis="x", alpha=0.22)
        panel.set_xlabel("Change in mean reward")
        panel.legend(fontsize=8, loc="best")
    ax.set_yticks(y, [f"{SCOPE_NAMES[scope]} · {SPLIT_NAMES[split]}" for scope, split in keys])
    ax.invert_yaxis()
    save(fig, destination, 4, "reward-components", "Where the additive arm gained or lost reward",
         "Each change compares the same additive-trained and correctness-only completions. Gated scoring drops the format payment on wrong answers.")


def outcome(row):
    if row["correct"]:
        return 0
    return 1 if row["format_ok"] else 2


def paired_outcomes(root, report, destination):
    labels = ("correct", "formatted_wrong", "wrong_failed_format")
    transitions = {}
    example = None
    pooled_rows = {arm: [] for arm in ARMS}
    reference_keys = None
    for seed in SEEDS:
        data = {arm: load_jsonl(root / f"seed-{seed}" / f"{arm}.jsonl") for arm in ARMS}
        keys = [(row["id"], row["numbers"], row["target"]) for row in data[CONTROL]]
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Held-out puzzle order differs between seeds.")
        reference_keys = keys
        for arm, rows in data.items():
            pooled_rows[arm].extend(rows)
            if ([(row["id"], row["numbers"], row["target"]) for row in rows] != keys
                    or len({key[0] for key in keys}) != len(keys)
                    or summarize_held_out(rows) != report[str(seed)]["metrics"][arm]):
                raise ValueError(f"Held-out raw completions differ from summary or puzzle order for seed {seed}, {arm}.")
        for arm in TREATMENTS:
            pairs = list(zip(data[CONTROL], data[arm], strict=True))
            for size in (3, 4):
                selected = [(before, after) for before, after in pairs if len(before["numbers"]) == size]
                counts = Counter((outcome(before), outcome(after)) for before, after in selected)
                transitions[seed, arm, size] = counts
                if (seed, arm, size) == (43, "additive", 4):
                    example = next((pair for pair in selected if pair[0]["id"] == 968
                                    and (outcome(pair[0]), outcome(pair[1])) == (0, 1)), None)
    pooled = {arm: summarize_held_out(rows) for arm, rows in pooled_rows.items()}
    if pooled != report["pooled"]["metrics"]:
        raise ValueError("Pooled held-out metrics differ from the raw completions.")
    with (destination / "puzzle-transitions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "arm", "puzzle_size", "control_outcome", "treatment_outcome", "count"])
        for (seed, arm, size), counts in transitions.items():
            for i, before in enumerate(labels):
                for j, after in enumerate(labels):
                    writer.writerow([seed, arm, size, before, after, counts[i, j]])

    counts = transitions[43, "additive", 4]
    changes = (
        ("Wrong, failed format → Formatted wrong", counts[2, 1], COLORS["additive"]),
        ("Correct → Formatted wrong", counts[0, 1], COLORS["additive"]),
        ("Formatted wrong → Correct", counts[1, 0], COLORS[CONTROL]),
        ("Formatted wrong → Wrong, failed format", counts[1, 2], COLORS[CONTROL]),
    )
    names, values, colors = zip(*changes, strict=True)
    fig, ax = plt.subplots(figsize=(10.7, 4.4))
    bars = ax.barh(names, values, color=colors, height=0.52)
    ax.bar_label(bars, padding=5, fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Puzzles")
    ax.grid(axis="x", alpha=0.18)
    save(fig, destination, 5, "puzzle-transitions", "Which answers changed on four-number puzzles?",
         "Seed 43, correctness-only → additive. Shows changes in outcome across the same 500 puzzles.")
    return transitions, example


def reward_sensitivity(report, destination):
    fig, axes = plt.subplots(1, len(SCOPES), sharey=True, figsize=(14, 5))
    points = np.linspace(0, 1, 201)
    crossings = {}
    colors = {"all": "#465870", "3": "#23675E", "4": "#98442A"}
    for ax, scope in zip(axes, SCOPES, strict=True):
        ax.axhline(0, color="#64748b", lw=1)
        ax.axvline(BONUS, color="#64748b", ls=":", lw=1.4)
        ax.text(BONUS + 0.02, 0.96, f"+{BONUS:g} used in training", transform=ax.get_xaxis_transform(),
                color="#475569", va="top", fontsize=8)
        for split in SPLITS:
            d = report[scope]["deltas_vs_correctness_only"]["additive"][split]
            intercept, slope = d["strict_solve_rate_pct"] / 100, d["format_rate_pct"] / 100
            ax.plot(points, intercept + points * slope, color=colors[split], label=SPLIT_NAMES[split])
            if slope and 0 <= (crossing := -intercept / slope) <= 1:
                crossings[scope, split] = crossing
                ax.scatter(crossing, 0, color=colors[split], s=27, zorder=3)
                if (scope, split) in {("43", "4"), ("pooled", "all")}:
                    ax.annotate(f"{crossing:.3f}", (crossing, 0), xytext=(5, 13),
                                textcoords="offset points", color=colors[split], fontsize=9,
                                arrowprops={"arrowstyle": "-", "color": colors[split], "lw": 0.8})
        ax.set_title(SCOPE_NAMES[scope])
        ax.set_xlabel("Format bonus used to rescore")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Change in mean additive reward")
    axes[-1].legend(fontsize=8)
    save(fig, destination, 6, "held-out-bonus-sensitivity", "How bonus size changes the held-out reward comparison",
         f"Rescores fixed completions with correctness = +1. Dotted line marks the +{BONUS:g} training bonus. Dots mark break-even points within the sweep.")
    return crossings


def write_report(destination, report, pilot, periods, transitions, example, crossings):
    pooled = report["pooled"]["metrics"]
    lines = [
        "# Objective 6 · Reward hacking analysis", "",
        "The analysis uses the saved pilot, 250 training updates per arm and seed, and one greedy held-out response per adapter and puzzle.",
        "The reward sweeps rescore saved responses and measure sensitivity to the scoring rule. Training used one positive format bonus.", "",
        "## 1. Reward weights and GRPO advantage", "",
        "[Pilot figure](01-pilot-reward-grid.png) · [Full reward grid](pilot-reward-grid.csv)", "",
        "The pilot uses the same sampled groups at every weight. A mixed group contains at least one correct and one wrong response.",
        "The CSV includes rankings comparing the highest-scoring wrong and correct responses in each mixed group.", "",
        "| Puzzle size | All-wrong groups with additive signal | Mean positive advantage at +0.01 | At +0.1 |",
        "|---|---:|---:|---:|",
    ]
    for size in (3, 4):
        p = pilot[size, False]
        lines.append(f"| {size} numbers | {p['signal_groups']}/{p['all_wrong']} | "
                     f"{p['low_advantage']:.3f} | {p['tested_advantage']:.3f} |")
    lines.extend([
        "", "A larger positive bonus barely changed the normalized advantage in these all-wrong groups. "
        "The correctness weight has no effect on this slice because every answer is wrong. The gated rule gives all wrong answers zero reward.",
        "The reward grid shows no observed ties or reversals in the sampled mixed groups. "
        "In principle, an additive bonus equal to or greater than the correctness weight can let a formatted wrong answer tie or outrank an unformatted correct one.", "",
        "## 2. Behavior during training", "",
        "[Training curves](02-training-behavior.png)", "",
        "| Seed | Arm | Strict solves, first → last window | Formatted wrong, first → last | Common additive reward, first → last |",
        "|---|---|---:|---:|---:|",
    ])
    for (seed, arm), (first, last) in periods.items():
        lines.append(f"| {seed} | {LABELS[arm]} | {first[0]:.0f}% → {last[0]:.0f}% | "
                     f"{first[2]:.0f}% → {last[2]:.0f}% | {first[3]:.3f} → {last[3]:.3f} |")
    lines.extend([
        "", "In both seeds, the additive arm produced more formatted wrong answers than the control in the last window. "
        "It also solved fewer and scored lower under the common additive rule in that window. "
        "Training windows summarize sampled rollouts on different puzzles, so the curves describe behavior over the training sequence.", "",
        "## 3. Held-out behavior", "", "[Held-out changes](03-held-out-effects.png)", "",
        "| Arm | Strict solves | Format compliance | Formatted wrong |", "|---|---:|---:|---:|",
    ])
    for arm in ARMS:
        rates = " | ".join(f"{pooled[arm]['all'][key]:.2f}%" for key, _ in METRICS)
        lines.append(f"| {LABELS[arm]} | {rates} |")
    lines.extend([
        "", "Correctness-only had the highest pooled solve rate. The additive arm improved format compliance and produced more formatted wrong responses overall.",
        "The seed-level markers show that the size and direction of the changes varied by puzzle count and seed.", "",
        "## 4. Reward decomposition", "", "[Reward components](04-reward-components.png)", "",
    ])
    for scope in ("pooled", "43"):
        split = "all" if scope == "pooled" else "4"
        d = report[scope]["deltas_vs_correctness_only"]["additive"][split]
        lines.append(f"- **{SCOPE_NAMES[scope]}, {SPLIT_NAMES[split].lower()}.** Mean reward change was "
                     f"{d['mean_additive_reward']:+.5f} under additive scoring and {d['mean_gated_reward']:+.5f} under gated scoring.")
    lines.extend([
        "", "Only seed 43's four-number subgroup met the reward-hacking criterion. The additive arm solved fewer puzzles and earned more under additive scoring because wrong answers received the format bonus.", "",
        "## 5. Paired puzzle outcomes", "", "[Puzzle outcome changes](05-puzzle-transitions.png) · [All transition counts](puzzle-transitions.csv)", "",
    ])
    counts = transitions[43, "additive", 4]
    lines.append(f"In that subgroup, {counts[2, 1]} already-wrong answers started passing the format check, "
                 f"{counts[0, 1]} correct answers became formatted wrong, and {counts[1, 0]} formatted wrong answer became correct.")
    if example is not None:
        before, after = example
        lines.extend(["", f"Puzzle {before['id']} supplied {before['numbers']} with target {before['target']}. "
                      f"The control answered `{extract_answer(before['output'])}`. The additive arm answered "
                      f"`{extract_answer(after['output'])}` and earned the format bonus despite an invalid solution."])
    lines.extend([
        "", "## 6. Sensitivity to bonus size", "", "[Held-out bonus sensitivity](06-held-out-bonus-sensitivity.png)", "",
        "The mean reward change under additive rescoring equals the change in solve rate plus the bonus times the change in format compliance. Both rates are expressed as proportions.",
        f"The observed outputs break even at about **{crossings['43', '4']:.3f}** for seed 43's four-number puzzles "
        f"and **{crossings['pooled', 'all']:.3f}** when pooled.", "",
        "These thresholds apply to the saved outputs. Testing whether other bonus sizes produce different learned behavior requires paired training at those weights. "
        "Two training seeds establish the observed subgroup result and leave its reproducibility open.", "",
    ])
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/countdown-objective6"))
    args = parser.parse_args()

    pilot = load_jsonl(args.results_dir / "reward_signal_validation" / "responses.jsonl")
    training = args.results_dir / "countdown-objective4-parallel"
    held_out = args.results_dir / "countdown-objective5"
    summary = read_json(held_out / "summary.json")
    if summary["smoke_test"] or summary["puzzles_per_seed"] != 1000:
        raise ValueError("Objective 6 requires the full Objective 5 evaluation.")
    report = {**summary["seeds"], "pooled": summary["pooled"]}
    for scope in SCOPES:
        result = report[scope]
        if summarize_comparison(result["metrics"]) != result["deltas_vs_correctness_only"]:
            raise ValueError(f"Held-out deltas differ from the metrics for {scope}.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#172332"})
    diagnostics = pilot_grid(pilot, args.out_dir)
    periods = training_curves(training, args.out_dir)
    held_out_effects(report, args.out_dir)
    reward_components(report, args.out_dir)
    transitions, example = paired_outcomes(held_out, report, args.out_dir)
    crossings = reward_sensitivity(report, args.out_dir)
    write_report(args.out_dir, report, diagnostics, periods, transitions, example, crossings)
    print(f"Saved six Objective 6 figures and interpretation to {args.out_dir}")


if __name__ == "__main__":
    main()


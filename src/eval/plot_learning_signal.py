"""
plot_learning_signal.py - chart of the GRPO learning signal (pass@4 experiment)
==============================================================================

WHAT
    Reads the two *_cot_k4_summary.json files (0.5B and 1.5B, 4 sampled answers
    per puzzle) and draws, for each level, the % of puzzles whose 4 answers are
    MIXED (some right, some wrong). Only those puzzles teach GRPO to solve.

RUN
    python -m src.eval.plot_learning_signal
"""
import json, os
import matplotlib.pyplot as plt

FOLDER = "results/Learning_signal_analysis_pass@4"      # where you put the k4 files
METRIC = "useful_groups_correctness"             # % of puzzles GRPO can learn from
MODELS = {"0.5B": "#3D63C4", "1.5B": "#1F7A57"}  # blue / green, colorblind-checked
LEVELS = [("2_numbers", "2 numbers"), ("3_numbers", "3 numbers"), ("all", "All puzzles")]

# 1) read the two summaries
vals = {}
for m in MODELS:
    path = f"{FOLDER}/Qwen2.5-{m}-Instruct_cot_k4_summary.json"
    if not os.path.exists(path):
        print("missing:", path); continue
    s = json.load(open(path))
    vals[m] = [s[lv][METRIC] for lv, _ in LEVELS]

# 2) table
print(f"{'model':<8}" + "".join(f"{name:>14}" for _, name in LEVELS))
for m, v in vals.items():
    print(f"{m:<8}" + "".join(f"{x:>13.1f}%" for x in v))

# 3) grouped bar chart
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
fig.patch.set_facecolor("#F7F6F2"); ax.set_facecolor("#F7F6F2")
w = 0.36
for j, (m, v) in enumerate(vals.items()):
    xs = [i + (j - 0.5) * w for i in range(len(LEVELS))]
    bars = ax.bar(xs, v, width=w, color=MODELS[m], label=f"Qwen2.5-{m}",
                  edgecolor="#F7F6F2", linewidth=2)
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x + 0.8, f"{x:.1f}%",
                ha="center", va="bottom", fontsize=11, color="#1E3B34")
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels([n for _, n in LEVELS], fontsize=12, color="#1E3B34")
ax.set_ylabel("Puzzles with mixed rewards (%)", fontsize=12, color="#4A5568")
ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.2)
ax.set_title("Learning signal for GRPO: groups of 4 sampled answers that are partly correct (CoT, test set)",
             fontsize=12, color="#1E3B34", loc="left", pad=14)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#C9C4BA"); ax.spines["bottom"].set_color("#C9C4BA")
ax.tick_params(colors="#4A5568")
ax.yaxis.grid(True, color="#E2DED5", linewidth=1); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=12, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FOLDER}/learning_signal_k4.png", facecolor=fig.get_facecolor())
plt.show()
print(f"saved -> {FOLDER}/learning_signal_k4.png")
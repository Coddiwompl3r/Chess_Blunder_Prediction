"""
exploratory_analysis.py
Phase 5: Visualize patterns in the dataset.

Each plot tests one of your research hypotheses.
Output saved to results/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

data_path    = os.path.join(project_dir, "data",    "chess_blunder_dataset.csv")
results_path = os.path.join(project_dir, "results")
os.makedirs(results_path, exist_ok=True)

df = pd.read_csv(data_path)

print(f"Dataset loaded: {len(df)} moves from {df['game_id'].nunique()} games")
print(f"Blunder rate  : {df['is_blunder'].mean()*100:.2f}%")
print(f"Mistake rate  : {df['is_mistake'].mean()*100:.2f}%")
print(f"\nQuality breakdown:")
print(df["quality"].value_counts())
print()

# ─────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "monospace",
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#1a1a1a",
    "axes.edgecolor":    "#444444",
    "text.color":        "white",
    "axes.labelcolor":   "white",
    "xtick.color":       "white",
    "ytick.color":       "white",
    "grid.color":        "#333333",
})

GOLD  = "#e8c547"
BLUE  = "#4a9eff"
RED   = "#ff4a4a"
GRAY  = "#888888"


# ─────────────────────────────────────────────
# PLOT 1: Move Quality Distribution
# What proportion of moves are good vs errors?
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

quality_counts = df["quality"].value_counts()
order  = ["good", "inaccuracy", "mistake", "blunder", "unknown"]
order  = [q for q in order if q in quality_counts.index]
counts = [quality_counts[q] for q in order]
colors = [BLUE, GOLD, "#ff8c00", RED, GRAY][:len(order)]

bars = ax.bar(order, counts, color=colors, alpha=0.85, edgecolor="#333", width=0.5)

for bar, count in zip(bars, counts):
    pct = count / len(df) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{count}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=10, color="white"
    )

ax.set_xlabel("Move Quality", labelpad=10)
ax.set_ylabel("Number of Moves")
ax.set_title(
    "Move Quality Distribution\n"
    "How often do elite players make errors?",
    fontsize=13, pad=15
)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(counts) * 1.2)

plt.tight_layout()
out = os.path.join(results_path, "01_move_quality_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 01_move_quality_distribution.png")


# ─────────────────────────────────────────────
# PLOT 2: Error Rate by Game Phase
# Hypothesis: Different phases produce different error rates
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

phase_order = ["opening", "middlegame", "endgame"]
phase_colors = [BLUE, GOLD, RED]

for ax, metric, title in zip(
    axes,
    ["is_blunder", "is_mistake"],
    ["Blunder Rate by Game Phase", "Mistake Rate by Game Phase"]
):
    rates  = []
    counts = []
    for phase in phase_order:
        subset = df[df["game_phase"] == phase]
        rates.append(subset[metric].mean() * 100)
        counts.append(len(subset))

    bars = ax.bar(phase_order, rates, color=phase_colors,
                   alpha=0.85, edgecolor="#333", width=0.45)

    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.2f}%\n(n={count})",
            ha="center", va="bottom", fontsize=9, color="white"
        )

    ax.set_xlabel("Game Phase", labelpad=8)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(rates) * 1.5 + 0.1)

plt.suptitle(
    "When Are Players Most Error-Prone?",
    fontsize=14, y=1.02
)
plt.tight_layout()
out = os.path.join(results_path, "02_error_rate_by_phase.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 02_error_rate_by_phase.png")


# ─────────────────────────────────────────────
# PLOT 3: Branching Factor vs Error Rate
# Hypothesis: More legal moves = more confusion = more errors
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

df["legal_bin"] = pd.cut(
    df["n_legal_moves"],
    bins=[0, 15, 25, 35, 45, 120],
    labels=["0–15", "15–25", "25–35", "35–45", "45+"]
)

bin_data = df.groupby("legal_bin", observed=True).agg(
    mistake_rate=("is_mistake", "mean"),
    count=("is_mistake", "count")
).reset_index()

bin_data["mistake_pct"] = bin_data["mistake_rate"] * 100

bars = ax.bar(
    range(len(bin_data)),
    bin_data["mistake_pct"],
    color=GOLD, alpha=0.85, edgecolor="#333"
)

for i, (bar, row) in enumerate(zip(bars, bin_data.itertuples())):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{row.mistake_pct:.1f}%\n(n={row.count})",
        ha="center", va="bottom", fontsize=9, color="white"
    )

ax.set_xticks(range(len(bin_data)))
ax.set_xticklabels(bin_data["legal_bin"].astype(str))
ax.set_xlabel("Number of Legal Moves (Branching Factor)", labelpad=10)
ax.set_ylabel("Mistake Rate (%)")
ax.set_title(
    "Complexity Hypothesis: Does More Choice Cause More Errors?\n"
    "Mistake rate vs. number of legal moves",
    fontsize=12, pad=15
)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(results_path, "03_complexity_vs_error_rate.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 03_complexity_vs_error_rate.png")


# ─────────────────────────────────────────────
# PLOT 4: Evaluation Volatility vs Error Rate
# Hypothesis: Unstable positions produce more mistakes
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

df["vol_bin"] = pd.cut(
    df["eval_volatility"],
    bins=[0, 20, 50, 100, 200, 2000],
    labels=["0–20", "20–50", "50–100", "100–200", "200+"]
)

vol_data = df.groupby("vol_bin", observed=True).agg(
    mistake_rate=("is_mistake", "mean"),
    count=("is_mistake", "count")
).reset_index()

vol_data["mistake_pct"] = vol_data["mistake_rate"] * 100

bars = ax.bar(
    range(len(vol_data)),
    vol_data["mistake_pct"],
    color=RED, alpha=0.8, edgecolor="#333"
)

for bar, row in zip(bars, vol_data.itertuples()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{row.mistake_pct:.1f}%\n(n={row.count})",
        ha="center", va="bottom", fontsize=9, color="white"
    )

ax.set_xticks(range(len(vol_data)))
ax.set_xticklabels(vol_data["vol_bin"].astype(str))
ax.set_xlabel("Evaluation Volatility (std dev of last 5 evals)", labelpad=10)
ax.set_ylabel("Mistake Rate (%)")
ax.set_title(
    "Volatility Hypothesis: Do Unstable Positions Cause More Errors?\n"
    "Mistake rate vs. position evaluation volatility",
    fontsize=12, pad=15
)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(results_path, "04_volatility_vs_error_rate.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 04_volatility_vs_error_rate.png")


# ─────────────────────────────────────────────
# PLOT 5: Error Rate Per Game (Who Played Best?)
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

game_data = df.groupby(["game_id", "white_name", "black_name"]).agg(
    mistake_rate=("is_mistake", "mean"),
    total_moves=("is_mistake", "count"),
    blunders=("is_blunder", "sum")
).reset_index()

game_data["mistake_pct"] = game_data["mistake_rate"] * 100
game_data["label"] = game_data.apply(
    lambda r: f"{r['white_name'].split()[-1]} vs\n{r['black_name'].split()[-1]}", axis=1
)

bars = ax.bar(
    range(len(game_data)),
    game_data["mistake_pct"],
    color=[BLUE, GOLD, RED, "#a855f7", "#22c55e"],
    alpha=0.85, edgecolor="#333"
)

for bar, row in zip(bars, game_data.itertuples()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{row.mistake_pct:.1f}%\n({int(row.blunders)} blunders)",
        ha="center", va="bottom", fontsize=8, color="white"
    )

ax.set_xticks(range(len(game_data)))
ax.set_xticklabels(game_data["label"], fontsize=9)
ax.set_xlabel("Game", labelpad=10)
ax.set_ylabel("Mistake Rate (%)")
ax.set_title(
    "Error Rate Per Game\n"
    "Which games contained the most mistakes?",
    fontsize=12, pad=15
)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(results_path, "05_error_rate_per_game.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 05_error_rate_per_game.png")


# ─────────────────────────────────────────────
# SUMMARY STATISTICS (for your paper)
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("SUMMARY STATISTICS FOR RESEARCH PAPER")
print("="*50)

numeric_cols = [
    "n_legal_moves", "n_pieces", "eval_volatility",
    "eval_before_cp", "move_number"
]

for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  mean   : {df[col].mean():.2f}")
    print(f"  std    : {df[col].std():.2f}")
    print(f"  min    : {df[col].min():.2f}")
    print(f"  max    : {df[col].max():.2f}")

print(f"\nCorrelation with is_mistake:")
for col in numeric_cols:
    corr = df[col].corr(df["is_mistake"])
    print(f"  {col:<22}: {corr:+.4f}")

print(f"\nAll plots saved to: {results_path}")



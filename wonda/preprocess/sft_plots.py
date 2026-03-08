"""Plot utilities for SFT dataset stats (token length, grade, speedup)."""

import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_sft_dataset_stats(
    final_dataset: Dataset,
    tokenizer: AutoTokenizer,
    kept_grades: list,
    invariant_token_lengths: list,
    min_grade: int = 1,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    """Single figure with 4 subplots: token length, grade distribution, invariant token length, speedup."""
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not plot_path:
        return
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    token_lengths = [len(tokenizer.encode(s["text"])) for s in final_dataset]
    grade_counts = Counter(kept_grades)
    sorted_grades = sorted(grade_counts.keys())
    grade_labels = [str(g) for g in sorted_grades]
    grade_heights = [grade_counts[g] for g in sorted_grades]
    speedups = [s.get("speedup", 0.0) for s in final_dataset]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"SFT dataset stats — {name}", fontsize=13, fontweight="medium", y=1.02)
    color_main = "#2e86ab"
    color_mean = "#e94f37"
    grid_alpha = 0.25

    # Subplot 1: full-text token length
    ax1 = axes[0, 0]
    ax1.hist(token_lengths, bins=min(60, max(20, len(token_lengths) // 10)), color=color_main, alpha=0.85, edgecolor="white", linewidth=0.5)
    if token_lengths:
        mean_t = sum(token_lengths) / len(token_lengths)
        ax1.axvline(mean_t, color=color_mean, linestyle="--", linewidth=1.5, label=f"Mean: {mean_t:.0f}")
    ax1.set_xlabel("Token length (full text)")
    ax1.set_ylabel("Count")
    ax1.set_title("Full-text tokens")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=grid_alpha)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot 2: grade distribution
    ax2 = axes[0, 1]
    bars = ax2.bar(grade_labels, grade_heights, color=color_main, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, h in zip(bars, grade_heights):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(grade_heights) * 0.02), str(h), ha="center", va="bottom", fontsize=9, fontweight="medium")
    ax2.set_xlabel("Quality grade")
    ax2.set_ylabel("Count")
    ax2.set_title("Grade distribution")
    ax2.grid(True, axis="y", alpha=grid_alpha)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Subplot 3: invariant-only token length
    ax3 = axes[1, 0]
    ax3.hist(invariant_token_lengths, bins=min(50, max(15, len(invariant_token_lengths) // 5)), color=color_main, alpha=0.85, edgecolor="white", linewidth=0.5)
    if invariant_token_lengths:
        mean_i = sum(invariant_token_lengths) / len(invariant_token_lengths)
        ax3.axvline(mean_i, color=color_mean, linestyle="--", linewidth=1.5, label=f"Mean: {mean_i:.0f}")
    ax3.set_xlabel("Token length (invariant only)")
    ax3.set_ylabel("Count")
    ax3.set_title("Invariant tokens")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=grid_alpha)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Subplot 4: speedup distribution (grade >= min_grade) + per-grade stats
    ax4 = axes[1, 1]
    speedups_ge = [s for g, s in zip(kept_grades, speedups) if g >= min_grade]
    if speedups_ge:
        n_bins = min(40, max(15, len(speedups_ge) // 5))
        ax4.hist(speedups_ge, bins=n_bins, color=color_main, alpha=0.85, edgecolor="white", linewidth=0.5)
        mean_s = sum(speedups_ge) / len(speedups_ge)
        sorted_s = sorted(speedups_ge)
        mid = len(sorted_s) // 2
        median_s = (sorted_s[mid - 1] + sorted_s[mid]) / 2 if len(sorted_s) % 2 == 0 and sorted_s else (sorted_s[mid] if sorted_s else 0)
        ax4.axvline(mean_s, color=color_mean, linestyle="--", linewidth=1.5, label=f"Mean: {mean_s:.2f}x")
        ax4.axvline(median_s, color="#333", linestyle=":", linewidth=1.5, label=f"Median: {median_s:.2f}x")
        ax4.set_xlabel("Speedup")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Speedup (grade ≥ {min_grade})")
        ax4.legend(loc="upper right", fontsize=8)
        lines = [f"Together (≥g{min_grade}): n={len(speedups_ge)}  min={min(speedups_ge):.2f}  max={max(speedups_ge):.2f}  avg={mean_s:.2f}  median={median_s:.2f}"]
        for g in sorted_grades:
            if g < min_grade:
                continue
            sg = [s for gr, s in zip(kept_grades, speedups) if gr == g]
            if not sg:
                continue
            ss = sorted(sg)
            m = len(ss) // 2
            med = (ss[m - 1] + ss[m]) / 2 if len(ss) % 2 == 0 else ss[m]
            lines.append(f"G{g}: n={len(sg)}  min={min(sg):.2f}  max={max(sg):.2f}  avg={sum(sg)/len(sg):.2f}  median={med:.2f}")
        ax4.text(0.98, 0.97, "\n".join(lines), transform=ax4.transAxes, ha="right", va="top", fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    else:
        ax4.text(0.5, 0.5, "No speedup data\n(grade ≥ min_grade)", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title(f"Speedup (grade ≥ {min_grade})")
    ax4.grid(True, alpha=grid_alpha)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Dataset stats plot saved to {plot_path}")


def plot_token_distribution(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    """Plot token length distribution for a dataset (single histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    token_lengths = [len(tokenizer.encode(s["text"])) for s in dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution - {name}")
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    plt.axvline(mean_len, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_len:.1f}")
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    plt.text(
        0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}",
        transform=plt.gca().transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Token length distribution plot saved to {plot_path}")


def plot_invariant_token_distribution(
    token_lengths: list,
    name: str = "dataset",
    plot_path: str | None = None,
) -> None:
    """Plot distribution of invariant-only token lengths (single histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color="coral", edgecolor="black", alpha=0.7)
    plt.xlabel("Invariant Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Invariant Token Length Distribution - {name}")
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    min_len = min(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    plt.axvline(mean_len, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_len:.1f}")
    plt.text(
        0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}",
        transform=plt.gca().transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Invariant token length distribution plot saved to {plot_path}")

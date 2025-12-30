"""
Script to compute statistics on cleaned invariants.

Stats computed:
- Character length (min, max, avg, median)
- Number of disjunctions || (min, max, avg, median)
- Number of conjunctions && (min, max, avg, median)

Also generates a combined histogram plot with subplots for each metric.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def count_operators(inv: str) -> tuple[int, int]:
    """Count disjunctions (||) and conjunctions (&&) in an invariant."""
    disjunctions = inv.count("||")
    conjunctions = inv.count("&&")
    return disjunctions, conjunctions


def compute_stats(values: list[int | float]) -> dict:
    """Compute min, max, avg, median for a list of values."""
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def print_stats(name: str, stats: dict) -> None:
    """Pretty print statistics."""
    print(f"  {name}:")
    print(f"    Min:    {stats['min']}")
    print(f"    Max:    {stats['max']}")
    print(f"    Avg:    {stats['avg']}")
    print(f"    Median: {stats['median']}")


def plot_histograms(
    char_lengths: list[int],
    pretty_char_lengths: list[int],
    disjunction_counts: list[int],
    conjunction_counts: list[int],
    output_path: Path,
    dataset_name: str,
) -> None:
    """Generate a 2x2 subplot figure with all histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Calculate compression rate for subtitle
    compression_rate = None
    if char_lengths and pretty_char_lengths:
        avg_orig = statistics.mean(char_lengths)
        avg_pretty = statistics.mean(pretty_char_lengths)
        compression_rate = round((1 - avg_pretty / avg_orig) * 100, 1)

    # --- Char length ---
    ax = axes[0, 0]
    if char_lengths:
        cap = np.percentile(char_lengths, 99)
        capped = [v for v in char_lengths if v <= cap]
        stats = compute_stats(char_lengths)
        ax.hist(capped, bins=50, edgecolor="black", alpha=0.7, color="#2E86AB")
        ax.axvline(
            stats["avg"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['avg']:.1f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median: {stats['median']:.1f}",
        )
        ax.legend(loc="upper right", fontsize=9)
        # Add min/max as text box
        textstr = f"Min: {stats['min']}\nMax: {stats['max']}"
        ax.text(
            0.98,
            0.65,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Original Invariant Length (99th pctl)")
    ax.grid(axis="y", alpha=0.3)

    # --- Pretty char length ---
    ax = axes[0, 1]
    if pretty_char_lengths:
        cap = np.percentile(pretty_char_lengths, 99)
        capped = [v for v in pretty_char_lengths if v <= cap]
        stats = compute_stats(pretty_char_lengths)
        ax.hist(capped, bins=50, edgecolor="black", alpha=0.7, color="#A23B72")
        ax.axvline(
            stats["avg"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['avg']:.1f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median: {stats['median']:.1f}",
        )
        ax.legend(loc="upper right", fontsize=9)
        # Add min/max as text box
        textstr = f"Min: {stats['min']}\nMax: {stats['max']}"
        ax.text(
            0.98,
            0.65,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_title("Pretty Invariant Length (99th pctl)")
    else:
        ax.text(
            0.5,
            0.5,
            "No pretty invariants",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Pretty Invariant Length")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)

    # --- Disjunctions ---
    ax = axes[1, 0]
    if disjunction_counts:
        stats = compute_stats(disjunction_counts)
        if max(disjunction_counts) > 0:
            cap = np.percentile(disjunction_counts, 99)
            capped = [v for v in disjunction_counts if v <= cap]
            ax.hist(
                capped,
                bins=min(50, int(cap) + 1),
                edgecolor="black",
                alpha=0.7,
                color="#F18F01",
            )
        else:
            ax.hist(
                disjunction_counts,
                bins=10,
                edgecolor="black",
                alpha=0.7,
                color="#F18F01",
            )
        ax.axvline(
            stats["avg"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['avg']:.1f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median: {stats['median']:.1f}",
        )
        ax.legend(loc="upper right", fontsize=9)
        # Add min/max as text box
        textstr = f"Min: {stats['min']}\nMax: {stats['max']}"
        ax.text(
            0.98,
            0.65,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.set_xlabel("Number of Disjunctions (||)")
    ax.set_ylabel("Frequency")
    ax.set_title("Disjunction Count (99th pctl)")
    ax.grid(axis="y", alpha=0.3)

    # --- Conjunctions ---
    ax = axes[1, 1]
    if conjunction_counts:
        stats = compute_stats(conjunction_counts)
        if max(conjunction_counts) > 0:
            cap = np.percentile(conjunction_counts, 99)
            capped = [v for v in conjunction_counts if v <= cap]
            ax.hist(
                capped,
                bins=min(50, int(cap) + 1),
                edgecolor="black",
                alpha=0.7,
                color="#C73E1D",
            )
        else:
            ax.hist(
                conjunction_counts,
                bins=10,
                edgecolor="black",
                alpha=0.7,
                color="#C73E1D",
            )
        ax.axvline(
            stats["avg"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['avg']:.1f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median: {stats['median']:.1f}",
        )
        ax.legend(loc="upper right", fontsize=9)
        # Add min/max as text box
        textstr = f"Min: {stats['min']}\nMax: {stats['max']}"
        ax.text(
            0.98,
            0.65,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.set_xlabel("Number of Conjunctions (&&)")
    ax.set_ylabel("Frequency")
    ax.set_title("Conjunction Count (99th pctl)")
    ax.grid(axis="y", alpha=0.3)

    # Main title with subtitle for compression rate
    fig.suptitle(
        f"Invariant Statistics - {dataset_name}", fontsize=14, fontweight="bold"
    )
    if compression_rate is not None:
        fig.text(
            0.5,
            0.965,
            f"Compression Rate (Original → Pretty): {compression_rate}%",
            ha="center",
            fontsize=11,
            style="italic",
            color="#555555",
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  Saved plot: {output_path}")


def analyze_invariants(json_path: str | Path, plot: bool = True) -> None:
    """Load cleaned invariants JSON and print statistics."""
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    # Collect all invariants
    char_lengths = []
    pretty_char_lengths = []
    disjunction_counts = []
    conjunction_counts = []

    total_invariants = 0

    for entry in data:
        for inv_info in entry.get("invariants", []):
            # Original invariant
            inv = inv_info.get("invariant", "")
            if inv:
                total_invariants += 1
                char_lengths.append(len(inv))
                disj, conj = count_operators(inv)
                disjunction_counts.append(disj)
                conjunction_counts.append(conj)

            # Pretty invariant (if available)
            pretty_inv = inv_info.get("pretty_invariant", "")
            if pretty_inv:
                pretty_char_lengths.append(len(pretty_inv))

    print("=" * 60)
    print(f"Invariant Statistics for: {json_path.name}")
    print("=" * 60)
    print(f"\nTotal entries: {len(data)}")
    print(f"Total invariants: {total_invariants}")

    print("\n--- Original Invariants ---")
    print_stats("Character Length", compute_stats(char_lengths))
    print_stats("Disjunctions (||)", compute_stats(disjunction_counts))
    print_stats("Conjunctions (&&)", compute_stats(conjunction_counts))

    if pretty_char_lengths:
        print("\n--- Pretty Invariants ---")
        print_stats("Character Length", compute_stats(pretty_char_lengths))

        # Compression ratio
        if char_lengths:
            avg_orig = statistics.mean(char_lengths)
            avg_pretty = statistics.mean(pretty_char_lengths)
            compression = round((1 - avg_pretty / avg_orig) * 100, 1)
            print(f"\n  Avg compression (pretty vs original): {compression}%")

    print("=" * 60)

    # Generate plot
    if plot:
        output_dir = json_path.parent / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = json_path.stem
        plot_histograms(
            char_lengths,
            pretty_char_lengths,
            disjunction_counts,
            conjunction_counts,
            output_dir / "inv_stats_histograms.png",
            dataset_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute invariant statistics")
    parser.add_argument(
        "json_file",
        nargs="?",
        default="data/train/invbench-train-uautomizer25-k1-full/invbench-train-uautomizer25-k1-full-cleaned.json",
        help="Path to the invariants JSON file",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating histogram plot",
    )

    args = parser.parse_args()
    analyze_invariants(args.json_file, plot=not args.no_plot)

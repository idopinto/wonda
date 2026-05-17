"""
Post-hoc timeout sweep analysis (R2-Q1).

Simulates how verification performance varies with timeout by filtering
per-instance results from Weave based on verification_time <= T.

Usage (from project root):
    uv run python scripts/rebuttal/timeout_sweep.py \
        --weave-project ip-ai/eval-inv-gen-new-latest-trails-runlim \
        --call-id 019bd9cf-d247-7d31-91c0-4c0883c4bff7
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_DIR = Path(__file__).resolve().parent / "figures"

TIMEOUTS = [15, 30, 60, 90, 120, 180, 300, 450, 600]

PALETTE = {
    "blue":   "#3A86FF",
    "purple": "#8338EC",
    "pink":   "#FF006E",
    "orange": "#FB5607",
    "teal":   "#06D6A0",
    "grey":   "#6C757D",
    "dark":   "#212529",
    "red":    "#D62828",
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ---------------------------------------------------------------------------
# Weave data fetching
# ---------------------------------------------------------------------------

def fetch_scorer_data(weave_project: str, call_id: str) -> tuple[list[dict], str]:
    """
    Fetch per-instance scorer results from a Weave evaluation call.

    Returns (instances, model_display_name).
    Each instance has: verification_time, verification_time_e2e,
    correctness_score, usefulness_score, speedup, speedup_e2e,
    has_speedup, has_speedup_e2e, median_timing.
    """
    import weave  # type: ignore[import-not-found]

    print(f"Connecting to Weave project: {weave_project}")
    client = weave.init(weave_project)

    eval_call = client.get_call(call_id)
    trace_id = eval_call.trace_id
    display_name = getattr(eval_call, "display_name", None) or call_id[:12]
    print(f"Evaluation call: {eval_call.op_name}")
    print(f"Trace ID: {trace_id}")

    print("Fetching trace calls...")
    all_trace_calls = list(client.get_calls(filter={"trace_ids": [trace_id]}))
    scorer_calls = [
        c for c in all_trace_calls if "InvGenScorer.score" in c.op_name
    ]
    print(f"  Scorer calls: {len(scorer_calls)}")

    def _float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    instances = []
    for sc in scorer_calls:
        output = sc.output or {}
        instances.append({
            "correctness_score": bool(output.get("correctness_score", False)),
            "usefulness_score": bool(output.get("usefulness_score", False)),
            "verification_time": _float(output.get("verification_time")),
            "verification_time_e2e": _float(output.get("verification_time_e2e")),
            "speedup": _float(output.get("speedup")),
            "speedup_e2e": _float(output.get("speedup_e2e")),
            "has_speedup": bool(output.get("has_speedup", False)),
            "has_speedup_e2e": bool(output.get("has_speedup_e2e", False)),
            "median_timing": _float(output.get("median_timing")),
        })

    print(f"  Total instances: {len(instances)}")
    return instances, display_name


# ---------------------------------------------------------------------------
# Timeout sweep
# ---------------------------------------------------------------------------

def compute_sweep(instances: list[dict], timeouts: list[float]) -> dict:
    """
    For each timeout T, compute correctness rate and speedup rate.

    Returns dict with keys:
        correctness_rate, correctness_rate_e2e,
        speedup_rate, speedup_rate_e2e
    Each is a list of floats (one per timeout).
    """
    n = len(instances)
    correctness_rates = []
    correctness_rates_e2e = []
    speedup_rates = []
    speedup_rates_e2e = []

    for T in timeouts:
        correct_at_T = sum(
            1 for r in instances
            if r["correctness_score"] and r["verification_time"] <= T
        )
        correct_at_T_e2e = sum(
            1 for r in instances
            if r["correctness_score"] and r["verification_time_e2e"] <= T
        )
        speedup_at_T = sum(
            1 for r in instances
            if r["usefulness_score"] and r["verification_time"] <= T and r["speedup"] > 1
        )
        speedup_at_T_e2e = sum(
            1 for r in instances
            if r["usefulness_score"] and r["verification_time_e2e"] <= T and r["speedup_e2e"] > 1
        )

        correctness_rates.append(correct_at_T / n * 100)
        correctness_rates_e2e.append(correct_at_T_e2e / n * 100)
        speedup_rates.append(speedup_at_T / n * 100)
        speedup_rates_e2e.append(speedup_at_T_e2e / n * 100)

    return {
        "correctness_rate": correctness_rates,
        "correctness_rate_e2e": correctness_rates_e2e,
        "speedup_rate": speedup_rates,
        "speedup_rate_e2e": speedup_rates_e2e,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(timeouts: list[float], sweep: dict, model_name: str, n: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Verification Performance vs. Timeout — {model_name}  ({n} instances)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # --- Correctness rate ---
    ax1.plot(timeouts, sweep["correctness_rate"], "o-",
             color=PALETTE["blue"], linewidth=2, markersize=6,
             label="Correctness rate")
    ax1.plot(timeouts, sweep["correctness_rate_e2e"], "s--",
             color=PALETTE["blue"], linewidth=2, markersize=6, alpha=0.6,
             label="Correctness rate (e2e)")

    for i, T in enumerate(timeouts):
        ax1.annotate(f"{sweep['correctness_rate'][i]:.1f}%",
                     (T, sweep["correctness_rate"][i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color=PALETTE["dark"])

    ax1.set_xlabel("Timeout (seconds)")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Correctness Rate vs. Timeout")
    ax1.set_xscale("log")
    ax1.set_xticks(timeouts)
    ax1.set_xticklabels([str(t) for t in timeouts])
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9)

    # --- Speedup rate ---
    ax2.plot(timeouts, sweep["speedup_rate"], "o-",
             color=PALETTE["teal"], linewidth=2, markersize=6,
             label="Speedup rate")
    ax2.plot(timeouts, sweep["speedup_rate_e2e"], "s--",
             color=PALETTE["teal"], linewidth=2, markersize=6, alpha=0.6,
             label="Speedup rate (e2e)")

    for i, T in enumerate(timeouts):
        ax2.annotate(f"{sweep['speedup_rate'][i]:.1f}%",
                     (T, sweep["speedup_rate"][i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color=PALETTE["dark"])

    ax2.set_xlabel("Timeout (seconds)")
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Speedup Rate vs. Timeout")
    ax2.set_xscale("log")
    ax2.set_xticks(timeouts)
    ax2.set_xticklabels([str(t) for t in timeouts])
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out = FIG_DIR / f"timeout_sweep_{safe_name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_sweep_table(timeouts: list[float], sweep: dict, n: int):
    print()
    print(f"  {'Timeout':>8}  {'Correct':>9}  {'Correct(e2e)':>13}  "
          f"{'Speedup':>9}  {'Speedup(e2e)':>13}")
    print("  " + "-" * 60)
    for i, T in enumerate(timeouts):
        print(f"  {T:>7.0f}s  "
              f"{sweep['correctness_rate'][i]:>8.1f}%  "
              f"{sweep['correctness_rate_e2e'][i]:>12.1f}%  "
              f"{sweep['speedup_rate'][i]:>8.1f}%  "
              f"{sweep['speedup_rate_e2e'][i]:>12.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc timeout sweep analysis (R2-Q1 rebuttal)"
    )
    parser.add_argument(
        "--weave-project", required=True,
        help="Weave project path",
    )
    parser.add_argument(
        "--call-id", required=True,
        help="Weave evaluation call ID",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Display name for the model (auto-detected from Weave if omitted)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    instances, model_name_detected = fetch_scorer_data(args.weave_project, args.call_id)
    model_name = args.model_name or model_name_detected
    n = len(instances)
    print(f"Model: {model_name}")

    sweep = compute_sweep(instances, TIMEOUTS)
    print_sweep_table(TIMEOUTS, sweep, n)

    print("Generating plot...")
    plot_sweep(TIMEOUTS, sweep, model_name, n)
    print("Done.")


if __name__ == "__main__":
    main()

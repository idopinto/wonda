"""
Multi-model post-hoc timeout sweep analysis (R2-Q1).

Plots correctness rate and speedup rate vs. timeout for multiple models
on the same figure.

Usage (from project root):
    uv run python scripts/rebuttal/timeout_sweep_multi.py
"""

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

WEAVE_PROJECT = "ip-ai/eval-inv-gen-new-latest-trails-runlim"

MODELS = [
    {"call_id": "019be131-79fa-7b05-9b95-1b67ae3f7e7f", "name": "Qwen3-4B (base)"},
    {"call_id": "019bd9cf-d247-7d31-91c0-4c0883c4bff7", "name": "Qwen3-4B SFT v2.2"},
    {"call_id": "019bc319-9cf9-728b-95dc-13230ab4d144", "name": "Qwen3-8B (base)"},
    {"call_id": "019bd8f1-1063-7004-a87d-6cc6890694bf", "name": "Qwen3-8B SFT v2.2"},
]

LINE_COLORS = ["#3A86FF", "#3A86FF", "#8338EC", "#8338EC"]
LINE_MARKERS = ["o", "o", "s", "s"]
LINE_STYLES = ["--", "-", "--", "-"]

PALETTE = {
    "grey":   "#6C757D",
    "dark":   "#212529",
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
# Weave data fetching (reused from timeout_sweep.py)
# ---------------------------------------------------------------------------

def fetch_scorer_data(client, call_id: str) -> list[dict]:
    eval_call = client.get_call(call_id)
    trace_id = eval_call.trace_id

    all_trace_calls = list(client.get_calls(filter={"trace_ids": [trace_id]}))
    scorer_calls = [
        c for c in all_trace_calls if "InvGenScorer.score" in c.op_name
    ]

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

    return instances


def compute_sweep(instances: list[dict], timeouts: list[float]) -> dict:
    n = len(instances)
    avg_baseline = sum(r["median_timing"] for r in instances) / n
    correctness_rates = []
    speedup_rates = []
    vbp_values = []

    for T in timeouts:
        correct_at_T = sum(
            1 for r in instances
            if r["correctness_score"] and r["verification_time"] <= T
        )
        speedup_at_T = sum(
            1 for r in instances
            if r["usefulness_score"] and r["verification_time"] <= T and r["speedup"] > 1
        )
        vbp_at_T = sum(
            min(r["verification_time"], r["median_timing"])
            if r["usefulness_score"] and r["verification_time"] <= T
            else r["median_timing"]
            for r in instances
        ) / n

        correctness_rates.append(correct_at_T / n * 100)
        speedup_rates.append(speedup_at_T / n * 100)
        vbp_values.append(vbp_at_T)

    return {
        "correctness_rate": correctness_rates,
        "speedup_rate": speedup_rates,
        "vbp": vbp_values,
        "avg_baseline": avg_baseline,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multi_sweep(timeouts: list[float], all_sweeps: list[dict]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Verification Performance vs. Timeout — Base vs. Fine-Tuned  (123 instances)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for i, entry in enumerate(all_sweeps):
        name = entry["name"]
        sweep = entry["sweep"]
        color = LINE_COLORS[i % len(LINE_COLORS)]
        marker = LINE_MARKERS[i % len(LINE_MARKERS)]
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        alpha = 0.5 if ls == "--" else 1.0

        ax1.plot(timeouts, sweep["correctness_rate"],
                 color=color, linewidth=2, markersize=6, label=name,
                 marker=marker, linestyle=ls, alpha=alpha)
        ax2.plot(timeouts, sweep["speedup_rate"],
                 color=color, linewidth=2, markersize=6, label=name,
                 marker=marker, linestyle=ls, alpha=alpha)
        ax3.plot(timeouts, sweep["vbp"],
                 color=color, linewidth=2, markersize=6, label=name,
                 marker=marker, linestyle=ls, alpha=alpha)

    rate_vals = []
    for entry in all_sweeps:
        rate_vals.extend(entry["sweep"]["correctness_rate"])
        rate_vals.extend(entry["sweep"]["speedup_rate"])
    y_max_rate = max(rate_vals) + 8

    for ax, title in [(ax1, "Correctness Rate vs. Timeout"),
                       (ax2, "Speedup Rate vs. Timeout")]:
        ax.set_xlabel("Timeout (seconds)")
        ax.set_ylabel("Rate (%)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(timeouts)
        ax.set_xticklabels([str(t) for t in timeouts], rotation=45)
        ax.set_ylim(0, y_max_rate)
        ax.legend(fontsize=9, loc="lower right")

    baseline = all_sweeps[0]["sweep"]["avg_baseline"]
    ax3.axhline(y=baseline, color=PALETTE["grey"], linestyle=":", linewidth=1.5,
                label=f"Baseline avg ({baseline:.1f}s)")
    ax3.set_xlabel("Timeout (seconds)")
    ax3.set_ylabel("VBP (seconds)")
    ax3.set_title("Virtual Best Performance vs. Timeout")
    ax3.set_xscale("log")
    ax3.set_xticks(timeouts)
    ax3.set_xticklabels([str(t) for t in timeouts], rotation=45)
    ax3.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "timeout_sweep_multi.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_multi_table(timeouts: list[float], all_sweeps: list[dict]):
    header = f"  {'Timeout':>8}"
    for entry in all_sweeps:
        short = entry["name"][:12]
        header += f"  {short + ' C':>14}  {short + ' S':>14}  {short + ' V':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ti, T in enumerate(timeouts):
        row = f"  {T:>7.0f}s"
        for entry in all_sweeps:
            s = entry["sweep"]
            row += (f"  {s['correctness_rate'][ti]:>13.1f}%"
                    f"  {s['speedup_rate'][ti]:>13.1f}%"
                    f"  {s['vbp'][ti]:>13.1f}s")
        print(row)

    print(f"\n  Baseline avg_median_timing: {all_sweeps[0]['sweep']['avg_baseline']:.2f}s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import weave  # type: ignore[import-not-found]

    _apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Weave project: {WEAVE_PROJECT}")
    client = weave.init(WEAVE_PROJECT)

    all_sweeps = []
    for model in MODELS:
        name = model["name"]
        call_id = model["call_id"]
        print(f"\nFetching {name} ({call_id[:12]}...)...")
        instances = fetch_scorer_data(client, call_id)
        print(f"  {len(instances)} instances")
        sweep = compute_sweep(instances, TIMEOUTS)
        all_sweeps.append({"name": name, "sweep": sweep})

    print("\n" + "=" * 70)
    print("  TIMEOUT SWEEP — MULTI-MODEL COMPARISON")
    print("=" * 70)
    print_multi_table(TIMEOUTS, all_sweeps)

    print("Generating plot...")
    plot_multi_sweep(TIMEOUTS, all_sweeps)
    print("Done.")


if __name__ == "__main__":
    main()

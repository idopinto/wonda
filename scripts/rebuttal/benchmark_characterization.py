"""
Benchmark characterization for ICML 2026 rebuttal.

Uses pycparser (via AstProgram) for precise AST-based analysis of:
- Program sizes (lines of code)
- Loop nesting depths
- Variable counts

Usage (from project root):
    uv run python scripts/rebuttal/benchmark_characterization.py
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from pycparser import c_ast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wonda.core.ast_program import AstProgram  # noqa: E402

HF_DATASET_NAME = "idopinto/wonda-eval-benchmark-full"
BENCHMARK_PATH = ROOT / "data" / "eval" / "wonda-eval-benchmark-full" / "wonda-eval-benchmark-full.json"
CODE2INV_BASELINE_PATH = ROOT / "data" / "eval" / "code2inv" / "baseline_uauto25" / "baseline_dataset.json"
TRAIN_DATASET_PATH = ROOT / "data" / "train" / "wonda-train-dataset-full-raw" / "wonda-train-dataset-full-raw.json"
DIFFICULTY_THRESHOLD = 15
FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":   "#3A86FF",
    "purple": "#8338EC",
    "pink":   "#FF006E",
    "orange": "#FB5607",
    "yellow": "#FFBE0B",
    "teal":   "#06D6A0",
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
# AST visitors
# ---------------------------------------------------------------------------

class VariableCollector(c_ast.NodeVisitor):
    """Count variable declarations inside function bodies."""

    def __init__(self):
        self.count = 0
        self._in_func = False

    def visit_FuncDef(self, node: c_ast.FuncDef):
        self._in_func = True
        self.generic_visit(node)
        self._in_func = False

    def visit_Decl(self, node: c_ast.Decl):
        if not self._in_func or node.name is None:
            return
        if isinstance(node.type, c_ast.FuncDecl):
            return
        self.count += 1


class LoopAnalyzer(c_ast.NodeVisitor):
    """Measure maximum loop nesting depth."""

    def __init__(self):
        self.max_depth = 0
        self._cur_depth = 0

    def _enter_loop(self, node: Any):
        self._cur_depth += 1
        self.max_depth = max(self.max_depth, self._cur_depth)
        self.generic_visit(node)
        self._cur_depth -= 1

    def visit_While(self, node: c_ast.While):
        self._enter_loop(node)

    def visit_For(self, node: c_ast.For):
        self._enter_loop(node)

    def visit_DoWhile(self, node: c_ast.DoWhile):
        self._enter_loop(node)


def _is_constant(node) -> bool:
    """Check if an AST subtree is a pure constant expression."""
    if isinstance(node, c_ast.Constant):
        return True
    if isinstance(node, c_ast.UnaryOp) and node.op in {'-', '+', '~'}:
        return _is_constant(node.expr)
    if isinstance(node, c_ast.Cast):
        return _is_constant(node.expr)
    return False


class NonlinearDetector(c_ast.NodeVisitor):
    """Detect variable*variable, variable/variable, or variable%variable."""

    def __init__(self):
        self.has_nonlinear = False

    def visit_BinaryOp(self, node: c_ast.BinaryOp):
        if node.op in {'*', '/', '%'}:
            if not _is_constant(node.left) and not _is_constant(node.right):
                self.has_nonlinear = True
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Data loading & analysis
# ---------------------------------------------------------------------------

def load_eval_programs(threshold: float, split: str = "hard") -> list[dict]:
    def _filter(median_timing: float) -> bool:
        if split == "hard":
            return median_timing > threshold
        elif split == "easy":
            return median_timing <= threshold
        else:
            return True

    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET_NAME, split="full")
        filtered = ds.filter(lambda x: _filter(x["median_timing"]))
        programs = [
            {
                "file": row["file"],
                "original_program": row["original_program"],
                "program_for_llm": row["program_for_llm"],
                "program_for_baseline": row["program_for_baseline"],
                "median_timing": row["median_timing"],
            }
            for row in filtered
        ]
        print(f"Loaded {len(programs)} {split} programs from HF ({HF_DATASET_NAME})")
        return programs
    except Exception as e:
        print(f"HF load failed ({e}), falling back to local JSON")
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        return [
            {
                "file": d["file"],
                "original_program": d["original_program"],
                "program_for_llm": d.get("program_for_llm", ""),
                "program_for_baseline": d.get("program_for_baseline", ""),
                "median_timing": d["timings"]["median"],
            }
            for d in data if _filter(d["timings"]["median"])
        ]


def load_train_programs(split: str = "all") -> list[dict]:
    with open(TRAIN_DATASET_PATH) as f:
        data = json.load(f)
    if split != "all":
        data = [d for d in data if d.get("split") == split]
    programs = [
        {
            "file": d["file"],
            "original_program": d["original_program"],
            "program_for_llm": d.get("program_for_llm", d["original_program"]),
            "program_for_baseline": d["program_for_baseline"],
            "median_timing": d["timings"]["median"],
        }
        for d in data
        if d.get("program_for_baseline")
    ]
    label = f"split={split}" if split != "all" else "all splits"
    print(f"Loaded {len(programs)} training programs ({label}) from {TRAIN_DATASET_PATH}")
    return programs


def load_code2inv_programs() -> list[dict]:
    with open(CODE2INV_BASELINE_PATH) as f:
        data = json.load(f)
    programs = [
        {
            "file": d["file"],
            "original_program": d["original_program"],
            "program_for_llm": d["original_program"],
            "program_for_baseline": d["original_program"],
            "median_timing": d.get("median_timing"),
        }
        for d in data
    ]
    print(f"Loaded {len(programs)} Code2Inv programs from {CODE2INV_BASELINE_PATH}")
    return programs


def loc(code: str) -> int:
    return len([l for l in code.split("\n") if l.strip()])


def analyze_program(entry: dict) -> dict | None:
    try:
        prog = AstProgram().from_code(entry["original_program"])
        prog.process(print_ast=False)
    except Exception as e:
        return {"error": str(e), "file": entry["file"]}

    var_collector = VariableCollector()
    var_collector.visit(prog.marked_ast)

    loop_analyzer = LoopAnalyzer()
    loop_analyzer.visit(prog.marked_ast)

    nonlinear_detector = NonlinearDetector()
    nonlinear_detector.visit(prog.marked_ast)

    return {
        "file": entry["file"],
        "loc_original": loc(entry["original_program"]),
        "loc_llm": loc(entry.get("program_for_llm") or prog.llm_code or ""),
        "loc_baseline": loc(entry.get("program_for_baseline") or ""),
        "max_nesting": loop_analyzer.max_depth,
        "num_variables": var_collector.count,
        "has_nonlinear": nonlinear_detector.has_nonlinear,
        "median_timing": entry.get("median_timing"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar_label(ax, rects, fmt="{:.0f}"):
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2, h + 0.5,
                fmt.format(h), ha="center", va="bottom", fontsize=9, color=PALETTE["dark"],
            )


def _stats_text(vals: list) -> str:
    """One-line summary: mean, median, min, max, std."""
    return (f"mean={statistics.mean(vals):.1f}  med={statistics.median(vals):.0f}  "
            f"min={min(vals)}  max={max(vals)}  std={statistics.stdev(vals):.1f}")


def _plot_single(results: list[dict], loc_key: str, label: str, filename: str,
                 num_instances: int | None = None):
    n = len(results)
    instances_str = f", {num_instances} instances" if num_instances is not None and num_instances != n else ""
    has_timing = any(r.get("median_timing") is not None for r in results)
    ncols = 5 if has_timing else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5.2))
    fig.suptitle(f"Benchmark Characterization — {label}  ({n} programs{instances_str})",
                 fontsize=14, fontweight="bold", y=1.02)

    # --- LOC histogram ---
    ax = axes[0]
    locs = [r[loc_key] for r in results]
    bins = np.arange(0, max(locs) + 30, 30)
    counts, edges, patches = ax.hist(locs, bins=bins, color=PALETTE["blue"], edgecolor="white", linewidth=0.6, alpha=0.85)
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.3,
                    str(int(count)), ha="center", va="bottom", fontsize=8, color=PALETTE["dark"])
    ax.axvline(statistics.median(locs), color=PALETTE["pink"], ls="--", lw=1.5,
               label=f"median = {statistics.median(locs):.0f}")
    ax.set_xticks(bins)
    ax.set_xticklabels([str(int(b)) for b in bins], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Lines of Code")
    ax.set_ylabel("Programs")
    ax.set_title("Program Size")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, _stats_text(locs), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=PALETTE["grey"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9))

    # --- Nesting depth bar ---
    ax = axes[1]
    nestings = [r["max_nesting"] for r in results]
    nesting_cnt = Counter(nestings)
    x = sorted(nesting_cnt.keys())
    y = [nesting_cnt[k] for k in x]
    bars = ax.bar([str(d) for d in x], y, color=PALETTE["orange"], edgecolor="white", linewidth=0.6)
    _bar_label(ax, bars)
    ax.set_xlabel("Max Loop Nesting Depth")
    ax.set_ylabel("Programs")
    ax.set_title("Loop Nesting Depth")
    ax.text(0.97, 0.95, _stats_text(nestings), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=PALETTE["grey"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9))

    # --- Variable count histogram ---
    ax = axes[2]
    vars_ = [r["num_variables"] for r in results]
    bins_v = np.arange(0, max(vars_) + 4, 4)
    counts_v, edges_v, patches_v = ax.hist(vars_, bins=bins_v, color=PALETTE["purple"], edgecolor="white", linewidth=0.6, alpha=0.85)
    for count, patch in zip(counts_v, patches_v):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.3,
                    str(int(count)), ha="center", va="bottom", fontsize=8, color=PALETTE["dark"])
    ax.axvline(statistics.median(vars_), color=PALETTE["pink"], ls="--", lw=1.5,
               label=f"median = {statistics.median(vars_):.0f}")
    ax.set_xticks(bins_v)
    ax.set_xticklabels([str(int(b)) for b in bins_v], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Variable Declarations")
    ax.set_ylabel("Programs")
    ax.set_title("Variable Count")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, _stats_text(vars_), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=PALETTE["grey"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9))

    # --- Nonlinear arithmetic bar ---
    ax = axes[3]
    nl_counts = Counter("Yes" if r["has_nonlinear"] else "No" for r in results)
    labels_nl = ["No", "Yes"]
    values_nl = [nl_counts.get(k, 0) for k in labels_nl]
    colors_nl = [PALETTE["teal"], PALETTE["pink"]]
    bars_nl = ax.bar(labels_nl, values_nl, color=colors_nl, edgecolor="white", linewidth=0.6)
    _bar_label(ax, bars_nl)
    ax.set_xlabel("Has Nonlinear Arithmetic")
    ax.set_ylabel("Programs")
    ax.set_title("Nonlinear Arithmetic")
    nl_note = "Nonlinear = var*var, var/var, or var%var"
    ax.text(0.97, 0.95, nl_note, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=PALETTE["grey"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9))

    # --- Baseline median timing histogram (log scale) ---
    if has_timing:
        ax = axes[4]
        timings = [r["median_timing"] for r in results if r.get("median_timing") is not None]
        log_bins = np.logspace(np.log10(max(1, min(timings))), np.log10(max(timings) + 1), 20)
        counts_t, edges_t, patches_t = ax.hist(timings, bins=log_bins,
                                                color=PALETTE["blue"], edgecolor="white",
                                                linewidth=0.6, alpha=0.85)
        for count, patch in zip(counts_t, patches_t):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.3,
                        str(int(count)), ha="center", va="bottom", fontsize=7, color=PALETTE["dark"])
        med = statistics.median(timings)
        mean_ = statistics.mean(timings)
        p25 = sorted(timings)[int(len(timings) * 0.25)]
        p75 = sorted(timings)[int(len(timings) * 0.75)]
        ax.axvline(med, color=PALETTE["pink"], ls="--", lw=1.8,
                   label=f"Median ({med:.1f}s)")
        ax.axvline(mean_, color=PALETTE["orange"], ls=":", lw=1.8,
                   label=f"Mean ({mean_:.1f}s)")
        ax.axvspan(p25, p75, alpha=0.08, color=PALETTE["grey"], label=f"IQR [{p25:.0f}s, {p75:.0f}s]")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}s"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(axis="x", which="both", labelsize=8, rotation=45)
        nice_ticks = [t for t in [5, 10, 20, 30, 50, 100, 200, 300, 500, 600]
                      if min(timings) <= t <= max(timings) * 1.1]
        ax.set_xticks(nice_ticks)
        ax.set_xlabel("Baseline Median Time (seconds, log scale)")
        ax.set_ylabel("Programs")
        ax.set_title("Baseline Timing Distribution")
        ax.legend(fontsize=8.5)
        ax.text(0.97, 0.95, _stats_text(timings), transform=ax.transAxes,
                fontsize=7.5, va="top", ha="right", color=PALETTE["grey"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.9))

    plt.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_characterization(results: list[dict], split: str = "hard"):
    n = len(results)
    suffix = f"_{split}" if split != "hard" else ""
    label_extra = f" [{split} split]" if split != "hard" else ""
    _plot_single(results, "loc_original", f"Original program (original_program){label_extra}",
                 f"benchmark_characterization{suffix}.png", num_instances=n)
    _plot_single(results, "loc_llm", f"LLM-facing code (program_for_llm){label_extra}",
                 f"benchmark_characterization_llm{suffix}.png", num_instances=n)
    _plot_single(results, "loc_baseline", f"Baseline code (program_for_baseline){label_extra}",
                 f"benchmark_characterization_baseline{suffix}.png", num_instances=n)


def plot_code2inv_characterization(results: list[dict]):
    _plot_single(results, "loc_baseline", "Code2Inv benchmark",
                 "benchmark_characterization_code2inv.png")


def plot_train_characterization(results: list[dict], split: str):
    suffix = f"_{split}" if split != "all" else ""
    n_prog = len(results)
    _plot_single(results, "loc_baseline",
                 f"Training data — program_for_baseline  (split={split}, {n_prog} programs)",
                 f"benchmark_characterization_train{suffix}.png")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def fmt(val: float, d: int = 1) -> str:
    return f"{val:.{d}f}"


def print_summary(results: list[dict]):
    n = len(results)

    props = [
        ("LOC (original)",    [r["loc_original"] for r in results]),
        ("LOC (baseline)",    [r["loc_baseline"] for r in results]),
        ("Max nesting depth", [r["max_nesting"] for r in results]),
        ("Variable count",    [r["num_variables"] for r in results]),
    ]

    print()
    print("=" * 60)
    print("  BENCHMARK CHARACTERIZATION  (AST-based)")
    print(f"  {n} programs  |  difficulty_threshold = {DIFFICULTY_THRESHOLD}s")
    print("=" * 60)

    hdr = f"  {'Property':<22} {'Mean':>7} {'Med':>7} {'Min':>5} {'Max':>5} {'Std':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, vals in props:
        s = sorted(vals)
        print(f"  {name:<22} {fmt(statistics.mean(s)):>7} {fmt(statistics.median(s)):>7}"
              f" {fmt(min(s)):>5} {fmt(max(s)):>5} {fmt(statistics.stdev(s)):>7}")

    nesting = Counter(r["max_nesting"] for r in results)
    print()
    print("  Nesting depth breakdown:")
    for depth in sorted(nesting):
        cnt = nesting[depth]
        print(f"    depth {depth}:  {cnt} programs ({100*cnt/n:.0f}%)")

    nl_count = sum(1 for r in results if r["has_nonlinear"])
    print()
    print("  Nonlinear arithmetic (var*var, var/var, var%var):")
    print(f"    nonlinear:  {nl_count} programs ({100*nl_count/n:.0f}%)")
    print(f"    linear:     {n - nl_count} programs ({100*(n-nl_count)/n:.0f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code2inv", action="store_true",
                        help="Characterize the Code2Inv benchmark")
    parser.add_argument("--train", action="store_true",
                        help="Characterize the training dataset")
    parser.add_argument("--split", default="hard", choices=["all", "easy", "hard"],
                        help="Split to use: hard (default), easy, or all")
    args = parser.parse_args()

    _apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.code2inv:
        programs = load_code2inv_programs()
    elif args.train:
        programs = load_train_programs(args.split)
    else:
        programs = load_eval_programs(DIFFICULTY_THRESHOLD, split=args.split)

    print(f"Analysing {len(programs)} programs via AST...")

    results = []
    errors = []
    for p in programs:
        r = analyze_program(p)
        if r is None or "error" in r:
            errors.append(r or {"file": p["file"], "error": "unknown"})
        else:
            results.append(r)

    if errors:
        print(f"  Parse failures: {len(errors)}")
        for e in errors:
            print(f"    - {e['file']}: {e.get('error', '?')[:70]}")

    print_summary(results)

    print("Generating plot...")
    if args.code2inv:
        plot_code2inv_characterization(results)
    elif args.train:
        plot_train_characterization(results, args.split)
    else:
        plot_characterization(results, split=args.split)
    print("Done.")


if __name__ == "__main__":
    main()

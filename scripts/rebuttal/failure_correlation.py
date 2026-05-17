"""
Failure mode correlation with program properties (R2-W3).

Fetches per-instance evaluation results from Weave, joins them with
AST-based program properties (LOC, nesting depth, variable count), and
produces grouped bar charts showing total vs failed instances per
property bucket.

Usage (from project root):
    # v0 model:
    uv run python scripts/rebuttal/failure_correlation.py \
        --weave-project ip-ai/eval-inv-gen-new-latest-trails-runlim \
        --call-id 019bec48-cf5a-7307-a910-190899d41094

    # v2.2 model:
    uv run python scripts/rebuttal/failure_correlation.py \
        --weave-project ip-ai/eval-inv-gen-new-latest-trails-runlim \
        --call-id 019bd9cf-d247-7d31-91c0-4c0883c4bff7
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycparser import c_ast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wonda.core.ast_program import AstProgram  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
BENCHMARK_PATH = ROOT / "data" / "eval" / "wonda-eval-benchmark-full" / "wonda-eval-benchmark-full.json"


def _build_baseline_loc_lookup() -> dict[str, int]:
    """Build a map from original_program text -> loc(program_for_baseline)."""
    import json
    try:
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)
        return {
            entry["original_program"]: loc(entry.get("program_for_baseline") or "")
            for entry in data
            if entry.get("program_for_baseline")
        }
    except Exception as e:
        print(f"Warning: could not load benchmark for baseline LoC lookup ({e})")
        return {}

# ---------------------------------------------------------------------------
# Style (shared with benchmark_characterization.py)
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
# AST visitors (identical to benchmark_characterization.py)
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


def loc(code: str) -> int:
    return len([line for line in code.split("\n") if line.strip()])


# ---------------------------------------------------------------------------
# Weave data fetching
# ---------------------------------------------------------------------------

def fetch_weave_instances(
    weave_project: str, call_id: str
) -> tuple[list[dict], str]:
    """
    Fetch per-instance results from a Weave evaluation call.

    Returns:
        (instances, model_display_name)
        Each instance dict has: original_program, target_marker,
        validation_score, correctness_score, usefulness_score, final_decision
    """
    import weave  # type: ignore[import-not-found]

    print(f"Connecting to Weave project: {weave_project}")
    client = weave.init(weave_project)

    eval_call = client.get_call(call_id)
    trace_id = eval_call.trace_id
    display_name = getattr(eval_call, "display_name", None) or call_id[:12]
    print(f"Evaluation call: {eval_call.op_name}")
    print(f"Trace ID: {trace_id}")

    # Bulk fetch all calls in the trace, then partition locally
    print("Fetching trace calls...")
    all_trace_calls = list(client.get_calls(filter={"trace_ids": [trace_id]}))

    scorer_calls = [
        c for c in all_trace_calls if "InvGenScorer.score" in c.op_name
    ]
    decide_calls = [
        c for c in all_trace_calls if "DecisionProcedure.decide" in c.op_name
    ]
    print(f"  Scorer calls: {len(scorer_calls)}")
    print(f"  Decide calls: {len(decide_calls)}")

    # Index decide calls by parent_id (parent = scorer call)
    decide_by_parent = {dc.parent_id: dc for dc in decide_calls}

    instances = []
    for sc in scorer_calls:
        output = sc.output or {}
        validation_score = bool(output.get("validation_score", False))
        correctness_score = bool(output.get("correctness_score", False))
        usefulness_score = bool(output.get("usefulness_score", False))

        # Get final_decision from the nested decide call
        final_decision = "N/A"
        dc = decide_by_parent.get(sc.id)
        if dc and dc.output:
            final_decision = str(getattr(dc.output, "final_decision", "N/A"))

        has_speedup = bool(output.get("has_speedup", False))
        has_speedup_e2e = bool(output.get("has_speedup_e2e", False))

        instances.append({
            "original_program": str(sc.inputs.get("original_program", "")),
            "target_marker": str(sc.inputs.get("target_marker", "")),
            "validation_score": validation_score,
            "correctness_score": correctness_score,
            "usefulness_score": usefulness_score,
            "final_decision": final_decision,
            "has_speedup": has_speedup,
            "has_speedup_e2e": has_speedup_e2e,
        })

    print(f"  Total instances fetched: {len(instances)}")
    return instances, display_name


# ---------------------------------------------------------------------------
# AST analysis per instance
# ---------------------------------------------------------------------------

def analyze_instances(instances: list[dict]) -> list[dict]:
    """
    Compute AST properties for each instance. Caches analysis by
    original_program to avoid redundant work across markers.
    LoC is taken from program_for_baseline (consistent with benchmark characterization).
    """
    baseline_loc_lookup = _build_baseline_loc_lookup()
    cache: dict[int, dict] = {}  # hash(program) -> {loc, max_nesting, num_variables}
    results = []
    errors = 0

    for inst in instances:
        prog_str = inst["original_program"]
        prog_hash = hash(prog_str)

        if prog_hash not in cache:
            try:
                prog = AstProgram().from_code(prog_str)
                prog.process(print_ast=False)

                var_collector = VariableCollector()
                var_collector.visit(prog.marked_ast)

                loop_analyzer = LoopAnalyzer()
                loop_analyzer.visit(prog.marked_ast)

                nonlinear_detector = NonlinearDetector()
                nonlinear_detector.visit(prog.marked_ast)

                baseline_loc = baseline_loc_lookup.get(prog_str) or loc(prog_str)

                cache[prog_hash] = {
                    "loc": baseline_loc,
                    "max_nesting": loop_analyzer.max_depth,
                    "num_variables": var_collector.count,
                    "has_nonlinear": nonlinear_detector.has_nonlinear,
                }
            except Exception:
                errors += 1
                cache[prog_hash] = None

        props = cache[prog_hash]
        if props is None:
            continue

        results.append({**inst, **props})

    if errors:
        print(f"  AST parse errors: {errors} (skipped)")
    return results


# ---------------------------------------------------------------------------
# Per-program aggregation
# ---------------------------------------------------------------------------

STAGE_RANK = {"Success": 3, "Sufficiency fail": 2, "Correctness fail": 1, "Validation fail": 0}


def aggregate_per_program(results: list[dict]) -> list[dict]:
    """
    Aggregate per-instance results to per-program. Each program appears once.
    The stage classification uses the *best* outcome across all markers.
    A program is 'Success' if at least one marker succeeded (all_fail = failed).
    """
    from collections import defaultdict
    programs: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        programs[hash(r["original_program"])].append(r)

    aggregated = []
    for prog_hash, instances in programs.items():
        best_stage = "Validation fail"
        for inst in instances:
            stage = _classify_stage(inst)
            if STAGE_RANK[stage] > STAGE_RANK[best_stage]:
                best_stage = stage

        ref = instances[0]
        aggregated.append({
            "loc": ref["loc"],
            "max_nesting": ref["max_nesting"],
            "num_variables": ref["num_variables"],
            "has_nonlinear": ref["has_nonlinear"],
            "num_markers": len(instances),
            # Use best stage across markers for classification
            "validation_score": STAGE_RANK[best_stage] >= STAGE_RANK["Correctness fail"],
            "correctness_score": STAGE_RANK[best_stage] >= STAGE_RANK["Sufficiency fail"],
            "usefulness_score": best_stage == "Success",
            "has_speedup": any(i["has_speedup"] for i in instances),
            "has_speedup_e2e": any(i["has_speedup_e2e"] for i in instances),
            "final_decision": "TRUE" if best_stage == "Success" else "UNKNOWN",
        })

    return aggregated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STAGE_COLORS = {
    "Validation fail":  PALETTE["grey"],
    "Correctness fail": PALETTE["red"],
    "Sufficiency fail": PALETTE["orange"],
    "Success":          PALETTE["teal"],
}


def _classify_stage(r: dict) -> str:
    if not r["validation_score"]:
        return "Validation fail"
    if not r["correctness_score"]:
        return "Correctness fail"
    if not r["usefulness_score"]:
        return "Sufficiency fail"
    return "Success"


STAGE_ORDER = ["Success", "Sufficiency fail", "Correctness fail", "Validation fail"]


def _make_stacked_histogram(
    ax,
    results: list[dict],
    val_key: str,
    bins: np.ndarray,
    xlabel: str,
    title: str,
):
    """
    Draw a stacked bar histogram with pipeline stage breakdown per bin.
    """
    by_stage = {s: [] for s in STAGE_ORDER}
    for r in results:
        by_stage[_classify_stage(r)].append(r[val_key])

    stage_counts = {}
    for stage in STAGE_ORDER:
        stage_counts[stage], _ = np.histogram(by_stage[stage], bins=bins)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.75

    bottom = np.zeros(len(bin_centers))
    for stage in STAGE_ORDER:
        counts = stage_counts[stage]
        ax.bar(
            bin_centers, counts, width, bottom=bottom,
            color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.6,
            label=stage,
        )
        bottom += counts

    # Annotate total count on top of each bar
    totals = bottom
    max_h = max(totals) if len(totals) else 1
    for i, t in enumerate(totals):
        if t > 0:
            ax.text(
                bin_centers[i], t + max_h * 0.02, f"{int(t)}",
                ha="center", va="bottom", fontsize=8,
                color=PALETTE["dark"],
            )

    ax.set_xticks(bins)
    ax.set_xticklabels([str(int(b)) for b in bins], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Instances")
    ax.set_title(title)


def _make_stacked_categorical(
    ax,
    results: list[dict],
    val_key: str,
    xlabel: str,
    title: str,
):
    """
    Draw stacked bars for categorical data with pipeline stage breakdown.
    """
    by_stage = {s: [] for s in STAGE_ORDER}
    for r in results:
        by_stage[_classify_stage(r)].append(r[val_key])

    all_vals = [r[val_key] for r in results]
    categories = sorted(set(all_vals))
    x = np.arange(len(categories))
    width = 0.6

    stage_counts = {}
    for stage in STAGE_ORDER:
        cnt = Counter(by_stage[stage])
        stage_counts[stage] = [cnt.get(k, 0) for k in categories]

    bottom = np.zeros(len(categories))
    for stage in STAGE_ORDER:
        counts = np.array(stage_counts[stage])
        ax.bar(
            x, counts, width, bottom=bottom,
            color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.6,
            label=stage,
        )
        bottom += counts

    totals = bottom
    max_h = max(totals) if len(totals) else 1
    for i, t in enumerate(totals):
        if t > 0:
            ax.text(
                x[i], t + max_h * 0.02, f"{int(t)}",
                ha="center", va="bottom", fontsize=8,
                color=PALETTE["dark"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Instances")
    ax.set_title(title)


def plot_failure_correlation(results: list[dict], model_name: str,
                             view_label: str = "instances", suffix: str = ""):
    """Create the 1x4 stacked bar chart figure with pipeline breakdown."""
    n_all = len(results)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.8))
    fig.suptitle(
        f"Failure Mode Breakdown by Program Properties — {model_name}  ({n_all} {view_label})\n"
        f"Success = correct & conclusive  |  Sufficiency fail = correct but inconclusive  |  Correctness fail = incorrect invariant\n"
        f"Nonlinear = program contains var*var, var/var, or var%var operations",
        fontsize=11, fontweight="bold", y=1.06,
    )

    all_locs = [r["loc"] for r in results]
    loc_bins = np.arange(0, max(all_locs) + 30, 30)
    _make_stacked_histogram(
        axes[0], results, "loc", loc_bins,
        "Lines of Code", "Program Size",
    )

    _make_stacked_categorical(
        axes[1], results, "max_nesting",
        "Max Loop Nesting Depth", "Loop Nesting Depth",
    )

    all_vars = [r["num_variables"] for r in results]
    var_bins = np.arange(0, max(all_vars) + 4, 4)
    _make_stacked_histogram(
        axes[2], results, "num_variables", var_bins,
        "Variable Declarations", "Variable Count",
    )

    # Tag each result with a label for the categorical plot
    for r in results:
        r["_nonlinear_label"] = "Yes" if r["has_nonlinear"] else "No"
    _make_stacked_categorical(
        axes[3], results, "_nonlinear_label",
        "Has Nonlinear Arithmetic", "Nonlinear Arithmetic",
    )

    # Shared legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out = FIG_DIR / f"failure_correlation_{safe_name}{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_failure_summary(results: list[dict], view_label: str = "instances"):
    all_instances = results
    failed = [r for r in results if not r["usefulness_score"]]
    n_all = len(all_instances)
    n_fail = len(failed)

    print()
    print("=" * 70)
    print(f"  FAILURE CORRELATION ANALYSIS  ({view_label})")
    print(f"  {n_all} {view_label} | {n_all - n_fail} success | {n_fail} failed "
          f"({n_fail / n_all * 100:.1f}%)")
    print("=" * 70)

    # Decision breakdown
    decisions = Counter(r["final_decision"] for r in results)
    print("\n  Decision breakdown:")
    for dec in sorted(decisions.keys()):
        cnt = decisions[dec]
        print(f"    {dec:<12} {cnt:>4}  ({cnt / n_all * 100:.1f}%)")

    # Pipeline funnel: validation -> correctness -> sufficiency -> speedup
    n_valid = sum(1 for r in results if r["validation_score"])
    n_correct = sum(1 for r in results if r["correctness_score"])
    n_sufficient = sum(1 for r in results if r["usefulness_score"])
    n_speedup = sum(1 for r in results if r["has_speedup"])
    n_speedup_e2e = sum(1 for r in results if r["has_speedup_e2e"])

    def _funnel_row(stage: str, passed: int, total: int):
        lost = total - passed
        pct = passed / n_all * 100 if n_all else 0
        lost_pct = lost / n_all * 100 if n_all else 0
        print(f"    {stage:<22} {passed:>4}/{n_all}  ({pct:>5.1f}%)   "
              f"lost {lost:>3} ({lost_pct:>5.1f}%)")

    print(f"\n  Evaluation pipeline (cumulative pass / lost at each stage):")
    _funnel_row("Validation",  n_valid,     n_all)
    _funnel_row("Correctness", n_correct,   n_valid)
    _funnel_row("Sufficiency",  n_sufficient, n_correct)
    _funnel_row("Speedup",     n_speedup,   n_sufficient)
    _funnel_row("Speedup (e2e)", n_speedup_e2e, n_sufficient)

    # Correlation table
    print(f"\n  {'Property Range':<28} {'Total':>6} {'Failed':>7} {'Rate':>7}")
    print("  " + "-" * 50)

    def _rate_row(label: str, total: int, failed_count: int):
        rate = failed_count / total * 100 if total > 0 else 0
        print(f"  {label:<28} {total:>6} {failed_count:>7} {rate:>6.1f}%")

    # LOC ranges
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 500)]:
        a = [r for r in all_instances if lo <= r["loc"] < hi]
        f = [r for r in failed if lo <= r["loc"] < hi]
        if a:
            _rate_row(f"LOC [{lo}, {hi})", len(a), len(f))

    # Nesting depth
    for d in sorted(set(r["max_nesting"] for r in all_instances)):
        a = [r for r in all_instances if r["max_nesting"] == d]
        f = [r for r in failed if r["max_nesting"] == d]
        _rate_row(f"Nesting depth = {d}", len(a), len(f))

    # Variable count ranges
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50)]:
        a = [r for r in all_instances if lo <= r["num_variables"] < hi]
        f = [r for r in failed if lo <= r["num_variables"] < hi]
        if a:
            _rate_row(f"Variables [{lo}, {hi})", len(a), len(f))

    # Nonlinear arithmetic
    a_nl = [r for r in all_instances if r["has_nonlinear"]]
    f_nl = [r for r in failed if r["has_nonlinear"]]
    a_lin = [r for r in all_instances if not r["has_nonlinear"]]
    f_lin = [r for r in failed if not r["has_nonlinear"]]
    _rate_row("Nonlinear = Yes", len(a_nl), len(f_nl))
    _rate_row("Nonlinear = No", len(a_lin), len(f_lin))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Failure correlation analysis (R2-W3 rebuttal)"
    )
    parser.add_argument(
        "--weave-project", required=True,
        help="Weave project path, e.g. ip-ai/eval-inv-gen-new-latest-trails-runlim",
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

    # Step 1: Fetch from Weave
    instances, model_name_detected = fetch_weave_instances(args.weave_project, args.call_id)
    model_name = args.model_name or model_name_detected
    print(f"Model: {model_name}")

    # Step 2 & 3: Classify + compute AST properties
    print("Analysing programs via AST...")
    results = analyze_instances(instances)
    print(f"  Analysed {len(results)} instances")

    # Step 4: Per-instance view (123 instances)
    print_failure_summary(results, view_label="instances")
    print("Generating per-instance plot...")
    plot_failure_correlation(results, model_name,
                             view_label="instances", suffix="_instances")

    # Step 5: Per-program view (72 programs, success = at least one marker succeeded)
    programs = aggregate_per_program(results)
    print_failure_summary(programs, view_label="programs")
    print("Generating per-program plot...")
    plot_failure_correlation(programs, model_name,
                             view_label="programs", suffix="_programs")
    print("Done.")


if __name__ == "__main__":
    main()

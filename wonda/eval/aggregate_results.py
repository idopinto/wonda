"""
Standalone script to aggregate results from multiple evaluation runs.

Can be used to:
1. Re-aggregate existing experiment results
2. Combine results from different experiment folders
3. Compute statistics with different confidence levels

Usage:
    uv run -m wonda.eval.aggregate_results /path/to/experiment_dir
    uv run -m wonda.eval.aggregate_results /path/to/experiment_dir --confidence_level=0.99
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_summaries(experiment_dir: Path) -> list[dict]:
    """Load all summary JSON files from the experiment directory."""
    summaries = []
    for summary_file in sorted(experiment_dir.glob("summary_run_*.json")):
        with open(summary_file) as f:
            summaries.append(json.load(f))
        print(f"Loaded: {summary_file.name}")
    return summaries


def aggregate_summaries(summaries: list[dict], confidence_level: float = 0.95) -> dict:
    """
    Aggregate multiple run summaries into mean ± std with confidence intervals.
    
    Args:
        summaries: List of summary dictionaries from each run.
        confidence_level: Confidence level for CI (default 0.95 for 95% CI).
    
    Returns:
        Dictionary with aggregated statistics for each metric.
    """
    if not summaries:
        return {}
    
    # Get the InvGenScorer metrics (main scorer)
    scorer_key = "InvGenScorer"
    scorer_summaries = [s.get(scorer_key, {}) for s in summaries]
    
    # Get all numeric metrics
    all_keys = set()
    for s in scorer_summaries:
        all_keys.update(s.keys())
    
    n = len(scorer_summaries)
    aggregated = {"n_runs": n, "confidence_level": confidence_level}
    
    for key in sorted(all_keys):
        values = [s.get(key) for s in scorer_summaries if key in s]
        
        # Skip non-numeric values
        if not values or not all(isinstance(v, (int, float)) for v in values):
            continue
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        se = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        
        # Confidence interval using t-distribution
        if len(values) > 1 and se > 0:
            ci = stats.t.interval(confidence_level, df=len(values)-1, loc=mean, scale=se)
            ci_low, ci_high = ci
            # Safety check: if CI computation returned NaN, fall back to mean
            if np.isnan(ci_low) or np.isnan(ci_high):
                ci_low, ci_high = mean, mean
        else:
            # If all values are identical (se=0) or only one value, CI is just the mean
            ci_low, ci_high = mean, mean

        # Clamp CIs to sensible ranges for readability
        if key.endswith("_rate"):
            ci_low = max(0.0, float(ci_low))
            ci_high = min(1.0, float(ci_high))
        elif key.endswith("_count"):
            max_obs = float(np.max(values)) if len(values) else float(mean)
            ci_low = max(0.0, float(ci_low))
            ci_high = min(max_obs, float(ci_high))
        elif any(s in key for s in ["timing", "time", "vbp", "vbs", "speedup"]):
            ci_low = max(0.0, float(ci_low))
            ci_high = max(0.0, float(ci_high))
        
        aggregated[key] = {
            "mean": float(mean),
            "std": float(std),
            "se": float(se),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "values": values.tolist(),
        }
    
    return aggregated


def format_results(aggregated: dict) -> str:
    """Format aggregated results as a readable string."""
    lines = [
        "",
        "=" * 80,
        "AGGREGATED RESULTS",
        "=" * 80,
        f"Number of runs: {aggregated.get('n_runs', 'N/A')}",
        f"Confidence level: {aggregated.get('confidence_level', 0.95) * 100:.0f}%",
        "",
        "-" * 80,
        f"{'Metric':<25} {'Mean':>12} {'± Std':>12} {'95% CI':>25}",
        "-" * 80,
    ]
    
    # Key metrics to highlight
    key_metrics = [
        # Counts
        "validation_count", "correctness_count", "usefulness_count",
        "speedup_count", "speedup_e2e_count",

        # Rates
        "validation_rate", "correctness_rate", "usefulness_rate",
        "speedup_rate", "speedup_e2e_rate",

        # Performance
        "vbp", "vbp_e2e", "avg_median_timing",
        "speedup_gt1", "speedup_gt1_e2e",
        "speedup_all", "speedup_all_e2e",
    ]
    
    for metric in key_metrics:
        if metric in aggregated:
            data = aggregated[metric]
            ci_str = f"[{data['ci_low']:.4f}, {data['ci_high']:.4f}]"
            lines.append(
                f"{metric:<25} {data['mean']:>12.4f} {data['std']:>12.4f} {ci_str:>25}"
            )
    
    lines.extend(["-" * 80, ""])
    
    return "\n".join(lines)


def format_latex_table(aggregated: dict) -> str:
    """Format results as a LaTeX table row for papers."""
    key_metrics = [
        ("validation_rate", "Val."),
        ("correctness_rate", "Corr."),
        ("usefulness_rate", "Useful"),
        ("speedup_all", "Speedup"),
        ("vbp", "VBP"),
    ]
    
    lines = [
        "",
        "% LaTeX table format: Metric & Mean ± Std",
        "% Add to your table with: \\input{aggregated_latex.tex}",
        "",
    ]
    
    cells = []
    for metric, _ in key_metrics:
        if metric in aggregated:
            data = aggregated[metric]
            cells.append(f"${data['mean']:.2f} \\pm {data['std']:.2f}$")
        else:
            cells.append("-")
    
    lines.append(" & ".join(cells) + " \\\\")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results from multiple runs")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: experiment_dir/aggregated_results.json)")
    parser.add_argument("--latex", action="store_true", help="Also output LaTeX table format")
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory not found: {experiment_dir}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Aggregating results from: {experiment_dir}")
    print(f"Confidence level: {args.confidence_level * 100:.0f}%")
    print(f"{'='*60}\n")
    
    # Load summaries
    summaries = load_summaries(experiment_dir)
    
    if not summaries:
        print("No summary files found!")
        return 1
    
    # Aggregate
    aggregated = aggregate_summaries(summaries, args.confidence_level)
    
    # Save results
    output_path = Path(args.output) if args.output else experiment_dir / "aggregated_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {output_path}")
    
    # Print formatted results
    print(format_results(aggregated))
    
    # LaTeX output
    if args.latex:
        latex_output = format_latex_table(aggregated)
        print(latex_output)
        
        latex_path = experiment_dir / "aggregated_latex.tex"
        with open(latex_path, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table saved to: {latex_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

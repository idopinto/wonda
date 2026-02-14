"""
Multi-run evaluation script with statistical aggregation.

Runs the evaluation multiple times in fresh subprocesses to ensure no memory
accumulation between runs. Each run saves its summary to an experiment folder,
and results are aggregated with mean ± std and confidence intervals.

Usage:
    uv run -m wonda.eval.run_multi_eval --num_runs=3 [HYDRA_OVERRIDES]
    
Example:
    uv run -m wonda.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config
    uv run -m wonda.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config
    uv run -m wonda.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_oss_20b_config
    uv run -m wonda.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config
    uv run -m wonda.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_5.2_config
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

import configs.global_config as GC
from wonda.eval.system_metadata import collect_system_metadata, extract_override_value
from wonda.eval.aggregate_results import (
    load_summaries,
    aggregate_summaries,
    format_results,
    aggregated_table_rows,
)


def run_single_evaluation(run_id: int, experiment_dir: Path, hydra_args: list[str]) -> Path:
    """
    Run a single evaluation in a fresh subprocess.
    
    Args:
        run_id: The run number (0-indexed).
        experiment_dir: Directory to save results.
        hydra_args: Additional Hydra config overrides.
    
    Returns:
        Path to the saved summary JSON file.
    """
    summary_path = experiment_dir / f"summary_run_{run_id}.json"
    # Save the fully-resolved evaluate Hydra config alongside the experiment
    config_path = experiment_dir / f"evaluate_config_run_{run_id}.yaml"
    
    cmd = [
        "uv", "run", "-m", "wonda.eval.evaluate",
        f"+run_id={run_id}",
        f"+summary_output_path={summary_path}",
        f"+config_output_path={config_path}",
        *hydra_args
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting Run {run_id + 1}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Propagate parent env (incl. WEAVE_PARALLELISM / CUDA_VISIBLE_DEVICES if set) into subprocess.
    result = subprocess.run(cmd, cwd=GC.ROOT_DIR, env=os.environ.copy())
    
    if result.returncode != 0:
        print(f"Warning: Run {run_id + 1} exited with code {result.returncode}")
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Run multiple evaluations and aggregate results")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of evaluation runs")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for experiment folder")
    parser.add_argument("--version", type=str, default=None, help="Version label in folder name (e.g. v0, v1, v2.1, v2.2, v2.3)")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI")
    
    # Parse known args, pass the rest through to Hydra *inside* evaluate
    args, hydra_args = parser.parse_known_args()

    # Ensure a stable default for reproducibility/metadata and propagate into subprocesses.
    # (evaluate.py also sets a default, but that happens in the child process.)
    # Model inference can run in parallel; verifier calls are serialized via semaphore
    # in uautomizer.py (VERIFIER_MAX_CONCURRENT env var, default=1).
    os.environ.setdefault("WEAVE_PARALLELISM", GC.WEAVE_PARALLELISM)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = extract_override_value(hydra_args, "models") or "unknown_model"
    # Strip Hydra config suffix for cleaner folder names (e.g. qwen3_0.6b_nt_config -> qwen3_0.6b_nt)
    model_label = model_name.removesuffix("_config") if model_name else "unknown_model"
    version_part = f"{args.version}_" if args.version else ""
    exp_name = args.experiment_name or f"multi_eval_{model_label}_{version_part}{args.num_runs}_runs_{timestamp}"
    experiment_dir = GC.EXPERIMENTS_DIR / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*80}")
    print(f"Multi-Run Evaluation")
    print(f"{'#'*80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Hydra overrides: {hydra_args}")
    print(f"{'#'*80}\n")
    
    # Save run configuration
    config = {
        "num_runs": args.num_runs,
        "hydra_args": hydra_args,
        "timestamp": timestamp,
        "confidence_level": args.confidence_level,
        "system": collect_system_metadata(hydra_args),
    }
    if args.version is not None:
        config["version"] = args.version
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run evaluations
    for run_id in range(args.num_runs):
        run_single_evaluation(run_id, experiment_dir, hydra_args)
    
    # Load and aggregate results
    print(f"\n{'='*60}")
    print("Loading summaries...")
    print(f"{'='*60}")
    
    summaries = load_summaries(experiment_dir)
    
    if not summaries:
        print("No summaries found!")
        sys.exit(1)
    
    aggregated = aggregate_summaries(summaries, args.confidence_level)
    
    # Save aggregated results
    aggregated_path = experiment_dir / "aggregated_results.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {aggregated_path}")

    # Print formatted results
    formatted_table = format_results(aggregated)
    print(formatted_table)

    # Save the final table to Weave as an artifact (if Weave is enabled)
    _publish_aggregated_table_to_weave(
        aggregated=aggregated,
        formatted_table=formatted_table,
        experiment_name=exp_name,
    )


def _load_weave_config() -> dict | None:
    """Load weave section from eval config. Returns None if file missing or invalid."""
    config_path = GC.ROOT_DIR / "configs" / "eval" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        cfg = OmegaConf.load(config_path)
        weave_cfg = OmegaConf.to_container(cfg.get("weave", {}), resolve=True)
        return weave_cfg if isinstance(weave_cfg, dict) else None
    except Exception:
        return None


def _publish_aggregated_table_to_weave(
    aggregated: dict,
    formatted_table: str,
    experiment_name: str,
) -> None:
    """Publish the multi-run aggregated results table to Weave as an artifact."""
    weave_cfg = _load_weave_config()
    if not weave_cfg or weave_cfg.get("skip_weave", True):
        return
    try:
        import weave
    except ImportError:
        print("Weave not available; skipping artifact publish.")
        return
    entity = weave_cfg.get("entity", "")
    project = weave_cfg.get("project_name", "eval-wonda")
    if not entity or not project:
        return
    try:
        weave.init(f"{entity}/{project}")
    except Exception as e:
        print(f"Could not initialize Weave: {e}. Skipping artifact publish.")
        return
    table_rows = aggregated_table_rows(aggregated)
    if not table_rows:
        return
    try:
        table = weave.Table(table_rows)
        name = f"multi_eval_aggregated_{experiment_name}"
        ref = weave.publish(table, name=name)
        print(f"\nAggregated table published to Weave: {name}")
        # Also publish the raw aggregated dict and formatted text as a single artifact for reference
        summary_artifact = {
            "experiment_name": experiment_name,
            "n_runs": aggregated.get("n_runs"),
            "confidence_level": aggregated.get("confidence_level"),
            "formatted_table": formatted_table,
        }
        weave.publish(summary_artifact, name=f"{name}_summary")
    except Exception as e:
        print(f"Failed to publish Weave artifact: {e}")


if __name__ == "__main__":
    main()

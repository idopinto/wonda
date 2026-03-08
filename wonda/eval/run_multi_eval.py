"""
Multi-run evaluation controller with statistical aggregation.

This module is the Hydra entrypoint for evaluation. It composes config once,
creates per-run config files, and launches one fresh worker process per run.

Usage:
    uv run -m wonda.eval.run_multi_eval [HYDRA_OVERRIDES]

Example:
    uv run -m wonda.eval.run_multi_eval multi_run.num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config
"""

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from wonda.eval.aggregate_results import (
    aggregated_table_rows,
    aggregate_summaries,
    format_results,
    load_summaries,
)
from wonda.eval.system_metadata import collect_system_metadata


def _resolve_experiment_root(path_value: str | None) -> Path:
    root = path_value or "eval_experiments"
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = GC.ROOT_DIR / root_path
    return root_path


def run_single_evaluation(run_id: int, experiment_dir: Path, run_cfg: DictConfig) -> Path:
    """Run a single evaluation in a fresh subprocess."""
    summary_path = experiment_dir / f"summary_run_{run_id}.json"
    config_path = experiment_dir / f"evaluate_config_run_{run_id}.yaml"

    run_cfg = copy.deepcopy(run_cfg)
    run_cfg.run_id = run_id
    run_cfg.summary_output_path = str(summary_path)
    run_cfg.config_output_path = str(config_path)
    OmegaConf.save(run_cfg, config_path, resolve=True)

    cmd = [
        sys.executable, "-m", "wonda.eval.evaluate",
        "--config", str(config_path),
    ]

    print(f"\n{'=' * 60}")
    print(f"Starting Run {run_id + 1}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    # Propagate parent env (incl. WEAVE_PARALLELISM / CUDA_VISIBLE_DEVICES if set) into subprocess.
    result = subprocess.run(cmd, cwd=GC.ROOT_DIR, env=os.environ.copy())

    if result.returncode != 0:
        print(f"Warning: Run {run_id + 1} exited with code {result.returncode}")

    return summary_path

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # Ensure a stable default for reproducibility/metadata and propagate into subprocesses.
    os.environ.setdefault("WEAVE_PARALLELISM", GC.WEAVE_PARALLELISM)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = cfg.models.get("label", "unknown_model")
    version_suffix = ""
    if cfg.models.get("eval_ft_model", False):
        sft_version = cfg.models.ft_model.get("sft_version")
        if sft_version:
            version_suffix = f"_{sft_version}"
    exp_name = cfg.multi_run.get("experiment_name") or f"me_{model_label}{version_suffix}_{cfg.multi_run.num_runs}_runs_{timestamp}"
    experiment_root = _resolve_experiment_root(cfg.multi_run.get("experiment_root_dir"))
    project_name = cfg.weave.get("project_name", "eval-wonda")
    project_dir_name = f"{project_name}-test" if cfg.weave.get("test_mode", False) else project_name
    experiment_dir = experiment_root / project_dir_name / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 80}")
    print("Multi-Run Evaluation")
    print(f"{'#' * 80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Number of runs: {cfg.multi_run.num_runs}")
    print(f"Hydra overrides: {sys.argv[1:]}")
    print(f"{'#' * 80}\n")

    run_cfg_dict = OmegaConf.to_container(copy.deepcopy(cfg), resolve=False)
    if isinstance(run_cfg_dict, dict):
        run_cfg_dict.pop("multi_run", None)
    run_cfg = OmegaConf.create(run_cfg_dict)
    run_cfg.summary_output_dir = str(experiment_dir)
    run_cfg.config_output_path = None
    run_cfg.summary_output_path = None
    run_cfg.run_id = None

    config = {
        "num_runs": cfg.multi_run.num_runs,
        "hydra_args": sys.argv[1:],
        "timestamp": timestamp,
        "confidence_level": cfg.multi_run.confidence_level,
        "system": collect_system_metadata(sys.argv[1:]),
        "weave_project_name": project_name,
        "experiment_project_dir": project_dir_name,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for run_id in range(cfg.multi_run.num_runs):
        run_single_evaluation(run_id, experiment_dir, run_cfg)

    print(f"\n{'=' * 60}")
    print("Loading summaries...")
    print(f"{'=' * 60}")

    summaries = load_summaries(experiment_dir)
    if not summaries:
        print("No summaries found!")
        sys.exit(1)

    aggregated = aggregate_summaries(summaries, cfg.multi_run.confidence_level)
    aggregated_path = experiment_dir / "aggregated_results.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {aggregated_path}")

    formatted_table = format_results(aggregated)
    print(formatted_table)
    formatted_table_path = experiment_dir / "aggregated_results.txt"
    formatted_table_path.write_text(formatted_table + "\n")
    print(f"Formatted results saved to: {formatted_table_path}")

if __name__ == "__main__":
    main()

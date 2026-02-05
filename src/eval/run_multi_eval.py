"""
Multi-run evaluation script with statistical aggregation.

Runs the evaluation multiple times in fresh subprocesses to ensure no memory
accumulation between runs. Each run saves its summary to an experiment folder,
and results are aggregated with mean ± std and confidence intervals.

Usage:
    uv run -m src.eval.run_multi_eval --num_runs=3 [HYDRA_OVERRIDES]
    
Example:
    uv run -m src.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config
    uv run -m src.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config
    uv run -m src.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_oss_20b_config
    uv run -m src.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config
    uv run -m src.eval.run_multi_eval --num_runs=3 dataset.limit=-1 models=gpt_5.2_config
"""

import argparse
import json
import os
import subprocess
import sys
import platform
import socket
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import configs.global_config as GC


def _safe_run(cmd: list[str]) -> str | None:
    """Run a command and return stdout (stripped) or None on failure."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        out = (completed.stdout or "").strip()
        return out if out else None
    except Exception:
        return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except Exception:
        return None


def _parse_cpu_model_from_proc_cpuinfo() -> str | None:
    txt = _read_text("/proc/cpuinfo")
    if not txt:
        return None
    # Common fields: "model name" (x86), "Hardware" (ARM)
    m = re.search(r"^model name\s*:\s*(.+)$", txt, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^Hardware\s*:\s*(.+)$", txt, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def _mem_total_gb_from_proc_meminfo() -> float | None:
    txt = _read_text("/proc/meminfo")
    if not txt:
        return None
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB$", txt, flags=re.MULTILINE)
    if not m:
        return None
    kb = int(m.group(1))
    return kb / (1024 * 1024)


def _nvidia_smi_gpus() -> list[dict]:
    """
    Return GPU metadata via nvidia-smi if available.
    Example fields: name, vram_total_mb, driver_version.
    """
    q = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    out = _safe_run(q)
    if not out:
        return []
    gpus: list[dict] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem_mb, driver = parts[0], parts[1], parts[2]
        try:
            mem_mb_f = float(mem_mb)
        except Exception:
            mem_mb_f = None
        gpus.append(
            {
                "name": name,
                "vram_total_mb": mem_mb_f,
                "driver_version": driver,
            }
        )
    return gpus


def _extract_override_value(overrides: list[str], key: str) -> str | None:
    """Extract 'key=value' from Hydra override strings (best-effort)."""
    prefix = f"{key}="
    for o in overrides:
        if o.startswith(prefix):
            return o[len(prefix) :]
    return None


def collect_system_metadata(hydra_overrides: list[str]) -> dict:
    """Collect host/system metadata for reproducibility (best-effort, no hard deps)."""
    cpu_model = _parse_cpu_model_from_proc_cpuinfo()
    mem_gb = _mem_total_gb_from_proc_meminfo()
    meta = {
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": cpu_model,
            "cores_logical": os.cpu_count(),
        },
        "memory": {
            "ram_total_gb": mem_gb,
        },
        "gpu": {
            "nvidia": _nvidia_smi_gpus(),
        },
        "env": {
            "weave_parallelism": os.environ.get("WEAVE_PARALLELISM"),
        },
    }

    # Optional: physical core count if psutil exists
    try:
        import psutil  # type: ignore

        meta["cpu"]["cores_physical"] = psutil.cpu_count(logical=False)
        meta["memory"]["ram_available_gb"] = psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    return meta


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
    config_path = experiment_dir / "evaluate_config.yaml"
    
    cmd = [
        "uv", "run", "-m", "src.eval.evaluate",
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
        # - Rates: [0, 1]
        # - Counts: [0, max observed]
        # - Timings / speedups: >= 0
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

def main():
    parser = argparse.ArgumentParser(description="Run multiple evaluations and aggregate results")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of evaluation runs")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name for experiment folder")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI")
    
    # Parse known args, pass the rest through to Hydra *inside* evaluate
    args, hydra_args = parser.parse_known_args()

    # Ensure a stable default for reproducibility/metadata and propagate into subprocesses.
    # (evaluate.py also sets a default, but that happens in the child process.)
    # Model inference can run in parallel; verifier calls are serialized via semaphore
    # in uautomizer_runlim.py (VERIFIER_MAX_CONCURRENT env var, default=1).
    os.environ.setdefault("WEAVE_PARALLELISM", GC.WEAVE_PARALLELISM)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or f"multi_eval_{args.num_runs}_{timestamp}"
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
    print(format_results(aggregated))


if __name__ == "__main__":
    main()

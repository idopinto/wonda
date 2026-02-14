"""
Single entry point for evaluation (loop invariant generation with VBS/VBP metrics).
num_runs=1: one in-process eval, summary to file. num_runs>=2: N subprocess evals, then aggregate + optional W&B artifact.
Version in paths: from models.ft_model.sft_version when models.eval_ft_model=true.

  uv run -m wonda.eval.evaluate [HYDRA_OVERRIDES]
  uv run -m wonda.eval.evaluate dataset.limit=5 models=qwen3_0.6b_nt_config
  uv run -m wonda.eval.evaluate multi_run.num_runs=2 dataset.limit=5 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version=v2.1
"""

import asyncio
import functools
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import weave
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from weave import Evaluation

import configs.global_config as GC
from wonda.eval.aggregate_results import (
    load_summaries,
    aggregate_summaries,
    format_results,
    aggregated_table_rows,
)
from wonda.eval.eval_data import get_evaluation_dataset, preprocess_for_model
from wonda.eval.models.model_factory import ModelFactory
from wonda.eval.inv_gen_scorer import InvGenScorer
from wonda.eval.system_metadata import collect_system_metadata, extract_override_value
from wonda.verifiers.uautomizer import UAutomizerVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
os.environ.setdefault("WEAVE_PARALLELISM", os.environ.get("WEAVE_PARALLELISM", GC.WEAVE_PARALLELISM))
logger.info(f"WEAVE_PARALLELISM={os.environ['WEAVE_PARALLELISM']}")


def setup_prompts(prompts_config: dict, eval_per_marker: bool = True) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    sk = "per_marker_system_prompt" if eval_per_marker else "default_system_prompt"
    uk = "per_marker_user_prompt_template" if eval_per_marker else "default_user_prompt_template"
    return weave.StringPrompt(prompts_config[sk]), weave.StringPrompt(prompts_config[uk])


def run_single_eval(cfg: DictConfig) -> dict:
    """One evaluation run; returns summary (caller saves to file)."""
    if cfg.get("config_output_path"):
        p = Path(str(cfg.config_output_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(OmegaConf.to_yaml(cfg) if p.suffix in {".yaml", ".yml"} else json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
        logger.info(f"Saved config to {p}")

    if not cfg.weave.get("skip_weave", False):
        try:
            proj = f"{cfg.weave.entity}/{cfg.weave.project_name}" + ("-test" if cfg.weave.test_mode else "")
            weave.init(proj)
        except Exception as e:
            logger.warning(f"Weave init failed: {e}. Continuing without.")
    else:
        logger.info("Weave disabled (skip_weave=true)")

    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name, limit=cfg.dataset.limit, prefix=cfg.dataset.prefix,
        split=cfg.dataset.split, re_split=cfg.dataset.re_split,
        difficulty_threshold=cfg.dataset.difficulty_threshold, eval_per_marker=cfg.eval_per_marker,
    )
    system_prompt, user_prompt_template = setup_prompts(cfg.prompts, cfg.eval_per_marker)
    model = ModelFactory.create(cfg.models, system_prompt, user_prompt_template)
    run_id = cfg.get("run_id")
    display_name = f"{model.get_run_name()}-run{run_id}" if run_id is not None else model.get_run_name()
    logger.info(f"Weave display name: {display_name}")

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
        arch=cfg.scorer.verifier.arch, timeout_seconds=cfg.scorer.verifier.timeout_seconds,
        version=cfg.scorer.verifier.version, memory_limit_mb=GC.MEMORY_LIMIT_MB,
    )
    scorer = InvGenScorer(verifier=verifier, baseline_is_timeout=cfg.scorer.baseline_is_timeout, model_name=model.get_display_name())
    evaluation = Evaluation(dataset=dataset, scorers=[scorer], preprocess_model_input=functools.partial(preprocess_for_model, eval_per_marker=cfg.eval_per_marker))

    logger.info(f"Dataset size: {len(dataset)}")
    t0 = time.time()
    summary = asyncio.run(evaluation.evaluate(model, __weave={"display_name": display_name}))
    logger.info(f"Evaluation done in {time.time() - t0:.2f}s ({ (time.time() - t0) / len(dataset):.2f}s/example)")
    return summary


def _model_label(cfg: DictConfig, hydra_args: list[str]) -> str:
    name = extract_override_value(hydra_args, "models") or OmegaConf.select(cfg, "hydra.runtime.choices.models", default=None) or "default"
    return (name.removesuffix("_config") if name else "unknown").replace("/", "_")


def _version_label(cfg: DictConfig) -> str | None:
    if not OmegaConf.select(cfg, "models.eval_ft_model", default=False):
        return None
    v = OmegaConf.select(cfg, "models.ft_model.sft_version", default=None)
    return str(v) if v is not None else None


def _run_one_subprocess(run_id: int, experiment_dir: Path, hydra_args: list[str]) -> Path:
    summary_path = experiment_dir / f"summary_run_{run_id}.json"
    config_path = experiment_dir / f"evaluate_config_run_{run_id}.yaml"
    overrides = hydra_args + ["multi_run.num_runs=1", f"+run_id={run_id}", f"+summary_output_path={summary_path}", f"+config_output_path={config_path}"]
    cmd = ["uv", "run", "-m", "wonda.eval.evaluate", *overrides]
    print(f"\n{'='*60}\nStarting Run {run_id + 1}\nCommand: {' '.join(cmd)}\n{'='*60}\n")
    r = subprocess.run(cmd, cwd=GC.ROOT_DIR, env=os.environ.copy())
    if r.returncode != 0:
        print(f"Warning: Run {run_id + 1} exited with code {r.returncode}")
    return summary_path


def _publish_table_artifact(aggregated: dict, experiment_name: str) -> None:
    path = GC.ROOT_DIR / "configs" / "eval" / "config.yaml"
    if not path.exists():
        return
    try:
        w = OmegaConf.to_container(OmegaConf.load(path).get("weave", {}), resolve=True)
    except Exception:
        return
    if not isinstance(w, dict) or w.get("skip_weave", True) or not w.get("entity") or not w.get("project_name"):
        return
    rows = aggregated_table_rows(aggregated)
    if not rows:
        return
    try:
        import wandb
    except ImportError:
        print("wandb not available; skipping artifact.")
        return
    project = w.get("project_name", "eval-wonda") + "-tables-artifacts"
    run = None
    try:
        run = wandb.init(project=project, entity=w.get("entity"), job_type="multi_eval", name=experiment_name)
        tbl = wandb.Table(columns=["metric", "mean", "std", "ci_low", "ci_high"], data=[[r["metric"], r["mean"], r["std"], r["ci_low"], r["ci_high"]] for r in rows])
        art = wandb.Artifact(name=f"multi_eval_aggregated_{experiment_name}".replace(" ", "_"), type="eval_results", metadata={"experiment_name": experiment_name, "n_runs": aggregated.get("n_runs"), "confidence_level": aggregated.get("confidence_level")})
        art.add(tbl, "aggregated_metrics")
        run.log_artifact(art)
        print(f"\nAggregated table logged as W&B artifact: {art.name}")
    except Exception as e:
        print(f"Failed to log W&B artifact: {e}")
    finally:
        if run is not None:
            run.finish()


# Overrides from CLI; set in __main__ before Hydra-decorated main() runs so multi-run subprocesses get the same args.
_HYDRA_OVERRIDES: list[str] = []


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_args = _HYDRA_OVERRIDES or []
    os.environ.setdefault("WEAVE_PARALLELISM", GC.WEAVE_PARALLELISM)

    num_runs = int(cfg.get("multi_run", {}).get("num_runs", 1) or 1)
    model_label = _model_label(cfg, hydra_args)
    version = _version_label(cfg)

    if num_runs == 1:
        summary_path = cfg.get("summary_output_path")
        if not summary_path:
            out_dir = Path(cfg.get("summary_output_dir", str(GC.EXPERIMENTS_DIR)))
            ver = f"-{version}" if version else ""
            summary_path = out_dir / f"{cfg.weave.project_name}-{model_label}{ver}-{time.strftime('%Y%m%d_%H%M%S')}.json"
            logger.info(f"Summary path: {summary_path}")
        print("=" * 100)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 100)
        summary = run_single_eval(cfg)
        p = Path(str(summary_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {p}")
        return

    # Multi-run
    multi = cfg.get("multi_run", {})
    confidence = float(multi.get("confidence_level", 0.95) or 0.95)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver_part = f"{version}_" if version else ""
    exp_name = multi.get("experiment_name") or getattr(multi, "experiment_name", None) or f"multi_eval_{model_label}_{ver_part}{num_runs}_runs_{ts}"
    exp_dir = GC.EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}\nMulti-Run Evaluation\n{'#'*80}\nExperiment: {exp_dir}\nRuns: {num_runs}\nOverrides: {hydra_args}\n{'#'*80}\n")

    run_config = {"num_runs": num_runs, "hydra_args": hydra_args, "timestamp": ts, "confidence_level": confidence, "system": collect_system_metadata(hydra_args)}
    if version is not None:
        run_config["version"] = version
    (exp_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    for i in range(num_runs):
        _run_one_subprocess(i, exp_dir, hydra_args)

    summaries = load_summaries(exp_dir)
    if not summaries:
        print("No summaries found!")
        sys.exit(1)
    aggregated = aggregate_summaries(summaries, confidence)
    out = exp_dir / "aggregated_results.json"
    out.write_text(json.dumps(aggregated, indent=2))
    print(f"\nAggregated results: {out}\n{format_results(aggregated)}")
    _publish_table_artifact(aggregated, exp_name)


if __name__ == "__main__":
    _HYDRA_OVERRIDES[:] = sys.argv[1:] or []
    logger.info("Starting evaluation...")
    main()

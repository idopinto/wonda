"""
Single-run evaluation worker (loop invariant generation with VBS/VBP metrics).

This module intentionally does not use Hydra. It accepts a fully-resolved config
file produced by the multi-run controller and executes exactly one run.

Usage:
  uv run -m wonda.eval.evaluate --config /abs/path/to/evaluate_config_run_0.yaml
"""

import asyncio
import argparse
import functools
import json
import logging
import os
import time
from pathlib import Path

import weave
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from weave import Evaluation

import configs.global_config as GC
from wonda.eval.eval_data import get_evaluation_dataset, preprocess_for_model
from wonda.eval.models.model_factory import ModelFactory
from wonda.eval.inv_gen_scorer import InvGenScorer
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
    logger.info(f"Evaluation done in {time.time() - t0:.2f}s ({(time.time() - t0) / len(dataset):.2f}s/example)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single evaluation from a resolved config file")
    parser.add_argument("--config", required=True, help="Path to resolved evaluation config YAML/JSON")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    model_label = cfg.models.get("label", "default")
    version = str(cfg.models.ft_model.sft_version) if cfg.models.eval_ft_model else None
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


if __name__ == "__main__":
    logger.info("Starting evaluation...")
    main()

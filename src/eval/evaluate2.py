"""
Evaluation script for loop invariant generation with VBS (Virtual Best Solver) metrics.

This module orchestrates the evaluation pipeline with focus on VBS metrics:
1. Load configuration via Hydra (../../configs/eval/config.yaml)
2. Initialize Weave for experiment tracking
3. Load and preprocess the evaluation dataset
4. Load the model (base or fine-tuned with LoRA)
5. Setup prompts and create the InvariantGeneratorModel
6. Run evaluation using Weave's Evaluation framework with UAutomizer verification
7. Collect results and generate visualization plots
8. Compute VBS/VBP metrics (Virtual Best Solver / Virtual Best Performance)

Key Metrics:
- vbs: min(verification_time, baseline) when useful, else baseline
- vbs_e2e: min(verification_time + model_latency, baseline) when useful, else baseline
- vbp: average of vbs across all examples (Virtual Best Performance)
- speedup: baseline / verification_time (when useful)
- speedup_e2e: baseline / (verification_time + model_latency) (when useful)
"""

import asyncio
import functools
import json
import logging
import os
import time
from pathlib import Path

import hydra
import weave
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from weave import Evaluation

import configs.global_config as GC
from src.eval.models.model_factory import ModelFactory
from src.eval.preprocess_eval_data import (get_evaluation_dataset,
                                           preprocess_for_model)
from src.eval.inv_gen_scorer import InvGenScorer
from src.verifiers.uautomizer_runlim import UAutomizerVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Set Weave parallelism (must be set before weave.init() or any weave operations)
# Weave reads this at import/init time, so it must be set early
# Model inference can run in parallel; verifier calls are serialized via semaphore
# in uautomizer_runlim.py (VERIFIER_MAX_CONCURRENT env var, default=1).
weave_parallelism = os.environ.get("WEAVE_PARALLELISM", "8")
os.environ["WEAVE_PARALLELISM"] = weave_parallelism
logger.info(f"WEAVE_PARALLELISM set to: {os.environ['WEAVE_PARALLELISM']}")

# Verify it's actually set (check before any weave imports are used)
if "WEAVE_PARALLELISM" not in os.environ:
    logger.warning("WARNING: WEAVE_PARALLELISM not found in environment!")
else:
    logger.info(f"✓ Verified WEAVE_PARALLELISM={os.environ['WEAVE_PARALLELISM']} is set")

# To verify parallelism is actually working:
# 1. Check logs for concurrent execution (multiple items processing simultaneously)
# 2. Compare timing: parallel should be ~N times faster than sequential for I/O-bound tasks
# 3. Run: uv run -m src.eval.verify_parallelism (test script)
# 4. Monitor CPU usage: should see multiple processes/threads active

def setup_prompts(
    eval_per_marker: bool,
    prompts_config: dict,
) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    """
    Setup prompts based on configuration.

    Args:
        eval_per_marker: Whether to evaluate per marker.
        prompts_config: Dictionary containing prompts settings.
    
    Returns:
        Tuple of (system_prompt, user_prompt_template).
    """
    system_key = (
        "per_marker_system_prompt"
        if eval_per_marker
        else "default_system_prompt"
    )
    user_key = (
        "per_marker_user_prompt_template"
        if eval_per_marker
        else "default_user_prompt_template"
    )
    return weave.StringPrompt(
        prompts_config[system_key]
    ), weave.StringPrompt(prompts_config[user_key])

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config_pm")
def main(cfg: DictConfig):
    """Main evaluation entry point with VBS metrics."""
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    
    # Get output path for saving summary (optional)
    summary_output_path = cfg.get("summary_output_path", None)
    # Optional: save the fully-resolved Hydra config for reproducibility
    config_output_path = cfg.get("config_output_path", None)
    # Optional: identify this run when doing multi-run evaluation (for Weave UI)
    run_id = cfg.get("run_id", None)
    
    # Auto-generate summary output path if not provided
    if summary_output_path is None:
        summary_output_dir = Path(cfg.get("summary_output_dir", str(GC.EXPERIMENTS_DIR)))
        summary_output_path = summary_output_dir / f"{cfg.weave.project_name}-{time.strftime('%Y%m%d_%H%M%S')}.json"
        logger.info(f"Auto-generated summary output path: {summary_output_path}")

    # Save resolved config early (so we have it even if eval crashes)
    if config_output_path:
        out_path = Path(str(config_output_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        # If user gave a YAML path, write YAML; otherwise write JSON.
        if out_path.suffix in {".yaml", ".yml"}:
            out_path.write_text(OmegaConf.to_yaml(cfg))
        else:
            out_path.write_text(json.dumps(resolved, indent=2))
        logger.info(f"Saved resolved config to: {out_path}")
    
    # Initialize Weave only if not skipped and if connection is available
    skip_wandb = cfg.weave.get("skip_wandb", False)
    if not skip_wandb:
        try:
            client = weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Weave (connection timeout or other error): {e}")
            logger.warning("Continuing without Weave logging...")
            skip_wandb = True
    else:
        logger.info("Weave/W&B logging is disabled (skip_wandb=true)")

    eval_per_marker = cfg.eval_per_marker
    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split,
        re_split=cfg.dataset.re_split,
        difficulty_threshold=cfg.dataset.difficulty_threshold,
        eval_per_marker=eval_per_marker,
    )
    
    system_prompt, user_prompt_template = setup_prompts(eval_per_marker, cfg.prompts)
    invariant_generator = ModelFactory.create(
        cfg.models, system_prompt, user_prompt_template
    )
    logger.info(f"Run name: {invariant_generator.get_run_name()}")
    logger.info(f"Display name: {invariant_generator.get_display_name()}")
    logger.info(f"Evaluating per marker: {eval_per_marker}")

    # Make Weave UI names unique per trial when running multiple evaluations
    weave_display_name = invariant_generator.get_run_name()
    if run_id is not None:
        weave_display_name = f"{weave_display_name}-run{run_id}"
    logger.info(f"Weave display name: {weave_display_name}")
    
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
        arch=cfg.scorer.verifier.arch,
        timeout_seconds=cfg.scorer.verifier.timeout_seconds,
        version=cfg.scorer.verifier.version,
    )
    
    scorer = InvGenScorer(
        verifier=verifier,
        baseline_is_timeout=cfg.scorer.baseline_is_timeout,
    )

    preprocess_hook = functools.partial(
        preprocess_for_model, eval_per_marker=eval_per_marker
    )
    
    evaluation = Evaluation(
        dataset=dataset, scorers=[scorer], preprocess_model_input=preprocess_hook
    )

    # Verify parallelism one more time right before evaluation
    actual_parallelism = os.environ.get("WEAVE_PARALLELISM", "not set")
    logger.info(f"Starting evaluation with WEAVE_PARALLELISM={actual_parallelism}")
    logger.info(f"Dataset size: {len(dataset)} examples")
    
    # Time the evaluation to verify parallelism (parallel should be faster)
    eval_start_time = time.time()
    
    # Run evaluation and capture the summary
    summary = asyncio.run(
        evaluation.evaluate(
            invariant_generator, 
            __weave={"display_name": weave_display_name},
        )
    )
    
    eval_duration = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")
    logger.info(f"Average time per example: {eval_duration / len(dataset):.2f} seconds")
    # Save summary to file
    if summary_output_path:
        output_path = Path(summary_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {output_path}")
   
if __name__ == "__main__":
    logger.info("Starting VBS evaluation...")
    main()

# Example usage:
# uv run -m src.eval.evaluate2 models=qwen3_0.6b_nt_config
# uv run -m src.eval.evaluate_vbs models=qwen_config dataset.limit=1
# uv run -m src.eval.evaluate_vbs models=qwen_no_think_config dataset.limit=1


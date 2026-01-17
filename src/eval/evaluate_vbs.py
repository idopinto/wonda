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

import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from src.eval.models.model_factory import ModelFactory
from src.eval.preprocess_eval_data import (
    get_evaluation_dataset,
    preprocess_for_model,
)
from src.eval.scorer import InvariantScorer, ResultCollector
from src.eval.statistics import (
    aggregate_trial_results,
    log_trial_summary,
    log_aggregated_summary,
    METRICS_TO_AGGREGATE,
)
from src.verifiers.uautomizer import UAutomizerVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables
load_dotenv()
# os.environ["WEAVE_PARALLELISM"] = "1"


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


def save_summaries_incremental(
    all_trial_summaries: list[dict],
    run_name: str,
    stats_dir: Path,
    trial_num: int = None,
    is_final: bool = False
) -> Path:
    """
    Save summaries incrementally after each trial.
    
    Args:
        all_trial_summaries: List of all summaries collected so far
        run_name: Name of the evaluation run
        stats_dir: Directory to save stats files
        trial_num: Current trial number (None for single trial)
        is_final: Whether this is the final save (includes aggregation if multiple trials)
    
    Returns:
        Path to the saved file
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if trial_num is not None:
        filename = f"vbs_stats_{run_name}_trial{trial_num}_{timestamp}.json"
    else:
        filename = f"vbs_stats_{run_name}_{timestamp}.json"
    
    stats_path = stats_dir / filename
    
    stats_data = {
        "metadata": {
            "run_name": run_name,
            "trial_num": trial_num,
            "is_final": is_final,
            "num_summaries": len(all_trial_summaries),
            "timestamp": timestamp,
        },
        "raw_trial_summaries": all_trial_summaries,
    }
    
    # If this is final and we have multiple summaries, include aggregation
    if is_final and len(all_trial_summaries) > 1:
        from src.eval.statistics import aggregate_trial_results
        aggregated = aggregate_trial_results(all_trial_summaries)
        stats_data["statistics"] = {}
        for metric_name, metric_key in METRICS_TO_AGGREGATE:
            stats_data["statistics"][metric_key] = {
                "name": metric_name,
                "mean": float(aggregated.get(f"{metric_key}_mean", 0.0)),
                "std": float(aggregated.get(f"{metric_key}_std", 0.0)),
                "ci_lower": float(aggregated.get(f"{metric_key}_ci_lower", 0.0)),
                "ci_upper": float(aggregated.get(f"{metric_key}_ci_upper", 0.0)),
                "min": float(aggregated.get(f"{metric_key}_min", 0.0)),
                "max": float(aggregated.get(f"{metric_key}_max", 0.0)),
            }
        stats_data["metadata"]["num_trials"] = aggregated["num_trials"]
    
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    
    return stats_path


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config_l40s")
def main(cfg: DictConfig):
    """Main evaluation entry point with VBS metrics."""
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    
    # Initialize Weave only if not skipped and if connection is available
    skip_wandb = cfg.weave.get("skip_wandb", False)
    if not skip_wandb:
        try:
            weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Weave (connection timeout or other error): {e}")
            logger.warning("Continuing without Weave logging...")
            skip_wandb = True
    else:
        logger.info("Weave/W&B logging is disabled (skip_wandb=true)")
    
    dataset = get_evaluation_dataset(
        dataset_name=cfg.dataset.name,
        limit=cfg.dataset.limit,
        prefix=cfg.dataset.prefix,
        split=cfg.dataset.split,
        re_split=cfg.dataset.re_split,
        difficulty_threshold=cfg.dataset.difficulty_threshold,
        eval_per_marker=cfg.eval_per_marker,
    )
    
    system_prompt, user_prompt_template = setup_prompts(cfg.eval_per_marker, cfg.prompts)
    eval_ft_model = cfg.models.get("eval_ft_model", False)
    invariant_generator = ModelFactory.create(
        cfg.models, system_prompt, user_prompt_template, eval_ft_model=eval_ft_model
    )
    
    logger.info(f"Run name: {invariant_generator.get_run_name()}")
    logger.info(f"Display name: {invariant_generator.get_display_name()}")
    logger.info(f"Evaluating fine-tuned model: {eval_ft_model}")
    
    result_collector = ResultCollector()
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.scorer.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.scorer.verifier.property,
        arch=cfg.scorer.verifier.arch,
        timeout_seconds=cfg.scorer.verifier.timeout_seconds,
        version=cfg.scorer.verifier.version,
    )
    
    scorer = InvariantScorer(
        verifier=verifier,
        collector=result_collector,
        baseline_is_timeout=cfg.scorer.baseline_is_timeout,
    )

    preprocess_hook = functools.partial(
        preprocess_for_model, eval_per_marker=cfg.eval_per_marker
    )

    # Get number of trials from config (default to 1)
    num_trials = cfg.statistics.get("num_trials", 1)
    confidence_level = cfg.statistics.get("confidence_level", 0.95)
    
    if num_trials > 1:
        logger.info(f"Running {num_trials} trials for statistical significance (confidence level: {confidence_level})")
    
    all_trial_summaries = []
    all_trial_results = []
    
    # Run multiple trials
    for trial_num in range(1, num_trials + 1):
        logger.info(f"{'='*80}")
        logger.info(f"Trial {trial_num}/{num_trials}")
        logger.info(f"{'='*80}")
        
        # Create a fresh collector for each trial (or reuse original for single trial)
        if num_trials == 1:
            trial_collector = result_collector
            trial_scorer = scorer
        else:
            trial_collector = ResultCollector()
            trial_scorer = InvariantScorer(
                verifier=verifier,
                collector=trial_collector,
                baseline_is_timeout=cfg.scorer.baseline_is_timeout,
            )
        
        evaluation = Evaluation(
            dataset=dataset, scorers=[trial_scorer], preprocess_model_input=preprocess_hook
        )
        
        # Run evaluation
        display_name = (
            f"{invariant_generator.get_run_name()}_trial{trial_num}" 
            if num_trials > 1 
            else invariant_generator.get_run_name()
        )
        asyncio.run(
            evaluation.evaluate(
                invariant_generator, 
                __weave={"display_name": display_name}
            )
        )
        
        # Get results for this trial
        trial_plot_results = trial_collector.get_results()
        all_trial_results.append(trial_plot_results)
        logger.info(f"Trial {trial_num}: Collected {len(trial_plot_results)} results")
        
        # Compute summary for this trial
        if trial_plot_results:
            trial_summary = trial_scorer.summarize(trial_plot_results)
            all_trial_summaries.append(trial_summary)
            log_trial_summary(trial_summary, trial_num if num_trials > 1 else 0)
            
            # Save summaries incrementally after each trial
            stats_dir = GC.EXPERIMENTS_DIR / "eval_stats"
            run_name = invariant_generator.get_run_name()
            is_final = (trial_num == num_trials)
            saved_path = save_summaries_incremental(
                all_trial_summaries,
                run_name,
                stats_dir,
                trial_num=trial_num if num_trials > 1 else None,
                is_final=is_final
            )
            logger.info(f"Saved summaries after trial {trial_num} to: {saved_path}")
    
    # Use last trial's results for any additional processing
    plot_results = all_trial_results[-1] if all_trial_results else []
    total_results = sum(len(r) for r in all_trial_results)
    logger.info(f"Total collected {total_results} results across {num_trials} trial(s)")
    
    # If multiple trials, aggregate statistics
    if num_trials > 1 and all_trial_summaries:
        aggregated = aggregate_trial_results(all_trial_summaries)
        log_aggregated_summary(aggregated)
        
        # Save statistical results to file
        stats_dir = GC.EXPERIMENTS_DIR / "eval_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = invariant_generator.get_run_name()
        stats_filename = f"vbs_stats_{run_name}_{timestamp}.json"
        stats_path = stats_dir / stats_filename
        
        # Prepare results dictionary with metadata
        stats_data = {
            "metadata": {
                "run_name": run_name,
                "display_name": invariant_generator.get_display_name(),
                "model_config": (
                    cfg.models.get("ft_model", {}).get("id", "unknown")
                    if cfg.models.get("eval_ft_model", False)
                    else cfg.models.get("base_model", {}).get("id", "unknown")
                ),
                "num_trials": aggregated["num_trials"],
                "confidence_level": confidence_level,
                "timestamp": timestamp,
                "dataset": {
                    "name": cfg.dataset.name,
                    "split": cfg.dataset.split,
                    "limit": cfg.dataset.limit,
                },
            },
            "statistics": {},
            "raw_trial_summaries": all_trial_summaries,
        }
        
        # Add all metrics with their statistics
        for metric_name, metric_key in METRICS_TO_AGGREGATE:
            stats_data["statistics"][metric_key] = {
                "name": metric_name,
                "mean": float(aggregated.get(f"{metric_key}_mean", 0.0)),
                "std": float(aggregated.get(f"{metric_key}_std", 0.0)),
                "ci_lower": float(aggregated.get(f"{metric_key}_ci_lower", 0.0)),
                "ci_upper": float(aggregated.get(f"{metric_key}_ci_upper", 0.0)),
                "min": float(aggregated.get(f"{metric_key}_min", 0.0)),
                "max": float(aggregated.get(f"{metric_key}_max", 0.0)),
            }
        
        # Save to JSON file
        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Statistical results saved to: {stats_path}")
    
    # Single trial summary save
    elif len(all_trial_summaries) == 1:
        stats_dir = GC.EXPERIMENTS_DIR / "eval_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = invariant_generator.get_run_name()
        stats_filename = f"vbs_stats_{run_name}_{timestamp}.json"
        stats_path = stats_dir / stats_filename
        
        summary = all_trial_summaries[0]
        stats_data = {
            "metadata": {
                "run_name": run_name,
                "display_name": invariant_generator.get_display_name(),
                "model_config": (
                    cfg.models.get("ft_model", {}).get("id", "unknown")
                    if cfg.models.get("eval_ft_model", False)
                    else cfg.models.get("base_model", {}).get("id", "unknown")
                ),
                "num_trials": 1,
                "timestamp": timestamp,
                "dataset": {
                    "name": cfg.dataset.name,
                    "split": cfg.dataset.split,
                    "limit": cfg.dataset.limit,
                },
            },
            "summary": summary,
        }
        
        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Results saved to: {stats_path}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    logger.info("Starting VBS evaluation...")
    main()

# Example usage:
# uv run -m src.eval.evaluate_vbs
# uv run -m src.eval.evaluate_vbs models=qwen_config dataset.limit=1
# uv run -m src.eval.evaluate_vbs models=qwen_no_think_config dataset.limit=1

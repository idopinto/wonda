"""
Main evaluation script for loop invariant generation.

This module orchestrates the evaluation pipeline:
1. Load configuration via Hydra (../../configs/eval/config.yaml)
2. Initialize Weave for experiment tracking
3. Load and preprocess the evaluation dataset
4. Load the model (base or fine-tuned with LoRA)
5. Setup prompts and create the InvariantGeneratorModel
6. Run evaluation using Weave's Evaluation framework with UAutomizer verification
7. Collect results and generate visualization plots - only if there are results to plot
8. Optionally upload plots to Weights & Biases - only if the user wants to upload them
"""

import asyncio
import functools
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any
from src.eval.models.model_factory import ModelFactory
import weave
from dotenv import load_dotenv
from weave import Evaluation
import hydra
from omegaconf import DictConfig, OmegaConf

import configs.global_config as GC
from src.eval.preprocess_eval_data import (
    get_evaluation_dataset,
    preprocess_for_model,
)
from src.eval.scorer import InvariantScorer, ResultCollector
from src.verifiers.uautomizer import UAutomizerVerifier
from src.eval.plot import plot_verification_vs_baseline
from src.eval.wandb_utils import upload_plot_to_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables
load_dotenv()


def setup_prompts(
    eval_per_marker: bool,
    prompts_config: dict,
) -> tuple[weave.StringPrompt, weave.StringPrompt]:
    """
    Setup prompts based on configuration.

    Args:
        prompts_config: Dictionary containing prompts settings.
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


def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Calculate confidence interval using t-distribution.
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    
    mean = statistics.mean(values)
    if len(values) == 1:
        return (mean, mean)
    
    std_err = statistics.stdev(values) / (len(values) ** 0.5)
    
    # Use t-distribution (approximate with z for large n, or use scipy if available)
    # For simplicity, using z-score approximation (1.96 for 95% CI)
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    
    margin = z_score * std_err
    return (mean - margin, mean + margin)


def aggregate_trial_results(trial_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple trials with statistical measures.
    
    Args:
        trial_summaries: List of summary dictionaries from each trial
    
    Returns:
        Dictionary with aggregated statistics (mean, std, CI, etc.)
    """
    if not trial_summaries:
        return {}
    
    # Metrics to aggregate
    metrics = [
        "validation_rate",
        "correctness_rate",
        "speedup_rate_no_gen",
        "speedup_rate_gen",
        "speedup_gt1_no_gen",
        "speedup_gt1_gen",
        "speedup_all_no_gen",
        "speedup_all_gen",
    ]
    
    aggregated = {
        "num_trials": len(trial_summaries),
        "total_examples": trial_summaries[0].get("total_examples", 0),
    }
    
    for metric in metrics:
        values = [s.get(metric, 0.0) for s in trial_summaries if metric in s]
        if values:
            aggregated[f"{metric}_mean"] = statistics.mean(values)
            aggregated[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            ci_lower, ci_upper = calculate_confidence_interval(values)
            aggregated[f"{metric}_ci_lower"] = ci_lower
            aggregated[f"{metric}_ci_upper"] = ci_upper
            aggregated[f"{metric}_min"] = min(values)
            aggregated[f"{metric}_max"] = max(values)
    
    return aggregated


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    print("=" * 100)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 100)
    weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")
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
    invariant_generator = ModelFactory.create(cfg.models, system_prompt, user_prompt_template, eval_ft_model=eval_ft_model)
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
    num_trials = cfg.models.get("trials", {}).get("num_trials", 1)
    confidence_level = cfg.models.get("trials", {}).get("confidence_level", 0.95)
    
    if num_trials > 1:
        logger.info(f"Running {num_trials} trials for statistical significance (confidence level: {confidence_level})")
    
    all_trial_summaries = []
    all_trial_results = []  # Store results per trial
    
    # Run multiple trials sequentially
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
        asyncio.run(
            evaluation.evaluate(
                invariant_generator, 
                __weave={"display_name": f"{invariant_generator.get_run_name()}_trial{trial_num}" if num_trials > 1 else invariant_generator.get_run_name()}
            )
        )
        
        # Get results for this trial
        trial_plot_results = trial_collector.get_results()
        all_trial_results.append(trial_plot_results)
        logger.info(f"Trial {trial_num}: Collected {len(trial_plot_results)} results")
        
        # Compute summary for this trial using the scorer's summarize method
        if trial_plot_results:
            trial_summary = trial_scorer.summarize(trial_plot_results)
            all_trial_summaries.append(trial_summary)
            logger.info(f"Trial {trial_num} summary: correctness_rate={trial_summary.get('correctness_rate', 0):.3f}, "
                       f"speedup_rate_gen={trial_summary.get('speedup_rate_gen', 0):.3f}")
    
    # Use last trial's results for plotting
    if all_trial_results:
        plot_results = all_trial_results[-1]  # Use last trial's results
        total_results = sum(len(r) for r in all_trial_results)
        logger.info(f"Total collected {total_results} results across {num_trials} trial(s)")
        logger.info(f"Using {len(plot_results)} results for plotting (from trial {num_trials})")
    else:
        plot_results = []
        logger.warning("No results collected from any trial")
    
    # If multiple trials, aggregate statistics
    if num_trials > 1 and all_trial_summaries:
        logger.info("="*80)
        logger.info("STATISTICAL SUMMARY ACROSS TRIALS")
        logger.info("="*80)
        aggregated = aggregate_trial_results(all_trial_summaries)
        
        # Log aggregated statistics
        logger.info(f"Number of trials: {aggregated['num_trials']}")
        logger.info(f"Total examples per trial: {aggregated['total_examples']}")
        logger.info("")
        logger.info("Metrics (mean ± std, 95% CI):")
        
        metrics_to_log = [
            ("Validation Rate", "validation_rate"),
            ("Correctness Rate", "correctness_rate"),
            ("Speedup Rate (w/o Model Latency)", "speedup_rate_no_gen"),
            ("Speedup Rate (w/ Model Latency)", "speedup_rate_gen"),
            ("Speedup >1 (w/o Model Latency)", "speedup_gt1_no_gen"),
            ("Speedup >1 (w/ Model Latency)", "speedup_gt1_gen"),
            ("Speedup All (w/o Model Latency)", "speedup_all_no_gen"),
            ("Speedup All (w/ Model Latency)", "speedup_all_gen"),
        ]
        
        for metric_name, metric_key in metrics_to_log:
            mean = aggregated.get(f"{metric_key}_mean", 0.0)
            std = aggregated.get(f"{metric_key}_std", 0.0)
            ci_lower = aggregated.get(f"{metric_key}_ci_lower", 0.0)
            ci_upper = aggregated.get(f"{metric_key}_ci_upper", 0.0)
            logger.info(f"  {metric_name}: {mean:.4f} ± {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        logger.info("="*80)
        
        # Save statistical results to file
        stats_dir = GC.EXPERIMENTS_DIR / "eval_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = invariant_generator.get_run_name()
        stats_filename = f"stats_{run_name}_{timestamp}.json"
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
                "total_examples_per_trial": aggregated["total_examples"],
                "confidence_level": cfg.models.get("trials", {}).get("confidence_level", 0.95),
                "timestamp": timestamp,
                "dataset": {
                    "name": cfg.dataset.name,
                    "split": cfg.dataset.split,
                    "limit": cfg.dataset.limit,
                },
            },
            "statistics": {}
        }
        
        # Add all metrics with their statistics
        for metric_name, metric_key in metrics_to_log:
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

    # # Generate and upload plot
    # if len(plot_results) > 0:
    #     logger.info("Generating evaluation plot...")
    #     plot_dir = GC.EXPERIMENTS_DIR / f"eval_plots_{invariant_generator.get_run_name()}"
    #     plot_dir.mkdir(parents=True, exist_ok=True)

    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     plot_path = (
    #         plot_dir / f"eval_plot_{cfg.weave.project_name}_{invariant_generator.get_run_name()}_{timestamp}.html"
    #     )

    #     plot_verification_vs_baseline(
    #         results=plot_results,
    #         model_name=invariant_generator.get_display_name(),
    #         baseline_name=f"UAutomizer{cfg.scorer.verifier.version}",
    #         split_name=cfg.dataset.split,
    #         plot_path=plot_path,
    #     )
    #     logger.info(f"Plot saved to: {plot_path}")

    #     # Upload to W&B (optional)
    #     if not cfg.weave.skip_wandb:
    #         upload_plot_to_wandb(
    #             plot_path=plot_path,
    #             project_name=cfg.weave.project_name,
    #             run_name=f"plot-{invariant_generator.get_run_name()}",
    #             artifact_name=f"eval-plot-{invariant_generator.get_run_name()}",
    #             entity=cfg.weave.entity,
    #         )
    # else:
    #     logger.warning("No results collected for plotting - skipping plot generation")

    # logger.info("Evaluation complete!")


if __name__ == "__main__":
    logger.info("Starting evaluation...")
    main()

# Example usage:
# uv run -m src.eval.evaluate 
# uv run -m src.eval.evaluate models=qwen_config dataset.limit=1
# uv run -m src.eval.evaluate models=qwen_no_think_config dataset.limit=1
# uv run -m src.eval.evaluate models=oss_config dataset.limit=1

"""
Statistics utilities for evaluation metrics aggregation.

Provides functions for:
- Confidence interval calculation
- Trial results aggregation across multiple runs
- Summary logging
"""

import logging
import statistics
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# Metrics configuration for consistent use across modules
METRICS_TO_AGGREGATE = [
    ("Validation Rate", "validation_rate"),
    ("Correctness Rate", "correctness_rate"),
    ("Usefulness Rate", "usefulness_rate"),
    ("Speedup Rate", "speedup_rate"),
    ("Speedup Rate (e2e)", "speedup_e2e_rate"),
    ("Speedup >1", "speedup_gt1"),
    ("Speedup >1 (e2e)", "speedup_gt1_e2e"),
    ("Speedup All", "speedup_all"),
    ("Speedup All (e2e)", "speedup_all_e2e"),
    ("VBP (Virtual Best Performance)", "vbp"),
    ("VBP (e2e)", "vbp_e2e"),
    ("Avg Baseline Timing", "avg_median_timing"),
]


def calculate_confidence_interval(
    values: List[float], confidence_level: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval using t-distribution approximation.
    
    Args:
        values: List of numeric values
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    
    mean = statistics.mean(values)
    std_err = statistics.stdev(values) / (len(values) ** 0.5)
    
    # Z-score approximation for common confidence levels
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    
    margin = z_score * std_err
    return (mean - margin, mean + margin)


def aggregate_trial_results(trial_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple trials with statistical measures.
    
    Uses the metric naming convention from scorer.py:
    - speedup/speedup_e2e (instead of _no_gen/_gen)
    - vbp/vbp_e2e (Virtual Best Performance)
    
    Args:
        trial_summaries: List of summary dictionaries from each trial
    
    Returns:
        Dictionary with aggregated statistics (mean, std, CI, etc.)
    """
    if not trial_summaries:
        return {}
    
    # Metrics to aggregate - aligned with scorer.summarize() output
    metrics = [
        "validation_rate",
        "correctness_rate",
        "usefulness_rate",
        "speedup_rate",
        "speedup_e2e_rate",
        "speedup_gt1",
        "speedup_gt1_e2e",
        "speedup_all",
        "speedup_all_e2e",
        "vbp",
        "vbp_e2e",
        "avg_median_timing",
    ]
    
    aggregated = {
        "num_trials": len(trial_summaries),
    }
    
    # Add count metrics (averaged across trials)
    count_metrics = [
        "validation_count",
        "correctness_count", 
        "usefulness_count",
        "speedup_count",
        "speedup_e2e_count",
    ]
    for metric in count_metrics:
        values = [s.get(metric, 0) for s in trial_summaries if metric in s]
        if values:
            aggregated[f"{metric}_mean"] = statistics.mean(values)
    
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


def log_trial_summary(summary: Dict[str, Any], trial_num: int = 0) -> None:
    """
    Log a trial summary dictionary in a readable format.
    
    Args:
        summary: Summary dictionary from scorer.summarize()
        trial_num: Trial number (0 for single trial, >0 for multi-trial)
    """
    prefix = f"Trial {trial_num}: " if trial_num > 0 else ""
    
    logger.info(f"{prefix}Validation: {summary.get('validation_count', 0)} ({summary.get('validation_rate', 0):.2%})")
    logger.info(f"{prefix}Correctness: {summary.get('correctness_count', 0)} ({summary.get('correctness_rate', 0):.2%})")
    logger.info(f"{prefix}Usefulness: {summary.get('usefulness_count', 0)} ({summary.get('usefulness_rate', 0):.2%})")
    logger.info(f"{prefix}Speedup >1: {summary.get('speedup_count', 0)} ({summary.get('speedup_rate', 0):.2%})")
    logger.info(f"{prefix}Speedup >1 (e2e): {summary.get('speedup_e2e_count', 0)} ({summary.get('speedup_e2e_rate', 0):.2%})")
    logger.info(f"{prefix}Avg Speedup (>1 only): {summary.get('speedup_gt1', 1.0):.2f}x")
    logger.info(f"{prefix}Avg Speedup (>1 only, e2e): {summary.get('speedup_gt1_e2e', 1.0):.2f}x")
    logger.info(f"{prefix}Avg Speedup (all): {summary.get('speedup_all', 1.0):.2f}x")
    logger.info(f"{prefix}Avg Speedup (all, e2e): {summary.get('speedup_all_e2e', 1.0):.2f}x")
    logger.info(f"{prefix}VBP: {summary.get('vbp', 0):.2f}s")
    logger.info(f"{prefix}VBP (e2e): {summary.get('vbp_e2e', 0):.2f}s")
    logger.info(f"{prefix}Avg Baseline: {summary.get('avg_median_timing', 0):.2f}s")


def log_aggregated_summary(aggregated: Dict[str, Any]) -> None:
    """
    Log aggregated statistics across multiple trials.
    
    Args:
        aggregated: Dictionary from aggregate_trial_results()
    """
    logger.info("="*80)
    logger.info("STATISTICAL SUMMARY ACROSS TRIALS")
    logger.info("="*80)
    
    logger.info(f"Number of trials: {aggregated.get('num_trials', 0)}")
    logger.info("")
    logger.info("Metrics (mean ± std, 95% CI):")
    
    for metric_name, metric_key in METRICS_TO_AGGREGATE:
        mean = aggregated.get(f"{metric_key}_mean", 0.0)
        std = aggregated.get(f"{metric_key}_std", 0.0)
        ci_lower = aggregated.get(f"{metric_key}_ci_lower", 0.0)
        ci_upper = aggregated.get(f"{metric_key}_ci_upper", 0.0)
        logger.info(f"  {metric_name}: {mean:.4f} ± {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    logger.info("="*80)

#!/usr/bin/env python3
"""
Build v1 training dataset: clean raw invariants and verify them with quality grades.

This script pre-processes the raw UAutomizer invariants offline:
1. Load raw training data with original invariants
2. For each invariant:
   - Normalize it using normalize_invariant(inv, pretty=True) to remove casts and tautologies
   - Verify the cleaned invariant (correctness + usefulness)
   - Assign a quality grade (0-3)
3. Save qualifying entries (grade > 0) to main output file
4. Save failed entries (grade == 0) to separate *-failed.json file

This replaces the dynamic cleaning in data_utils.py inv_gen_per_marker_task_v0_to_v1.

Output files:
    - Main output: <output_path>.json (successful invariants with grade > 0)
    - Failed output: <output_path>-failed.json (failed invariants with grade == 0)

Usage:
    uv run -m wonda.data_pipeline.build_v1_train_dataset dataset.limit=5
    uv run -m wonda.data_pipeline.build_v1_train_dataset dataset.limit=-1  # Process all
    sbatch scripts/preprocess/build_v1_train_dataset.sbatch
"""

import copy
import csv
import json
import logging
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from configs import global_config as GC
from wonda.data_pipeline.gt_invariant_normalization import normalize_invariant
from wonda.core.ast_program import AstProgram
from wonda.core.property import Property
from wonda.eval.validate import syntactic_validation
from wonda.verifiers.uautomizer_runlim import UAutomizerVerifier, VerifierCallReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class QualityGrade:
    """Quality grade result for a cleaned invariant."""
    is_valid: bool = field(default=False)
    is_correct: bool = field(default=False)
    does_target_property_still_hold: bool = field(default=False)
    has_speedup: bool = field(default=False)
    speedup: Optional[float] = field(default=None)
    quality_grade: int = field(default=-1)  # -1 = unchecked, 0 = invalid, 1-3 = valid grades
    # Verification timings
    correctness_time: Optional[float] = field(default=None)
    correctness_decision: Optional[str] = field(default=None)
    correctness_runlim: Optional[dict] = field(default=None)
    usefulness_time: Optional[float] = field(default=None)
    usefulness_decision: Optional[str] = field(default=None)
    usefulness_runlim: Optional[dict] = field(default=None)
    total_verification_time: Optional[float] = field(default=None)  # max of correctness and usefulness (parallel)


@dataclass
class InvariantContext:
    """Context for processing a single invariant."""
    entry: dict[str, Any]
    entry_idx: int
    inv_idx: int
    marker: str
    original_invariant: str
    cleaned_invariant: str
    target_invariant: str  # The one that will be verified (raw or cleaned based on mode)
    inv_mode: str  # "raw" or "cleaned"
    baseline_decision: str
    baseline_timing: float
    program: AstProgram
    original_program: str


# =============================================================================
# Helper Functions
# =============================================================================
def compute_quality_grade(result: QualityGrade) -> int:
    """Compute the quality grade of a cleaned invariant."""
    if not result.is_valid or not result.is_correct:
        return 0
    elif not result.does_target_property_still_hold:
        return 1
    elif not result.has_speedup:
        return 2
    else:
        return 3


def run_verifier(verifier: UAutomizerVerifier, program_str: str, kind: str, label: str = None) -> VerifierCallReport:
    """Write program to temp file and run verifier."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        program_path = code_dir / f"code_for_{kind}.c"
        with open(program_path, "w") as f:
            f.write(program_str)
        full_label = f"{label}/{kind}" if label else kind
        verifier_report: VerifierCallReport = verifier.verify(
            program_path=program_path,
            reports_dir=code_dir,
            timeout_seconds=verifier.timeout_seconds,
            label=full_label,
        )
        logger.debug(
            f"Verifier report for {kind}: {verifier_report.decision} "
            f"({verifier_report.decision_reason}) in {verifier_report.time_taken}s"
        )
        return verifier_report


def is_trivial_invariant(inv: str) -> bool:
    """Check if invariant is trivial (0, 1, true, false) - UAutomizer artifacts."""
    if not inv:
        return True
    normalized = inv.strip().lower().replace('(', '').replace(')', '').replace(' ', '')
    return normalized in ('0', '1', 'true', 'false', '')


# =============================================================================
# Verification
# =============================================================================
def verify_invariant(
    verifier: UAutomizerVerifier,
    cleaned_invariant: str,
    program: AstProgram,
    marker: str,
    baseline_decision: str,
    baseline_timing: float,
    label: str = None,
) -> QualityGrade:
    """Verify a cleaned invariant (correctness + usefulness in parallel)."""
    result = QualityGrade()
    predicate = Property(content=cleaned_invariant, marker_name=marker)

    # Syntactic validation
    result.is_valid = syntactic_validation(predicate.content)
    if not result.is_valid:
        result.quality_grade = 0
        logger.warning(f"Syntactically invalid: '{predicate.content[:50]}...'")
        return result

    # Filter trivial invariants (UAutomizer artifacts)
    if is_trivial_invariant(cleaned_invariant):
        result.quality_grade = 0
        logger.warning(f"Trivial invariant rejected: '{cleaned_invariant}'")
        return result

    # Prepare verification programs
    correctness_program = program.get_program_with_assertion(
        predicate=predicate, assumptions=[], for_llm=False
    )
    usefulness_program = program.get_program_with_assertion(
        predicate=program.get_target_assert(), assumptions=[predicate], for_llm=False
    )

    # Run both checks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        correctness_future = executor.submit(run_verifier, verifier, correctness_program, "correctness", label)
        usefulness_future = executor.submit(run_verifier, verifier, usefulness_program, "usefulness", label)

        correctness_result = correctness_future.result()
        usefulness_result = usefulness_future.result()

    # Process results
    result.is_correct = correctness_result.decision == "TRUE"
    result.does_target_property_still_hold = (
        usefulness_result.decision in {"TRUE", "FALSE"}
        and usefulness_result.decision == baseline_decision
    )
    
    # Save verification timings, decisions, and runlim outputs
    result.correctness_time = correctness_result.time_taken
    result.correctness_decision = correctness_result.decision
    result.correctness_runlim = correctness_result.runlim
    result.usefulness_time = usefulness_result.time_taken
    result.usefulness_decision = usefulness_result.decision
    result.usefulness_runlim = usefulness_result.runlim
    
    # Total verification time is max of both (they run in parallel)
    total_verification_time = max(correctness_result.time_taken, usefulness_result.time_taken)
    result.total_verification_time = total_verification_time
    result.has_speedup = total_verification_time < baseline_timing
    result.speedup = (baseline_timing / total_verification_time) if total_verification_time > 0 else 0.0
    result.quality_grade = compute_quality_grade(result)

    return result


# =============================================================================
# Entry Creation
# =============================================================================
def create_entry(
    ctx: InvariantContext,
    quality: QualityGrade,
) -> dict[str, Any]:
    """Create a processed entry with gt_invariant containing the cleaned invariant."""
    entry = copy.deepcopy(ctx.entry)
    
    # Compute virtual best time: min(baseline, verification) if grade >= 2, else baseline
    # Grade >= 2 means: correct AND target property still holds (usable invariant)
    baseline_time = ctx.baseline_timing
    verification_time = quality.total_verification_time
    
    if quality.quality_grade >= 2 and verification_time is not None:
        virtual_best_time = min(baseline_time, verification_time)
    else:
        virtual_best_time = baseline_time
    
    entry["gt_invariant"] = {
        "marker": ctx.marker,
        "original_invariant": ctx.original_invariant,
        "cleaned_invariant": ctx.cleaned_invariant,
        "target_invariant": ctx.target_invariant,  # The one that was verified
        "inv_mode": ctx.inv_mode,  # "raw" or "cleaned"
        "is_valid": quality.is_valid,
        "is_correct": quality.is_correct,
        "does_target_property_still_hold": quality.does_target_property_still_hold,
        "has_speedup": quality.has_speedup,
        "speedup": quality.speedup,
        "quality_grade": quality.quality_grade,
        # Verification timings and runlim outputs
        "baseline_time": baseline_time,
        "correctness_time": quality.correctness_time,
        "correctness_decision": quality.correctness_decision,
        "correctness_runlim": quality.correctness_runlim,
        "usefulness_time": quality.usefulness_time,
        "usefulness_decision": quality.usefulness_decision,
        "usefulness_runlim": quality.usefulness_runlim,
        "total_verification_time": quality.total_verification_time,
        # Virtual best: min(baseline, verification) if usable, else baseline
        "virtual_best_time": virtual_best_time,
    }
    return entry


# =============================================================================
# Processing Functions
# =============================================================================
@dataclass
class ProcessResult:
    """Result of processing an invariant."""
    entry: Optional[dict[str, Any]] = None
    quality_grade: int = -1
    key: Optional[tuple[str, str, str]] = None  # (file, marker, original_inv) for tracking


@dataclass
class InvariantTask:
    """Task for parallel processing of an invariant."""
    ctx: InvariantContext
    key: tuple[str, str, str]  # (file, marker, original_inv)


def process_invariant(
    ctx: InvariantContext,
    verifier: UAutomizerVerifier,
    cfg: DictConfig,
) -> ProcessResult:
    """Process a single invariant: verify target_invariant (raw or cleaned based on mode)."""
    logger.info(
        f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: "
        f"marker={ctx.marker}, mode={ctx.inv_mode}, target='{ctx.target_invariant[:60]}...'"
    )

    if not cfg.pipeline.verify:
        # Skip verification, just return the entry with unchecked quality
        quality = QualityGrade(is_valid=True, quality_grade=-1)
        return ProcessResult(entry=create_entry(ctx, quality), quality_grade=-1)

    inv_label = f"entry_{ctx.entry_idx}/inv_{ctx.inv_idx}"
    quality = verify_invariant(
        verifier,
        ctx.target_invariant,
        ctx.program,
        ctx.marker,
        ctx.baseline_decision,
        ctx.baseline_timing,
        label=inv_label,
    )

    logger.info(
        f"Entry {ctx.entry_idx}, Inv {ctx.inv_idx}: "
        f"grade={quality.quality_grade}, correct={quality.is_correct}, "
        f"useful={quality.does_target_property_still_hold}, speedup={quality.speedup:.2f}x"
        if quality.speedup else f"grade={quality.quality_grade}"
    )

    # Always create entry, caller decides what to do based on quality_grade
    return ProcessResult(entry=create_entry(ctx, quality), quality_grade=quality.quality_grade)


def process_task(task: InvariantTask, verifier: UAutomizerVerifier, cfg: DictConfig) -> ProcessResult:
    """Process a single invariant task (wrapper for parallel execution)."""
    result = process_invariant(task.ctx, verifier, cfg)
    result.key = task.key
    return result


# =============================================================================
# Resume Support
# =============================================================================
def get_checkpoint_path(output_path: Path) -> Path:
    """Get the checkpoint file path based on the output path."""
    return output_path.parent / f".{output_path.stem}.checkpoint.json"


def build_processed_keys(processed: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Build a set of (file, marker, original_invariant) keys from processed entries."""
    keys = set()
    for entry in processed:
        gt_inv = entry.get("gt_invariant", {})
        key = (
            entry.get("file", ""),
            gt_inv.get("marker", ""),
            gt_inv.get("original_invariant", ""),
        )
        keys.add(key)
    return keys


def load_attempted_keys(checkpoint_path: Path) -> set[tuple[str, str, str]]:
    """Load set of all attempted (file, marker, original_invariant) keys from checkpoint."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                content = f.read().strip()
                if not content:
                    return set()
                data = json.loads(content)
            keys = {tuple(k) for k in data.get("attempted_keys", [])}
            logger.info(f"Loaded {len(keys)} attempted keys from checkpoint")
            return keys
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return set()


def save_checkpoint(checkpoint_path: Path, attempted_keys: set[tuple[str, str, str]]) -> None:
    """Save the set of attempted keys to checkpoint file (atomic write)."""
    data = {"attempted_keys": [list(k) for k in attempted_keys]}
    temp_file = checkpoint_path.with_suffix(".checkpoint.json.tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    temp_file.rename(checkpoint_path)


def load_existing_progress(output_path: Path) -> list[dict[str, Any]]:
    """Load existing processed entries from output file if it exists."""
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} existing processed entries from {output_path}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load existing progress: {e}. Starting fresh.")
    return []


# =============================================================================
# Main Pipeline
# =============================================================================
def get_failed_output_path(output_path: Path) -> Path:
    """Get the failed invariants output path based on the main output path."""
    return output_path.parent / f"{output_path.stem}-failed.json"


def build_v1_train_dataset(
    data: list[dict[str, Any]],
    cfg: DictConfig,
    verifier: UAutomizerVerifier,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    """Main preprocessing pipeline: clean + verify invariants with resume support and parallel processing.
    
    Returns:
        Tuple of (successful_entries, failed_entries, output_path)
    """
    max_workers = cfg.pipeline.get("max_parallel_workers", 4)
    inv_mode = cfg.pipeline.get("inv_mode", "cleaned")  # "raw" or "cleaned"
    
    if inv_mode not in ("raw", "cleaned"):
        raise ValueError(f"Invalid inv_mode: {inv_mode}. Must be 'raw' or 'cleaned'")
    
    logger.info(f"Processing {len(data)} entries with {max_workers} parallel workers, inv_mode={inv_mode}")

    # Create output path with v0/v1 subfolder based on inv_mode
    base_output_path = Path(cfg.output.preprocessed_invariants_path)
    version_folder = "v0" if inv_mode == "raw" else "v1"
    output_dir = base_output_path.parent / version_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{base_output_path.stem}_{version_folder}.json"
    checkpoint_path = get_checkpoint_path(output_path)
    failed_output_path = get_failed_output_path(output_path)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output path: {output_path}")

    # Load existing progress for resume capability
    processed = load_existing_progress(output_path)
    failed = load_existing_progress(failed_output_path)

    # Load attempted keys from checkpoint (includes both successes AND failures)
    attempted_keys = load_attempted_keys(checkpoint_path)

    # If no checkpoint exists, rebuild from successful + failed entries
    if not attempted_keys:
        attempted_keys = build_processed_keys(processed)
        attempted_keys.update(build_processed_keys(failed))
        if attempted_keys:
            logger.info(f"Rebuilt {len(attempted_keys)} attempted keys from output files")

    # Thread-safe lock for updating shared state
    lock = threading.Lock()
    
    # Statistics
    skipped_count = 0
    skipped_baseline = 0
    skipped_no_invariant = 0

    # Collect all tasks first
    tasks: list[InvariantTask] = []
    logger.info("Collecting tasks...")
    
    for i, entry in enumerate(tqdm(data, desc="Collecting tasks")):
        baseline_timing = entry.get("timings", {}).get("median", 0.0)
        baseline_decision = entry.get("baseline_decision", "UNKNOWN")
        original_program = entry.get("original_program", "")
        file_name = entry.get("file", "")

        # Skip invalid entries
        if baseline_decision not in {"TRUE", "FALSE"}:
            skipped_baseline += 1
            continue

        if not entry.get("invariants"):
            skipped_no_invariant += 1
            continue

        # Parse program once per entry
        program = AstProgram().from_code(original_program)
        program.process(print_ast=False)

        for j, inv in enumerate(entry["invariants"]):
            original_inv = inv.get("invariant", "")
            marker = inv.get("marker", "")

            if not original_inv or not marker:
                skipped_no_invariant += 1
                continue

            # Check if this invariant was already attempted (resume support)
            key = (file_name, marker, original_inv)
            if key in attempted_keys:
                skipped_count += 1
                continue

            # Clean the invariant (even if using raw mode, for reference)
            cleaned_inv = normalize_invariant(original_inv, pretty=True)
            
            # Select target invariant based on mode
            target_inv = original_inv if inv_mode == "raw" else cleaned_inv

            ctx = InvariantContext(
                entry=entry,
                entry_idx=i,
                inv_idx=j,
                marker=marker,
                original_invariant=original_inv,
                cleaned_invariant=cleaned_inv,
                target_invariant=target_inv,
                inv_mode=inv_mode,
                baseline_decision=baseline_decision,
                baseline_timing=baseline_timing,
                program=program,
                original_program=original_program,
            )

            tasks.append(InvariantTask(ctx=ctx, key=key))

    logger.info(f"Collected {len(tasks)} tasks to process")
    logger.info(f"Skipped: resumed={skipped_count}, invalid_baseline={skipped_baseline}, no_invariant={skipped_no_invariant}")

    if not tasks:
        logger.info("No tasks to process")
        return processed, failed, output_path

    # Save interval (save every N completed tasks)
    save_interval = max(1, min(10, len(tasks) // 10))
    completed_since_save = 0

    def save_progress():
        """Save current progress to files."""
        with open(output_path, "w") as f:
            json.dump(processed, f, indent=2)
        with open(failed_output_path, "w") as f:
            json.dump(failed, f, indent=2)
        save_checkpoint(checkpoint_path, attempted_keys)

    # Process tasks in parallel
    logger.info(f"Processing {len(tasks)} tasks with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_task, task, verifier, cfg): task
            for task in tasks
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing"):
            task = future_to_task[future]
            
            try:
                result = future.result()
                
                with lock:
                    if result.entry is not None:
                        if result.quality_grade > 0 or result.quality_grade == -1:
                            # Successful (grade > 0) or unverified (grade == -1)
                            processed.append(result.entry)
                        else:
                            # Failed (grade == 0)
                            failed.append(result.entry)

                    # Mark this invariant as attempted
                    attempted_keys.add(result.key)
                    
                    # Periodic save
                    completed_since_save += 1
                    if completed_since_save >= save_interval:
                        save_progress()
                        completed_since_save = 0

            except Exception as e:
                logger.error(f"Task {task.key}: Failed with error: {type(e).__name__}: {e}")
                with lock:
                    attempted_keys.add(task.key)

    # Final save
    with lock:
        save_progress()

    logger.info(f"Total successful entries (grade > 0): {len(processed)}")
    logger.info(f"Total failed entries (grade == 0): {len(failed)}")
    logger.info(f"Failed invariants saved to: {failed_output_path}")

    # Push to hub
    if cfg.output.push_to_hub:
        # Push successful entries
        dataset = Dataset.from_list(processed)
        hub_name = f"{cfg.output.hub_organization}/{cfg.output.hub_name}"
        logger.info(f"Pushing successful entries to: {hub_name}")
        DatasetDict({"train": dataset}).push_to_hub(hub_name)
        
        # Push failed entries to separate dataset
        if failed:
            failed_dataset = Dataset.from_list(failed)
            failed_hub_name = f"{cfg.output.hub_organization}/{cfg.output.hub_name}-failed"
            logger.info(f"Pushing failed entries to: {failed_hub_name}")
            DatasetDict({"train": failed_dataset}).push_to_hub(failed_hub_name)

    return processed, failed, output_path


# =============================================================================
# Statistics and Plotting
# =============================================================================
def save_grade_distribution(
    processed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    output_path: Path,
    inv_mode: str,
) -> None:
    """Save grade distribution plot and data."""
    # Combine all entries
    all_entries = processed + failed
    
    # Count grades
    grade_counts = Counter()
    for entry in all_entries:
        gt_inv = entry.get("gt_invariant", {})
        grade = gt_inv.get("quality_grade", -1)
        grade_counts[grade] += 1
    
    # Ensure all grades 0-3 are present
    for grade in [0, 1, 2, 3]:
        if grade not in grade_counts:
            grade_counts[grade] = 0
    
    # Sort by grade
    grades = sorted([g for g in grade_counts.keys() if g >= 0])
    counts = [grade_counts[g] for g in grades]
    total = sum(counts)
    
    # Log statistics
    logger.info(f"Grade distribution ({inv_mode} mode):")
    for grade, count in zip(grades, counts):
        pct = (count / total * 100) if total > 0 else 0
        logger.info(f"  Grade {grade}: {count} ({pct:.1f}%)")
    
    # Save data as CSV
    stats_dir = output_path.parent / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = stats_dir / f"grade_distribution_{inv_mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grade", "count", "percentage"])
        for grade, count in zip(grades, counts):
            pct = (count / total * 100) if total > 0 else 0
            writer.writerow([grade, count, f"{pct:.2f}"])
        writer.writerow(["total", total, "100.00"])
    logger.info(f"Grade distribution data saved to: {csv_path}")
    
    # Save plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grades, counts, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / total * 100) if total > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Quality Grade', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Quality Grade Distribution ({inv_mode} mode)\nTotal: {total} invariants', fontsize=14)
    plt.xticks(grades, ['0\n(Invalid/\nIncorrect)', '1\n(Correct but\nNot Useful)', 
                        '2\n(Useful but\nNo Speedup)', '3\n(Full\nSpeedup)'])
    plt.tight_layout()
    
    plot_path = stats_dir / f"grade_distribution_{inv_mode}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Grade distribution plot saved to: {plot_path}")


def save_speedup_distribution(
    processed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    output_path: Path,
    inv_mode: str,
) -> None:
    """Save speedup distribution plot and data."""
    # Combine all entries and extract speedups
    all_entries = processed + failed
    speedups = []
    
    for entry in all_entries:
        gt_inv = entry.get("gt_invariant", {})
        speedup = gt_inv.get("speedup")
        if speedup is not None and speedup > 0:
            speedups.append(speedup)
    
    if not speedups:
        logger.warning("No speedup data available for plotting")
        return
    
    speedups = np.array(speedups)
    
    # Compute statistics
    stats = {
        "count": len(speedups),
        "mean": float(np.mean(speedups)),
        "median": float(np.median(speedups)),
        "std": float(np.std(speedups)),
        "min": float(np.min(speedups)),
        "max": float(np.max(speedups)),
        "p25": float(np.percentile(speedups, 25)),
        "p75": float(np.percentile(speedups, 75)),
        "speedup_gt_1": int(np.sum(speedups > 1)),
        "speedup_gt_1_pct": float(np.sum(speedups > 1) / len(speedups) * 100),
    }
    
    # Log statistics
    logger.info(f"Speedup statistics ({inv_mode} mode):")
    logger.info(f"  Count: {stats['count']}")
    logger.info(f"  Mean: {stats['mean']:.2f}x")
    logger.info(f"  Median: {stats['median']:.2f}x")
    logger.info(f"  Std: {stats['std']:.2f}")
    logger.info(f"  Min: {stats['min']:.2f}x")
    logger.info(f"  Max: {stats['max']:.2f}x")
    logger.info(f"  P25: {stats['p25']:.2f}x")
    logger.info(f"  P75: {stats['p75']:.2f}x")
    logger.info(f"  Speedup > 1: {stats['speedup_gt_1']} ({stats['speedup_gt_1_pct']:.1f}%)")
    
    # Save data as CSV
    stats_dir = output_path.parent / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats_csv_path = stats_dir / f"speedup_stats_{inv_mode}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    logger.info(f"Speedup statistics saved to: {stats_csv_path}")
    
    # Save raw speedup values
    speedups_csv_path = stats_dir / f"speedups_{inv_mode}.csv"
    with open(speedups_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speedup"])
        for s in speedups:
            writer.writerow([f"{s:.4f}"])
    logger.info(f"Raw speedup data saved to: {speedups_csv_path}")
    
    # Save plot - histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(speedups, bins=50, edgecolor='black', alpha=0.7)
    
    # Color bars based on speedup > 1 or not
    for i, patch in enumerate(patches):
        if bins[i] >= 1:
            patch.set_facecolor('#2ecc71')  # Green for speedup
        else:
            patch.set_facecolor('#e74c3c')  # Red for slowdown
    
    ax1.axvline(x=1, color='black', linestyle='--', linewidth=2, label='No speedup (1x)')
    ax1.axvline(x=stats['mean'], color='blue', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]:.2f}x')
    ax1.axvline(x=stats['median'], color='orange', linestyle='-', linewidth=2, label=f'Median: {stats["median"]:.2f}x')
    
    ax1.set_xlabel('Speedup (x)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Speedup Distribution ({inv_mode} mode)\nn={stats["count"]}', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Plot 2: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(speedups, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No speedup (1x)')
    
    # Add statistics text
    textstr = '\n'.join([
        f'Mean: {stats["mean"]:.2f}x',
        f'Median: {stats["median"]:.2f}x',
        f'Std: {stats["std"]:.2f}',
        f'Min: {stats["min"]:.2f}x',
        f'Max: {stats["max"]:.2f}x',
        f'P25: {stats["p25"]:.2f}x',
        f'P75: {stats["p75"]:.2f}x',
        f'Speedup>1: {stats["speedup_gt_1"]} ({stats["speedup_gt_1_pct"]:.1f}%)',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(1.3, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title(f'Speedup Box Plot ({inv_mode} mode)', fontsize=14)
    ax2.set_xticklabels([inv_mode])
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"speedup_distribution_{inv_mode}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Speedup distribution plot saved to: {plot_path}")


def save_virtual_best_statistics(
    processed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    output_path: Path,
    inv_mode: str,
) -> None:
    """Save virtual best performance statistics and plot.
    
    Virtual best = min(baseline_time, verification_time) if grade >= 2, else baseline_time
    Grade >= 2 means: correct AND target property still holds (usable invariant)
    This represents the best achievable performance by choosing the better approach per instance.
    """
    all_entries = processed + failed
    
    baseline_times = []
    verification_times = []
    virtual_best_times = []
    
    for entry in all_entries:
        gt_inv = entry.get("gt_invariant", {})
        baseline_time = gt_inv.get("baseline_time")
        verification_time = gt_inv.get("total_verification_time")
        virtual_best_time = gt_inv.get("virtual_best_time")
        
        if baseline_time is not None and baseline_time > 0:
            baseline_times.append(baseline_time)
        if verification_time is not None and verification_time > 0:
            verification_times.append(verification_time)
        if virtual_best_time is not None and virtual_best_time > 0:
            virtual_best_times.append(virtual_best_time)
    
    if not baseline_times or not virtual_best_times:
        logger.warning("No timing data available for virtual best statistics")
        return
    
    baseline_times = np.array(baseline_times)
    virtual_best_times = np.array(virtual_best_times)
    verification_times = np.array(verification_times) if verification_times else None
    
    # Compute aggregate statistics
    total_baseline = float(np.sum(baseline_times))
    total_virtual_best = float(np.sum(virtual_best_times))
    total_verification = float(np.sum(verification_times)) if verification_times is not None else None
    
    overall_speedup = total_baseline / total_virtual_best if total_virtual_best > 0 else 0
    time_saved = total_baseline - total_virtual_best
    time_saved_pct = (time_saved / total_baseline * 100) if total_baseline > 0 else 0
    
    # Count how many times virtual best chose invariant vs baseline
    chose_invariant = sum(1 for b, v in zip(baseline_times, virtual_best_times) if v < b)
    chose_baseline = len(virtual_best_times) - chose_invariant
    
    stats = {
        "count": len(baseline_times),
        "total_baseline_time": total_baseline,
        "total_virtual_best_time": total_virtual_best,
        "total_verification_time": total_verification,
        "overall_speedup": overall_speedup,
        "time_saved_seconds": time_saved,
        "time_saved_percent": time_saved_pct,
        "chose_invariant_count": chose_invariant,
        "chose_baseline_count": chose_baseline,
        "chose_invariant_pct": (chose_invariant / len(virtual_best_times) * 100) if virtual_best_times.size > 0 else 0,
        "mean_baseline_time": float(np.mean(baseline_times)),
        "mean_virtual_best_time": float(np.mean(virtual_best_times)),
        "median_baseline_time": float(np.median(baseline_times)),
        "median_virtual_best_time": float(np.median(virtual_best_times)),
    }
    
    # Log statistics
    logger.info(f"Virtual Best Performance Statistics ({inv_mode} mode):")
    logger.info(f"  Total instances: {stats['count']}")
    logger.info(f"  Total baseline time: {stats['total_baseline_time']:.2f}s")
    logger.info(f"  Total virtual best time: {stats['total_virtual_best_time']:.2f}s")
    logger.info(f"  Overall speedup: {stats['overall_speedup']:.2f}x")
    logger.info(f"  Time saved: {stats['time_saved_seconds']:.2f}s ({stats['time_saved_percent']:.1f}%)")
    logger.info(f"  Chose invariant: {stats['chose_invariant_count']} ({stats['chose_invariant_pct']:.1f}%)")
    logger.info(f"  Chose baseline: {stats['chose_baseline_count']}")
    
    # Save statistics as CSV
    stats_dir = output_path.parent / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv_path = stats_dir / f"virtual_best_stats_{inv_mode}.csv"
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
    logger.info(f"Virtual best statistics saved to: {stats_csv_path}")
    
    # Save raw data
    raw_csv_path = stats_dir / f"virtual_best_times_{inv_mode}.csv"
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline_time", "virtual_best_time", "chose"])
        for b, v in zip(baseline_times, virtual_best_times):
            chose = "invariant" if v < b else "baseline"
            writer.writerow([f"{b:.4f}", f"{v:.4f}", chose])
    logger.info(f"Virtual best raw data saved to: {raw_csv_path}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart comparing totals
    ax1 = axes[0]
    categories = ['Baseline', 'Virtual Best']
    times = [total_baseline, total_virtual_best]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax1.bar(categories, times, color=colors, edgecolor='black')
    
    # Add labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Total Time (seconds)', fontsize=12)
    ax1.set_title(f'Total Time Comparison ({inv_mode} mode)\n'
                  f'Speedup: {overall_speedup:.2f}x | Time Saved: {time_saved:.1f}s ({time_saved_pct:.1f}%)',
                  fontsize=12)
    
    # Plot 2: Pie chart showing choice distribution
    ax2 = axes[1]
    sizes = [chose_invariant, chose_baseline]
    labels = [f'Invariant\n({chose_invariant}, {stats["chose_invariant_pct"]:.1f}%)',
              f'Baseline\n({chose_baseline}, {100-stats["chose_invariant_pct"]:.1f}%)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='', startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Virtual Best Choice Distribution\n(n={stats["count"]})', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = stats_dir / f"virtual_best_comparison_{inv_mode}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Virtual best comparison plot saved to: {plot_path}")


# =============================================================================
# Entry Point
# =============================================================================
def load_data(path: Path, limit: int = -1) -> list[dict[str, Any]]:
    """Load JSON data with optional limit."""
    with open(path) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="build_v1_train_dataset",
)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
    )

    data = load_data(Path(cfg.dataset.name), cfg.dataset.limit)

    successful, failed, output_path = build_v1_train_dataset(data, cfg, verifier)
    logger.info(f"Done! Successful: {len(successful)}, Failed: {len(failed)}")
    
    # Save statistics and plots
    inv_mode = cfg.pipeline.get("inv_mode", "cleaned")
    save_grade_distribution(successful, failed, output_path, inv_mode)
    save_speedup_distribution(successful, failed, output_path, inv_mode)
    save_virtual_best_statistics(successful, failed, output_path, inv_mode)


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m wonda.data_pipeline.build_v1_train_dataset dataset.limit=5 max_
# uv run -m wonda.data_pipeline.build_v1_train_dataset dataset.limit=-1  # Process all
# uv run -m wonda.data_pipeline.build_v1_train_dataset pipeline.verify=false  # Skip verification
# sbatch --job-name=build-v0-train-dataset scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=-1 pipeline.inv_mode=raw
# sbatch --job-name=build-v1-train-dataset scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=-1 pipeline.inv_mode=cleaned

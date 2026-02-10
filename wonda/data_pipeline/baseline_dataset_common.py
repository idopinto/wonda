#!/usr/bin/env python3
"""Shared utilities for building baseline datasets (train and eval).

Runs UAutomizer on each program in a benchmark suite, collects timing and
invariant data, and packages the results into a HuggingFace DatasetDict
split by difficulty (easy / hard).
"""

import json
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from tqdm import tqdm

from configs import global_config as GC
from wonda.core.ast_program import AstProgram
from wonda.verifiers.uautomizer_runlim import UAutomizerVerifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Program discovery
# ---------------------------------------------------------------------------

def fetch_original_programs(
    original_programs_dir: Path, prefix: str = "", limit: int = -1
) -> list[Path]:
    """Return ``*.c`` files from *original_programs_dir*, optionally filtered."""
    c_files = list(original_programs_dir.glob(f"{prefix}*.c"))
    if not c_files:
        logger.error(f"No C files found in {original_programs_dir}")
        sys.exit(1)

    logger.info(f"Found {len(c_files)} C files in {original_programs_dir}")
    if limit >= 0:
        c_files = c_files[:limit]
        logger.info(f"Processing first {len(c_files)} files (limit={limit})")
    return c_files


# ---------------------------------------------------------------------------
# Invariant extraction (version-dependent)
# ---------------------------------------------------------------------------

def extract_invariants_from_log(log_file: Path) -> list[dict[str, Any]]:
    """Extract invariants from a UAutomizer v23 log file."""
    try:
        invariants: list[dict[str, Any]] = []
        with open(log_file, "r", encoding="utf-8", errors="ignore") as logf:
            for line in logf:
                if "InvariantResult [Line:" not in line:
                    continue
                idx1 = line.find("InvariantResult [Line:")
                idx2 = line.find("]:", idx1)
                if idx1 == -1 or idx2 == -1:
                    continue
                try:
                    line_num = int(
                        line[idx1 + len("InvariantResult [Line:") : idx2].strip()
                    )
                except ValueError:
                    continue
                next_line = next(logf, None)
                if next_line is None:
                    continue
                tag = "Derived loop invariant:"
                inv_idx = next_line.find(tag)
                if inv_idx != -1:
                    inv_val = next_line[inv_idx + len(tag) :].strip()
                    if inv_val:
                        invariants.append({"line": line_num, "invariant": inv_val})
        return invariants
    except Exception as e:
        logger.warning(f"Could not extract invariants from {log_file}: {e}")
        return []


def extract_invariants_from_witness(witness_yml: Path) -> list[dict[str, Any]]:
    """Extract invariants from a UAutomizer v24+ witness YAML file."""
    try:
        invariants: list[dict[str, Any]] = []
        with open(witness_yml, "r", encoding="utf-8", errors="ignore") as f:
            yml_data = yaml.safe_load(f)

        for invariant_dict in yml_data[0]["content"]:
            inv = invariant_dict["invariant"]
            if inv.get("type") != "loop_invariant":
                continue
            location = inv.get("location")
            line = location.get("line")
            value = inv.get("value")
            if line is not None and value is not None:
                invariants.append({"line": line, "invariant": value})
        return invariants
    except Exception as e:
        logger.warning(f"Could not extract invariants from {witness_yml}: {e}")
        return []


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def get_ast_program(program_path: Path) -> AstProgram:
    """Parse and process a C program into an ``AstProgram``."""
    ast_program = AstProgram().from_file_path(program_path)
    ast_program.process(print_ast=False)
    return ast_program


def map_invariant_line_to_marker(
    invariant_line: int, program_code: str, available_markers: list[str]
) -> Optional[str]:
    """Map a UAutomizer invariant line number to the nearest loop marker.

    Markers are inserted at the start of loop bodies.  Returns the marker
    whose line is closest to (and ``<=``) *invariant_line*, or ``None`` if
    no markers exist.
    """
    marker_positions: dict[str, int] = {}
    lines = program_code.split("\n")

    for marker in available_markers:
        if marker == "TARGET_ASSERT_MARKER":
            continue
        for line_idx, line in enumerate(lines, start=1):
            if marker in line and marker not in marker_positions:
                marker_positions[marker] = line_idx

    if not marker_positions:
        return None

    # Prefer the marker closest to (and <=) the invariant line.
    candidates = {
        m: pos for m, pos in marker_positions.items() if pos <= invariant_line
    }
    if not candidates:
        return min(marker_positions.items(), key=lambda x: x[1])[0]
    return max(candidates.items(), key=lambda x: x[1])[0]


def get_invariant_with_marker(
    invariant: dict[str, Any], program_code: str, available_markers: list[str]
) -> dict[str, Any]:
    """Return a copy of *invariant* with an added ``marker`` key."""
    marker = map_invariant_line_to_marker(
        invariant["line"], program_code, available_markers
    )
    result = invariant.copy()
    if marker:
        result["marker"] = marker
    return result


# ---------------------------------------------------------------------------
# Single-program verification
# ---------------------------------------------------------------------------

def process_program(
    cfg: DictConfig, program_path: Path, verifier: UAutomizerVerifier
) -> dict[str, Any]:
    """Run UAutomizer on a single C program and collect results.

    Returns a dict with timing, decision, and (optionally) extracted
    invariant information, or ``{}`` on failure.
    """
    try:
        ast_program = get_ast_program(program_path)
    except Exception as e:
        logger.error(f"Error getting AST program: {e}")
        return {}

    if len(ast_program.assertions) == 0:
        logger.error("No target assertion found in the program")
        return {}
    if len(ast_program.assertions) > 1:
        logger.error("Multiple target assertions found in the program")
        return {}

    target_assert = ast_program.get_target_assert()
    program_for_baseline = ast_program.get_program_with_assertion(
        property_to_assert=target_assert, assumptions=[], for_llm=False
    )

    result: dict[str, Any] = {
        "file": program_path.name,
        "original_program": ast_program.code,
        "program_for_baseline": program_for_baseline,
        "program_for_llm": ast_program.llm_code,
        "baseline_decision": "UNKNOWN",
        "optional_details": [],
        "timings": {"all": [], "median": 0.0},
        "runlim": [],
    }

    timing_decision_pairs: list[tuple[float, str]] = []

    with tempfile.TemporaryDirectory(
        prefix=f"baseline_{program_path.stem}_"
    ) as temp_reports_dir:
        temp_reports_path = Path(temp_reports_dir)
        baseline_path = temp_reports_path / f"base_{program_path.stem}.c"
        baseline_path.write_text(program_for_baseline)

        for _ in tqdm(
            range(cfg.verifier.k),
            desc=f"Running UAutomizer {verifier.version}",
            leave=False,
        ):
            logger.info(f"Verifying: {baseline_path.name}")
            report = verifier.verify(
                program_path=baseline_path,
                reports_dir=temp_reports_path,
                timeout_seconds=cfg.verifier.timeout,
            )
            timing_decision_pairs.append((report.time_taken, report.decision))
            logger.info(
                f"Report: {report.decision} ({report.decision_reason}) "
                f"in {report.time_taken}s"
            )
            result["timings"]["all"].append(report.time_taken)
            result["runlim"].append(report.to_dict().get("runlim", {}))

        # Determine median timing and corresponding decision.
        all_timings = result["timings"]["all"]
        median_time = statistics.median(all_timings)
        result["timings"]["median"] = median_time

        decisions_at_median = [
            d for t, d in timing_decision_pairs if t == median_time
        ]
        if decisions_at_median:
            result["baseline_decision"] = max(
                set(decisions_at_median), key=decisions_at_median.count
            )
        elif median_time >= cfg.verifier.timeout:
            result["baseline_decision"] = "TIMEOUT"
        else:
            closest_pair = min(
                timing_decision_pairs, key=lambda x: abs(x[0] - median_time)
            )
            result["baseline_decision"] = closest_pair[1]

        result["optional_details"].append(report.decision_reason)

        # Extract invariants from the last verifier run.
        if not cfg.processing.ignore_invariants:
            if verifier.version == "23":
                log_file = temp_reports_path / f"base_{program_path.stem}.log"
                invariants = extract_invariants_from_log(log_file)
            elif verifier.version in ["24", "25", "26"]:
                witness_yml = (
                    temp_reports_path / f"base_{program_path.stem}_witness.yml"
                )
                invariants = extract_invariants_from_witness(witness_yml)

            available_markers = ast_program.get_available_markers()
            program_with_markers = ast_program.marked_code_from_ast
            result["invariants"] = [
                get_invariant_with_marker(inv, program_with_markers, available_markers)
                for inv in invariants
            ]

    return result


# ---------------------------------------------------------------------------
# Full-suite baseline run
# ---------------------------------------------------------------------------

def run_uautomizer_as_baseline(
    cfg: DictConfig, verifier: UAutomizerVerifier, results_path: Path
) -> list[dict[str, Any]]:
    """Run UAutomizer on every program in the configured dataset.

    Results are incrementally written to *results_path* after each program.
    """
    original_programs_dir = GC.DATASET_DIR / cfg.dataset.type / "orig_programs"
    programs = fetch_original_programs(
        original_programs_dir, cfg.dataset.prefix, cfg.dataset.limit
    )
    results: list[dict[str, Any]] = []
    start_time = time.time()

    for program_path in tqdm(programs, desc="Processing C programs"):
        result = process_program(cfg=cfg, program_path=program_path, verifier=verifier)
        if result:
            median_time = result["timings"]["median"]
            result["split"] = (
                "easy" if median_time <= cfg.processing.difficulty_threshold else "hard"
            )
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    return results


# ---------------------------------------------------------------------------
# HuggingFace dataset creation
# ---------------------------------------------------------------------------

def create_hf_base_dataset(
    results_path: Path, output_dir: Path, push_to_hub: bool = False
) -> DatasetDict:
    """Create a HuggingFace DatasetDict from baseline evaluation JSON."""
    logger.info(f"Creating dataset from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)

    filtered = [
        e for e in results if e["baseline_decision"] in {"TRUE", "FALSE", "TIMEOUT"}
    ]
    logger.info(
        f"Kept {len(filtered)}/{len(results)} results "
        f"(dropped {len(results) - len(filtered)} with errors)"
    )

    results_dict: dict[str, list] = {
        "id": list(range(1, len(filtered) + 1)),
        "file": [e["file"] for e in filtered],
        "original_program": [e["original_program"] for e in filtered],
        "program_for_baseline": [e["program_for_baseline"] for e in filtered],
        "baseline_decision": [e["baseline_decision"] for e in filtered],
        "timings": [e["timings"]["all"] for e in filtered],
        "median_timing": [e["timings"]["median"] for e in filtered],
        "program_for_llm": [e["program_for_llm"] for e in filtered],
        "split": [e["split"] for e in filtered],
    }

    if filtered and "invariants" in filtered[0]:
        results_dict["invariants"] = [
            [
                {
                    "line": inv["line"],
                    "invariant": inv["invariant"],
                    "marker": inv.get("marker", ""),
                }
                for inv in e.get("invariants", [])
            ]
            for e in filtered
        ]

    dataset = Dataset.from_dict(results_dict)
    easy = dataset.filter(lambda x: x["split"] == "easy")
    hard = dataset.filter(lambda x: x["split"] == "hard")

    dataset_dict = DatasetDict({"full": dataset, "easy": easy, "hard": hard})
    logger.info(f"Created: {len(dataset)} full, {len(easy)} easy, {len(hard)} hard")

    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved to: {output_dir}")

    if push_to_hub:
        dataset_dict.push_to_hub(f"idopinto/{output_dir.name}")
        logger.info(
            f"Pushed to: https://huggingface.co/datasets/idopinto/{output_dir.name}"
        )

    return dataset_dict

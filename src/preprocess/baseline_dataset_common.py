#!/usr/bin/env python3
"""Common functions for building baseline datasets (both train and eval)."""

import json
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from pycparser import c_parser
from tqdm import tqdm

from configs import global_config as GC
from src.verifiers.uautomizer_runlim import UAutomizerVerifier
from src.preprocess.program import Program as AstProgram

# logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
parser = c_parser.CParser()


def fetch_original_programs(
    original_programs_dir: Path, prefix: str = "", limit: int = -1
) -> List[Path]:
    """
    Get all C files in the original programs directory.
    Args:
        original_programs_dir: Path to the original programs directory.
        prefix: Prefix to filter files by (used for testing specific files).
        limit: Limit the number of files to process. (limit for testing)
    Returns:
        List[Path]: List of C files.
    """
    c_files = list(original_programs_dir.glob(f"{prefix}*.c"))
    if not c_files:
        logger.error(f"Error: No C files found in {original_programs_dir}")
        sys.exit(1)
    total_files = len(c_files)

    logger.info(f"Found {total_files} C files in {original_programs_dir}")
    if limit >= 0:
        c_files = c_files[:limit]
        logger.info(f"Processing first {len(c_files)} files (--limit={limit})")
    return c_files


def extract_invariants_from_log(log_file: Path) -> List[Dict[str, Any]]:
    """
    Extract invariants from UAutomizer log file. (2023 version, doesn't have witness file)
    Args:
        log_file: Path to the UAutomizer log file.
    Returns:
        List[Dict[str, Any]]: List of invariants.
    """
    try:
        invariants = []
        with open(log_file, "r", encoding="utf-8", errors="ignore") as logf:
            for line in logf:
                if "InvariantResult [Line:" in line:
                    idx1 = line.find("InvariantResult [Line:")
                    idx2 = line.find("]:", idx1)
                    if idx1 != -1 and idx2 != -1:
                        try:
                            line_num_str = line[
                                idx1 + len("InvariantResult [Line:") : idx2
                            ]
                            line_num = int(line_num_str.strip())
                        except Exception:
                            continue
                        next_line = next(logf, None)
                        if next_line is not None:
                            invariant_tag = "Derived loop invariant:"
                            inv_idx = next_line.find(invariant_tag)
                            if inv_idx != -1:
                                inv_val = next_line[
                                    inv_idx + len(invariant_tag) :
                                ].strip()
                                if inv_val:
                                    invariants.append(
                                        {"line": line_num, "invariant": inv_val}
                                    )
        return invariants
    except Exception as e:
        logger.warning(f"Could not extract invariants from {log_file}: {e}")
        return []


def extract_invariants_from_witness(witness_yml: Path) -> List[Dict[str, Any]]:
    """
    Extract invariants from UAutomizer witness.yml file. (2024+ version, has witness files)
    Pattern to match in the witness file:
      - InvariantResult [Line: X]: Loop Invariant
        Derived loop invariant: <invariant_text>
    Args:
        witness_yml: Path to the UAutomizer witness.yml file.
    Returns:
        List[Dict[str, Any]]: List of invariants.

    """
    try:
        invariants = []
        with open(witness_yml, "r", encoding="utf-8", errors="ignore") as f:
            yml_data = yaml.safe_load(f)

        content = yml_data[0]["content"]
        for invariant_dict in content:
            invariant = invariant_dict["invariant"]
            if invariant.get("type") == "loop_invariant":
                location = invariant.get("location")
                line = location.get("line")
                value = invariant.get("value")
                if line is not None and value is not None:
                    invariants.append({"line": line, "invariant": value})
        return invariants
    except Exception as e:
        logger.warning(f"Could not extract invariants from {witness_yml}: {e}")
        return []


def get_ast_program(program_path: Path) -> AstProgram:
    ast_program = AstProgram().from_file_path(program_path)
    ast_program.process(print_ast=False)
    return ast_program


def map_invariant_line_to_marker(
    invariant_line: int, program_code: str, available_markers: List[str]
) -> Optional[str]:
    """
    Map a UAutomizer invariant line number to the corresponding loop marker.

    UAutomizer reports invariants with line numbers from the original program.
    Markers are inserted at the start of loop bodies. This function finds which
    marker is closest to (or at) the given line number.

    Args:
        invariant_line: Line number from UAutomizer invariant (1-indexed)
        program_code: The program code with markers (e.g., program_for_llm or marked_code_from_ast)
        available_markers: List of available marker names (e.g., ["INVARIANT_MARKER_1", "INVARIANT_MARKER_2"])

    Returns:
        Marker name if found, None otherwise
    """

    # Find all markers and their line numbers in the program
    marker_positions = {}
    lines = program_code.split("\n")

    for marker in available_markers:
        if marker == "TARGET_ASSERT_MARKER":
            continue  # Skip target marker
        # Find all occurrences of this marker
        for line_idx, line in enumerate(lines, start=1):
            if marker in line:
                # Use the first occurrence (markers are at loop starts)
                if marker not in marker_positions:
                    marker_positions[marker] = line_idx

    if not marker_positions:
        return None

    # Find the marker that is closest to (and <=) the invariant line
    # Markers are inserted at loop starts, so invariant should be >= marker line
    candidates = {
        marker: line_num
        for marker, line_num in marker_positions.items()
        if line_num <= invariant_line
    }

    if not candidates:
        # If no marker is before the invariant line, use the first marker
        # (invariant might be at the very start)
        return min(marker_positions.items(), key=lambda x: x[1])[0]

    # Return the marker with the highest line number that's still <= invariant_line
    return max(candidates.items(), key=lambda x: x[1])[0]


def get_invariant_with_marker(
    invariant: Dict[str, Any], program_code: str, available_markers: List[str]
) -> Dict[str, Any]:
    """
    Add marker information to a UAutomizer invariant.

    Args:
        invariant: Dict with "line" and "invariant" keys
        program_code: Program code with markers
        available_markers: List of available marker names

    Returns:
        Dict with added "marker" key, or original dict if no match found
    """
    marker = map_invariant_line_to_marker(
        invariant["line"], program_code, available_markers
    )
    result = invariant.copy()
    if marker:
        result["marker"] = marker
    return result


def process_program(
    cfg: DictConfig, program_path: Path, verifier: UAutomizerVerifier
) -> Dict[str, Any]:
    """
    Process a single C program with UAutomizer.

    Returns:
        Dictionary with timing and invariant information
    """
    try:
        ast_program = get_ast_program(program_path)
    except Exception as e:
        logger.error(f"Error getting AST program: {e}")
        return {}
    if len(ast_program.assertions) == 0:
        logger.error("No target assertion found in the program")
        return {}
    elif len(ast_program.assertions) > 1:
        logger.error("Multiple target assertions found in the program")
        return {}
    target_assert = ast_program.get_target_assert()
    program_for_baseline = ast_program.get_program_with_assertion(
        predicate=target_assert, assumptions=[], for_llm=False
    )
    result = {
        "file": program_path.name,
        "original_program": ast_program.code,
        "program_for_baseline": program_for_baseline,
        "program_for_llm": ast_program.llm_code,
        "baseline_decision": "UNKNOWN",
        "optional_details": [],
        "timings": {
            "all": [],
            "median": 0.0,
        },  # Store all timings as list to handle duplicates
        "runlim": [],
        # "program_with_all_markers": ast_program.marked_code_from_ast,
        # "markers": ast_program.get_available_markers(),
    }
    timing_decision_pairs = []  # List of (time, decision) tuples
    with tempfile.TemporaryDirectory(
        prefix=f"baseline_{program_path.stem}_"
    ) as temp_reports_dir:
        temp_reports_path = Path(temp_reports_dir)
        program_for_baseline_path = temp_reports_path / f"base_{program_path.stem}.c"
        program_for_baseline_path.write_text(program_for_baseline)
        # print("program_for_baseline")
        # print(program_for_baseline)
        for i in tqdm(
            range(cfg.verifier.k),
            desc=f"Running UAutomizer {verifier.version}",
            leave=False,
        ):
            logger.info(f"Verifying the program: {program_for_baseline_path.name}")
            report = verifier.verify(
                program_path=program_for_baseline_path,
                reports_dir=temp_reports_path,
                timeout_seconds=cfg.verifier.timeout,
            )
            timing_decision_pairs.append((report.time_taken, report.decision))
            logger.info(
                f"Report: {report.decision} ({report.decision_reason}) in {report.time_taken} seconds"
            )
            result["timings"]["all"].append(report.time_taken)
            result["runlim"].append(report.to_dict().get("runlim", {}))

        # Calculate median from all timing values (including duplicates)
        all_timings = result["timings"]["all"]
        result["timings"]["median"] = statistics.median(all_timings)

        # Find the decision corresponding to the median time
        # If median time appears multiple times, use the most common decision at that time
        median_time = result["timings"]["median"]
        decisions_at_median = [
            decision for time, decision in timing_decision_pairs if time == median_time
        ]
        if decisions_at_median:
            # Use the most common decision at median time, or first if all same
            result["baseline_decision"] = max(
                set(decisions_at_median), key=decisions_at_median.count
            )
        elif median_time >= cfg.verifier.timeout:
            result["baseline_decision"] = "TIMEOUT"
        else:
            # Fallback: use decision from closest timing
            closest_pair = min(
                timing_decision_pairs, key=lambda x: abs(x[0] - median_time)
            )
            result["baseline_decision"] = closest_pair[1]

        result["optional_details"].append(report.decision_reason)

        if not cfg.processing.ignore_invariants:
            # Extracting the invariant only from the last verifier run.
            # prefix = "orig" if cfg.processing.verify_original_programs else "normalized"
            if verifier.version == "23":
                log_file = temp_reports_path / f"base_{program_path.stem}.log"
                invariants = extract_invariants_from_log(log_file)
            elif verifier.version in ["24", "25", "26"]:
                witness_yml = (
                    temp_reports_path / f"base_{program_path.stem}_witness.yml"
                )
                invariants = extract_invariants_from_witness(witness_yml)

            # Map invariants to markers
            available_markers = ast_program.get_available_markers()
            # PATCH = (
            # "void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }\n"
            # "void assume(int cond) { if (!cond) { abort(); } }\n"
            # )
            program_with_markers = ast_program.marked_code_from_ast
            # print(program_with_markers)
            invariants_with_markers = [
                get_invariant_with_marker(inv, program_with_markers, available_markers)
                for inv in invariants
            ]
            result["invariants"] = invariants_with_markers

    return result


def run_uautomizer_as_baseline(
    cfg: DictConfig, verifier: UAutomizerVerifier, results_path: Path
) -> List[Dict[str, Any]]:
    """
    Run UAutomizer as a baseline for the given configuration.
    Args:
        verifier: UAutomizerVerifier instance.
        cfg: Hydra configuration.
        results_path: Path to the results file.
    Returns:
        List[Dict[str, Any]]: List of results.
    """
    original_programs_dir = GC.DATASET_DIR / cfg.dataset.type / "orig_programs"
    programs = fetch_original_programs(
        original_programs_dir, cfg.dataset.prefix, cfg.dataset.limit
    )
    results = []
    start_time = time.time()
    for i, program_path in enumerate(tqdm(programs, desc="Processing C programs")):
        result = process_program(cfg=cfg, program_path=program_path, verifier=verifier)
        if result:
            median_time = result["timings"]["median"]
            result["split"] = (
                "easy" if median_time <= cfg.processing.difficulty_threshold else "hard"
            )
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time} seconds")
    return results


def create_hf_base_dataset(
    results_path: Path, output_dir: Path, push_to_hub: bool = False
) -> DatasetDict:
    """Create HF dataset from evaluation JSON. Optionally push to Hub."""
    logger.info(f"Creating evaluation dataset from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} results")
    filtered_results = [
        e for e in results if e["baseline_decision"] in {"TRUE", "FALSE", "TIMEOUT"}
    ]
    logger.info(
        f"Filtered {len(results) - len(filtered_results)} results out of {len(results)} examples that had errors. kept {len(filtered_results)} results."
    )
    results_dict = {
        "id": [i + 1 for i in range(len(filtered_results))],
        "file": [e["file"] for e in filtered_results],
        "original_program": [f"{e['original_program']}" for e in filtered_results],
        "program_for_baseline": [
            f"{e['program_for_baseline']}" for e in filtered_results
        ],
        "baseline_decision": [e["baseline_decision"] for e in filtered_results],
        "timings": [e["timings"]["all"] for e in filtered_results],
        "median_timing": [e["timings"]["median"] for e in filtered_results],
        "program_for_llm": [f"{e['program_for_llm']}" for e in filtered_results],
        "split": [e["split"] for e in filtered_results],
    }
    if "invariants" in filtered_results[0]:
        results_dict["invariants"] = [
            [
                {
                    "line": inv["line"],
                    "invariant": inv["invariant"],
                    "marker": inv.get("marker", ""),
                }
                for inv in e.get("invariants", [])
            ]
            for e in filtered_results
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

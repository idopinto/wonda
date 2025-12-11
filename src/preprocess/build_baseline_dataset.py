#!/usr/bin/env python3
import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from datasets import Dataset, DatasetDict
from pycparser import c_parser
from tqdm import tqdm

from src.preprocess.sft_tasks import task_generate_all_invariants, task_generate_one_invariant
from src.utils.paths import DATASET_DIR, PROPERTIES_DIR, UAUTOMIZER_PATHS
from src.utils.plain_verifier import run_uautomizer
from src.utils.program import Program
from src.utils.rewriter import Rewriter
from src.utils.utils import write_file

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = c_parser.CParser()

@dataclass
class BaselineConfig:
    """Configuration class for the baseline evaluation or training dataset creation."""
    # Core settings
    dataset_type: str = "evaluation" # 'training' or 'evaluation'
    uautomizer_version: str = "25" # '23', '24', '25', '26'. specific version can be found in the README.md
    arch: str = "32bit" # '32bit' or '64bit'
    
    # Paths
    original_programs_dir: Path = field(default=None)
    property_file_path: Path = PROPERTIES_DIR / "unreach-call.prp"

    # Timing & thresholds
    timeout_threshold: float = 600.0 # verifier timeout value
    difficulty_threshold: int = 30 # median time threshold for easy/hard split
    k: int = 1 # number of times to run UAutomizer for each file (to avoid outliers)
    
    # Filters
    limit: int = -1 # limit the number of files to process (useful for testing)
    prefix: str = "" # prefix to filter files by
    
    # Behavioral flags
    verify_original_programs: bool = False # verify original programs instead of normalized
    ignore_invariants: bool = False # skip invariant extraction from log/witness files

    def __post_init__(self):
        if self.original_programs_dir is None:
            self.original_programs_dir = DATASET_DIR / self.dataset_type / "orig_programs"

def fetch_original_programs(original_programs_dir: Path, prefix: str = "", limit: int = -1) -> List[Path]:
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
                            line_num_str = line[idx1 + len("InvariantResult [Line:"):idx2]
                            line_num = int(line_num_str.strip())
                        except Exception:
                            continue
                        next_line = next(logf, None)
                        if next_line is not None:
                            invariant_tag = "Derived loop invariant:"
                            inv_idx = next_line.find(invariant_tag)
                            if inv_idx != -1:
                                inv_val = next_line[inv_idx + len(invariant_tag):].strip()
                                if inv_val:
                                    invariants.append({
                                        "line": line_num,
                                        "invariant": inv_val
                                    })
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
        
        content = yml_data[0]['content']
        for invariant_dict in content:
            invariant = invariant_dict['invariant']
            if invariant.get('type') == 'loop_invariant':
                location = invariant.get('location')
                line = location.get('line')
                value = invariant.get('value')
                if line is not None and value is not None:
                    invariants.append({
                        "line": line,
                        "invariant": value
                    })
        return invariants
    except Exception as e:
        logger.warning(f"Could not extract invariants from {witness_yml}: {e}")
        return []

def check_correct_syntax(program_str: str) -> bool:
    """
    Check if the program has correct C syntax using pycparser.
    
    Assumes:
        - Program is already preprocessed (no #include, #define, etc. from rewriter)
        - Program uses C89/C90 standard (pycparser limitation)
        - No GCC/Clang extensions are used
    
    Args:
        program_str: String representation of the program.
    Returns:
        bool: True if the program has correct C syntax, False otherwise.
    """
    try:
        parser.parse(program_str)
        return True
    except Exception:
        return False
       
def normalize_program(file_path: Path) -> Tuple[str, Dict[str, bool]]:
    """
    Normalize a C file with rewriter and checks for the following issues:
        1. there is no target assertion in the program
        2. there are multiple target assertions in the program (should be only one)
        3. the program has invalid C syntax (checked by pycparser)
        if any of the issues are found, returning empty string and the issues found
    Args:
        file_path: Path to the C file.
    Returns:
        Tuple[str, Dict[str, bool]]: The normalized program string, and a dictionary with the issues found.
    """
    r = Rewriter(file_path)
    program = Program(r.lines_to_verify, r.replacement)
    problems = {
        "no_target_assertion": False,
        "multiple_target_assertions": False,
        "bad_syntax": False,
    }
    program_str = ""
    if len(program.assertions) == 0:
        problems["no_target_assertion"] = True
        return program_str, problems

    if len(program.assertions) > 1:
        problems["multiple_target_assertions"] = True
        return program_str, problems
    
    target_assert = program.assertions[0] # assuming there is only one assertion in the program
    program_str = program.get_program_with_assertion(predicate=target_assert, 
                                                    assumptions=[],
                                                    assertion_points={},
                                                    forGPT=False)
    if not check_correct_syntax(program_str):
        problems["bad_syntax"] = True
        return "", problems
    return program_str, problems

def process_program(program_path: Path, config: BaselineConfig) -> Dict[str, Any]:
    """
    Process a single C program with UAutomizer.
    
    Returns:
        Dictionary with timing and invariant information
    """
    orig_program_str = program_path.read_text()
    normalized_program_str, problems = normalize_program(program_path)
    
    result = {
            "file": program_path.name,
            "original_program": orig_program_str,
            "normalized_program": normalized_program_str,
            "baseline_decision": "UNKNOWN",
            "optional_details": [],
            "timings": {
                "all": {},
                "median": 0.0
            },
    }
    if problems["no_target_assertion"] or problems["bad_syntax"]:
        msgs = []
        if problems["no_target_assertion"]:
            msgs.append("no_target_assertion")
        if problems["multiple_target_assertions"]:
            msgs.append("multiple_target_assertions")
        if problems["bad_syntax"]:
            msgs.append("bad_syntax")
        result["optional_details"] = msgs
    else:
        decision_time = {}
        with tempfile.TemporaryDirectory(prefix=f"baseline_{program_path.stem}_") as temp_reports_dir:
            temp_reports_path = Path(temp_reports_dir)
            # Write programs to temp dir
            orig_program_path = temp_reports_path / f"orig_{program_path.stem}.c"
            normalized_program_path = temp_reports_path / f"normalized_{program_path.stem}.c"
            write_file(orig_program_path, orig_program_str)
            write_file(normalized_program_path, normalized_program_str)
            try:
                for i in tqdm(range(config.k), desc=f"Running UAutomizer{config.uautomizer_version}", leave=False):
                    program_to_verify = orig_program_path if config.verify_original_programs else normalized_program_path
                    logger.info(f"Verifying the program: {program_to_verify.name}")
                    report = run_uautomizer(
                        program_path=program_to_verify,
                        property_file_path=config.property_file_path,
                        reports_dir=temp_reports_path,
                        timeout_seconds=config.timeout_threshold,
                        uautomizer_path=UAUTOMIZER_PATHS[config.uautomizer_version],
                        arch=config.arch
                    )
                    decision_time[report.time_taken] = report.decision
                    logger.info(f"Report: {report.decision} ({report.decision_reason}) in {report.time_taken} seconds")
                    result["timings"]["all"][report.time_taken] = report.decision
                result["timings"]["median"] = statistics.median(list(result["timings"]["all"].keys()))
                # {600: TIMEOUT, 550: TRUE, 600: TIMEOUT} -> median
                logger.info(f"Decision time dict: {decision_time}")
                if result["timings"]["median"] < config.timeout_threshold:
                    result["baseline_decision"] = decision_time[result["timings"]["median"]]
                else:
                    result["baseline_decision"] = "TIMEOUT"
                result["optional_details"].append(report.decision_reason)
                
                if not config.ignore_invariants:
                    # Extracting the invariant only from the last verifier run.
                    prefix = 'orig' if config.verify_original_programs else 'normalized' 
                    if config.uautomizer_version == "23":
                        log_file = temp_reports_path / f"{prefix}_{program_path.stem}.log"
                        invariants = extract_invariants_from_log(log_file)
                    elif config.uautomizer_version in ["24", "25", "26"]:
                        witness_yml = temp_reports_path / f"{prefix}_{program_path.stem}_witness.yml"
                        invariants = extract_invariants_from_witness(witness_yml)
                    result["invariants"] = invariants

            except Exception as e:
                logger.error(f"Error processing {program_path.name}: {e}")
                result["baseline_decision"] = "ERROR"
                result["optional_details"].append(str(e))

    return result

def run_uautomizer_as_baseline(config: BaselineConfig, results_path: Path) -> List[Dict[str, Any]]:
    """
    Run UAutomizer as a baseline for the given configuration.
    Args:
        config: BaselineConfig configuration.
        results_path: Path to the results file.
    Returns:
        List[Dict[str, Any]]: List of results.
    """
    programs = fetch_original_programs(config.original_programs_dir, config.prefix, config.limit)
    results = []
    start_time = time.time()
    for i, program_path in enumerate(tqdm(programs, desc="Processing C programs")):
        result = process_program(
            program_path=program_path,
            config=config
        )
        if result:
            median_time = result["timings"]["median"]
            result["split"] = "easy" if median_time <= config.difficulty_threshold else "hard"
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time} seconds")
    return results

def create_evaluation_dataset(results_path: Path, output_dir: Path, push_to_hub: bool = False) -> DatasetDict:
    """Create HF dataset from evaluation JSON. Optionally push to Hub."""
    logger.info(f"Creating evaluation dataset from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} results")
    filtered_results = [e for e in results if e['baseline_decision'] in {"TRUE", "FALSE", "TIMEOUT"}]
    logger.info(f"Filtered {len(results) - len(filtered_results)} results out of {len(results)} examples that had errors. kept {len(filtered_results)} results.")
    results_dict = {
        "id": [i+1 for i in range(len(filtered_results))],
        "file": [e["file"] for e in filtered_results],
        "original_program": [f"{e['original_program']}" for e in filtered_results],
        "normalized_program": [f"{e['normalized_program']}" for e in filtered_results],
        "baseline_decision": [e["baseline_decision"] for e in filtered_results],
        "median_timing": [e["timings"]["median"] for e in filtered_results],
        "split": [e["split"] for e in filtered_results],
    }
    if "invariants" in filtered_results[0]:
        results_dict["invariants"] = [[{"line": inv["line"], "invariant": inv["invariant"]} for inv in e.get("invariants", [])] for e in filtered_results]
    dataset = Dataset.from_dict(results_dict)

    easy = dataset.filter(lambda x: x["split"] == "easy")
    hard = dataset.filter(lambda x: x["split"] == "hard")
    
    dataset_dict = DatasetDict({"easy": easy, "hard": hard})
    logger.info(f"Created: {len(dataset)} total, {len(easy)} easy, {len(hard)} hard")
    
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved to: {output_dir}")
    
    if push_to_hub:
        dataset_dict.push_to_hub(f"idopinto/{output_dir.name}")
        logger.info(f"Pushed to: https://huggingface.co/datasets/{output_dir.name}")
    
    return dataset_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline data collection from training or evaluation dataset."
    )
    parser.add_argument("--dataset-type", type=str, required=False, default="evaluation", choices=["training", "evaluation"], help="Dataset type (training or evaluation)")
    parser.add_argument("--uautomizer-version", type=str, required=False, default="25", choices=["23", "24", "25", "26"], help="UAutomizer version (23, 24, 25, or 26). specific versions detailed in the README.md file.")
    parser.add_argument("--k", type=int, required=False, default=3, help="Number of times to run UAutomizer for each file (default: 1)")
    parser.add_argument("--limit", type=int, required=False, default=-1, help="Limit the number of files to process (useful for testing). Processes first k files.")
    parser.add_argument("--prefix", type=str, required=False, default="", help="Filter files by prefix (used for testing specific files).")
    parser.add_argument("--verify-original-programs", action="store_true", help="Verify the original programs with UAutomizer and not the normalized programs (default: verify the normalized programs)")
    parser.add_argument("--ignore-invariants", action="store_true", help="Do not extract invariants from the log file or the witness file (default: try to extract from the .log (2023 version) or witness file (2024+))")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset to the Hugging Face hub")
    return parser.parse_args()
    
def main():
    args = parse_args()
    baseline_config = BaselineConfig(
        dataset_type=args.dataset_type,
        uautomizer_version=args.uautomizer_version,
        k=args.k,
        limit=args.limit,
        prefix=args.prefix,
        verify_original_programs=args.verify_original_programs,
        ignore_invariants=args.ignore_invariants,
    )
    logger.info(baseline_config)
    output_dir = DATASET_DIR / args.dataset_type / f"invbench-{args.dataset_type}-uautomizer{args.uautomizer_version}-k{args.k}-test"
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{output_dir.name}.json"
    run_uautomizer_as_baseline(config=baseline_config, results_path=results_path)
    if args.dataset_type == "evaluation":
        create_evaluation_dataset(results_path=results_path, output_dir=output_dir, push_to_hub=args.push_to_hub)
    elif args.dataset_type == "training":
        output_dir_all_invariants = output_dir / f"invbench-{args.dataset_type}-uautomizer{args.uautomizer_version}-k{args.k}-gen-all"
        output_dir_one_invariant = output_dir / f"invbench-{args.dataset_type}-uautomizer{args.uautomizer_version}-k{args.k}-gen-one"
        task_generate_all_invariants(results_path=results_path, output_dir=output_dir_all_invariants, push_to_hub=args.push_to_hub)
        task_generate_one_invariant(results_path=results_path, output_dir=output_dir_one_invariant, push_to_hub=args.push_to_hub)

if __name__ == "__main__":
    main()


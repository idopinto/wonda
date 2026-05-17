"""
Data loading and preprocessing module for evaluation.

Handles dataset loading from HuggingFace or local C files and example preprocessing.
"""

import copy as py_copy
import json
import logging
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from wonda.core.ast_program import AstProgram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def expand_dataset_per_marker(dataset: list[dict]) -> list[dict]:
    """
    Expand the dataset into one example per loop marker (INVARIANT_MARKER_k).
    """
    logger.info(f"Expanding dataset to per-marker examples. Original size: {len(dataset)}")
    expanded = []
    for ex in dataset:
        prog = preprocess_example(ex)["program"]
        k_max = prog.num_loops
        if k_max == 0:
            continue
        for k in range(1, k_max + 1):
            ex_k = dict(ex)
            ex_k["original_program"] = py_copy.deepcopy(prog).code
            ex_k["target_marker"] = f"INVARIANT_MARKER_{k}"
            expanded.append(ex_k)
    logger.info(f"Expanded {len(dataset)} examples to {len(expanded)} per-marker examples")
    return expanded

def load_local_evaluation_dataset(
    local_dir: str,
    default_median_timing: float = 600.0,
    baseline_json: Optional[str] = None,
) -> List[dict]:
    """Load C/I source files from a local directory into the same schema as the HF dataset.

    When *baseline_json* points to a baseline_dataset.json produced by the
    baseline-timing scripts, per-program ``median_timing`` values are read from
    that file.  Programs not found in the JSON fall back to *default_median_timing*.
    """
    from configs import global_config as GC

    local_path = Path(local_dir)
    if not local_path.is_absolute():
        local_path = GC.ROOT_DIR / local_path

    src_files = sorted([*local_path.glob("*.c"), *local_path.glob("*.i")])
    if not src_files:
        raise FileNotFoundError(f"No .c or .i files found in {local_path}")

    timing_lookup: dict[str, float] = {}
    if baseline_json:
        bj_path = Path(baseline_json)
        if not bj_path.is_absolute():
            bj_path = GC.ROOT_DIR / bj_path
        with open(bj_path) as f:
            for entry in json.load(f):
                timing_lookup[entry["file"]] = entry["median_timing"]
        logger.info(f"Loaded baseline timings for {len(timing_lookup)} programs from {bj_path}")

    dataset = []
    for src_file in src_files:
        dataset.append({
            "file": src_file.name,
            "original_program": src_file.read_text().strip(),
            "median_timing": timing_lookup.get(src_file.name, default_median_timing),
        })
    logger.info(f"Loaded {len(dataset)} source files from {local_path}")
    return dataset


def get_evaluation_dataset(
    dataset_name: str,
    limit: int = -1,
    prefix: Optional[str] = None,
    split: str = "test",
    re_split: bool = True,
    difficulty_threshold: int = 15,
    eval_per_marker: bool = True,
    local_dir: Optional[str] = None,
    default_median_timing: float = 600.0,
    baseline_json: Optional[str] = None,
) -> List[dict]:
    """
    Load and filter evaluation dataset from HuggingFace or a local directory of C files.

    Args:
        dataset_name: HuggingFace dataset identifier (ignored when local_dir is set).
        limit: Maximum number of examples to return (-1 for all).
        prefix: Filter examples by file prefix (use "all" to skip filtering).
        split: Dataset split to load.
        local_dir: Path to a directory of .c files. When set, loads from disk
            instead of HuggingFace and skips re_split.
        default_median_timing: Fallback median_timing for local files (default 600s).
        baseline_json: Path to baseline_dataset.json with per-program median_timing
            values. Used only when local_dir is set.

    Returns:
        List of dataset examples as dictionaries.
    """
    if local_dir:
        dataset = load_local_evaluation_dataset(local_dir, default_median_timing, baseline_json)
    elif re_split:
        dataset = load_dataset(dataset_name, split="full")
        easy = dataset.filter(lambda x: x["median_timing"] <= difficulty_threshold)
        hard = dataset.filter(lambda x: x["median_timing"] > difficulty_threshold)
        dataset = easy if split == "easy" else hard
        logger.info(f"Split dataset into easy and hard. Easy size: {len(easy)}, Hard size: {len(hard)}, Total size: {len(dataset)}")
        logger.info(f"Loading {split} split")
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Filter by prefix if specified
    if prefix and prefix != "all":
        dataset = [sample for sample in dataset if sample["file"].startswith(prefix)]
    elif prefix != "all":
        # Convert to list if not already filtered
        dataset = list(dataset)
    logger.info(f"Loaded {len(dataset)} examples")
    if eval_per_marker and dataset and "target_marker" not in dataset[0]:
        dataset = expand_dataset_per_marker(dataset)
    else:
        logger.info(f"Dataset is already expanded into one example per loop marker (INVARIANT_MARKER_k) (per-marker mode)")
    # Apply limit if specified
    if limit > 0:
        dataset = dataset[:limit]
    logger.info(f"Final dataset size: {len(dataset)}. (--limit={limit})")
    return dataset


def preprocess_example(example: dict) -> dict:
    """
    Preprocess example for the new marker-based evaluation pipeline.

    Returns an AST-based Program that already ran .process(), so it has:
    - llm_code containing INVARIANT_MARKER_k and the target assert at its original location
    - target assertion stored as Property(marker_name=TARGET_ASSERT_MARKER)
    """
    code = str(example["original_program"])
    program = AstProgram().from_code(code)
    program.process(print_ast=False)
    return {
        "program": program,
        "baseline_timing": example["median_timing"],
    }


def preprocess_for_model(example: dict, eval_per_marker: bool) -> dict:
    """
    One preprocess function that supports both modes.

    - Always returns an AST-based Program under key `program`
    - Always returns `baseline_timing`
    - If eval_per_marker=True, also returns `target_marker` (required by per-marker prompt)
    """
    out = preprocess_example(example)
    if eval_per_marker:
        target_marker = example["target_marker"]
        out["target_marker"] = str(target_marker)
    return out

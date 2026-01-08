"""
Data loading and preprocessing module for evaluation.

Handles dataset loading from HuggingFace and example preprocessing.
"""

from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from src.utils.program import Program
from src.utils.program_normalizer import ProgramNormalizer

# New AST-based pipeline (used by src/new_eval/*)
from src.preprocess.program import Program as AstProgram
import copy as py_copy

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def expand_dataset_per_marker(dataset: list[dict]) -> list[dict]:
    logger.info(f"Expanding dataset to per-marker examples. Original size: {len(dataset)}")
    expanded = []
    for ex in dataset:
        prog = preprocess_example(ex)["program"]
        k_max = _get_num_loops(prog)
        if k_max == 0:
            continue
        for k in range(1, k_max + 1):
            ex_k = dict(ex)
            ex_k["original_program"] = py_copy.deepcopy(prog).code
            ex_k["target_marker"] = f"INVARIANT_MARKER_{k}"
            expanded.append(ex_k)
    logger.info(f"Expanded {len(dataset)} examples to {len(expanded)} per-marker examples")
    return expanded

def _get_num_loops(program: Program) -> int:
    return program.num_loops

def get_evaluation_dataset(
    dataset_name: str,
    limit: int = -1,
    prefix: Optional[str] = None,
    split: str = "test",
    re_split: bool = False,
    difficulty_threshold: int = 15,
    eval_per_marker: bool = False,   # If True, expand the dataset into one example per loop marker (INVARIANT_MARKER_k).
) -> List[dict]:
    """
    Load and filter evaluation dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier.
        limit: Maximum number of examples to return (-1 for all).
        prefix: Filter examples by file prefix (use "all" to skip filtering).
        split: Dataset split to load.

    Returns:
        List of dataset examples as dictionaries.
    """
    if re_split:
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
        logger.info(f"Dataset is already per-marker")
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
    - target assertion stored as Predicate(marker_name=TARGET_ASSERT_MARKER)
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

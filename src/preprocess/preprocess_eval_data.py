"""
Data loading and preprocessing module for evaluation.

Handles dataset loading from HuggingFace and example preprocessing.
"""
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

from src.utils.program import Program
from src.utils.program_normalizer import ProgramNormalizer


def get_evaluation_dataset(
    dataset_name: str,
    limit: int = -1,
    prefix: Optional[str] = None,
    split: str = "test"
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
    dataset = load_dataset(dataset_name, split=split)
    
    # Filter by prefix if specified
    if prefix and prefix != "all":
        dataset = [sample for sample in dataset if sample["file"].startswith(prefix)]
    elif prefix != "all":
        # Convert to list if not already filtered
        dataset = list(dataset)
    
    # Apply limit if specified
    if limit > 0:
        dataset = dataset[:limit]
    
    return dataset


def preprocess_example(example: dict) -> dict:
    """
    Preprocess a dataset example for model input.
    
    Converts the raw C program into a Program object suitable for
    the invariant generator model.
    
    Args:
        example: Dataset example containing 'original_program' and 'median_timing'.
        
    Returns:
        Dict with 'program' (Program object) and 'baseline_timing' (float).
    """
    code = str(example["original_program"])
    
    normalizer = ProgramNormalizer(code=code)
    program = Program(normalizer.lines_to_verify, normalizer.replacement)
    
    return {
        "program": program,
        "baseline_timing": example["median_timing"]
    }


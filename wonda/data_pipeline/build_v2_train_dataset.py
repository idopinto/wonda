#!/usr/bin/env python3
"""
Build v2 training dataset from preprocessed JSON and upload to Hugging Face.

This script:
1. Loads the parallel-latest JSON file with ground-truth invariants
2. Filters out trivial invariants (0, 1, true, false)
3. Creates a HuggingFace dataset with train split
4. Optionally pushes to HuggingFace Hub
"""

import argparse
import json
import logging
from pathlib import Path
from datasets import Dataset, DatasetDict

from configs import global_config as GC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_degenerate_invariant(inv: str) -> bool:
    """Check if invariant is degenerate (0, 1, true, false) - UAutomizer artifacts."""
    if not inv:
        return True
    normalized = inv.strip().lower().replace('(', '').replace(')', '').replace(' ', '').replace('!', '')
    return normalized in ('0', '1', 'true', 'false', '')


def load_and_filter_data(json_path: Path) -> list[dict]:
    """Load JSON data and filter out entries with degenerate invariants."""
    logger.info(f"Loading data from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries")
    
    # Filter out entries with degenerate gt_invariant
    filtered = []
    degenerate_count = 0
    for entry in data:
        if 'gt_invariant' in entry and entry['gt_invariant']:
            inv = entry['gt_invariant'].get('target_invariant', '')
            if is_degenerate_invariant(inv):
                degenerate_count += 1
                continue
        filtered.append(entry)
    
    logger.info(f"Filtered out {degenerate_count} entries with degenerate invariants")
    logger.info(f"Remaining: {len(filtered)} entries")
    return filtered


def create_hf_train_dataset(
    data: list[dict],
    output_dir: Path,
    dataset_name: str,
    push_to_hub: bool = False,
) -> DatasetDict:
    """Create HuggingFace dataset from training data."""
    logger.info(f"Creating HF dataset with {len(data)} entries")
    
    # Build dataset dict
    dataset_dict = {
        "id": [],
        "file": [],
        "original_program": [],
        "program_for_llm": [],
        "split": [],
        "gt_invariant_marker": [],
        "gt_invariant_content": [],
        "gt_invariant_quality_grade": [],
        "gt_invariant_is_correct": [],
        "gt_invariant_speedup": [],
    }
    
    for i, entry in enumerate(data):
        gt = entry.get('gt_invariant', {}) or {}
        
        dataset_dict["id"].append(i + 1)
        dataset_dict["file"].append(entry.get("file", ""))
        dataset_dict["original_program"].append(entry.get("original_program", ""))
        dataset_dict["program_for_baseline"].append(entry.get("program_for_baseline", ""))
        dataset_dict["program_for_llm"].append(entry.get("program_for_llm", ""))
        dataset_dict["baseline_decision"].append(entry.get("baseline_decision", ""))
        dataset_dict["split"].append(entry.get("split", ""))
        dataset_dict["gt_invariant_marker"].append(gt.get("marker", ""))
        dataset_dict["gt_invariant_content"].append(gt.get("target_invariant", ""))
        dataset_dict["gt_invariant_quality_grade"].append(gt.get("quality_grade", 0))
        dataset_dict["gt_invariant_is_correct"].append(gt.get("is_correct", False))
        dataset_dict["gt_invariant_speedup"].append(gt.get("speedup", 0.0))
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split by quality grade or difficulty
    grade3 = dataset.filter(lambda x: x["gt_invariant_quality_grade"] == 3)
    grade2 = dataset.filter(lambda x: x["gt_invariant_quality_grade"] >= 2)
    
    dataset_dict_hf = DatasetDict({
        "train": dataset,
        "grade3": grade3,
        "grade2": grade2,
    })
    
    logger.info(f"Created splits: train={len(dataset)}, grade3={len(grade3)}, grade2={len(grade2)}")
    
    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict_hf.save_to_disk(output_dir)
    logger.info(f"Saved to: {output_dir}")
    
    # Push to Hub
    if push_to_hub:
        hub_name = f"idopinto/{dataset_name}"
        dataset_dict_hf.push_to_hub(hub_name)
        logger.info(f"Pushed to: https://huggingface.co/datasets/{hub_name}")
    
    return dataset_dict_hf


def main():
    parser = argparse.ArgumentParser(description="Build v2 training dataset and upload to HuggingFace")
    parser.add_argument(
        "--input",
        type=str,
        default="data/train/invbench-train-uautomizer25-k1-full-raw/invbench-train-uautomizer25-k1-full-parallel-latest.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train/invbench-train-v2",
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="invbench-train-v2",
        help="Dataset name for HuggingFace Hub",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub",
    )
    args = parser.parse_args()
    
    # Resolve paths
    input_path = GC.ROOT_DIR / args.input
    output_dir = GC.ROOT_DIR / args.output_dir
    
    # Load and filter data
    data = load_and_filter_data(input_path)
    
    # Create and upload dataset
    create_hf_train_dataset(
        data=data,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        push_to_hub=args.push_to_hub,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m wonda.data_pipeline.build_v2_train_dataset --push-to-hub
# uv run -m wonda.data_pipeline.build_v2_train_dataset --input data/train/custom.json --dataset-name invbench-train-custom --push-to-hub
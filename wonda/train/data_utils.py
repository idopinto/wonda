import logging

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sft_dataset(
    hf_repo: str | None = None,
    json_path: str | None = None,
    limit: int = -1,
    min_grade: int | None = None,
) -> Dataset:
    """Load a pre-built SFT dataset from HuggingFace Hub or local JSON.

    Exactly one of hf_repo or json_path must be provided.
    If min_grade is set and the dataset has a quality_grade column, keep only samples with quality_grade >= min_grade
    (e.g. load wonda-qwen-nt-sft-v2-g1 then filter to grade >= 2).
    """
    if hf_repo and json_path:
        raise ValueError("Provide exactly one of hf_repo or json_path")
    if not hf_repo and not json_path:
        raise ValueError("Provide either hf_repo or json_path")

    if json_path:
        logger.info(f"Loading SFT dataset from local JSON: {json_path}")
        dataset = load_dataset("json", data_files=json_path, split="train")
    else:
        logger.info(f"Loading SFT dataset from HuggingFace: {hf_repo}")
        dataset = load_dataset(hf_repo, split="train")

    if min_grade is not None and "quality_grade" in dataset.column_names:
        n_before = len(dataset)
        dataset = dataset.filter(lambda x: x["quality_grade"] >= min_grade)
        logger.info(f"Filtered to quality_grade >= {min_grade}: {len(dataset)} / {n_before} samples")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    logger.info(f"Loaded {len(dataset)} SFT samples")
    return dataset


def split_dataset(dataset: Dataset, split_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets."""
    train_size = int(len(dataset) * split_ratio)
    train_dataset = dataset.select(range(train_size))
    validation_dataset = dataset.select(range(train_size, len(dataset)))
    logger.info(
        f"Split dataset into train and validation. Train: {len(train_dataset)}, Validation: {len(validation_dataset)}"
    )
    return train_dataset, validation_dataset


def log_dataset_stats(dataset: Dataset, tokenizer: AutoTokenizer, name: str = "dataset") -> dict:
    """Compute and log statistics for a dataset split. Returns stats dict."""
    from collections import Counter

    stats = {"name": name, "total": len(dataset)}

    if len(dataset) == 0:
        logger.info(f"[{name}] Empty dataset")
        return stats

    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
    stats["token_length"] = {
        "mean": sum(token_lengths) / len(token_lengths),
        "min": min(token_lengths),
        "max": max(token_lengths),
    }

    grade_col = "quality_grade" if "quality_grade" in dataset.column_names else ("grade" if "grade" in dataset.column_names else None)
    if grade_col:
        grades = [sample[grade_col] for sample in dataset]
        grade_counts = Counter(grades)
        stats["grade"] = {
            "mean": sum(grades) / len(grades),
            "distribution": dict(sorted(grade_counts.items())),
        }

    if "invariant_token_length" in dataset.column_names:
        inv_lengths = [sample["invariant_token_length"] for sample in dataset]
        stats["invariant_token_length"] = {
            "mean": sum(inv_lengths) / len(inv_lengths),
            "min": min(inv_lengths),
            "max": max(inv_lengths),
        }

    if "speedup" in dataset.column_names:
        if grade_col:
            speedups = [sample["speedup"] for sample in dataset if sample[grade_col] > 1]
        else:
            speedups = [sample["speedup"] for sample in dataset]
        if speedups:
            speedups_gt1 = [s for s in speedups if s > 1.0]
            stats["speedup"] = {
                "count_used": len(speedups),
                "mean": sum(speedups) / len(speedups),
                "min": min(speedups),
                "max": max(speedups),
                "count_gt1": len(speedups_gt1),
                "mean_gt1": sum(speedups_gt1) / len(speedups_gt1) if speedups_gt1 else 0.0,
            }
        else:
            stats["speedup"] = {"count_used": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "count_gt1": 0, "mean_gt1": 0.0}

    logger.info(f"[{name}] Total samples: {stats['total']}")
    logger.info(
        f"[{name}] Token length - Mean: {stats['token_length']['mean']:.1f}, "
        f"Min: {stats['token_length']['min']}, Max: {stats['token_length']['max']}"
    )
    if "grade" in stats:
        logger.info(
            f"[{name}] Grade - Mean: {stats['grade']['mean']:.2f}, "
            f"Distribution: {stats['grade']['distribution']}"
        )
    if "invariant_token_length" in stats:
        logger.info(
            f"[{name}] Invariant token length - Mean: {stats['invariant_token_length']['mean']:.1f}, "
            f"Min: {stats['invariant_token_length']['min']}, Max: {stats['invariant_token_length']['max']}"
        )
    if "speedup" in stats:
        s = stats["speedup"]
        logger.info(
            f"[{name}] Speedup (grade>1 only) - Mean(all): {s['mean']:.2f}x, Mean(>1): {s['mean_gt1']:.2f}x, "
            f"Min: {s['min']:.2f}x, Max: {s['max']:.2f}x, Count>1: {s['count_gt1']}/{s['count_used']}"
        )

    return stats

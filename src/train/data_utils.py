import logging
from os import path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoTokenizer

from src.preprocess.clean_invariants import clean_invariant
from src.preprocess.program import Program as AstProgram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_trivial_invariant(invariant: str)->bool:
    """Check if invariant is trivial (0, 1, true, false) - UAutomizer artifacts."""
    if not invariant:
        return True
    normalized = invariant.strip().lower().replace('(', '').replace(')', '').replace(' ', '')
    return normalized in ('0', '1', 'true', 'false', '')

def split_dataset(dataset: Dataset, split_ratio: float = 0.8)->(Dataset, Dataset):
    """Split dataset into train and validation sets."""
    train_size = int(len(dataset) * split_ratio)
    train_dataset = dataset.select(range(train_size))
    validation_dataset = dataset.select(range(train_size, len(dataset)))
    logger.info(f"Split dataset into train and validation sets. Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")
    return train_dataset, validation_dataset


def log_dataset_stats(dataset: Dataset, tokenizer: AutoTokenizer, name: str = "dataset") -> dict:
    """Compute and log statistics for a dataset split. Returns stats dict."""
    from collections import Counter
    
    stats = {"name": name, "total": len(dataset)}
    
    if len(dataset) == 0:
        logger.info(f"[{name}] Empty dataset")
        return stats
    
    # Token length stats (full text)
    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
    stats["token_length"] = {
        "mean": sum(token_lengths) / len(token_lengths),
        "min": min(token_lengths),
        "max": max(token_lengths)
    }
    
    # Grade stats (if available)
    if "grade" in dataset.column_names:
        grades = [sample["grade"] for sample in dataset]
        grade_counts = Counter(grades)
        stats["grade"] = {
            "mean": sum(grades) / len(grades),
            "distribution": dict(sorted(grade_counts.items()))
        }
    
    # Invariant token length stats (if available)
    if "invariant_token_length" in dataset.column_names:
        inv_lengths = [sample["invariant_token_length"] for sample in dataset]
        stats["invariant_token_length"] = {
            "mean": sum(inv_lengths) / len(inv_lengths),
            "min": min(inv_lengths),
            "max": max(inv_lengths)
        }
    
    # Speedup stats (if available) - exclude grade 1 since it's not useful
    if "speedup" in dataset.column_names:
        # Only consider samples with grade > 1 for speedup calculation (grade 1 = not useful)
        if "grade" in dataset.column_names:
            speedups = [sample["speedup"] for sample in dataset if sample["grade"] > 1]
        else:
            speedups = [sample["speedup"] for sample in dataset]
        
        if speedups:
            speedups_gt1 = [s for s in speedups if s > 1.0]
            stats["speedup"] = {
                "count_used": len(speedups),  # samples used for speedup calculation (grade > 1)
                "mean": sum(speedups) / len(speedups),
                "min": min(speedups),
                "max": max(speedups),
                "count_gt1": len(speedups_gt1),
                "mean_gt1": sum(speedups_gt1) / len(speedups_gt1) if speedups_gt1 else 0.0
            }
        else:
            stats["speedup"] = {"count_used": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "count_gt1": 0, "mean_gt1": 0.0}
    
    # Log the stats
    logger.info(f"[{name}] Total samples: {stats['total']}")
    logger.info(f"[{name}] Token length - Mean: {stats['token_length']['mean']:.1f}, Min: {stats['token_length']['min']}, Max: {stats['token_length']['max']}")
    if "grade" in stats:
        logger.info(f"[{name}] Grade - Mean: {stats['grade']['mean']:.2f}, Distribution: {stats['grade']['distribution']}")
    if "invariant_token_length" in stats:
        logger.info(f"[{name}] Invariant token length - Mean: {stats['invariant_token_length']['mean']:.1f}, Min: {stats['invariant_token_length']['min']}, Max: {stats['invariant_token_length']['max']}")
    if "speedup" in stats:
        logger.info(f"[{name}] Speedup (grade>1 only) - Mean(all): {stats['speedup']['mean']:.2f}x, Mean(>1): {stats['speedup']['mean_gt1']:.2f}x, Min: {stats['speedup']['min']:.2f}x, Max: {stats['speedup']['max']:.2f}x, Count>1: {stats['speedup']['count_gt1']}/{stats['speedup']['count_used']}")
    
    return stats

def inv_gen_per_marker_task_v2(raw_dataset: Dataset, tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str, max_length: int = 1024, min_grade: int = 1, inv_mode: str = "simplified", output_dir: str = None, hf_repo_name: str = None)->Dataset:
    # assuming already in per marker format
    logger.info(f"Creating SFT train data from {raw_dataset}. task: inv_gen_per_marker_task_v2 for {inv_mode} mode")
    if output_dir:
        token_plot_path = path.join(output_dir, f"token_length_distribution_per_marker_{inv_mode}_v2.{min_grade}.png")
        grade_plot_path = path.join(output_dir, f"grade_distribution_per_marker_{inv_mode}_v2.{min_grade}.png")
        inv_token_plot_path = path.join(output_dir, f"invariant_token_length_distribution_{inv_mode}_v2.{min_grade}.png")
    else:
        token_plot_path = None
        grade_plot_path = None
        inv_token_plot_path = None
    samples = []
    kept_grades = []  # Track grades for kept samples
    invariant_token_lengths = []  # Track invariant-only token lengths
    max_tokens = 0
    skipped_baseline = 0
    skipped_grade = 0
    skipped_no_invariant = 0
    skipped_length = 0
    skipped_trivial = 0
    for sample in tqdm(raw_dataset):
        if sample.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            skipped_baseline += 1
            continue
        if not sample.get("gt_invariant"):
            skipped_no_invariant += 1
            continue
        invariant = sample["gt_invariant"]
        if invariant.get("quality_grade", -1) < min_grade:
            skipped_grade += 1
            continue
        if is_trivial_invariant(invariant.get("target_invariant", "")):
            skipped_trivial += 1
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        user_prompt = user_prompt_template.format(
                program=ast_program.llm_code, target_marker=invariant.get("marker", "")
            )
        if inv_mode == "simplified":
            content = invariant.get("target_invariant", "")
        elif inv_mode == "pretty":
            content = invariant.get("pretty_invariant", "")
        elif inv_mode == "raw":
            content = invariant.get("original_invariant", "")
        else:
            raise ValueError(f"Invalid mode: {inv_mode}. can only be 'simplified', 'pretty', or 'raw'")
        answer = f'```json\n{{"marker":"{invariant.get("marker", "")}","content":"{content}"}}\n```'
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if len(tokenizer.encode(text)) > max_length - 1:
            skipped_length += 1
            logger.info(f"Skipping sample {sample['file']} because token length is greater than {max_length - 1}")
            # logger.info(f"text: {text}")
            continue
        inv_token_len = len(tokenizer.encode(content))
        speedup = invariant.get("speedup", 0.0)
        samples.append({
            "text": text,
            "grade": invariant.get("quality_grade", -1),
            "invariant_token_length": inv_token_len,
            "speedup": speedup
        })
        kept_grades.append(invariant.get("quality_grade", -1))
        invariant_token_lengths.append(inv_token_len)
        max_tokens = max(max_tokens, len(tokenizer.encode(text)))
    logger.info(f"Filtering stats: skipped_baseline={skipped_baseline}, skipped_grade={skipped_grade}, skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, skipped_trivial={skipped_trivial}, kept={len(samples)}")
    logger.info(f"Max tokens: {max_tokens}")
    
    final_dataset = Dataset.from_list(samples)
    
    if len(samples) > 0 and token_plot_path:
        plot_token_distribution(final_dataset, tokenizer, name=f"per_marker_{inv_mode}_v2.{min_grade}", plot_path=token_plot_path)
    
    if len(kept_grades) > 0 and grade_plot_path:
        plot_grade_distribution(kept_grades, name=f"per_marker_{inv_mode}_v2.{min_grade}", plot_path=grade_plot_path)
    
    if len(invariant_token_lengths) > 0 and inv_token_plot_path:
        plot_invariant_token_distribution(invariant_token_lengths, name=f"per_marker_{inv_mode}_v2.{min_grade}", plot_path=inv_token_plot_path)
    
    # Upload to HuggingFace if repo name is provided
    if hf_repo_name and len(samples) > 0:
        upload_dataset_to_hf(final_dataset, hf_repo_name)
    
    return final_dataset


def inv_gen_per_marker_task_v0_to_v1(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
    max_length: int = 1024,
    inv_mode: str = "raw", # "raw" or "pretty" only 
    output_dir: str = None,
) -> Dataset:
    if output_dir:
        plot_path = path.join(output_dir, f"token_length_distribution_per_marker_{inv_mode}.png")
    else:
        plot_path = None
    logger.info(f"Creating SFT train data from {raw_dataset}. task: inv_gen_per_marker for {inv_mode} mode")
    samples = []
    max_tokens = 0
    skipped_baseline = 0
    skipped_no_invariant = 0
    skipped_length = 0
    for sample in tqdm(raw_dataset, desc="Processing samples"):
        # Skip results with no invariants or with invalid baseline decision
        if sample.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            skipped_baseline += 1
            continue
        if not sample.get("invariants"):
            skipped_no_invariant += 1
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        invariants = sample["invariants"]
        for invariant in invariants:
            if "invariant" not in invariant:
                skipped_no_invariant += 1
                continue
            user_prompt = user_prompt_template.format(
                program=ast_program.llm_code, target_marker=invariant["marker"]
            )
            if inv_mode == "pretty":
                content = clean_invariant(invariant["invariant"], pretty=True)
            elif inv_mode == "raw":
                content = invariant["invariant"]
            else:
                raise ValueError(f"Invalid invariant mode: {inv_mode}. can only be 'raw' (v0) or 'pretty' (v1)")
            answer = f'```json\n{{"marker":"{invariant["marker"]}","content":"{content}"}}\n```'
            # logger.info(f"Answer: {answer}")
            conversation = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if len(tokenizer.encode(text)) > max_length - 1:
                skipped_length += 1
                logger.info(f"Skipping sample {sample['file']} because token length is greater than {max_length - 1}")
                logger.info(f"text: {text}")
                continue
            samples.append({"text": text})
            max_tokens = max(max_tokens, len(tokenizer.encode(text)))
    logger.info(f"Filtering stats: skipped_baseline={skipped_baseline}, skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, kept={len(samples)}")
    logger.info(f"Max tokens: {max_tokens}")
    if len(samples) > 0 and plot_path:
        plot_token_distribution(Dataset.from_list(samples), tokenizer, name=f"per_marker_{inv_mode}", plot_path=plot_path)
    return Dataset.from_list(samples)



def try_load_from_hf(repo_name: str) -> Dataset | None:
    """Try to load dataset from HuggingFace Hub. Returns None if not found."""
    try:
        logger.info(f"Attempting to load processed dataset from HuggingFace: {repo_name}")
        dataset = load_dataset(repo_name, split="train")
        logger.info(f"Successfully loaded dataset from {repo_name} with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.info(f"Dataset not found on HuggingFace ({repo_name}): {e}")
        return None


def load_training_data(dataset_name: str, tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str, limit: int = -1, inv_mode: str = "raw", split: str = "full", max_length: int = 1024, json_path: str = None, min_grade: int = 1, output_dir: str = None, hf_upload_repo: str = None, force_reprocess: bool = False)->Dataset:
    # Try to load from HuggingFace first if repo is specified and not forcing reprocess
    if hf_upload_repo and not force_reprocess:
        cached_dataset = try_load_from_hf(hf_upload_repo)
        if cached_dataset is not None:
            return cached_dataset
        logger.info("Will process raw dataset and upload to HuggingFace after processing")
    
    # Load and process raw dataset
    if json_path:
        raw_dataset = load_dataset("json", data_files=json_path, split="train")
    else:
        raw_dataset = load_dataset(dataset_name, split=split)
    raw_dataset = raw_dataset.shuffle() # think about sorting by baseline timing (curriculum learning)
    if limit > 0:
        raw_dataset = raw_dataset.select(range(limit))
    if inv_mode == "simplified":
        dataset = inv_gen_per_marker_task_v2(raw_dataset, tokenizer, system_prompt, user_prompt_template, max_length=max_length, inv_mode=inv_mode, min_grade=min_grade, output_dir=output_dir, hf_repo_name=hf_upload_repo)
    else:
        dataset = inv_gen_per_marker_task_v0_to_v1(raw_dataset, tokenizer, system_prompt, user_prompt_template, max_length=max_length, inv_mode=inv_mode, output_dir=output_dir)

    return dataset

def plot_token_distribution(dataset: Dataset, tokenizer: AutoTokenizer, name: str = "dataset", plot_path: str = None)->None:
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Create parent directory if it doesn't exist
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
    
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution - {name}")
    
    # Add summary stats
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    max_len = max(token_lengths) if token_lengths else 0
    min_len = min(token_lengths) if token_lengths else 0
    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.1f}')
    
    plt.text(0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}", 
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Token length distribution plot saved to {plot_path}")


def plot_grade_distribution(grades: list, name: str = "dataset", plot_path: str = None)->None:
    """Plot the distribution of quality grades for kept samples."""
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Create parent directory if it doesn't exist
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    grade_counts = Counter(grades)
    sorted_grades = sorted(grade_counts.keys())
    counts = [grade_counts[g] for g in sorted_grades]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(g) for g in sorted_grades], counts, color='steelblue', edgecolor='black')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Quality Grade")
    plt.ylabel("Frequency")
    plt.title(f"Grade Distribution (Kept Samples) - {name}")
    
    # Add summary stats
    total = len(grades)
    mean_grade = sum(grades) / total if total > 0 else 0
    plt.text(0.95, 0.95, f"Total: {total}\nMean: {mean_grade:.2f}", 
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Grade distribution plot saved to {plot_path}")


def plot_invariant_token_distribution(token_lengths: list, name: str = "dataset", plot_path: str = None)->None:
    """Plot the distribution of invariant-only token lengths."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Create parent directory if it doesn't exist
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='coral', edgecolor='black', alpha=0.7)
    
    plt.xlabel("Invariant Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Invariant Token Length Distribution - {name}")
    
    # Add summary stats
    total = len(token_lengths)
    mean_len = sum(token_lengths) / total if total > 0 else 0
    max_len = max(token_lengths) if token_lengths else 0
    min_len = min(token_lengths) if token_lengths else 0
    plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_len:.1f}')
    
    plt.text(0.95, 0.95, f"Total: {total}\nMean: {mean_len:.1f}\nMin: {min_len}\nMax: {max_len}", 
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Invariant token length distribution plot saved to {plot_path}")


def upload_dataset_to_hf(dataset: Dataset, repo_name: str, private: bool = False)->None:
    """Upload dataset to HuggingFace Hub."""
    
    logger.info(f"Uploading dataset to HuggingFace Hub: {repo_name}")
    try:
        dataset.push_to_hub(repo_name, private=private)
        logger.info(f"Successfully uploaded dataset to {repo_name}")
    except Exception as e:
        logger.error(f"Failed to upload dataset to HuggingFace: {e}")
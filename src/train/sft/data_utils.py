from os import path
from datasets import Dataset, load_dataset
from src.preprocess.program import Program as AstProgram
from src.preprocess.clean_invariants import clean_invariant
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset(dataset: Dataset, split_ratio: float = 0.8)->(Dataset, Dataset):
    """Split dataset into train and validation sets."""
    train_size = int(len(dataset) * split_ratio)
    train_dataset = dataset.select(range(train_size))
    validation_dataset = dataset.select(range(train_size, len(dataset)))
    logger.info(f"Split dataset into train and validation sets. Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")
    return train_dataset, validation_dataset

def inv_gen_per_marker_task_v2(raw_dataset: Dataset, tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str, max_length: int = 1024, min_grade: int = 1, inv_mode: str = "simplified")->Dataset:
    # assuming already in per marker format
    logger.info(f"Creating SFT train data from {raw_dataset}. task: inv_gen_simplified")
    samples = []
    max_tokens = 0
    skipped_baseline = 0
    skipped_grade = 0
    skipped_no_invariant = 0
    skipped_length = 0
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
            logger.info(f"text: {text}")
            continue
        samples.append({"text": text})
        max_tokens = max(max_tokens, len(tokenizer.encode(text)))
    logger.info(f"Filtering stats: skipped_baseline={skipped_baseline}, skipped_grade={skipped_grade}, skipped_no_invariant={skipped_no_invariant}, skipped_length={skipped_length}, kept={len(samples)}")
    logger.info(f"Max tokens: {max_tokens}")
    if len(samples) > 0:
        plot_token_distribution(Dataset.from_list(samples), tokenizer, name=f"simplified_mode_{inv_mode}")
    return Dataset.from_list(samples)


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



def load_training_data(dataset_name: str, tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str, limit: int = -1, inv_mode: str = "raw", split: str = "full", max_length: int = 1024, json_path: str = None, min_grade: int = 1, output_dir: str = None)->Dataset:
    if json_path:
        raw_dataset = load_dataset("json", data_files=json_path, split="train")
    else:
        raw_dataset = load_dataset(dataset_name, split=split)
    raw_dataset = raw_dataset.shuffle()
    if limit > 0:
        raw_dataset = raw_dataset.select(range(limit))
    if inv_mode == "simplified":
        dataset = inv_gen_per_marker_task_v2(raw_dataset, tokenizer, system_prompt, user_prompt_template, max_length=max_length, inv_mode=inv_mode, min_grade=min_grade)
    else:
        dataset = inv_gen_per_marker_task_v0_to_v1(raw_dataset, tokenizer, system_prompt, user_prompt_template, max_length=max_length, inv_mode=inv_mode, output_dir=output_dir)

    return dataset

def plot_token_distribution(dataset: Dataset, tokenizer: AutoTokenizer, name: str = "dataset", plot_path: str = None)->None:
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Create parent directory if it doesn't exist
    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    
    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
    plt.hist(token_lengths, bins=100)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution - {name}")
    plt.savefig(plot_path)
    plt.close()
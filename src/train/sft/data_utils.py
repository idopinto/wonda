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

def inv_gen_per_marker_task(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_prompt_template: str,
    max_length: int = 1024,
    inv_mode: str = "raw" # "raw" or "pretty" or "simplified"
) -> Dataset:
    logger.info(f"Creating SFT train data from {raw_dataset}. task: inv_gen_per_marker")
    samples = []
    max_tokens = 0
    for sample in tqdm(raw_dataset):
        # Skip results with no invariants or with invalid baseline decision
        if sample.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            # logger.info(f"Skipping sample {sample['file']} because baseline decision is {sample.get('baseline_decision')}")
            continue
        if not sample.get("invariants"):
            # logger.info(f"Skipping sample {sample['file']} because no invariants")
            continue
        program_str = sample["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        invariants = sample["invariants"]
        for invariant in invariants:
            # target_label = invariant["marker"].split("_")[-1]
            user_prompt = user_prompt_template.format(
                program=ast_program.llm_code, target_marker=invariant["marker"]
            )
            if inv_mode == "simplified":
                if "simplified_invariant" in invariant:
                    content = invariant["simplified_invariant"]
                else:
                    raise ValueError(f"Simplified invariant not found for {invariant['marker']}.")
            elif inv_mode == "pretty":
                if "pretty_invariant" in invariant:
                    content = invariant["pretty_invariant"]
                else:
                    content = clean_invariant(invariant["invariant"], pretty=True)
            elif inv_mode == "raw":
                if "invariant" in invariant:
                    content = invariant["invariant"]
                else:
                    raise ValueError(f"Invariant not found for {invariant['marker']}.")
            else:
                raise ValueError(f"Invalid invariant mode: {inv_mode}. can only be 'raw', 'pretty', or 'simplified'")
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
                logger.info(f"Skipping sample {sample['file']} because token length is greater than {max_length - 1}")
                continue
            samples.append({"text": text})
            max_tokens = max(max_tokens, len(tokenizer.encode(text)))
    logger.info(f"Max tokens: {max_tokens}")
    plot_token_distribution(Dataset.from_list(samples), tokenizer)
    return Dataset.from_list(samples)



def load_training_data(dataset_name: str,tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str, limit: int = -1, inv_mode: str = "raw", split: str = "full", max_length: int = 1024)->Dataset:
    raw_dataset = load_dataset(dataset_name, split=split)
    raw_dataset = raw_dataset.shuffle()
    if limit > 0:
        raw_dataset = raw_dataset.select(range(limit))
    dataset = inv_gen_per_marker_task(raw_dataset, tokenizer, system_prompt, user_prompt_template, max_length=max_length, inv_mode=inv_mode)

    logger.info(f"Loaded dataset {dataset_name}...")
    logger.info(f"Dataset Size: {len(dataset)}")

    return dataset

def plot_token_distribution(dataset: Dataset, tokenizer: AutoTokenizer)->None:
    import matplotlib.pyplot as plt
    token_lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
    plt.hist(token_lengths, bins=100)
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title("Token Length Distribution")
    plt.savefig("plots/token_length_distribution.png")
    plt.close()
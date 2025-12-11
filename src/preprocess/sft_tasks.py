import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def task_generate_all_invariants(results_path: Path, output_dir: Path, push_to_hub: bool = False) -> DatasetDict:
    logger.info(f"Creating 'generate all invariants' task from {results_path}.") 
    with open(results_path, "r") as f:
        results = json.load(f)   # type: ignore
    system_prompt = "You are an expert C programmer and highly proficient in generating strong loop invariants for C programs that accelerates traditional verifiers."
    user_prompt_template = """Given the following C program, for each of the marked lines, generate a strong loop invariant.
```c
{program}
```

Format:
assert(<invariant_1>); // Line A
assert(<invariant_2>); // Line B

Example:
assert(a > 0 && a < 10); // Line A
assert(b > 0 && b < 20); // Line B
"""
    samples = []
    for result in tqdm(results):
        # Skip results with no invariants or with invalid baseline decision
        if result.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            continue
        if not result.get("invariants"):
            continue
        normalized_program_lines = result["normalized_program"].split("\n")
        invariants = result["invariants"]
        # Assign letters in ascending line order (A = first/top line)
        sorted_by_line_asc = sorted(invariants, key=lambda x: x['line'])
        line_map = {}
        for i, inv_data in enumerate(sorted_by_line_asc):
            line_map[inv_data['line']] = chr(ord('A') + i)
        # Insert markers from bottom to top to avoid offset issues
        sorted_by_line_desc = sorted(invariants, key=lambda x: x['line'], reverse=True)
        for inv_data in sorted_by_line_desc:
            line_number = inv_data['line']
            normalized_program_lines.insert(line_number, f"// Line {line_map[line_number]}")
        formatted_program = "\n".join(normalized_program_lines)
        # Build answer in ascending line order (A first)
        answer = "\n".join([f"assert({inv_data['invariant']}); // Line {line_map[inv_data['line']]}" for inv_data in sorted_by_line_asc])
        sample = {"messages": [
            {"role": "system", "content": system_prompt, "thinking": None},
            {"role": "user", "content": user_prompt_template.format(program=formatted_program), "thinking": None},
            {"role": "assistant", "content": answer, "thinking": None}
        ]}
        samples.append(sample)
    
    logger.info(f"Created {len(samples)} samples")
    if not samples:
        logger.warning("No samples created - all results were filtered out")
        return DatasetDict()
    
    dataset = Dataset.from_list(samples)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved to: {output_dir}")
    if push_to_hub:
        dataset_dict.push_to_hub(f"idopinto/{output_dir.name}")
        logger.info(f"Pushed to: https://huggingface.co/datasets/{output_dir.name}")
    return dataset_dict

def task_generate_one_invariant(results_path: Path, output_dir: Path, push_to_hub: bool = False) -> DatasetDict:
    logger.info(f"Creating 'generate one invariant' task from {results_path}") 
    with open(results_path, "r") as f:
        results = json.load(f)   # type: ignore
    system_prompt = "You are an expert C programmer and highly proficient in generating strong loop invariants for C programs that accelerates traditional verifiers."
    user_prompt_template = """Given the following C program, generate a strong loop invariant for the marked line with the comment // HERE:
```c
{program}
```

Format:
assert(<invariant>); 

Example:
assert(a > 0 && a < 10);
"""
    samples = []
    for result in tqdm(results):
        # Skip results with no invariants or with invalid result
        if result.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            continue
        # if not result.get("invariants"):
        #     continue
        for invariant in result["invariants"]:
            normalized_program_lines = result["normalized_program"].split("\n")
            line_number = invariant["line"]
            normalized_program_lines.insert(line_number, "// HERE")
            formatted_program = "\n".join(normalized_program_lines)
            sample = {"messages": [
                {"role": "system", "content": system_prompt, "thinking": None},
                {"role": "user", "content": user_prompt_template.format(
                    program=formatted_program
                ), "thinking": None},
                {"role": "assistant", "content": f"assert({invariant['invariant']});", "thinking": None}
            ]}
            samples.append(sample)
    
    logger.info(f"Created {len(samples)} samples")
    if not samples:
        logger.warning("No samples created - all results were filtered out")
        return DatasetDict()
    
    dataset = Dataset.from_list(samples)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(output_dir)
    logger.info(f"Saved to: {output_dir}")
    if push_to_hub:
        dataset_dict.push_to_hub(f"idopinto/{output_dir.name}")
        logger.info(f"Pushed to: https://huggingface.co/datasets/{output_dir.name}")
    return dataset_dict
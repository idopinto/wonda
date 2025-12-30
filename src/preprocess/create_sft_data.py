import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from src.preprocess.baseline_dataset_common import AstProgram, logger
from omegaconf import DictConfig
import hydra
from omegaconf import OmegaConf
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def per_location_task(
    cfg: DictConfig,
    results_path: Path,
    output_dir: Path,
    push_to_hub: bool = False,
) -> DatasetDict:
    logger.info(f"Creating SFT train data from {results_path}. task: per_location")
    with open(results_path, "r") as f:
        results = json.load(f)  # type: ignore
    results = results[:cfg.dataset.limit] if cfg.dataset.limit > 0 else results
    system_prompt = cfg.prompts.system
    user_prompt_template = cfg.prompts.per_location.user
    samples = []
    for result in tqdm(results):
        # Skip results with no invariants or with invalid baseline decision
        if result.get("baseline_decision") in {"UNKNOWN", "ERROR", "TIMEOUT"}:
            continue
        if not result.get("invariants"):
            continue
        program_str = result["original_program"]
        ast_program = AstProgram().from_code(program_str)
        ast_program.process(print_ast=False)
        # program_lines = program.split("\n")
        invariants = result["invariants"]
        for invariant in invariants:
            target_label = invariant["marker"].split("_")[-1]
            user_prompt = user_prompt_template.format(
                program=ast_program.llm_code, target_label=target_label
            )
            if "simplified_invariant" in invariant:
                content = invariant["simplified_invariant"]
            elif "pretty_invariant" in invariant:
                content = invariant["pretty_invariant"]
            else:
                content = invariant["invariant"]
            if "rationale" in invariant:
                rationale = invariant["rationale"]
            else:
                rationale = None
            answer = f'{{"marker":"{invariant["marker"]}","content":"{content}"}}'
            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "thinking": rationale,
                    },
                ]
            }
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
        logger.info(
            f"Pushed to: https://huggingface.co/datasets/idopinto/{output_dir.name}"
        )
    return dataset_dict

@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="create_sft_data")
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")    
    results_path = Path(cfg.dataset.preprocessed_invariants_path)
    
    output_dir = Path(cfg.output.per_location_task_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_location_task(cfg, results_path, output_dir, cfg.output.push_to_hub)

if __name__ == "__main__":
    main()

# Example usage:    
# uv run -m src.preprocess.create_sft_data
#!/usr/bin/env python3
"""
Preprocess ground truth invariants.

This script:
1. Loads training data with verbose invariants
2. Filters invariants based on complexity criteria
3. Uses Kimi K2 Thinking to generalize and simplifythem into compact formulas
4. Outputs a new JSON file with simplifiedinvariants in the same format

Usage:
    uv run -m src.preprocess.preprocess_gt_invariants
    uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=5
"""

import json
import logging
from pathlib import Path
import tempfile
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from src.preprocess import predicate
from together import Together
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from src.preprocess.clean_invariants import (
    clean_invariant,
    pretty_invariant_src,
)
import weave
from src.preprocess.program import Program, Predicate
from src.preprocess.syntactic_validation import syntactic_validation
from src.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport
from configs import global_config as GC
from dataclasses import dataclass, field
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def is_verbose_invariant(inv_text: str, cfg: DictConfig) -> bool:
    """Check if an invariant matches the verbose criteria for generalization."""
    disjunct_count = inv_text.count(" || ") + 1
    char_length = len(inv_text)

    return cfg.filtering.min_disjuncts <= disjunct_count <= cfg.filtering.max_disjuncts and cfg.filtering.min_char_length <= char_length <= cfg.filtering.max_char_length


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format few-shot examples for the prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"Input: {ex['input'].strip()}")
        formatted.append(f"Output: {ex['output']}")
        formatted.append(f"Explanation: {ex['explanation']}")
        formatted.append("")
    return "\n".join(formatted)


@dataclass
class SimplifyInvariantResult:
    full_prompt: str = field(default="")
    simplified_invariant: str = field(default="")
    rationale: str = field(default="")
    reasoning: str = field(default="")

@weave.op()
def simplify_invariant(
    client: Together,
    original_program: str,
    invariant: str,
    marker: str,
    cfg: DictConfig,
) -> SimplifyInvariantResult:
    """
    Use Kimi K2 Thinking to simplify a verbose invariant.

    Args:
        client: Together AI client
        program: The C program context (program_for_baseline)
        invariant: The verbose invariant to simplify
        cfg: Hydra configuration

    Returns dict with 'prompt', 'invariant', 'rationale', and 'reasoning' keys.
    """
    program = Program().from_code(original_program)
    program.process(print_ast=False)
    program_for_llm = program.llm_code
    few_shot = format_few_shot_examples(cfg.prompts.few_shot_examples)
    user_prompt = cfg.prompts.simplify_invariant_user_prompt.format(program=program_for_llm, invariant=invariant, marker=marker)
    user_content = f"""\n{few_shot}\n{user_prompt}"""

    messages = [
        {"role": "system", "content": cfg.prompts.simplify_invariant_system_prompt},
        {"role": "user", "content": user_content},
    ]
    full_prompt = '\n'.join([messages[0]['content'], messages[1]['content']])
    response = client.chat.completions.create(
        model=cfg.model.name,
        messages=messages,
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
        n=cfg.model.n,
    )

    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", "") or ""
    content = message.content or ""
    try:
        content = json.loads(content)
        simplified_invariant = content["simplified_invariant"]
        rationale = content["rationale"]
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON content: {content}")
        return SimplifyInvariantResult(full_prompt=full_prompt, simplified_invariant="", rationale="", reasoning=reasoning)
    return SimplifyInvariantResult(full_prompt=full_prompt, simplified_invariant=simplified_invariant, rationale=rationale, reasoning=reasoning)

def run_verifier(verifier: UAutomizerVerifier, program_str: str, kind: str) -> VerifierCallReport:
    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = Path(temp_dir)
        code_dir.mkdir(parents=True, exist_ok=True)
        program_path = code_dir / f"code_for_{kind}.c"
        with open(program_path, "w") as out_file:
            out_file.write(program_str)
        verifier_report: VerifierCallReport = verifier.verify(
            program_path=program_path,
            reports_dir=code_dir,
            timeout_seconds=verifier.timeout_seconds,
        )
        print(f"Verifier report: {verifier_report.decision} ({verifier_report.decision_reason}) in {verifier_report.time_taken} seconds")
        return verifier_report

@dataclass
class QualityGradeResult:
    is_valid: bool = field(default=False)
    is_correct: bool = field(default=False)
    does_target_property_still_hold: bool = field(default=False)
    has_speedup: bool = field(default=False)
    quality_grade: int = field(default=-1) # -1 means unknown (unchecked) quality.

@weave.op()
def decide_if_worth_keeping_invariant(verifier: UAutomizerVerifier, baseline_decision: str, baseline_timing: float, program: Program, predicate: Predicate) -> QualityGradeResult:
    """Decide if the invariant is worth keeping."""
    logger.info(f"Deciding if worth keeping invariant: {predicate}")
    quality_grade_result = QualityGradeResult()
    quality_grade_result.is_valid = syntactic_validation(predicate.content)
    if not quality_grade_result.is_valid:
        quality_grade_result.quality_grade = 0
        logger.warning(f"Invariant {predicate.content} is not valid, skipping")
        return quality_grade_result
    
    program_for_correctness = program.get_program_with_assertion(predicate=predicate, assumptions=[], for_llm=False)
    program_for_usefullness = program.get_program_with_assertion(predicate=program.get_target_assert(), assumptions=[predicate], for_llm=False)
    # Parallel evaluation: run both verifier queries concurrently
    # da = V(P, Ø, q): Check if q is an invariant
    # db = V(P, {q}, p*): Check if target property holds assuming q is true
    with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both verifier tasks
            correctness_future = executor.submit(
                run_verifier,
                verifier=verifier,
                program_str=program_for_correctness,
                kind="correctness",
            )

            usefulness_future = executor.submit(
                run_verifier,
                verifier=verifier,
                program_str=program_for_usefullness,
                kind="usefulness",
            )

            for future in as_completed([correctness_future, usefulness_future]):
                result = future.result()
                if future == correctness_future:
                    quality_grade_result.is_correct = result.decision == "TRUE"
                elif future == usefulness_future:
                    quality_grade_result.does_target_property_still_hold = result.decision in  {"TRUE", "FALSE"} and result.decision == baseline_decision
                    quality_grade_result.has_speedup = result.time_taken < baseline_timing

    if not quality_grade_result.is_correct:
        quality_grade_result.quality_grade = 0
    elif not quality_grade_result.does_target_property_still_hold:
        quality_grade_result.quality_grade = 1
    elif not quality_grade_result.has_speedup:
        quality_grade_result.quality_grade = 2
    else:
        quality_grade_result.quality_grade = 3
    return quality_grade_result

@weave.op()
def preprocess_all_gt_invariants(
    data: list[dict[str, Any]],
    client: Together,
    cfg: DictConfig,
    verifier: UAutomizerVerifier,
) -> list[dict[str, Any]]:
    """Preprocess all ground truth invariants in the data. expanding to one invariant per entry."""
    logger.info(f"Processing {len(data)} entries")
    output_path = Path(cfg.output.preprocessed_invariants_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data = []
    for i, entry in enumerate(tqdm(data, desc="preprocessing invariants")):
        original_program = entry.get("original_program", "")
        baseline_timing = entry.get("timings", {}).get("median", 0.0)
        baseline_decision = entry.get("baseline_decision", "UNKNOWN")
        if baseline_decision not in {"TRUE", "FALSE"} or not entry.get("invariants", []): # only process entries with a valid baseline decision and invariants
            logger.warning(f"Entry {i} has unknown baseline decision or no invariants, skipping")
            continue

        for j, inv in enumerate(entry["invariants"]):
            logger.info(f"Entry {i}, Invariant {j}: Original invariant:\n{inv['invariant']}\nBaseline timing:\n{baseline_timing}\nBaseline decision:\n{baseline_decision}")
            marker = inv.get("marker", "")
            invariant_entry = {
                "marker": inv.get("marker", ""),
                "original_invariant": inv.get("invariant", ""),
                "pretty_invariant": "",
                "simplified_invariant": {
                    "content": "",
                    "rationale": "",
                    "is_correct": False,
                    "does_target_property_still_hold": False,
                    "has_speedup": False,
                    "quality_grade": -1,
                }
            }
            if cfg.pipeline.prettify:
                invariant_entry["pretty_invariant"] = clean_invariant(inv["invariant"], pretty=True)
                logger.info(f"Entry {i}, Invariant {j}: Pretty cleaned invariant: {invariant_entry['pretty_invariant']}")

            invariant_to_simplify = invariant_entry['pretty_invariant'] if cfg.pipeline.prettify else invariant_entry['original_invariant']
            if cfg.pipeline.simplify and is_verbose_invariant(invariant_to_simplify, cfg):
                simplify_result = simplify_invariant(client, original_program, invariant_to_simplify, marker, cfg)
                invariant_entry["simplified_invariant"]["content"] = simplify_result.simplified_invariant
                invariant_entry["simplified_invariant"]["rationale"] = simplify_result.rationale
                if cfg.pipeline.verify:
                    predicate = Predicate(content=simplify_result.simplified_invariant, marker_name=marker)
                    program = Program().from_code(original_program)
                    program.process(print_ast=False)
                    quality_grade_result = decide_if_worth_keeping_invariant(verifier, baseline_decision, baseline_timing, program, predicate)   
                    logger.info(f"Entry {i}, Invariant {j}: Quality grade result: {quality_grade_result}")
                    if quality_grade_result.quality_grade > 0:
                        invariant_entry["simplified_invariant"]["is_correct"] = quality_grade_result.is_correct
                        invariant_entry["simplified_invariant"]["does_target_property_still_hold"] = quality_grade_result.does_target_property_still_hold
                        invariant_entry["simplified_invariant"]["has_speedup"] = quality_grade_result.has_speedup
                        invariant_entry["simplified_invariant"]["quality_grade"] = quality_grade_result.quality_grade
                        entry["gt_invariant"] = invariant_entry
                        processed_data.append(entry)
                    else:
                        logger.warning(f"Entry {i}, Invariant {j}: Simplified invariant is not worth keeping, skipping")
                else:
                    logger.info(f"Warning: Skipping verifying simplified invariant. the invariant might not be correct or useful.")
                    entry["gt_invariant"] = invariant_entry
                    processed_data.append(entry)
            with open(output_path, "w") as f:
                json.dump(processed_data, f, indent=2)

    # Convert to HuggingFace Dataset and push to hub if configured
    if cfg.output.push_to_hub:
        dataset = Dataset.from_list(processed_data)
        dataset_dict = DatasetDict({"train": dataset})
        
        # Determine hub name from output path or use a default
        full_hub_name = f"{cfg.output.hub_organization}/{cfg.output.hub_name}"
        logger.info(f"Pushing dataset to HuggingFace Hub: {full_hub_name}")
        dataset_dict.push_to_hub(full_hub_name)
        logger.info(f"Pushed to: https://huggingface.co/datasets/{full_hub_name}")


def plot_statistics(data: list[dict[str, Any]]) -> None:
    """Plot statistics of the data."""
    import matplotlib.pyplot as plt
    # Plot the distribution of quality grades
    quality_grades = [inv["simplified_invariant"]["quality_grade"] for entry in data for inv in entry["invariants"] if "simplified_invariant" in inv]
    plt.hist(quality_grades, bins=range(4))
    plt.xlabel("Quality Grade")
    plt.ylabel("Count")
    plt.title("Distribution of Quality Grades")
    plt.savefig("plots/quality_grades_distribution.png")
    plt.close()

@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="preprocess_gt_invariants",
)
def main(cfg: DictConfig) -> None:
    """Main function for preprocess ground truth invariants."""

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
    )
    # Load training data
    data_path = Path(cfg.dataset.name)
    with open(data_path, 'r') as f:
        data = json.load(f)
    data = data[: cfg.dataset.limit] if cfg.dataset.limit > 0 else data
    logger.info(f"Loaded {len(data)} entries from {data_path}")
    # if cfg.pipeline.generalize_invariants:
    if cfg.weave.get("log_to_weave", False):
        weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")

    client = Together()
    logger.info(f"Preprocessing all ground truth invariants and saving to {cfg.output.preprocessed_invariants_path}")
    preprocess_all_gt_invariants(data, client, cfg, verifier)
    plot_statistics(data)

if __name__ == "__main__":
    main()

# Example usage:
# uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=1
# uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=-1  # Process all

# Config Location:
# configs/preprocess/preprocess_gt_invariants.yaml

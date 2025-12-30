#!/usr/bin/env python3
"""
Preprocess ground truth invariants.

This script:
1. Loads training data with verbose invariants
2. Filters invariants based on complexity criteria
3. Uses Kimi K2 Thinking to generalize them into compact formulas
4. Outputs a new JSON file with generalized invariants in the same format

Usage:
    uv run -m src.preprocess.preprocess_gt_invariants
    uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=5
"""

from curses import OK
from dis import pretty_flags
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

from src.preprocess.clean_invariants import (
    clean_invariant,
    pretty_invariant_src,
)
import weave
from src.preprocess.program import Program
from src.verifiers.uautomizer import UAutomizerVerifier, VerifierCallReport
from configs import global_config as GC
from src.preprocess.predicate import Predicate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@weave.op()
def generate_cot_reasoning(
    client: Together,
    program: str,
    target_label: str,
    simplified_invariant: str,
    cfg: DictConfig,
) -> dict[str, str]:
    """
    Generate Chain-of-Thought reasoning for a simplified invariant.

    Args:
        client: Together AI client
        program: The C program with markers (program_for_llm)
        target_label: The target marker label (e.g., "1" for INVARIANT_MARKER_1)
        simplified_invariant: The simplified invariant to explain
        cfg: Hydra configuration

    Returns dict with 'reasoning' and 'content' keys.
    """
    # Build user prompt using CoT distillation prompts
    user_prompt = cfg.cot_distilation_prompts.user.format(
        program=program,
        target_label=target_label,
        ground_truth_invariant_content=simplified_invariant,
    )

    messages = [
        {"role": "system", "content": cfg.cot_distilation_prompts.system},
        {"role": "user", "content": user_prompt},
    ]

    # Make API call (non-streaming)
    response = client.chat.completions.create(
        model=cfg.model.name,
        messages=messages,
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
    )

    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", "") or ""
    content = message.content or ""

    return {
        "reasoning": reasoning,
        "content": content.strip(),
    }

def is_verbose_invariant(inv_text: str, cfg: DictConfig) -> bool:
    """Check if an invariant matches the verbose criteria for generalization."""
    disjunct_count = inv_text.count(" || ") + 1
    char_length = len(inv_text)

    return (
        cfg.filtering.min_disjuncts <= disjunct_count <= cfg.filtering.max_disjuncts
        and cfg.filtering.min_char_length
        <= char_length
        <= cfg.filtering.max_char_length
    )


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


@weave.op()
def generalize_invariant(
    client: Together,
    program: str,
    invariant: str,
    cfg: DictConfig,
) -> dict[str, str]:
    """
    Use Kimi K2 Thinking to generalize a verbose invariant.

    Args:
        client: Together AI client
        program: The C program context (program_for_baseline)
        invariant: The verbose invariant to generalize
        cfg: Hydra configuration

    Returns dict with 'reasoning' and 'result' keys.
    """
    # Build few-shot examples string
    few_shot = format_few_shot_examples(cfg.prompts.few_shot_examples)

    # Build user prompt with program context and few-shot examples
    user_prompt = cfg.prompts.user.format(program=program, invariant=invariant)
    user_content = f"""{few_shot}
{user_prompt}"""

    messages = [
        {"role": "system", "content": cfg.prompts.system},
        {"role": "user", "content": user_content},
    ]

    # Make API call (non-streaming)
    response = client.chat.completions.create(
        model=cfg.model.name,
        messages=messages,
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
    )

    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", "") or ""
    content = message.content or ""
    try:
        content = json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON content: {content}")
        return {
            "invariant": "",
            "rationale": "",
            "reasoning": reasoning,
        }
    return {
        "prompt": "\n".join([cfg.prompts.system, user_content]),
        "invariant": content["invariant"],
        "rationale": content["rationale"],
        "reasoning": reasoning,
    }

# @weave.op()
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

@weave.op()
def decide_if_worth_keeping_invariant(
    original_program: str,
    invariant: str,
    marker_name: str,
    verifier: UAutomizerVerifier,
) -> bool:
    """Decide if the invariant is worth keeping."""
    logger.info(f"Deciding if invariant {invariant} with marker {marker_name} is worth keeping")
    predicate = Predicate(content=invariant, marker_name=marker_name)
    logger.info(f"Verifying correctness of invariant {predicate.content} with marker {predicate.marker_name}")
    program = Program().from_code(original_program)
    program.process(print_ast=False)
    target_assert = program.get_target_assert()
    program_for_correctness = program.get_program_with_assertion(predicate=predicate, assumptions=[], for_llm=False)
    program_for_usefullness = program.get_program_with_assertion(predicate=target_assert, assumptions=[predicate], for_llm=False)
    is_correct = False
    is_useful = False
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
                    is_correct = result.decision == "TRUE"
                elif future == usefulness_future:
                    is_useful = result.decision == "TRUE"
    result = {
        "is_correct": is_correct,
        "is_useful": is_useful,
    }
    return result


@weave.op()
def preprocess_all_gt_invariants(
    data: list[dict[str, Any]],
    client: Together,
    cfg: DictConfig,
    verifier: UAutomizerVerifier,
) -> list[dict[str, Any]]:
    """Preprocess all ground truth invariants in the data. expanding to one invariant per entry."""
    logger.info(f"Processing {len(data)} entries")
    
    # Process entries
    processed_data = []
    processed_count = 0

    for entry in tqdm(data, desc="preprocessing invariants"):
        original_program = entry.get("original_program", "")
        # program_for_baseline = entry.get("program_for_baseline", "")
        program_for_llm = entry.get("program_for_llm", "")

        # Check if this entry has verbose invariants
        if "invariants" in entry and entry["invariants"]:
            for inv in entry["invariants"]:
                logger.info(f"Original invariant: {inv['invariant']}")
                pretty_inv = clean_invariant(inv["invariant"], pretty=True)
                logger.info(f"Pretty cleaned invariant: {pretty_inv}")
                inv["pretty_invariant"] = pretty_inv
                if cfg.pipeline.generalize_invariants and is_verbose_invariant(pretty_inv, cfg):
                    result_dict = generalize_invariant(client, program_for_llm, pretty_inv, cfg)
                    result = decide_if_worth_keeping_invariant(original_program, result_dict.get("invariant", ""), inv["marker"], verifier)    
                    ok = result["is_correct"] and result["is_useful"]
                    logger.info(f"Result: {result}. Keeping pretty invariant" if not ok else f"Result: {result}. Keeping simplified invariant")
                    inv["simplified_invariant"] = result_dict.get("invariant", "") if ok else inv["pretty_invariant"]
                    inv["rationale"] = result_dict.get("rationale", "") if ok else ""
                else:
                    logger.warning(f"Invariant {pretty_inv} is not verbose enough, keeping original")
                if cfg.pipeline.cot_distilation:
                    # Extract target label from marker (e.g., "INVARIANT_MARKER_1" -> "1")
                    marker = inv.get("marker", "")
                    if marker.startswith("INVARIANT_MARKER_"):
                        target_label = marker.split("_")[-1]
                        # Use simplified_invariant if available, otherwise use pretty_invariant
                        invariant_to_explain = inv.get("simplified_invariant", inv.get("pretty_invariant", pretty_inv))
                        result_dict = generate_cot_reasoning(client, program_for_llm, target_label, invariant_to_explain, cfg)
                        inv["cot_reasoning"] = result_dict["reasoning"]
                        inv["cot_content"] = result_dict["content"]
                    else:
                        logger.warning(f"Invalid marker format: {marker}, skipping CoT generation")
                processed_count += 1
                
        processed_data.append(entry)    
    output_path = Path(cfg.output.preprocessed_invariants_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Processed {processed_count} invariants")
    return processed_data

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
    data_path = Path(cfg.dataset.json_path)
    with open(data_path, 'r') as f:
        data = json.load(f)
    data = data[: cfg.dataset.limit]
    logger.info(f"Loaded {len(data)} entries from {data_path}")
    # if cfg.pipeline.generalize_invariants:
    if cfg.weave.log_to_weave:
        weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")

    client = Together()
    logger.info(f"Preprocessing all ground truth invariants and saving to {cfg.output.preprocessed_invariants_path}")
    preprocess_all_gt_invariants(data, client, cfg, verifier)


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=1
# uv run -m src.preprocess.preprocess_gt_invariants dataset.limit=-1  # Process all

# Config Location:
# configs/preprocess/preprocess_gt_invariants.yaml

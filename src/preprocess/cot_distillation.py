#!/usr/bin/env python3
"""
Generate Chain-of-Thought (CoT) reasoning for simplified invariants.

This script:
1. Loads training data with simplified invariants
2. For each simplified invariant, generates CoT reasoning using Kimi K2 Thinking
3. Outputs a new JSON file with CoT reasoning added to each invariant

Usage:
    uv run -m src.preprocess.cot_distillation
    uv run -m src.preprocess.cot_distillation dataset.limit=5
"""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from together import Together
from tqdm import tqdm

import weave

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
    # Build user prompt
    user_prompt = cfg.prompts.user.format(
        program=program,
        target_label=target_label,
        ground_truth_invariant_content=simplified_invariant,
    )

    messages = [
        {"role": "system", "content": cfg.prompts.system},
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


@weave.op()
def distill_cot_for_all_invariants(
    data: list[dict[str, Any]],
    save_path: str,
    client: Together,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    """Generate CoT reasoning for all simplified invariants in the data."""
    logger.info(f"Processing {len(data)} entries")
    
    # Process entries
    processed_data = []
    processed_count = 0

    for entry in tqdm(data, desc="generating CoT reasoning"):
        # Check if this entry has invariants
        if "invariants" in entry and entry["invariants"]:
            program = entry.get("program_for_llm", "")
            if not program:
                logger.warning(f"Entry {entry.get('file', 'unknown')} missing program_for_llm, skipping")
                processed_data.append(entry)
                continue

            # Process each invariant
            for inv in entry["invariants"]:
                # Check if simplified_invariant exists
                if "simplified_invariant" not in inv or not inv["simplified_invariant"]:
                    logger.warning(
                        f"Invariant {inv.get('marker', 'unknown')} missing simplified_invariant, skipping"
                    )
                    continue

                # Extract target label from marker (e.g., "INVARIANT_MARKER_1" -> "1")
                marker = inv.get("marker", "")
                if not marker.startswith("INVARIANT_MARKER_"):
                    logger.warning(f"Invalid marker format: {marker}, skipping")
                    continue

                target_label = marker.split("_")[-1]
                simplified_inv = inv["simplified_invariant"]

                # Generate CoT reasoning
                logger.info(
                    f"Generating CoT for marker {marker}, invariant: {simplified_inv}"
                )
                result_dict = generate_cot_reasoning(
                    client, program, target_label, simplified_inv, cfg
                )
                
                # Add CoT reasoning to the invariant
                inv["cot_reasoning"] = result_dict["reasoning"]
                inv["cot_content"] = result_dict["content"]
                processed_count += 1

            processed_data.append(entry)
        else:
            # Entry has no invariants, keep as is
            processed_data.append(entry)

    # Save results
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Processed {processed_count} invariants with CoT reasoning")

    return processed_data


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="cot_distilation",
)
def main(cfg: DictConfig) -> None:
    """Main function for CoT distillation."""

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Load training data
    data_path = Path(cfg.dataset.json_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Apply limit if specified (limit=-1 means process all)
    if cfg.dataset.limit > 0:
        data = data[: cfg.dataset.limit]
    # If limit is -1 or negative, process all (no slicing needed)
    
    logger.info(f"Loaded {len(data)} entries from {data_path}")

    if cfg.pipeline.distil_cot:
        if cfg.weave.log_to_weave:
            weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")

        client = Together()
        logger.info(
            f"Generating CoT reasoning for simplified invariants and saving to {cfg.output.cot_distilation_path}"
        )
        distill_cot_for_all_invariants(
            data, cfg.output.cot_distilation_path, client, cfg
        )


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m src.preprocess.cot_distillation dataset.limit=1
# uv run -m src.preprocess.cot_distillation dataset.limit=-1  # Process all

# Config Location:
# configs/preprocess/cot_distilation.yaml

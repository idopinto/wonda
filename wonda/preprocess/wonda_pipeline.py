#!/usr/bin/env python3
"""Run the full Wonda data pipeline (Stage 1 + Stage 2) from the paper.

Takes raw training data with verifier-generated invariants and produces
clean, verified invariant candidates suitable for SFT training.

Stage 1 — Normalization (gt_invariant_normalization.py):
  AST-based rewriting that strips redundant casts, eliminates tautologies /
  contradictions, and minimizes parentheses.

Stage 2 — Simplification (gt_invariant_simplification.py):
  For each invariant:
    - If verbose: generate N simplified candidates via an LLM, then verify
      each for correctness and sufficiency in parallel.
    - If compact: verify as-is.
  Filter by quality_grade > 0 and save qualifying entries.

Usage:
    for testing:
      uv run -m wonda.preprocess.wonda_pipeline dataset.limit=5
    for full run:
      uv run -m wonda.preprocess.wonda_pipeline dataset.limit=-1
    for slurm run:
      sbatch scripts/preprocess/wonda_pipeline.sbatch pipe.limit=-1
"""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import weave
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from together import Together
from tqdm import tqdm

load_dotenv()

from configs import global_config as GC
from wonda.core.ast_program import AstProgram
from wonda.preprocess.gt_invariant_normalization import normalize_invariant
from wonda.preprocess.gt_invariant_simplification import (
    InvariantContext,
    process_invariant,
)
from wonda.preprocess.pipe_utils import (
    build_processed_keys,
    get_checkpoint_path,
    load_attempted_keys,
    load_data,
    load_from_hf,
    load_existing_progress,
    save_checkpoint,
)

from wonda.verifiers.uautomizer import UAutomizerVerifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



# =============================================================================
# Main Pipeline
# =============================================================================

def run_wonda(
    data: list[dict[str, Any]],
    client: Together,
    cfg: DictConfig,
    verifier: UAutomizerVerifier,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Main wonda data pipeline with resume support and prompt caching.."""


    output_path = output_dir / f"{output_dir.name}.json"
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = get_checkpoint_path(output_path)
    logger.info(f"\033[34mOutput path: {output_path}\033[0m")
    logger.info(f"\033[34mCheckpoint path: {checkpoint_path}\033[0m")

    # Load existing progress for resume capability
    processed = load_existing_progress(output_path)

    # Load attempted keys from checkpoint (includes both successes AND failures)
    attempted_keys = load_attempted_keys(checkpoint_path)

    # If no checkpoint exists, rebuild from successful entries
    if not attempted_keys:
        attempted_keys = build_processed_keys(processed)
        if attempted_keys:
            logger.info(f"Rebuilt {len(attempted_keys)} attempted keys from output file")

    skipped_dict ={
      "false_decision": 0,
      "unknown_decision": 0,
      "true_decision_no_invariants": 0,
      "already_attempted": 0,
    }
    
    for i, entry in enumerate(tqdm(data, desc="Processing")):
        # baseline_timing = entry.get("timings", {}).get("median", 0.0)
        baseline_timing = entry.get("median_timing", 0.0)
        # print(f"Baseline timing: {baseline_timing}")
        baseline_decision = entry.get("baseline_decision", "TIMEOUT")
        original_program = entry.get("original_program", "")
        file_name = entry.get("file", "")
        
        # Skip invalid entries, must be TRUE otherwise there are no invariants to extract (even if FALSE).
        if baseline_decision != "TRUE" or not entry.get("invariants"):
            if baseline_decision == "FALSE":
                skipped_dict["false_decision"] += 1
            elif baseline_decision == "TIMEOUT":
                skipped_dict["unknown_decision"] += 1
            else:
                skipped_dict["true_decision_no_invariants"] += 1
            # logger.warning(f"Entry {i}: skipping (invalid baseline (TIMEOUT or FALSE) or no invariants). Skipped {skipped_dict['false_decision']} false decisions, {skipped_dict['unknown_decision']} unknown decisions, {skipped_dict['true_decision_no_invariants']} true decisions with no invariants so far.")
            continue
        # Parse program once per entry
        program = AstProgram().from_code(original_program)
        program.process(print_ast=False)
        
        for j, inv in enumerate(entry["invariants"]):
            original_inv = inv.get("invariant", "")
            marker = inv.get("marker", "")
            
            # Check if this invariant was already attempted (resume support)
            key = (file_name, marker, original_inv)
            if key in attempted_keys:
                skipped_dict["already_attempted"] += 1
                if skipped_dict["already_attempted"] % 100 == 0:
                    logger.info(f"Skipped {skipped_dict['already_attempted']} already-attempted invariants...")
                continue
            
            normalized_inv = ""
            if cfg.pipe.normalize:
                normalized_inv = normalize_invariant(original_inv, pretty=True)
            invariant_to_process = normalized_inv if cfg.pipe.normalize else original_inv
            ctx = InvariantContext(
                entry=entry,
                entry_idx=i,
                inv_idx=j,
                marker=marker,
                original_invariant=original_inv,
                pretty_invariant=normalized_inv,
                invariant_to_process=invariant_to_process,
                baseline_decision=baseline_decision,
                baseline_timing=baseline_timing,
                program=program,
                original_program=original_program,
            )
            
            logger.info(f"Entry {i+1}/{len(data)}, Inv {j+1}/{len(entry['invariants'])}: marker={ctx.marker}, timing={baseline_timing:.2f}s\ninvariant_to_process={ctx.invariant_to_process}")
            
            try:
                new_entries = process_invariant(ctx, client, verifier, cfg)
                processed.extend(new_entries)
                # Mark this invariant as attempted (success or failure)
                attempted_keys.add(key)
              # Save intermediate results after each invariant
                with open(output_path, "w") as f:
                   json.dump(processed, f, indent=2)
              
                # Save checkpoint with all attempted keys
                save_checkpoint(checkpoint_path, attempted_keys)
                
            except Exception as e:
                logger.error(f"Entry {i+1}/{len(data)}, Inv {j+1}/{len(entry['invariants'])}: Failed with error: {type(e).__name__}: {e}. Skipping...")
                # Still mark as attempted so we don't retry on crash
                attempted_keys.add(key)
                save_checkpoint(checkpoint_path, attempted_keys)
                continue
    
    if skipped_dict["already_attempted"] > 0:
        logger.info(f"Resumed run: skipped {skipped_dict['already_attempted']} already-attempted invariants")
    logger.info(f"Total qualifying entries: {len(processed)}")
    logger.info(f"Skipped {skipped_dict['false_decision']} false decisions, {skipped_dict['unknown_decision']} unknown decisions, {skipped_dict['true_decision_no_invariants']} true decisions with no invariants")
    
    # Push to hub (only when push_to_hub=true and hf_organization is set)
    if cfg.pipe.push_to_hub and cfg.pipe.get("hf_organization"):
        dataset = Dataset.from_list(processed)
        hub_name = f"{cfg.pipe.hf_organization}/{output_dataset_name}"
        logger.info(f"Pushing to: {hub_name}")
        DatasetDict({"train": dataset}).push_to_hub(hub_name)
    elif cfg.pipe.push_to_hub:
        logger.warning("push_to_hub is true but pipe.hf_organization is not set. Skipping push.")
    
    return processed



@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="wonda_data_pipeline",
)
def main(cfg: DictConfig) -> None:
    # logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"\033[34mTest mode: {cfg.test_mode}\033[0m")
    test_suffix = "-test" if cfg.test_mode else ""
    output_dataset_name = f"{cfg.pipe.dataset_name}-{cfg.pipe.output_suffix}{test_suffix}"
    output_dir = GC.TRAINING_DATASET_DIR / output_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
    )

    input_dataset_name = f"{cfg.pipe.dataset_name}-{cfg.pipe.input_suffix}"
    # input_path = GC.TRAINING_DATASET_DIR / input_dataset_name / f"{input_dataset_name}.json"
    # data = load_data(input_path, cfg.pipe.limit)
    hf_org = cfg.pipe.get("hf_organization")
    if not hf_org:
        raise ValueError(
            "pipe.hf_organization is not set. Set it to your HuggingFace org or use 'idopinto' to read the public raw dataset."
        )
    data = load_from_hf(f"{hf_org}/{cfg.pipe.dataset_name}-{cfg.pipe.input_suffix}", split="full", limit=cfg.pipe.limit)
    logger.info(f"\033[34mProcessing {len(data)} entries from {input_dataset_name}.\033[0m")

    weave_project = f"{cfg.weave.project_name}{test_suffix}"
    if cfg.weave.get("use_weave", False) and cfg.weave.get("entity"):
        weave.init(f"{cfg.weave.entity}/{weave_project}")
        logger.info(f"\033[34mWeave initialized ({weave_project})\033[0m")
    elif cfg.weave.get("use_weave", False):
        logger.info("Weave skipped: set weave.entity to your W&B entity to enable.")

    client = Together()
    logger.info(f"\033[34mTogether client initialized. using model {cfg.model.name} with temperature {cfg.model.temperature}, max tokens {cfg.model.max_tokens}, n={cfg.model.n} candidates.\033[0m")
    logger.info(f"\033[34mWonda data pipeline started\033[0m")
    run_wonda(data, client, cfg, verifier, output_dir)
    logger.info(f"\033[34mWonda data pipeline completed\033[0m")

if __name__ == "__main__":
    main()

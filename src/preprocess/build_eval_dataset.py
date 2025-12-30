#!/usr/bin/env python3
"""Build baseline evaluation dataset using AST-based program processing."""

import logging
from pathlib import Path
import json
from datasets import Dataset, DatasetDict

import hydra
from omegaconf import DictConfig, OmegaConf

from configs import global_config as GC
from src.verifiers.uautomizer import UAutomizerVerifier
from src.preprocess.baseline_dataset_common import (
    run_uautomizer_as_baseline,
    logger,
    create_hf_base_dataset,
)

# logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="build_eval_dataset",
)
def main(cfg: DictConfig):
    # """Main function for building evaluation dataset."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    test_suffix = "test" if cfg.output.test_mode else "full"
    output_dir = (
        GC.DATASET_DIR
        / "eval"
        / f"invbench-eval-uautomizer{cfg.verifier.version}-k{cfg.verifier.k}-{test_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{output_dir.name}.json"
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Results path: {results_path}")

    property_path = GC.PROPERTIES_DIR / cfg.verifier.property
    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=property_path,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
    )
    run_uautomizer_as_baseline(cfg=cfg, verifier=verifier, results_path=results_path)

    create_hf_base_dataset(
        results_path=results_path,
        output_dir=output_dir,
        push_to_hub=cfg.output.push_to_hub,
    )


if __name__ == "__main__":
    main()

# Example usage:
# uv run -m src.preprocess.build_eval_dataset output.test_mode=true dataset.limit=1 verifier.k=3 processing.ignore_invariants=true output.push_to_hub=true

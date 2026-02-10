#!/usr/bin/env python3
"""Build the baseline evaluation dataset using UAutomizer.

Usage:
    uv run -m wonda.preprocess.build_eval_dataset output.test_mode=true dataset.limit=1
    uv run -m wonda.preprocess.build_eval_dataset verifier.k=3 output.push_to_hub=true
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from configs import global_config as GC
from wonda.preprocess.baseline_dataset_common import (
    create_hf_base_dataset,
    logger,
    run_uautomizer_as_baseline,
)
from wonda.verifiers.uautomizer_runlim import UAutomizerVerifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="build_eval_dataset",
)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    test_suffix = "-test" if cfg.output.test_mode else ""
    output_dir = GC.DATASET_DIR / "eval" / f"wonda-eval-benchmark-full{test_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{output_dir.name}.json"
    logger.info(f"Output: {output_dir}")

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
        memory_limit_mb=GC.MEMORY_LIMIT_MB,
    )

    run_uautomizer_as_baseline(cfg=cfg, verifier=verifier, results_path=results_path)
    create_hf_base_dataset(
        results_path=results_path,
        output_dir=output_dir,
        push_to_hub=cfg.output.push_to_hub,
    )


if __name__ == "__main__":
    main()

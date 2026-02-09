#!/usr/bin/env python3
"""Run the full Wonda data pipeline (Stage 1 + Stage 2).

Takes raw training data with verifier-generated invariants and produces
clean, verified invariant candidates suitable for SFT training.

Stage 1 — Normalization (gt_invariant_normalization):
  AST-based rewriting that strips redundant casts, eliminates tautologies /
  contradictions, and minimizes parentheses.

Stage 2 — Simplification (gt_invariant_simplification):
  For each invariant:
    - If verbose: generate N simplified candidates via an LLM, then verify
      each for correctness and usefulness in parallel.
    - If compact: verify as-is.
  Filter by quality_grade > 0 and save qualifying entries.

Usage:
    uv run -m wonda.data_pipeline.run_wonda_pipeline dataset.limit=5
    uv run -m wonda.data_pipeline.run_wonda_pipeline dataset.limit=-1
    sbatch scripts/preprocess/preprocess_gt_invariants_parallel.sbatch dataset.limit=-1
"""

import logging
from pathlib import Path

import hydra
import weave
from omegaconf import DictConfig, OmegaConf
from together import Together

from configs import global_config as GC
from wonda.data_pipeline.gt_invariant_simplification import (
    load_data,
    run_wonda,
)
from wonda.verifiers.uautomizer import UAutomizerVerifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="preprocess_gt_invariants_parallel",
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    verifier = UAutomizerVerifier(
        uautomizer_path=GC.UAUTOMIZER_PATHS[cfg.verifier.version],
        property_file_path=GC.PROPERTIES_DIR / cfg.verifier.property,
        arch=cfg.verifier.arch,
        timeout_seconds=cfg.verifier.timeout,
        version=cfg.verifier.version,
    )

    data = load_data(Path(cfg.dataset.name), cfg.dataset.limit)

    if cfg.weave.get("use_weave", False):
        weave.init(f"{cfg.weave.entity}/{cfg.weave.project_name}")

    client = Together()
    run_wonda(data, client, cfg, verifier)


if __name__ == "__main__":
    main()

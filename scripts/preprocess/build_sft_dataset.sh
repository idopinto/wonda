#!/bin/bash
# Build SFT-ready training dataset from raw or v2 datasets.
#
# Usage:
#   ./scripts/preprocess/build_sft_dataset.sh
#   ./scripts/preprocess/build_sft_dataset.sh --config-name=build_sft_dataset dataset.inv_mode=simplified dataset.min_grade=3
#   ./scripts/preprocess/build_sft_dataset.sh dataset.json_path=data/train/sft-ready/sft_dataset_simplified_g2.json

set -e

cd "$(dirname "$0")/../.."

echo "=== Building SFT dataset (preprocess step) ==="

uv run -m wonda.data_pipeline.build_sft_dataset "$@"

echo "=== Done ==="

#!/bin/bash
# Build SFT-ready training dataset from raw or v2 datasets.
#
# Usage:
#   ./scripts/preprocess/build_sft_dataset.sh
#   ./scripts/preprocess/build_sft_dataset.sh dataset.version=v0 dataset.input_repo=idopinto/wonda-train-dataset-full-raw dataset.split=full
#   ./scripts/preprocess/build_sft_dataset.sh dataset.version=v1 dataset.input_repo=idopinto/wonda-train-dataset-full-raw dataset.split=full
#   ./scripts/preprocess/build_sft_dataset.sh dataset.version=v2 dataset.min_grade=1
#   ./scripts/preprocess/build_sft_dataset.sh dataset.version=v2 dataset.min_grade=2
#   ./scripts/preprocess/build_sft_dataset.sh dataset.version=v2 dataset.min_grade=3


set -e

cd "$(dirname "$0")/../.."

echo "=== Building SFT dataset (preprocess step) ==="

uv run -m wonda.preprocess.build_sft_dataset "$@"

echo "=== Done ==="

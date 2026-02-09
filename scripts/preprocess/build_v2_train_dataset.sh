#!/bin/bash
# Build v2 training dataset and upload to HuggingFace
#
# Usage:
#   ./scripts/preprocess/build_v2_train_dataset.sh                    # Just build locally
#   ./scripts/preprocess/build_v2_train_dataset.sh --push-to-hub      # Build and push to HF

set -e

cd "$(dirname "$0")/../.."

echo "=== Building v2 Training Dataset ==="
echo "Input: data/train/invbench-train-uautomizer25-k1-full-raw/invbench-train-uautomizer25-k1-full-parallel-latest.json"
echo "Output: data/train/invbench-train-v2"

uv run -m wonda.data_pipeline.build_v2_train_dataset \
    --input "data/train/invbench-train-uautomizer25-k1-full-raw/invbench-train-uautomizer25-k1-full-parallel-latest.json" \
    --output-dir "data/train/invbench-train-v2" \
    --dataset-name "invbench-train-v2" \
    "$@"

echo "=== Done ==="

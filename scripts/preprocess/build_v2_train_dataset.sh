#!/bin/bash
# Run the Wonda data pipeline (Stage 2: ground-truth invariant simplification)
#
# Usage:
#   ./scripts/preprocess/build_v2_train_dataset.sh                       # Process all
#   ./scripts/preprocess/build_v2_train_dataset.sh dataset.limit=5       # Process 5 samples

set -e

cd "$(dirname "$0")/../.."

echo "=== Running Wonda Pipeline (AST-based Invariant Normalization + LLM-based Invariant Simplification) ==="

uv run -m wonda.data_pipeline.wonda_pipeline "$@"

echo "=== Done ==="

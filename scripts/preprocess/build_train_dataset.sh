#!/bin/bash
#SBATCH --job-name=build-train-dataset
#SBATCH --output=slurm/build_train_dataset_%j.out
#SBATCH --error=slurm/build_train_dataset_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Script to build baseline training dataset using AST-based program processing
# Usage: sbatch scripts/preprocess/build_train_dataset.sh [HYDRA_OVERRIDES]
#
# This script runs UAutomizer verification on training programs sequentially
# to ensure accurate timing measurements. Creates baseline dataset for training.
# Runs on H200 to match the compute node used for LLM evaluation.
#
# Hydra config overrides (pass as arguments):
#   dataset.limit=N                      Limit number of files to process, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Filter files by prefix (used for testing)
#   verifier.k=N                         Number of times to run UAutomizer per file (default: 3)
#   verifier.timeout=N                   Timeout per verification in seconds (default: 600.0)
#   processing.ignore_invariants=true    Skip extracting invariants from logs
#   output.test_mode=true                Run in test mode (affects output directory naming)
#   output.push_to_hub=true             Push datasets to Hugging Face hub
#
# Examples:
#   sbatch scripts/preprocess/build_train_dataset.sh output.test_mode=true dataset.limit=1 verifier.k=1
#   sbatch scripts/preprocess/build_train_dataset.sh output.test_mode=false output.push_to_hub=true

echo "Starting train dataset build..."
set -e

# Project root - defaults to the directory from which sbatch was submitted
# Override by setting PROJECT_ROOT before calling sbatch
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Building Training Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Hydra overrides: $@"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  GPUs: $SLURM_GPUS"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# Run train dataset builder
uv run -m wonda.data_pipeline.build_raw_train_dataset "$@"

echo ""
echo "=========================================="
echo "Train dataset build completed!"
echo "End time: $(date)"
echo "=========================================="

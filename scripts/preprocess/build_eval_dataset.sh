#!/bin/bash
#SBATCH --job-name=build-eval-dataset
#SBATCH --output=slurm/build_eval_dataset_%j.out
#SBATCH --error=slurm/build_eval_dataset_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Script to build baseline evaluation dataset using AST-based program processing
# Usage: sbatch scripts/preprocess/build_eval_dataset.sh [HYDRA_OVERRIDES]
#
# This script runs UAutomizer verification on evaluation programs sequentially
# to ensure accurate timing measurements for speedup calculations.
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
#   sbatch scripts/preprocess/build_eval_dataset.sh output.test_mode=true dataset.limit=1 verifier.k=3
#   sbatch scripts/preprocess/build_eval_dataset.sh output.test_mode=false output.push_to_hub=true

echo "Starting eval dataset build..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Building Evaluation Dataset"
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

# Run eval dataset builder
uv run -m src.preprocess.build_eval_dataset "$@"

echo ""
echo "=========================================="
echo "Eval dataset build completed!"
echo "End time: $(date)"
echo "=========================================="

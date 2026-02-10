#!/bin/bash
#SBATCH --job-name=preprocess-gt-invariants
#SBATCH --output=slurm/preprocess_gt_invariants_%j.out
#SBATCH --error=slurm/preprocess_gt_invariants_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Script to preprocess ground truth invariants using Kimi K2 Thinking
# Usage: sbatch scripts/preprocess/preprocess_gt_invariants.sh [HYDRA_OVERRIDES]
#
# This script:
# 1. Loads training data with verbose invariants
# 2. Filters invariants based on complexity criteria
# 3. Uses Kimi K2 Thinking to generalize them into compact formulas
# 4. Outputs a new JSON file with generalized invariants
#
# Hydra config overrides (pass as arguments):
#   dataset.limit=N                      Limit number of entries to process, -1 for all (default: 10)
#   dataset.json_path=PATH                Path to input JSON file
#   output.preprocessed_invariants_path=PATH  Path to output JSON file
#   pipeline.generalize_invariants=true  Enable invariant generalization (default: true)
#   verifier.timeout=N                   Timeout per verification in seconds (default: 600.0)
#   weave.log_to_weave=true              Enable Weave logging (default: true)
#
# Examples:
#   sbatch scripts/preprocess/preprocess_gt_invariants.sh dataset.limit=1
#   sbatch scripts/preprocess/preprocess_gt_invariants.sh dataset.limit=-1  # Process all

echo "Starting ground truth invariants preprocessing..."
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
echo "Preprocessing Ground Truth Invariants"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Hydra overrides: $@"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# Run ground truth invariants preprocessor
uv run -m wonda.preprocess.gt_invariant_simplification "$@"

echo ""
echo "=========================================="
echo "Ground truth invariants preprocessing completed!"
echo "End time: $(date)"
echo "=========================================="

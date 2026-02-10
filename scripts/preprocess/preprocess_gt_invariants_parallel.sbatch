#!/bin/bash
#SBATCH --job-name=preprocess-gt-invariants-parallel
#SBATCH --output=slurm/preprocess_gt_invariants_parallel_%j.out
#SBATCH --error=slurm/preprocess_gt_invariants_parallel_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Script to preprocess ground truth invariants with parallel N-candidate generation
# Usage: sbatch scripts/preprocess/preprocess_gt_invariants_parallel.sbatch [HYDRA_OVERRIDES]
#
# This script:
# 1. Loads training data with verbose invariants
# 2. Generates N simplified invariant candidates per verbose invariant
# 3. Verifies all N candidates in parallel (correctness + usefulness)
# 4. Filters and saves candidates with quality_grade > 0 as separate entries
#
# Resource allocation for 5 candidates (default):
#   - 5 outer workers × 2 inner checks = 10 max concurrent verifier processes
#   - 16 CPUs: ~1.6 cores per verifier process
#   - 64G RAM: ~6.4 GB per verifier process
#
# Hydra config overrides (pass as arguments):
#   dataset.limit=N                           Limit number of entries to process, -1 for all
#   model.n=N                                 Number of candidates to generate per invariant (default: 5)
#   verifier.max_parallel_workers=N           Should equal model.n (default: 5)
#   verifier.timeout=N                        Timeout per verification in seconds (default: 600)
#   output.preprocessed_invariants_path=PATH  Path to output JSON file
#   weave.use_weave=true                      Enable Weave logging
#
# Examples:
#   sbatch scripts/preprocess/preprocess_gt_invariants_parallel.sbatch dataset.limit=10
#   sbatch scripts/preprocess/preprocess_gt_invariants_parallel.sbatch dataset.limit=-1  # Process all

echo "Starting parallel ground truth invariants preprocessing..."
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
echo "Preprocessing Ground Truth Invariants (Parallel)"
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

# Run parallel ground truth invariants preprocessor
uv run -m wonda.preprocess.gt_invariant_simplification "$@"

echo ""
echo "=========================================="
echo "Parallel ground truth invariants preprocessing completed!"
echo "End time: $(date)"
echo "=========================================="

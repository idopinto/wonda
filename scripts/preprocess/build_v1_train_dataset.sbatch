#!/bin/bash
#SBATCH --job-name=build-v1-train-dataset
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --exclude=joey-01

# Script to build v1 (cleaned) training dataset with parallel processing
# Usage: sbatch scripts/preprocess/build_v1_train_dataset.sbatch [HYDRA_OVERRIDES]
#
# This script:
# 1. Loads raw training data with UAutomizer invariants
# 2. Cleans each invariant using clean_invariant(inv, pretty=True)
#    - Removes integral casts (long long, unsigned int, etc.)
#    - Removes tautologies (x <= x, 0 < 1, etc.)
#    - Pretty-prints with minimal parentheses
# 3. Verifies each cleaned invariant in parallel (correctness + usefulness)
# 4. Assigns quality grades (0-3)
# 5. Saves qualifying entries (quality_grade > 0) and failed entries separately
#
# Resource allocation for 8 parallel workers (default):
#   - 8 outer workers × 2 inner checks = 16 max concurrent verifier processes
#   - 64 CPUs: ~4 cores per verifier process
#   - 256G RAM: ~16 GB per verifier process
#
# Hydra config overrides (pass as arguments):
#   dataset.limit=N                           Limit number of entries to process, -1 for all
#   verifier.timeout=N                        Timeout per verification in seconds (default: 600)
#   pipeline.verify=false                     Skip verification (just clean invariants)
#   pipeline.max_parallel_workers=N           Number of parallel workers (default: 8)
#   pipeline.inv_mode=raw|cleaned             Which invariant to verify: "raw" (v0) or "cleaned" (v1)
#   output.preprocessed_invariants_path=PATH  Path to output JSON file
#   output.push_to_hub=true                   Push to HuggingFace Hub
#
# Examples:
#   sbatch scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=10
#   sbatch scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=-1  # Process all
#   sbatch scripts/preprocess/build_v1_train_dataset.sbatch pipeline.inv_mode=raw  # Verify raw (v0) invariants
#   sbatch scripts/preprocess/build_v1_train_dataset.sbatch pipeline.inv_mode=cleaned  # Verify cleaned (v1) invariants
#   sbatch scripts/preprocess/build_v1_train_dataset.sbatch pipeline.max_parallel_workers=4  # Fewer workers

# # Verify cleaned invariants (v1)
# sbatch --job-name=build-v1-train-dataset scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=-1 pipeline.inv_mode=cleaned

# # Verify raw invariants (v0)
# sbatch --job-name=build-v0-train-dataset scripts/preprocess/build_v1_train_dataset.sbatch dataset.limit=-1 pipeline.inv_mode=raw

echo "Starting v1 training dataset build..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Building v1 (Cleaned) Training Dataset"
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

# Run v1 training dataset builder
uv run -m src.preprocess.build_v1_train_dataset "$@"

echo ""
echo "=========================================="
echo "v1 training dataset build completed!"
echo "End time: $(date)"
echo "=========================================="

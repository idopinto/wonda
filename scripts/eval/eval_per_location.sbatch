#!/bin/bash
#SBATCH --job-name=eval-per-loc
#SBATCH --output=slurm/eval_per_location_%j.out
#SBATCH --error=slurm/eval_per_location_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Script to run per-location evaluation using HuggingFace client
# Usage: sbatch scripts/eval/eval_per_location.sbatch [HYDRA_OVERRIDES]
# This script evaluates invariant generation at each candidate location separately.
# Works best on single h200 gpu; if unavailable, consider 2 l40s to avoid OOM.
#
# Hydra config overrides (pass as arguments):
#   dataset.split=easy|hard              Data split (default: hard)
#   dataset.limit=N                      Limit number of tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Prefix for dataset files
#   weave.skip_wandb=true                Skip uploading plot to W&B
#   model.eval_finetuned_model=true      Evaluate SFT model instead of base model
#
# Examples:
#   sbatch scripts/eval/eval_per_location.sbatch dataset.split=easy dataset.limit=1
#   sbatch scripts/eval/eval_per_location.sbatch dataset.split=hard
#   sbatch scripts/eval/eval_per_location.sbatch model.eval_finetuned_model=true

echo "Starting per-location evaluation..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen 

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Starting Per-Location Evaluation"
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

# Run per-location evaluation with HF client
uv run -m src.eval.evaluate_per_location model.client=hf "$@"

echo ""
echo "=========================================="
echo "Per-location evaluation completed!"
echo "End time: $(date)"
echo "=========================================="


# sbatch scripts/eval/eval_per_location.sbatch dataset.split=hard dataset.limit=-1 model.reasoning_effort=low
# sbatch scripts/eval/eval_per_location.sbatch dataset.split=hard model.eval_finetuned_model=true model.reasoning_effort=low

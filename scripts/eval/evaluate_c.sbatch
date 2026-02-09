#!/bin/bash
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Script to run evaluation using OpenRouter Client (gpt5.2 evaluation) 
# Usage: sbatch scripts/eval/evaluate_c.sbatch [HYDRA_OVERRIDES]
# This scripts doesn't require GPU but for best comparsion use the same hardware used for uautomizer baseline.
# Hydra config overrides (pass as arguments):
#   dataset.split=easy|hard              Data split (default: hard)
#   dataset.limit=N                      Limit number of tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Prefix for dataset files
#   weave.skip_wandb=true                Skip uploading scatter plot to W&B
#
# Examples:
#   sbatch scripts/eval/evaluate_c.sbatch dataset.split=easy dataset.limit=1
#   sbatch scripts/eval/evaluate_c.sbatch dataset.split=hard

echo "Starting OpenRouter evaluation..."
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
echo "Starting Evaluation (OpenRouter Client)"
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

# Run evaluation with OpenRouter client (override client=openrouter via Hydra)
uv run -m wonda.eval.evaluate "$@"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

# sbatch --job-name=eval_gpt5 scripts/eval/evaluate_c.sbatch models=openai_config dataset.limit=-1 
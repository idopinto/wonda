#!/bin/bash
#SBATCH --job-name=eval-hf
#SBATCH --output=slurm/eval_hf_%j.out
#SBATCH --error=slurm/eval_hf_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:l40s:2

# Script to run evaluation using HuggingFace client (loads model directly)
# Usage: sbatch scripts/eval/eval_hf.sbatch [HYDRA_OVERRIDES]
# This script works best on single h200 gpu (for gpt-oss evaluation), if it is not available consider 2 l40s to avoid OOM.
#
# Hydra config overrides (pass as arguments):
#   dataset.split=easy|hard              Data split (default: hard)
#   dataset.limit=N                      Limit number of tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Prefix for dataset files
#   weave.skip_wandb=true                Skip uploading plot to W&B
#   model.eval_finetuned_model=true      Evaluate SFT model instead of base model
#
# Examples:
#   sbatch scripts/eval/eval_hf.sbatch dataset.split=easy dataset.limit=1
#   sbatch scripts/eval/eval_hf.sbatch dataset.split=hard
#   sbatch scripts/eval/eval_hf.sbatch model.eval_finetuned_model=true

echo "Starting HF evaluation..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen 

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Starting Evaluation (HuggingFace Client)"
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

# Run evaluation with HF client (override client=hf via Hydra)
uv run -m src.eval.evaluate model.client=hf "$@"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="


# sbatch scripts/eval/eval_hf.sbatch dataset.split=hard dataset.limit -1
# sbatch scripts/eval/eval_hf.sbatch dataset.split=hard model.eval_finetuned_model=true
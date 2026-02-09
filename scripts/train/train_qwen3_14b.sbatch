#!/bin/bash
#SBATCH --job-name=train_qwen_14b
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable
# Script to train Qwen 14B model using SFT
# Usage: sbatch scripts/train/train_qwen3_14b.sbatch [HYDRA_OVERRIDES]
#
# Hydra config overrides (pass as arguments):
#   --config-name=qwen3_14b_nt_v0 | qwen3_14b_nt_v1 | qwen3_14b_nt_v2
#   dataset.limit=N                      Limit number of training samples, -1 for all (default: -1)
#   sft.num_train_epochs=N               Number of training epochs (default: 2)
#   wandb.use_wandb=true|false           Enable/disable wandb logging (default: true)
#
# Examples:
#   sbatch --job-name=train_qwen_14b_v0 scripts/train/train_qwen3_14b.sbatch
#   sbatch --job-name=train_qwen_14b_v1 scripts/train/train_qwen3_14b.sbatch
#   sbatch --job-name=train_qwen_14b_v2 scripts/train/train_qwen3_14b.sbatch

echo "Starting Qwen 14B training..."
set -e

# Project root - defaults to the directory from which sbatch was submitted
# Override by setting PROJECT_ROOT before calling sbatch
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job info
echo "=========================================="
echo "Starting Qwen 14B SFT Training"
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

# Run training with config
# uv run -m src.train.train_qwen --config-name=qwen3_14b_nt_v0 "$@"
uv run -m src.train.train_qwen --config-name=qwen3_14b_nt_v1 "$@"
# uv run -m src.train.train_qwen --config-name=qwen3_14b_nt_v2 dataset.min_grade=1 "$@"
# uv run -m src.train.train_qwen --config-name=qwen3_14b_nt_v2 dataset.min_grade=2 "$@"
# uv run -m src.train.train_qwen --config-name=qwen3_14b_nt_v2 dataset.min_grade=3 "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="

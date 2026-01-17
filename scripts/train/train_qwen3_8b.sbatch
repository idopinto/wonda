#!/bin/bash
#SBATCH --job-name=train-qwen
#SBATCH --output=slurm/train_qwen_%j.out
#SBATCH --error=slurm/train_qwen_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Script to train Qwen model using SFT
# Usage: sbatch scripts/train/train_qwen.sbatch [HYDRA_OVERRIDES]
#
# Hydra config overrides (pass as arguments):
#   --config-name=qwen_0_6b_v0|qwen_0_6b_v1|qwen_1_7b_v0|qwen_1_7b_v1
#   dataset.limit=N                      Limit number of training samples, -1 for all (default: -1)
#   sft.num_train_epochs=N               Number of training epochs (default: 2)
#   wandb.use_wandb=true|false           Enable/disable wandb logging (default: true)
#
# Examples:
#   sbatch scripts/train/train_qwen3_8b.sbatch
#   sbatch scripts/train/train_qwen.sbatch
#   sbatch scripts/train/train_qwen.sbatch dataset.limit=100 sft.num_train_epochs=1

echo "Starting Qwen training..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen 

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Starting Qwen SFT Training"
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
# uv run -m src.train.sft.train_qwen --config-name=qwen3_8b_nt_v0 "$@"
uv run -m src.train.sft.train_qwen --config-name=qwen3_8b_nt_v1 "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="

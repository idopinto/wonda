#!/bin/bash
#SBATCH --job-name=train_qwen
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1

# Script to train Qwen model using SFT
# Usage: sbatch scripts/train/train_qwen3_0.6b.sbatch[HYDRA_OVERRIDES]
#
# Hydra config overrides (pass as arguments):
#   --config-name= qwen_0_6b_v0 | qwen_0_6b_v1| qwen_1_7b_v0| qwen_1_7b_v1
#   dataset.limit=N                      Limit number of training samples, -1 for all (default: -1)
#   sft.num_train_epochs=N               Number of training epochs (default: 2)
#   wandb.use_wandb=true|false           Enable/disable wandb logging (default: true)
#
# Examples:
#   sbatch --job-name=train_qwen_0.6_v2.3 scripts/train/train_qwen3_0.6b.sbatch
#   sbatch --job-name=train_qwen_0.6_v2.2 scripts/train/train_qwen3_0.6b.sbatch
#   sbatch --job-name=train_qwen_0.6_v2.1 scripts/train/train_qwen3_0.6b.sbatch

echo "Starting Qwen training..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen 

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v0 "$@"
# uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v1 "$@"
# uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v2 dataset.min_grade=1 "$@"
# uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v2 dataset.min_grade=2 "$@"
uv run -m src.train.sft.train_qwen --config-name=qwen3_0.6b_nt_v2 dataset.min_grade=3 "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="

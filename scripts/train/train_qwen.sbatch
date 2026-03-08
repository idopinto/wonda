#!/bin/bash
#SBATCH --job-name=train_qwen
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Unified script to train any Qwen model size using SFT.
# Usage: sbatch --job-name=<name> scripts/train/train_qwen.sbatch <config_name> [hydra overrides...]
#
# Config name (first argument): qwen3_0.6b | qwen3_1.7b | qwen3_4b | qwen3_8b | qwen3_14b
#
# Hydra overrides (optional, pass after config name):
#   dataset.version=v0|v1|v2           Dataset version (default in config: v2)
#   dataset.min_grade=1|2|3           Min quality grade for v2 (optional)
#   dataset.limit=N                   Limit samples, -1 for all
#   sft.num_train_epochs=N            Training epochs
#   wandb.use_wandb=true|false        Enable/disable wandb
#
# Explicit commands (model size × version; v2 uses min_grade=2, others use defaults):
#   0.6B
#   sbatch --job-name=train_qwen_0.6b_v0 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v0
#   sbatch --job-name=train_qwen_0.6b_v1 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v1
#   sbatch --job-name=train_qwen_0.6b_v2 scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_qwen_0.6b_v2_test scripts/train/train_qwen.sbatch qwen3_0.6b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   1.7B
#   sbatch --job-name=train_qwen_1.7b_v0 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v0
#   sbatch --job-name=train_qwen_1.7b_v1 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v1
#   sbatch --job-name=train_qwen_1.7b_v2 scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_qwen_1.7b_v2_test scripts/train/train_qwen.sbatch qwen3_1.7b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   4B
#   sbatch --job-name=train_qwen_4b_v0 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v0
#   sbatch --job-name=train_qwen_4b_v1 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v1
#   sbatch --job-name=train_qwen_4b_v2 scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_qwen_4b_v2_test scripts/train/train_qwen.sbatch qwen3_4b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   8B
#   sbatch --job-name=train_qwen_8b_v0 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v0
#   sbatch --job-name=train_qwen_8b_v1 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v1
#   sbatch --job-name=train_qwen_8b_v2 scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_qwen_8b_v2_test scripts/train/train_qwen.sbatch qwen3_8b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   14B
#   sbatch --job-name=train_qwen_14b_v0 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v0
#   sbatch --job-name=train_qwen_14b_v1 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v1
#   sbatch --job-name=train_qwen_14b_v2 scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_qwen_14b_v2_test scripts/train/train_qwen.sbatch qwen3_14b dataset.version=v2 dataset.min_grade=2 test_mode=true

echo "Starting Qwen training..."
set -e

CONFIG_NAME="${1:?Usage: sbatch train_qwen.sbatch <config_name> [overrides...]}"
shift
OVERRIDES=("$@")

# Project root - defaults to the directory from which sbatch was submitted
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

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
echo "Config: $CONFIG_NAME"
echo "Hydra overrides: ${OVERRIDES[*]}"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  GPUs: $SLURM_GPUS"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# Run training
uv run -m wonda.train.train_qwen --config-name="$CONFIG_NAME" "${OVERRIDES[@]}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="

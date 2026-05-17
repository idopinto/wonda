#!/bin/bash
#SBATCH --job-name=train_llama
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Unified script to train Llama 3.1 models using SFT.
# Usage: sbatch --job-name=<name> scripts/train/train_llama.sbatch <config_name> [hydra overrides...]
#
# Config name (first argument): llama3_8b | llama3_8b_full
#
# Hydra overrides (optional, pass after config name):
#   dataset.version=v0|v1|v2           Dataset version (default: v2)
#   dataset.min_grade=1|2|3            Min quality grade for v2 (optional)
#   dataset.limit=N                    Limit samples, -1 for all
#   sft.num_train_epochs=N             Training epochs
#   wandb.use_wandb=true|false         Enable/disable wandb
#   test_mode=true|false               Quick test with 100 samples
#
# Explicit commands:
#   8B LoRA
#   sbatch --job-name=train_llama_8b_v2_g2 scripts/train/train_llama.sbatch llama3_8b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_llama_8b_v2_g3 scripts/train/train_llama.sbatch llama3_8b dataset.version=v2 dataset.min_grade=3
#   sbatch --job-name=train_llama_8b_v2_g2_test scripts/train/train_llama.sbatch llama3_8b dataset.version=v2 dataset.min_grade=2 test_mode=true
#
#   8B Full fine-tune
#   sbatch --job-name=train_llama_8b_full_v2_g2 scripts/train/train_llama.sbatch llama3_8b_full dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_llama_8b_full_v2_g3 scripts/train/train_llama.sbatch llama3_8b_full dataset.version=v2 dataset.min_grade=3
#   sbatch --job-name=train_llama_8b_full_v2_g2_test scripts/train/train_llama.sbatch llama3_8b_full dataset.version=v2 dataset.min_grade=2 test_mode=true

echo "Starting Llama training..."
set -e

CONFIG_NAME="${1:?Usage: sbatch train_llama.sbatch <config_name> [overrides...]}"
shift
OVERRIDES=("$@")

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Starting Llama 3.1 SFT Training"
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

uv run -m wonda.train.train_llama --config-name="$CONFIG_NAME" "${OVERRIDES[@]}"

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="

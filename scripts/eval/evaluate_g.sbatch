#!/bin/bash
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Script to run evaluation using HuggingFace client (loads model directly)
# Usage: sbatch scripts/eval/evaluate_g.sbatch [HYDRA_OVERRIDES]
# This script works best on single h200 gpu (for gpt-oss evaluation mainly), if it is not available consider 2 l40s to avoid OOM.
# For Qwen3 evaluation a single l40s is enough. 

# Hydra config overrides (pass as arguments):
#   dataset.split=easy|hard              Data split (default: hard)
#   dataset.limit=N                      Limit number of tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Prefix for dataset files
#   weave.skip_wandb=true                Skip uploading plot to W&B
#   models.eval_ft_model=true      Evaluate SFT model instead of base model
#
# Examples:
#   sbatch scripts/eval/evaluate_g.sbatch dataset.split=easy dataset.limit=1
#   sbatch scripts/eval/evaluate_g.sbatch dataset.split=hard
#   sbatch scripts/eval/evaluate_g.sbatch models.eval_ft_model=true

echo "Starting local evaluation..."
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
echo "Starting Evaluation (Locally)"
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

# Run evaluation locally (override client=hf via Hydra)
uv run -m wonda.eval.evaluate "$@"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

# sbatch --job-name=eval_qwen scripts/eval/evaluate_g.sbatch models=qwen_no_think_config dataset.limit=-1 
# sbatch --job-name=eval_qwenv0 scripts/eval/evaluate_g.sbatch models=qwen_no_think_config models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-0.6b-gen-inv-sft-v0-exp2
# sbatch --job-name=eval_qwenv1 scripts/eval/evaluate_g.sbatch models=qwen_no_think_config models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-0.6b-gen-inv-sft-v1-exp2


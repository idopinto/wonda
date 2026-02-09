#!/bin/bash
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=62
#SBATCH --mem=1T
#SBATCH --gres=gpu:l40s:1

# Script to run evaluation using HuggingFace client (loads model directly)
# Usage: sbatch scripts/eval/evaluate_g.sbatch [HYDRA_OVERRIDES]
# This script works best on single h200 gpu (for gpt-oss evaluation mainly), if it is not available consider 2 l40s to avoid OOM.
# For Qwen3 evaluation a single l40s is enough. 
# 4, 16
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

# export WEAVE_PARALLELISM=4
echo "WEAVE_PARALLELISM: $WEAVE_PARALLELISM"
echo "=========================================="
echo ""

# Run evaluation locally (override client=hf via Hydra)
uv run -m wonda.eval.evaluate_vbs "$@"
# uv run -m wonda.eval.evaluate_vbs "$@"
# uv run -m wonda.eval.evaluate_vbs "$@"


echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

# OpenAI- gpt-5.2
# sbatch --job-name=eval_gpt5 scripts/eval/evaluate_g_vbp.sbatch models=openai_config dataset.limit=-1 statistics.num_trials=5

# Qwen/Qwen3-0.6B
# sbatch --job-name=eval_qwen3 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config dataset.limit=-1 statistics.num_trials=5
# sbatch --job-name=eval_qwen3_v0 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-0.6b-gen-inv-sft-v0 statistics.num_trials=5
# sbatch --job-name=eval_qwen3_v1 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-0.6b-gen-inv-sft-v1 statistics.num_trials=5
# sbatch --job-name=eval_qwen3_v2 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-0.6b-gen-inv-sft-v2 statistics.num_trials=5


# Qwen3/Qwen3-8B
# sbatch --job-name=eval_qwen3_8b scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.base_model.id="Qwen/Qwen3-8B" statistics.num_trials=5
# sbatch --job-name=eval_qwen3_8b_v0 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.base_model.id="Qwen/Qwen3-8B" models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-8b-gen-inv-sft-v0 ft_model.is_lora=true statistics.num_trials=5
# sbatch --job-name=eval_qwen3_8b_v1 scripts/eval/evaluate_g_vbp.sbatch models=qwen_no_think_config models.base_model.id="Qwen/Qwen3-8B" models.eval_ft_model=true models.ft_model.id=idopinto/qwen3-8b-gen-inv-sft-v1 ft_model.is_lora=true statistics.num_trials=5
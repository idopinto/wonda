#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#
# Single-run evaluation. Client (hf / vllm / openrouter) is set by the model config.
#
# Usage:
#   sbatch --job-name=<name> scripts/eval/run_evaluation.sbatch [HYDRA_OVERRIDES]
#
# Hydra overrides (pass as arguments):
#   models=CONFIG                       Model config (default from config.yaml: qwen3_0.6b_nt_config)
#   models.client=hf|vllm|openrouter    Override client (each model config has its own default)
#   models.eval_ft_model=true            Evaluate fine-tuned model instead of base
#   models.ft_model.sft_version="v0"     SFT version when eval_ft_model=true
#   models.ft_model.is_lora=true         Set for 8B/14B fine-tuned models
#   dataset.split=easy|hard             Split (default: hard)
#   dataset.limit=N                     Limit tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX               Prefix for dataset files
#   weave.skip_wandb=true               Skip uploading to W&B
#
# Notes:
# - For large external models (e.g. GPT-OSS), prefer single h200 GPU or 2× l40s to avoid OOM.
# - CUDA_VISIBLE_DEVICES is set by SLURM on GPU nodes.
#
# ==========================================
# Explicit commands for each evaluation:
# ==========================================
#
# --- Qwen3-0.6B ---
#   sbatch --job-name=eval_qwen06 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_0.6b_nt_config
#   sbatch --job-name=eval_qwen06_v0 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
#   sbatch --job-name=eval_qwen06_v1 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"
#   sbatch --job-name=eval_qwen06_v22 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
#
# --- Qwen3-1.7B ---
#   sbatch --job-name=eval_qwen17 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_1.7b_nt_config
#   sbatch --job-name=eval_qwen17_v0 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
#   sbatch --job-name=eval_qwen17_v22 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
#
# --- Qwen3-4B ---
#   sbatch --job-name=eval_qwen4b scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_4b_nt_config
#   sbatch --job-name=eval_qwen4b_v0 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
#   sbatch --job-name=eval_qwen4b_v22 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
#
# --- Qwen3-8B ---
#   sbatch --job-name=eval_qwen8b scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_8b_nt_config
#   sbatch --job-name=eval_qwen8b_v0 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
#   sbatch --job-name=eval_qwen8b_v22 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true
#
# --- Qwen3-14B ---
#   sbatch --job-name=eval_qwen14b scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_14b_nt_config
#   sbatch --job-name=eval_qwen14b_v0 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
#   sbatch --job-name=eval_qwen14b_v22 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true
#
# --- External / baseline models (client from config: openrouter, together, etc.) ---
#   sbatch --job-name=eval_gpt5 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=gpt_5.2_config
#   sbatch --job-name=eval_oss20 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=gpt_oss_20b_config
#   sbatch --job-name=eval_oss120 scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=gpt_oss_120b_config
#   sbatch --job-name=eval_qwen80b scripts/eval/run_evaluation.sbatch dataset.limit=-1 models=qwen3_80b_nt_config

echo "Starting evaluation..."
set -e

# Project root: use SLURM submit dir when running under sbatch, else script-relative
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Evaluation (single run)"
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

# All args forwarded as Hydra overrides to evaluate
uv run -m wonda.eval.evaluate "$@"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

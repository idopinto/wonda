#!/bin/bash
#SBATCH --job-name=multi-eval
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#
# Run multiple independent eval runs (fresh subprocess per run) and aggregate stats.
#
# Usage:
#   sbatch --job-name=<name> scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 [HYDRA_OVERRIDES]
#
# Notes:
# - Weave concurrency is controlled via WEAVE_PARALLELISM (set inside run_multi_eval/evaluate).
# - CUDA_VISIBLE_DEVICES is set by SLURM on GPU nodes; no need to set it manually.
# - For 8B/14B fine-tuned models, add models.ft_model.is_lora=true.
# - For single-run with the same pipeline/artifacts, use multi_run.num_runs=1.
#
# ==========================================
# Explicit commands for each evaluation:
# ==========================================
#
# --- Qwen3-0.6B ---
# Base model (no-think)
#   sbatch --job-name=me_qwen06 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_nt_config
# Base model (think)
#   sbatch --job-name=me_qwen06_t scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_t_config
# Fine-tuned v0
#   sbatch --job-name=me_qwen06_v0 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
# Fine-tuned v1
#   sbatch --job-name=me_qwen06_v1 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"
# Fine-tuned v2.2
#   sbatch --job-name=me_qwen06_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
# Fine-tuned v2.3
#   sbatch --job-name=me_qwen06_v23 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.3"
#
# --- Qwen3-1.7B ---
# Base model
#   sbatch --job-name=me_qwen17 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config
# Fine-tuned v0
#   sbatch --job-name=me_qwen17_v0 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
# Fine-tuned v1
#   sbatch --job-name=me_qwen17_v1 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"
# Fine-tuned v2.2
#   sbatch --job-name=me_qwen17_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
#
# --- Qwen3-4B ---
# Base model
#   sbatch --job-name=me_qwen4b scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_4b_nt_config
# Fine-tuned v0
#   sbatch --job-name=me_qwen4b_v0 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"
# Fine-tuned v1
#   sbatch --job-name=me_qwen4b_v1 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"
# Fine-tuned v2.2
#   sbatch --job-name=me_qwen4b_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"
#
# --- Qwen3-8B ---
# Base model (no-think)
#   sbatch --job-name=me_qwen8b scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_nt_config
# Base model (think)
#   sbatch --job-name=me_qwen8b_t scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_t_config
# Fine-tuned v0 (LoRA)
#   sbatch --job-name=me_qwen8b_v0 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
# Fine-tuned v1 (LoRA)
#   sbatch --job-name=me_qwen8b_v1 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" models.ft_model.is_lora=true
# Fine-tuned v2.2 (LoRA)
#   sbatch --job-name=me_qwen8b_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true
# Fine-tuned v2.3 (LoRA)
#   sbatch --job-name=me_qwen8b_v23 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.3" models.ft_model.is_lora=true
#
# --- Qwen3-14B ---
# Base model (no-think)
#   sbatch --job-name=me_qwen14b scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config
# Base model (think)
#   sbatch --job-name=me_qwen14b_t scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_t_config
# Fine-tuned v0 (LoRA)
#   sbatch --job-name=me_qwen14b_v0 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
# Fine-tuned v1 (LoRA)
#   sbatch --job-name=me_qwen14b_v1 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" models.ft_model.is_lora=true
# Fine-tuned v2.2 (LoRA)
#   sbatch --job-name=me_qwen14b_v22 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true
#
# --- OpenAI / external baselines ---
# GPT-5.2
#   sbatch --job-name=me_gpt5 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=gpt_5.2_config
# GPT OSS 20B
#   sbatch --job-name=me_oss20 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=gpt_oss_20b_config
# GPT OSS 120B
#   sbatch --job-name=me_oss120 scripts/eval/run_multi_eval.sbatch multi_run.num_runs=5 dataset.limit=-1 models=gpt_oss_120b_config
# Qwen3-80B
#   sbatch --job-name=me_qwen80b scripts/eval/run_multi_eval.sbatch multi_run.num_runs=3 dataset.limit=-1 models=qwen3_80b_nt_config

echo "Starting multi-run evaluation..."
set -e

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Multi-Run Evaluation (run_multi_eval)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Args: $@"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  GPUs: $SLURM_GPUS"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

uv run -m wonda.eval.run_multi_eval "$@"

echo ""
echo "=========================================="
echo "Multi-run evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

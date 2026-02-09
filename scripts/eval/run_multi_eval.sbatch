#!/bin/bash
#SBATCH --job-name=multi-eval
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:1
#
# Run multiple independent eval runs (fresh subprocess per run) and aggregate stats.
#
# Usage:
#   sbatch scripts/eval/run_multi_eval.sbatch --num_runs=3 [HYDRA_OVERRIDES]
#
# Examples:

# Qwen3-0.6B
#   sbatch --job-name=me_qwen06_t scripts/eval/run_multi_eval.sbatch --num_runs=1 dataset.limit=1 models=qwen3_0.6b_t_config
#   sbatch --job-name=me_qwen06 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config
#   sbatch --job-name=me_qwen06_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" 
#   sbatch --job-name=me_qwen06_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"
#   sbatch --job-name=me_qwen06_v22 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"  
#   sbatch --job-name=me_qwen06_v23 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.3"  

# Qwen3-1.7B
# sbatch --job-name=me_qwen3_1.7b scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_1.7_nt_config
# sbatch --job-name=me_qwen3_1.7b_v22 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_1.7b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"  

# Qwen3-4B
# sbatch --job-name=me_qwen3_4b scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config
# sbatch --job-name=me_qwen3_4b_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0"  
# sbatch --job-name=me_qwen3_4b_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1"  
# sbatch --job-name=me_qwen3_4b_v22 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_4b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2"  



# Qwen3-8B
#   sbatch --job-name=me_qwen8b_t  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_t_config
#   sbatch --job-name=me_qwen8b  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config
#   sbatch --job-name=me_qwen8b_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
#   sbatch --job-name=me_qwen8b_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" models.ft_model.is_lora=true 
#   sbatch --job-name=me_qwen8b_v22 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true
#   sbatch --job-name=me_qwen8b_v23 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.3" models.ft_model.is_lora=true 

# Qwen3-14B
#   sbatch --job-name=me_qwen3_14b_t  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_14b_t_config
#   sbatch --job-name=me_qwen3_14b  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config
#   sbatch --job-name=me_qwen3_14b_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
#   sbatch --job-name=me_qwen3_14_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" models.ft_model.is_lora=true 
#   sbatch --job-name=me_qwen3_14_v22 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_14b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v2.2" models.ft_model.is_lora=true

# OpenAI models - baseline
#   sbatch --job-name=me_gpt5   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_5.2_config
#   sbatch --job-name=me_oss20   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_oss_20b_config
#   sbatch --job-name=me_oss120   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config
#   sbatch --job-name=qwen3_80b  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_80b_nt_config

# Notes:
# - You can control Weave concurrency with WEAVE_PARALLELISM (default is set inside run_multi_eval/evaluate).
# - CUDA_VISIBLE_DEVICES is set by SLURM on GPU nodes; no need to set it manually.

echo "Starting multi-run evaluation..."
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

# Run multi-eval; remaining args are forwarded as Hydra overrides to evaluate
uv run -m wonda.eval.run_multi_eval "$@"

echo ""
echo "=========================================="
echo "Multi-run evaluation completed!"
echo "End time: $(date)"
echo "=========================================="


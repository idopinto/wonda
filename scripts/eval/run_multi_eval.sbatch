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
#   sbatch --job-name=me_qwen06 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config
#   sbatch --job-name=me_qwen06_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" 
#   sbatch --job-name=me_qwen06_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_0.6b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" 
#   sbatch --job-name=me_qwen8b  scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=10 models=qwen3_8b_nt_config
#   sbatch --job-name=me_qwen8b_v0 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v0" models.ft_model.is_lora=true
#   sbatch --job-name=me_qwen8b_v1 scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=qwen3_8b_nt_config models.eval_ft_model=true models.ft_model.sft_version="v1" models.ft_model.is_lora=true 
#   sbatch --job-name=me_gpt5   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_5.2_config
# sbatch --job-name=me_oss20   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_oss_20b_config
# sbatch --job-name=me_oss120   scripts/eval/run_multi_eval.sbatch --num_runs=3 dataset.limit=-1 models=gpt_oss_120b_config
#
# Notes:
# - You can control Weave concurrency with WEAVE_PARALLELISM (default is set inside run_multi_eval/evaluate2).
# - CUDA_VISIBLE_DEVICES is set by SLURM on GPU nodes; no need to set it manually.

echo "Starting multi-run evaluation..."
set -e

# Set working directory
cd /cs/labs/guykatz/idopinto12/projects/inv-gen

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

# Run multi-eval; remaining args are forwarded as Hydra overrides to evaluate2
uv run -m src.eval.run_multi_eval "$@"

echo ""
echo "=========================================="
echo "Multi-run evaluation completed!"
echo "End time: $(date)"
echo "=========================================="


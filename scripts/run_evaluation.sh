#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm/eval_%j.out
#SBATCH --error=slurm/eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:l40s:2

# Script to run evaluation using the unified evaluator
# Usage: sbatch scripts/run_evaluation.sh [OPTIONS]
# This script works best on single h200 gpu (for gpt-oss evaluation), if it is not available consider 2 l40s to avoid OOM.
# Available arguments:
#   --split SPLIT                    Data split: "easy" or "hard" (default: "hard")
#   --limit N                        Limit number of tasks, -1 for all (default: -1)
#   --prefix PREFIX                  Prefix for dataset files (default: empty)
#   --skip_wandb                     Skip uploading plot to W&B
#   --eval-finetuned-model           Evaluate SFT model instead of base model
# For more configuration changes, modify: src/eval/config.py
# Examples:
# ======
#   sbatch scripts/run_evaluation.sh --split easy --limit 1 (for testing)
# ======
#   sbatch scripts/run_evaluation.sh --split easy
#   sbatch scripts/run_evaluation.sh --split hard 
# ======
#   sbatch scripts/run_evaluation.sh --split easy --eval-sft 
#   sbatch scripts/run_weave_eval.sh --split hard --eval-sft 

echo "Starting evaluation..."
set -e

# Set working directory (change to your project root)
cd /cs/labs/guykatz/idopinto12/projects/inv-gen 


# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Print job info
echo "=========================================="
echo "Starting Weave Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  GPUs: $SLURM_GPUS"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# Run evaluation
uv run -m src.eval.evaluate "$@"

echo ""
echo "=========================================="
echo "Weave Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="

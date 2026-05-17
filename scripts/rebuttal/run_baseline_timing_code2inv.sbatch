#!/bin/bash
#SBATCH --job-name=code2inv-baseline
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Measure UAutomizer baseline verification times on all Code2Inv programs.
#
# Usage:
#   sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch
#   sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --uautomizer-version 23
#   sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --uautomizer-version 25 --num-trials 3
#   sbatch scripts/rebuttal/run_baseline_timing_code2inv.sbatch --timeout 900

set -e

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Code2Inv Baseline Timing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Args: $@"
echo "=========================================="
echo ""

uv run python scripts/rebuttal/baseline_timing_code2inv.py "$@"

echo ""
echo "=========================================="
echo "Baseline timing completed!"
echo "End time: $(date)"
echo "=========================================="

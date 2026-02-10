#!/bin/bash
#SBATCH --job-name=wonda-pipeline
#SBATCH --output=slurm/wonda_pipeline_%j.out
#SBATCH --error=slurm/wonda_pipeline_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Run the full Wonda data pipeline (wonda_pipeline.py).
# Usage: sbatch scripts/preprocess/wonda_pipeline.sbatch [HYDRA_OVERRIDES]
#
# Pipeline: Stage 1 (normalize) + Stage 2 (simplify + verify). Uses config wonda_data_pipeline.
#
# Hydra overrides (pass as arguments):
#   pipe.limit=N                          Number of entries to process, -1 for all (default: -1)
#   pipe.normalize=true|false             Run normalization (default: true)
#   pipe.simplify=true|false              Run simplification (default: true)
#   pipe.verify=true|false                Run verification (default: true)
#   pipe.output_suffix=SUFFIX             Output dataset suffix (default: full-v2)
#   pipe.push_to_hub=true|false           Push result to Hugging Face (default: true)
#   model.n=N                             Candidates per invariant (default: 4)
#   model.name=MODEL                      LLM for simplification (default: moonshotai/Kimi-K2-Thinking)
#   verifier.timeout=N                    Timeout per verification in seconds (default: 600)
#   weave.use_weave=true|false            Enable Weave logging (default: true)
#   test_mode=true|false                  Limit data and add -test suffix (default: false)
#
# Examples:
#   sbatch scripts/preprocess/wonda_pipeline.sbatch pipe.limit=10
#   sbatch scripts/preprocess/wonda_pipeline.sbatch pipe.limit=-1
#   sbatch scripts/preprocess/wonda_pipeline.sbatch pipe.limit=5 test_mode=true

echo "Starting Wonda data pipeline..."
set -e

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo "Wonda Data Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Hydra overrides: $@"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

uv run -m wonda.preprocess.wonda_pipeline "$@"

echo ""
echo "=========================================="
echo "Wonda data pipeline completed!"
echo "End time: $(date)"
echo "=========================================="

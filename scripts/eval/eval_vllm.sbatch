#!/bin/bash
#SBATCH --job-name=eval-vllm
#SBATCH --output=slurm/eval_vllm_%j.out
#SBATCH --error=slurm/eval_vllm_%j.err
#SBATCH --gres=gpu:h200:1
#SBATCH --killable
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Script to run evaluation using vLLM server (OpenAI-compatible API)
# Usage: sbatch scripts/eval/eval_vllm.sbatch [HYDRA_OVERRIDES]
#
# Hydra config overrides (pass as arguments):
#   dataset.split=easy|hard              Data split (default: hard)
#   dataset.limit=N                      Limit number of tasks, -1 for all (default: -1)
#   dataset.prefix=PREFIX                Prefix for dataset files
#   weave.skip_wandb=true                Skip uploading plot to W&B
#
# Examples:
#   sbatch scripts/eval/eval_vllm.sbatch dataset.split=easy dataset.limit=1
#   sbatch scripts/eval/eval_vllm.sbatch dataset.split=hard

# Exit on error
set -e

# ============================================
# Configuration - modify these as needed
# ============================================
BASE_MODEL="openai/gpt-oss-20b"
LORA_ADAPTER="idopinto/gpt-oss-20b-rlinv-sft-sep"
ADAPTER_NAME="gen_inv_adapter"
MAX_LORA_RANK=8
VLLM_PORT=8000

# ============================================
# Setup
# ============================================
echo "=========================================="
echo "Starting Evaluation (vLLM Client)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Hydra overrides: $@"
echo "=========================================="

# Project root - defaults to the directory from which sbatch was submitted
# Override by setting PROJECT_ROOT before calling sbatch
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

# Activate virtual environment
source .venv/bin/activate

# ============================================
# Start vLLM server in background
# ============================================
echo "Starting vLLM server..."

vllm serve ${BASE_MODEL} \
    --enable-lora \
    --lora-modules ${ADAPTER_NAME}=${LORA_ADAPTER} \
    --max-lora-rank ${MAX_LORA_RANK} \
    --port ${VLLM_PORT} \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    &

VLLM_PID=$!
echo "vLLM server started with PID: ${VLLM_PID}"

# ============================================
# Wait for server to be ready
# ============================================
echo "Waiting for vLLM server to be ready..."

MAX_RETRIES=60  # Wait up to 5 minutes (60 * 5 seconds)
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting for server... (attempt $((RETRY_COUNT + 1))/${MAX_RETRIES})"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: vLLM server failed to start within timeout"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Verify models are loaded
echo "Available models:"
curl -s http://localhost:${VLLM_PORT}/v1/models | python -m json.tool

# ============================================
# Run evaluation with Hydra overrides
# ============================================
echo "Starting evaluation at $(date)"

# Pass model.client=vllm and vLLM config, plus any additional Hydra overrides from args
uv run -m src.eval.evaluate \
    model.client=vllm \
    model.vllm_base_url=http://localhost:${VLLM_PORT}/v1 \
    model.vllm_model=${ADAPTER_NAME} \
    "$@"

EVAL_EXIT_CODE=$?

# ============================================
# Cleanup
# ============================================
echo "Evaluation completed with exit code: ${EVAL_EXIT_CODE}"
echo "Shutting down vLLM server..."

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "End time: $(date)"
echo "=========================================="
exit $EVAL_EXIT_CODE

# sbatch scripts/eval/eval_vllm.sbatch dataset.split=hard dataset.limit -1
# sbatch scripts/eval/eval_vllm.sbatch dataset.split=hard dataset.limit -1 eval_finetuned_model=true
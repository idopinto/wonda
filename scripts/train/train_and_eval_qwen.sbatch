#!/bin/bash
#SBATCH --job-name=train-eval-qwen
#SBATCH --output=slurm/train_eval_qwen_%j.out
#SBATCH --error=slurm/train_eval_qwen_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --killable

# Script to train Qwen model and then evaluate it
# Usage: sbatch scripts/train/train_and_eval_qwen.sbatch [TRAIN_ARGS] [--] [EVAL_ARGS]
#
# The script will:
#   1. Train the model using train_qwen.py
#   2. Extract the output directory from training
#   3. Evaluate the trained model using evaluate.py
#
# Training args (before --): --config-name=qwen3_0.6b|qwen3_1.7b|qwen3_4b|qwen3_8b|qwen3_14b, dataset.version=v0|v1|v2, dataset.min_grade=2 (for v2), etc.
# Evaluation args (after --): models=qwen_no_think_config|..., dataset.limit=N, dataset.split=easy|hard, etc.
#
# Explicit commands (model size × version; v2 uses min_grade=2):
#   0.6B
#   sbatch --job-name=train_eval_0.6b_v0 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_0.6b dataset.version=v0
#   sbatch --job-name=train_eval_0.6b_v1 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_0.6b dataset.version=v1
#   sbatch --job-name=train_eval_0.6b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_0.6b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_eval_0.6b_v2_test scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_0.6b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   1.7B
#   sbatch --job-name=train_eval_1.7b_v0 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_1.7b dataset.version=v0
#   sbatch --job-name=train_eval_1.7b_v1 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_1.7b dataset.version=v1
#   sbatch --job-name=train_eval_1.7b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_1.7b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_eval_1.7b_v2_test scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_1.7b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   4B
#   sbatch --job-name=train_eval_4b_v0 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_4b dataset.version=v0
#   sbatch --job-name=train_eval_4b_v1 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_4b dataset.version=v1
#   sbatch --job-name=train_eval_4b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_4b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_eval_4b_v2_test scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_4b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   8B
#   sbatch --job-name=train_eval_8b_v0 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_8b dataset.version=v0
#   sbatch --job-name=train_eval_8b_v1 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_8b dataset.version=v1
#   sbatch --job-name=train_eval_8b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_8b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_eval_8b_v2_test scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_8b dataset.version=v2 dataset.min_grade=2 test_mode=true
#   14B
#   sbatch --job-name=train_eval_14b_v0 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_14b dataset.version=v0
#   sbatch --job-name=train_eval_14b_v1 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_14b dataset.version=v1
#   sbatch --job-name=train_eval_14b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_14b dataset.version=v2 dataset.min_grade=2
#   sbatch --job-name=train_eval_14b_v2_test scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_14b dataset.version=v2 dataset.min_grade=2 test_mode=true
# With evaluation overrides (e.g. limit eval samples):
#   sbatch --job-name=train_eval_0.6b_v2 scripts/train/train_and_eval_qwen.sbatch --config-name=qwen3_0.6b dataset.version=v2 dataset.min_grade=2 -- dataset.limit=10

echo "Starting Qwen training and evaluation pipeline..."
set -e

# Project root - defaults to the directory from which sbatch was submitted
# Override by setting PROJECT_ROOT before calling sbatch
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parse arguments - split training and evaluation overrides
TRAIN_ARGS=()
EVAL_ARGS=()
FOUND_SEPARATOR=false

for arg in "$@"; do
    if [ "$arg" == "--" ]; then
        FOUND_SEPARATOR=true
    elif [ "$FOUND_SEPARATOR" == true ]; then
        EVAL_ARGS+=("$arg")
    else
        TRAIN_ARGS+=("$arg")
    fi
done

# Print job info
echo "=========================================="
echo "Qwen Training + Evaluation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Training args: ${TRAIN_ARGS[@]}"
echo "Evaluation args: ${EVAL_ARGS[@]}"
echo "SLURM resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo "  GPUs: $SLURM_GPUS"
echo "  Time limit: $SLURM_TIMELIMIT"
echo "=========================================="
echo ""

# ============================================
# STEP 1: TRAINING
# ============================================
echo "=========================================="
echo "STEP 1: Training Model"
echo "=========================================="
echo "Starting training at $(date)"

# Run training and capture output to extract model path
TRAIN_OUTPUT=$(mktemp)
if uv run -m wonda.train.train_qwen "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_OUTPUT"; then
    echo "Training completed successfully at $(date)"
else
    echo "ERROR: Training failed!"
    rm -f "$TRAIN_OUTPUT"
    exit 1
fi

# Extract output directory from training output (train_qwen prints model_name and output_dir after derivation)
CONFIG_NAME=$(echo "${TRAIN_ARGS[@]}" | grep -oP '--config-name=\K[^\s]+' || echo "qwen3_0.6b")
VERSION=$(echo "${TRAIN_ARGS[@]}" | grep -oP 'dataset\.version=\K[^\s]+' || echo "v0")
MODEL_OUTPUT_DIR=$(grep -oP 'trained_models/qwen3-[^[:space:]]+-nt-gen-inv-sft-[^[:space:]]+' "$TRAIN_OUTPUT" | tail -1 || true)

# If not in log, try config file (unified configs have output_dir: null, so this often fails)
if [ -z "$MODEL_OUTPUT_DIR" ]; then
    CONFIG_FILE="configs/train/models/${CONFIG_NAME}.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        MODEL_OUTPUT_DIR=$(grep "^  output_dir:" "$CONFIG_FILE" | sed 's/.*output_dir:\s*["\s]*\([^"]*\)["\s]*/\1/')
    fi
fi

# Fallback: derive from config name + version (matches train_qwen derive_run_names)
if [ -z "$MODEL_OUTPUT_DIR" ] || [ "$MODEL_OUTPUT_DIR" = "null" ]; then
    SIZE_SLUG="${CONFIG_NAME#qwen3_}"
    MODEL_OUTPUT_DIR="trained_models/qwen3-${SIZE_SLUG}-nt-gen-inv-sft-${VERSION}"
    echo "WARNING: Could not extract output_dir from log/config, using: $MODEL_OUTPUT_DIR"
fi

echo "Model output directory: $MODEL_OUTPUT_DIR"

# Check if model was saved (look for pytorch_model.bin or adapter files)
if [ ! -d "$MODEL_OUTPUT_DIR" ]; then
    echo "ERROR: Model output directory not found: $MODEL_OUTPUT_DIR"
    rm -f "$TRAIN_OUTPUT"
    exit 1
fi

rm -f "$TRAIN_OUTPUT"

echo ""
echo "=========================================="
echo "STEP 2: Evaluating Trained Model"
echo "=========================================="
echo "Starting evaluation at $(date)"
echo "Using model from: $MODEL_OUTPUT_DIR"

# Determine evaluation model config (default to qwen_no_think_config)
EVAL_MODEL_CONFIG=$(echo "${EVAL_ARGS[@]}" | grep -oP 'models=\K[^\s]+' || echo "qwen_no_think_config")

# Run evaluation with the trained model
# Set eval_ft_model=true and update ft_model.id to point to the trained model
EVAL_CMD="uv run -m wonda.eval.evaluate models=${EVAL_MODEL_CONFIG} models.eval_ft_model=true models.ft_model.id=${MODEL_OUTPUT_DIR}"

# Add other evaluation args
if [ ${#EVAL_ARGS[@]} -gt 0 ]; then
    # Remove models= from EVAL_ARGS since we're setting it explicitly
    FILTERED_EVAL_ARGS=()
    for arg in "${EVAL_ARGS[@]}"; do
        if [[ ! "$arg" =~ ^models= ]]; then
            FILTERED_EVAL_ARGS+=("$arg")
        fi
    done
    EVAL_CMD="$EVAL_CMD ${FILTERED_EVAL_ARGS[@]}"
fi

echo "Evaluation command: $EVAL_CMD"
echo ""

if eval "$EVAL_CMD"; then
    echo "Evaluation completed successfully at $(date)"
else
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "End time: $(date)"
echo "=========================================="

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
# Usage: sbatch scripts/train/train_and_eval_qwen.sbatch [TRAIN_CONFIG_OVERRIDES] [EVAL_CONFIG_OVERRIDES]
#
# The script will:
#   1. Train the model using train_qwen.py
#   2. Extract the output directory from training
#   3. Evaluate the trained model using evaluate.py
#
# Hydra config overrides for training (pass before --):
#   --config-name=qwen_0_6b_v0|qwen_0_6b_v1|qwen_1_7b_v0|qwen_1_7b_v1
#   dataset.limit=N                      Limit number of training samples
#   sft.num_train_epochs=N               Number of training epochs
#
# Hydra config overrides for evaluation (pass after --):
#   models=qwen_no_think_config|qwen_config
#   dataset.limit=N                      Limit number of evaluation samples
#   dataset.split=easy|hard               Evaluation split
#
# Examples:
#   sbatch scripts/train/train_and_eval_qwen.sbatch
#   sbatch scripts/train/train_and_eval_qwen.sbatch --config-name=qwen_0_6b_v1 --models=qwen_no_think_config dataset.limit=10

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

# Extract output directory from training config or output
# Try to get it from the config file or from training output
MODEL_OUTPUT_DIR=$(grep -oP 'output_dir["\s:]+["\s]+[^"]*["\s]+' "$TRAIN_OUTPUT" | head -1 | sed 's/.*["\s]\([^"]*\)["\s].*/\1/' || echo "")

# If not found in output, try to extract from Hydra config
if [ -z "$MODEL_OUTPUT_DIR" ]; then
    # Get config name from args or use default
    CONFIG_NAME=$(echo "${TRAIN_ARGS[@]}" | grep -oP '--config-name=\K[^\s]+' || echo "qwen_0_6b_v0")
    CONFIG_FILE="configs/train/qwen/${CONFIG_NAME}.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        MODEL_OUTPUT_DIR=$(grep -A 1 "^sft:" "$CONFIG_FILE" | grep "output_dir:" | sed 's/.*output_dir:\s*["\s]*\([^"]*\)["\s]*/\1/')
    fi
fi

# If still not found, construct default path
if [ -z "$MODEL_OUTPUT_DIR" ]; then
    CONFIG_NAME=$(echo "${TRAIN_ARGS[@]}" | grep -oP '--config-name=\K[^\s]+' || echo "qwen_0_6b_v0")
    MODEL_OUTPUT_DIR="trained_models/qwen-3-0.6b-gen-inv-sft-v0"
    echo "WARNING: Could not extract output_dir, using default: $MODEL_OUTPUT_DIR"
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

#!/bin/bash
set -euo pipefail

# EUOS25 Multi-task Pipeline
# This script runs the complete multi-task training pipeline

# Default values
FORCE=false
CONF=""
PROCESSED_DIR=data/processed
MODEL_DIR=data/models/multitask
PRED_DIR=data/preds/multitask
SUBMISSION_DIR=data/submissions/multitask
RAW_DIR=data/raw
TASKS_SPEC=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      FORCE=true
      shift
      ;;
    --config)
      CONF="$2"
      shift 2
      ;;
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --pred-dir)
      PRED_DIR="$2"
      shift 2
      ;;
    --submission-dir)
      SUBMISSION_DIR="$2"
      shift 2
      ;;
    --data-dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --tasks)
      TASKS_SPEC="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --force              Force regeneration of all outputs (default: false)"
      echo "  --config PATH        Configuration file path (required)"
      echo "  --processed-dir DIR  Processed data directory (default: data/processed)"
      echo "  --model-dir DIR      Model output directory (default: data/models/multitask)"
      echo "  --pred-dir DIR       Prediction output directory (default: data/preds/multitask)"
      echo "  --submission-dir DIR Submission output directory (default: data/submissions/multitask)"
      echo "  --data-dir DIR       Raw data directory (default: data/raw)"
      echo "  --tasks TASKS        Space-separated list of tasks (if not in config)"
      echo "                       Available tasks: transmittance340, transmittance450,"
      echo "                       fluorescence340_450, fluorescence480"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$CONF" ]; then
  echo "Error: --config is required"
  echo "Use --help for usage information"
  exit 1
fi

# Extract tasks from config if not provided
if [ -z "$TASKS_SPEC" ]; then
  # Parse tasks from YAML config file
  TASKS_SPEC=$(python3 -c "
import yaml
import sys
with open('$CONF', 'r') as f:
    config = yaml.safe_load(f)
    tasks = config.get('tasks', [])
    if not tasks:
        print('Error: No tasks specified in config', file=sys.stderr)
        sys.exit(1)
    print(' '.join(tasks))
")
  if [ $? -ne 0 ]; then
    echo "Error: Could not extract tasks from config file"
    exit 1
  fi
fi

# Convert space-separated tasks to array
read -ra TASKS <<< "$TASKS_SPEC"

echo "==================================="
echo "EUOS25 Multi-task Pipeline"
if [ "$FORCE" = true ]; then
  echo "Mode: FORCE (will regenerate all outputs)"
else
  echo "Mode: REUSE (will skip existing outputs)"
fi
echo "Config: $CONF"
echo "Tasks: ${TASKS[*]}"
echo "==================================="

# Task identifier for file naming
TASK_ID=$(echo "${TASKS[*]}" | tr ' ' '_')

# File paths
PREPARED_FILES=()
FEATURE_FILES=()
SPLIT_FILES=()

for task in "${TASKS[@]}"; do
  PREPARED_FILES+=("$PROCESSED_DIR/train_${task}_prepared.csv")
  FEATURE_FILES+=("$PROCESSED_DIR/features_train_${task}.parquet")
  SPLIT_FILES+=("$PROCESSED_DIR/splits_${task}.json")
done

# Common features file for multi-task (uses first task's features)
FEATURES_COMMON="${FEATURE_FILES[0]}"

echo ""
echo "Step 1: Preparing training data for each task..."
PREPROCESS_NEEDED=false
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  PREPARED_FILE="${PREPARED_FILES[$i]}"

  if [ "$FORCE" = true ] || [ ! -f "$PREPARED_FILE" ]; then
    PREPROCESS_NEEDED=true
    break
  fi
done

if [ "$PREPROCESS_NEEDED" = true ]; then
  python scripts/preprocess_multitask.py \
    --tasks "${TASKS[@]}" \
    --input-dir "$RAW_DIR" \
    --output-dir "$PROCESSED_DIR"
else
  echo "  Skipping: All prepared files already exist"
fi

echo ""
echo "Step 2: Creating scaffold splits..."
SPLITS_NEEDED=false
for split_file in "${SPLIT_FILES[@]}"; do
  if [ "$FORCE" = true ] || [ ! -f "$split_file" ]; then
    SPLITS_NEEDED=true
    break
  fi
done

if [ "$SPLITS_NEEDED" = true ]; then
  python scripts/split_multitask.py \
    --tasks "${TASKS[@]}" \
    --input-dir "$PROCESSED_DIR" \
    --output-dir "$PROCESSED_DIR" \
    --folds 5 \
    --seed 42 \
    --scaffold-min-size 10
else
  echo "  Skipping: All split files already exist"
fi

echo ""
echo "Step 3: Building features..."
FEATURES_NEEDED=false
for feature_file in "${FEATURE_FILES[@]}"; do
  if [ "$FORCE" = true ] || [ ! -f "$feature_file" ]; then
    FEATURES_NEEDED=true
    break
  fi
done

if [ "$FEATURES_NEEDED" = true ]; then
  for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PREPARED_FILE="${PREPARED_FILES[$i]}"
    FEATURE_FILE="${FEATURE_FILES[$i]}"

    if [ "$FORCE" = true ] || [ ! -f "$FEATURE_FILE" ]; then
      echo "  Building features for $TASK..."
      uv run -m euos25.cli build-features \
        --input "$PREPARED_FILE" \
        --output "$FEATURE_FILE" \
        --config "$CONF"
    else
      echo "  Skipping $TASK: features already exist"
    fi
  done
else
  echo "  Skipping: All feature files already exist"
fi

echo ""
echo "Step 4: Training multi-task models..."

# Check if model exists
MODEL_EXISTS=false
if [ -d "$MODEL_DIR/$TASK_ID" ]; then
  # Check for checkpoint or model files
  if [ -n "$(find "$MODEL_DIR/$TASK_ID" -name '*.ckpt' -o -name 'model.txt' 2>/dev/null)" ]; then
    MODEL_EXISTS=true
  fi
fi

if [ "$FORCE" = true ] || [ "$MODEL_EXISTS" = false ]; then
  python scripts/train_multitask.py "$CONF" \
    --features "$FEATURES_COMMON" \
    --splits "${SPLIT_FILES[0]}" \
    --output "$MODEL_DIR" \
    --data-dir "$RAW_DIR"
else
  echo "  Skipping: Models already exist in $MODEL_DIR/$TASK_ID"
fi

echo ""
echo "Step 5: Preparing test data..."
PREPARED_TEST="$PROCESSED_DIR/test_prepared.csv"
RAW_TEST="$RAW_DIR/euos25_challenge_test.csv"

if [ "$FORCE" = true ] || [ ! -f "$PREPARED_TEST" ]; then
  uv run -m euos25.cli prepare \
    --input "$RAW_TEST" \
    --output "$PREPARED_TEST" \
    --normalize --deduplicate
else
  echo "  Skipping: $PREPARED_TEST already exists"
fi

echo ""
echo "Step 6: Building test features..."
FEATURES_TEST="$PROCESSED_DIR/features_test_multitask.parquet"

if [ "$FORCE" = true ] || [ ! -f "$FEATURES_TEST" ]; then
  uv run -m euos25.cli build-features \
    --input "$PREPARED_TEST" \
    --output "$FEATURES_TEST" \
    --config "$CONF"
else
  echo "  Skipping: $FEATURES_TEST already exists"
fi

echo ""
echo "Step 7: Generating OOF predictions..."
OOF_OUTPUT="$PRED_DIR/oof"
mkdir -p "$OOF_OUTPUT"

OOF_NEEDED=false
for task in "${TASKS[@]}"; do
  if [ "$FORCE" = true ] || [ ! -f "$OOF_OUTPUT/${task}_oof.csv" ]; then
    OOF_NEEDED=true
    break
  fi
done

if [ "$OOF_NEEDED" = true ]; then
  python scripts/infer_multitask.py "$CONF" \
    --features "$FEATURES_COMMON" \
    --splits "${SPLIT_FILES[0]}" \
    --model-dir "$MODEL_DIR" \
    --output "$OOF_OUTPUT" \
    --mode oof
else
  echo "  Skipping: All OOF predictions already exist"
fi

echo ""
echo "Step 8: Generating test predictions..."
TEST_OUTPUT="$PRED_DIR/test"
mkdir -p "$TEST_OUTPUT"

TEST_NEEDED=false
for task in "${TASKS[@]}"; do
  if [ "$FORCE" = true ] || [ ! -f "$TEST_OUTPUT/${task}_test.csv" ]; then
    TEST_NEEDED=true
    break
  fi
done

if [ "$TEST_NEEDED" = true ]; then
  python scripts/infer_multitask.py "$CONF" \
    --features "$FEATURES_TEST" \
    --model-dir "$MODEL_DIR" \
    --output "$TEST_OUTPUT" \
    --mode test \
    --n-folds 5
else
  echo "  Skipping: All test predictions already exist"
fi

echo ""
echo "Step 9: Creating submission files..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMISSION_OUTPUT="$SUBMISSION_DIR/$TIMESTAMP"
mkdir -p "$SUBMISSION_OUTPUT"

python scripts/submit_multitask.py \
  --pred-dir "$TEST_OUTPUT" \
  --output-dir "$SUBMISSION_OUTPUT" \
  --tasks "${TASKS[@]}" \
  --mode test

echo ""
echo "==================================="
echo "Multi-task pipeline completed successfully!"
echo "==================================="
echo "Models saved to: $MODEL_DIR/$TASK_ID"
echo "OOF predictions: $OOF_OUTPUT"
echo "Test predictions: $TEST_OUTPUT"
echo "Submissions saved to: $SUBMISSION_OUTPUT"
echo "Tasks: ${TASKS[*]}"
echo ""

#!/bin/bash
set -euo pipefail

# EUOS25 Full-scale Pipeline
# This script runs the complete pipeline with all 4 tasks (fluorescence340_450, fluorescence480, transmittance340, transmittance450)

# Default values
FORCE=false
CONF=configs/full.yaml
PROCESSED_DIR=data/processed
MODEL_DIR=data/models/full
PRED_DIR=data/preds/full
SUBMISSION_DIR=data/submissions
TASKS_SPEC="all"

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
    --tasks)
      TASKS_SPEC="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --force              Force regeneration of all outputs (default: false)"
      echo "  --config PATH        Configuration file path (default: configs/full.yaml)"
      echo "  --processed-dir DIR  Processed data directory (default: data/processed)"
      echo "  --model-dir DIR      Model output directory (default: data/models/full)"
      echo "  --pred-dir DIR       Prediction output directory (default: data/preds/full)"
      echo "  --submission-dir DIR Submission output directory (default: data/submissions)"
      echo "  --tasks TASKS        Comma-separated list of tasks or 'all' (default: all)"
      echo "                       Available tasks: fluo_340_450, fluo_480, trans_340, trans_450"
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

# Output directories (now using variables set above or defaults)

RAW_TEST=data/raw/euos25_challenge_test.csv
# Training data files
RAW_TRAIN_FLUO_340_450=data/raw/euos25_challenge_train_fluorescence340_450_extended.csv
RAW_TRAIN_FLUO_480=data/raw/euos25_challenge_train_fluorescence480_extended.csv
RAW_TRAIN_TRANS_340=data/raw/euos25_challenge_train_transmittance340_extended.csv
RAW_TRAIN_TRANS_450=data/raw/euos25_challenge_train_transmittance450_extended.csv

# Prepared data files
PREPARED_TEST=$PROCESSED_DIR/test_prepared.csv
PREPARED_TRAIN_FLUO_340_450=$PROCESSED_DIR/train_fluo_340_450_prepared.csv
PREPARED_TRAIN_FLUO_480=$PROCESSED_DIR/train_fluo_480_prepared.csv
PREPARED_TRAIN_TRANS_340=$PROCESSED_DIR/train_trans_340_prepared.csv
PREPARED_TRAIN_TRANS_450=$PROCESSED_DIR/train_trans_450_prepared.csv

# Feature files
FEATURES_TEST=$PROCESSED_DIR/features_test_full.parquet
FEATURES_TRAIN_FLUO_340_450=$PROCESSED_DIR/features_train_fluo_340_450.parquet
FEATURES_TRAIN_FLUO_480=$PROCESSED_DIR/features_train_fluo_480.parquet
FEATURES_TRAIN_TRANS_340=$PROCESSED_DIR/features_train_trans_340.parquet
FEATURES_TRAIN_TRANS_450=$PROCESSED_DIR/features_train_trans_450.parquet

# Split files
SPLITS_FLUO_340_450=$PROCESSED_DIR/splits_fluo_340_450.json
SPLITS_FLUO_480=$PROCESSED_DIR/splits_fluo_480.json
SPLITS_TRANS_340=$PROCESSED_DIR/splits_trans_340.json
SPLITS_TRANS_450=$PROCESSED_DIR/splits_trans_450.json

# Task mapping: task_name -> (task_name_in_code, label_col, train_file, prepared_file, feature_file, split_file)
# Using regular arrays instead of associative arrays for bash 3.2 compatibility
TASK_KEYS=("fluo_340_450" "fluo_480" "trans_340" "trans_450")
TASK_VALUES=(
  "y_fluo_any|Fluorescence|$RAW_TRAIN_FLUO_340_450|$PREPARED_TRAIN_FLUO_340_450|$FEATURES_TRAIN_FLUO_340_450|$SPLITS_FLUO_340_450"
  "y_fluo_any|Fluorescence|$RAW_TRAIN_FLUO_480|$PREPARED_TRAIN_FLUO_480|$FEATURES_TRAIN_FLUO_480|$SPLITS_FLUO_480"
  "y_trans_any|Transmittance|$RAW_TRAIN_TRANS_340|$PREPARED_TRAIN_TRANS_340|$FEATURES_TRAIN_TRANS_340|$SPLITS_TRANS_340"
  "y_trans_any|Transmittance|$RAW_TRAIN_TRANS_450|$PREPARED_TRAIN_TRANS_450|$FEATURES_TRAIN_TRANS_450|$SPLITS_TRANS_450"
)

# Helper function to get task info by key
get_task_info() {
  local key="$1"
  local i
  for i in "${!TASK_KEYS[@]}"; do
    if [ "${TASK_KEYS[$i]}" = "$key" ]; then
      echo "${TASK_VALUES[$i]}"
      return 0
    fi
  done
  return 1
}

# Build task arrays based on TASKS_SPEC
if [ "$TASKS_SPEC" = "all" ]; then
  TASKS=("fluo_340_450" "fluo_480" "trans_340" "trans_450")
else
  # Parse comma-separated task list
  OLD_IFS=$IFS
  IFS=',' read -ra TASKS <<< "$TASKS_SPEC"
  IFS=$OLD_IFS
  # Validate tasks
  for task in "${TASKS[@]}"; do
    task=$(echo "$task" | xargs)  # trim whitespace
    if ! get_task_info "$task" > /dev/null 2>&1; then
      echo "Error: Unknown task '$task'"
      echo "Available tasks: fluo_340_450, fluo_480, trans_340, trans_450"
      exit 1
    fi
  done
fi

# Build parallel arrays for selected tasks
TASK_NAMES=()
TRAIN_FILES=()
PREPARED_FILES=()
FEATURE_FILES=()
SPLIT_FILES=()
LABEL_COLS=()

for task in "${TASKS[@]}"; do
  task=$(echo "$task" | xargs)  # trim whitespace
  task_info=$(get_task_info "$task")
  OLD_IFS=$IFS
  IFS='|' read -ra INFO <<< "$task_info"
  IFS=$OLD_IFS
  TASK_NAMES+=("${INFO[0]}")
  LABEL_COLS+=("${INFO[1]}")
  TRAIN_FILES+=("${INFO[2]}")
  PREPARED_FILES+=("${INFO[3]}")
  FEATURE_FILES+=("${INFO[4]}")
  SPLIT_FILES+=("${INFO[5]}")
done

echo "==================================="
echo "EUOS25 Full-scale Pipeline"
if [ "$FORCE" = true ]; then
  echo "Mode: FORCE (will regenerate all outputs)"
else
  echo "Mode: REUSE (will skip existing outputs)"
fi
echo "Tasks: ${TASKS[*]}"
echo "==================================="

# Step 1: Prepare test data (shared across all tasks)
echo ""
echo "Step 1: Preparing test data..."
if [ "$FORCE" = true ] || [ ! -f "$PREPARED_TEST" ]; then
  uv run -m euos25.cli prepare \
    --input $RAW_TEST \
    --output $PREPARED_TEST \
    --normalize --deduplicate
else
  echo "  Skipping: $PREPARED_TEST already exists"
fi

# Step 2: Process each task independently
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TASK_NAME="${TASK_NAMES[$i]}"
  TRAIN_FILE="${TRAIN_FILES[$i]}"
  PREPARED_FILE="${PREPARED_FILES[$i]}"
  FEATURE_FILE="${FEATURE_FILES[$i]}"
  SPLIT_FILE="${SPLIT_FILES[$i]}"
  LABEL_COL="${LABEL_COLS[$i]}"

  echo ""
  echo "==================================="
  echo "Processing task: $TASK (task_name: $TASK_NAME)"
  echo "==================================="

  # Prepare training data
  echo ""
  echo "Step 2a: Preparing training data for $TASK..."
  if [ "$FORCE" = true ] || [ ! -f "$PREPARED_FILE" ]; then
    uv run -m euos25.cli prepare \
      --input "$TRAIN_FILE" \
      --output "$PREPARED_FILE" \
      --normalize --deduplicate
  else
    echo "  Skipping: $PREPARED_FILE already exists"
  fi

  # Create splits
  echo ""
  echo "Step 2b: Creating scaffold splits for $TASK..."
  if [ "$FORCE" = true ] || [ ! -f "$SPLIT_FILE" ]; then
    uv run -m euos25.cli make-splits \
      --input "$PREPARED_FILE" \
      --output "$SPLIT_FILE" \
      --folds 5 \
      --seed 42 \
      --scaffold-min-size 10 \
      --label-col "$LABEL_COL"
  else
    echo "  Skipping: $SPLIT_FILE already exists"
  fi

  # Build training features
  # Note: build-features command will add missing feature groups even if file exists
  echo ""
  echo "Step 2c: Building training features for $TASK..."
  if [ "$FORCE" = true ] || [ ! -f "$FEATURE_FILE" ]; then
    uv run -m euos25.cli build-features \
      --input "$PREPARED_FILE" \
      --output "$FEATURE_FILE" \
      --config $CONF
  else
    echo "  File exists, but checking for missing feature groups..."
    uv run -m euos25.cli build-features \
      --input "$PREPARED_FILE" \
      --output "$FEATURE_FILE" \
      --config $CONF
  fi

  # Train models
  echo ""
  echo "Step 2d: Training models for $TASK..."

  # Check if at least one model file exists
  MODEL_EXISTS=false
  if [ -f "$MODEL_DIR/$TASK/$TASK_NAME/lgbm/fold_0/model.txt" ]; then
    MODEL_EXISTS=true
  fi

  if [ "$FORCE" = true ] || [ "$MODEL_EXISTS" = false ]; then
    uv run -m euos25.cli train \
      --features "$FEATURE_FILE" \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --outdir "$MODEL_DIR/$TASK" \
      --data "$PREPARED_FILE" \
      --label-col "$LABEL_COL" \
      --task "$TASK_NAME"
  else
    echo "  Skipping: Models in $MODEL_DIR/$TASK/$TASK_NAME already exist"
  fi

  # Generate OOF predictions
  echo ""
  echo "Step 2e: Generating OOF predictions for $TASK..."

  OOF_OUTPUT="$PRED_DIR/$TASK/${TASK_NAME}_oof.csv"
  if [ "$FORCE" = true ] || [ ! -f "$OOF_OUTPUT" ]; then
    uv run -m euos25.cli infer \
      --features "$FEATURE_FILE" \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --model-dir "$MODEL_DIR/$TASK" \
      --outdir "$PRED_DIR/$TASK" \
      --mode oof \
      --task "$TASK_NAME"
  else
    echo "  Skipping: $OOF_OUTPUT already exists"
  fi
done

# Step 3: Build test features (shared features for all tasks)
# Note: build-features command will add missing feature groups even if file exists
echo ""
echo "Step 3: Building test features..."
if [ "$FORCE" = true ] || [ ! -f "$FEATURES_TEST" ]; then
  uv run -m euos25.cli build-features \
    --input $PREPARED_TEST \
    --output $FEATURES_TEST \
    --config $CONF
else
  echo "  File exists, but checking for missing feature groups..."
  uv run -m euos25.cli build-features \
    --input $PREPARED_TEST \
    --output $FEATURES_TEST \
    --config $CONF
fi

# Step 4: Generate test predictions for each task
echo ""
echo "Step 4: Generating test predictions..."
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TASK_NAME="${TASK_NAMES[$i]}"
  SPLIT_FILE="${SPLIT_FILES[$i]}"

  echo ""
  echo "Step 4a: Generating test predictions for $TASK..."

  TEST_OUTPUT="$PRED_DIR/$TASK/${TASK_NAME}_test.csv"
  if [ "$FORCE" = true ] || [ ! -f "$TEST_OUTPUT" ]; then
    uv run -m euos25.cli infer \
      --features $FEATURES_TEST \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --model-dir "$MODEL_DIR/$TASK" \
      --outdir "$PRED_DIR/$TASK" \
      --mode test \
      --task "$TASK_NAME"
  else
    echo "  Skipping: $TEST_OUTPUT already exists"
  fi
done

# Step 5: Create submissions for each task
echo ""
echo "Step 5: Creating submissions..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TASK_NAME="${TASK_NAMES[$i]}"

  echo ""
  echo "Step 5a: Creating submission for $TASK..."

  SUBMISSION_FILE="$SUBMISSION_DIR/${TASK}_${TIMESTAMP}.csv"
  # Submissions are always created with timestamp, so we don't skip them
  # But we can check if the input prediction file exists
  PRED_INPUT="$PRED_DIR/$TASK/${TASK_NAME}_test.csv"
  if [ ! -f "$PRED_INPUT" ]; then
    echo "  Warning: Prediction file $PRED_INPUT not found, skipping submission"
    continue
  fi

  uv run -m euos25.cli submit \
    --pred "$PRED_INPUT" \
    --out "$SUBMISSION_FILE"
done

# Step 6: Create final submission combining all tasks (only if --tasks=all)
if [ "$TASKS_SPEC" = "all" ]; then
  echo ""
  echo "Step 6: Creating final submission by combining all tasks..."
  FINAL_SUBMISSION_FILE="$SUBMISSION_DIR/submission_${TIMESTAMP}.csv"

  # Check if all individual submission files exist
  ALL_FILES_EXIST=true
  TRAN_340_FILE="$SUBMISSION_DIR/trans_340_${TIMESTAMP}.csv"
  TRAN_450_FILE="$SUBMISSION_DIR/trans_450_${TIMESTAMP}.csv"
  FLUO_480_FILE="$SUBMISSION_DIR/fluo_480_${TIMESTAMP}.csv"
  FLUO_340_450_FILE="$SUBMISSION_DIR/fluo_340_450_${TIMESTAMP}.csv"

  for SUBMISSION_FILE in "$TRAN_340_FILE" "$TRAN_450_FILE" "$FLUO_480_FILE" "$FLUO_340_450_FILE"; do
    if [ ! -f "$SUBMISSION_FILE" ]; then
      echo "  Warning: Submission file $SUBMISSION_FILE not found"
      ALL_FILES_EXIST=false
    fi
  done

  if [ "$ALL_FILES_EXIST" = true ]; then
    uv run -m euos25.cli submit-final \
      --trans-340 "$TRAN_340_FILE" \
      --trans-450 "$TRAN_450_FILE" \
      --fluo-480 "$FLUO_480_FILE" \
      --fluo-340-450 "$FLUO_340_450_FILE" \
      --out "$FINAL_SUBMISSION_FILE"
    echo "  Created: submission_${TIMESTAMP}.csv"
  else
    echo "  Warning: Could not create final submission due to missing files"
    ALL_FILES_EXIST=false
  fi
else
  ALL_FILES_EXIST=false
fi

echo ""
echo "==================================="
echo "Pipeline completed successfully!"
echo "==================================="
echo "Models saved to: $MODEL_DIR"
echo "Predictions saved to: $PRED_DIR"
echo "Submissions saved to: $SUBMISSION_DIR"
echo ""
echo "Submission files created:"
for TASK in "${TASKS[@]}"; do
  echo "  - ${TASK}_${TIMESTAMP}.csv"
done
if [ "$TASKS_SPEC" = "all" ] && [ "$ALL_FILES_EXIST" = true ]; then
  echo "  - submission_${TIMESTAMP}.csv (FINAL)"
fi
echo ""

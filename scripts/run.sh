#!/bin/bash
set -euo pipefail

# EUOS25 Full-scale Pipeline
# This script runs the complete pipeline with all 4 tasks (fluorescence340_450, fluorescence480, transmittance340, transmittance450)

CONF=configs/full.yaml
RAW_TEST=data/raw/euos25_challenge_test.csv

# Training data files
RAW_TRAIN_FLUO_340_450=data/raw/euos25_challenge_train_fluorescence340_450_extended.csv
RAW_TRAIN_FLUO_480=data/raw/euos25_challenge_train_fluorescence480_extended.csv
RAW_TRAIN_TRANS_340=data/raw/euos25_challenge_train_transmittance340_extended.csv
RAW_TRAIN_TRANS_450=data/raw/euos25_challenge_train_transmittance450_extended.csv

# Prepared data files
PREPARED_TEST=data/processed/test_prepared.csv
PREPARED_TRAIN_FLUO_340_450=data/processed/train_fluo_340_450_prepared.csv
PREPARED_TRAIN_FLUO_480=data/processed/train_fluo_480_prepared.csv
PREPARED_TRAIN_TRANS_340=data/processed/train_trans_340_prepared.csv
PREPARED_TRAIN_TRANS_450=data/processed/train_trans_450_prepared.csv

# Feature files
FEATURES_TEST=data/processed/features_test_full.parquet
FEATURES_TRAIN_FLUO_340_450=data/processed/features_train_fluo_340_450.parquet
FEATURES_TRAIN_FLUO_480=data/processed/features_train_fluo_480.parquet
FEATURES_TRAIN_TRANS_340=data/processed/features_train_trans_340.parquet
FEATURES_TRAIN_TRANS_450=data/processed/features_train_trans_450.parquet

# Split files
SPLITS_FLUO_340_450=data/processed/splits_fluo_340_450.json
SPLITS_FLUO_480=data/processed/splits_fluo_480.json
SPLITS_TRANS_340=data/processed/splits_trans_340.json
SPLITS_TRANS_450=data/processed/splits_trans_450.json

# Output directories
MODEL_DIR=data/models
PRED_DIR=data/preds
SUBMISSION_DIR=data/submissions

echo "==================================="
echo "EUOS25 Full-scale Pipeline"
echo "==================================="

# Step 1: Prepare test data (shared across all tasks)
echo ""
echo "Step 1: Preparing test data..."
uv run -m euos25.cli prepare \
  --input $RAW_TEST \
  --output $PREPARED_TEST \
  --normalize --deduplicate

# Step 2: Process each task independently
TASKS=("fluo_340_450" "fluo_480" "trans_340" "trans_450")
TRAIN_FILES=("$RAW_TRAIN_FLUO_340_450" "$RAW_TRAIN_FLUO_480" "$RAW_TRAIN_TRANS_340" "$RAW_TRAIN_TRANS_450")
PREPARED_FILES=("$PREPARED_TRAIN_FLUO_340_450" "$PREPARED_TRAIN_FLUO_480" "$PREPARED_TRAIN_TRANS_340" "$PREPARED_TRAIN_TRANS_450")
FEATURE_FILES=("$FEATURES_TRAIN_FLUO_340_450" "$FEATURES_TRAIN_FLUO_480" "$FEATURES_TRAIN_TRANS_340" "$FEATURES_TRAIN_TRANS_450")
SPLIT_FILES=("$SPLITS_FLUO_340_450" "$SPLITS_FLUO_480" "$SPLITS_TRANS_340" "$SPLITS_TRANS_450")
LABEL_COLS=("Fluorescence" "Fluorescence" "Transmittance" "Transmittance")

for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  TRAIN_FILE="${TRAIN_FILES[$i]}"
  PREPARED_FILE="${PREPARED_FILES[$i]}"
  FEATURE_FILE="${FEATURE_FILES[$i]}"
  SPLIT_FILE="${SPLIT_FILES[$i]}"
  LABEL_COL="${LABEL_COLS[$i]}"

  echo ""
  echo "==================================="
  echo "Processing task: $TASK"
  echo "==================================="

  # Prepare training data
  echo ""
  echo "Step 2a: Preparing training data for $TASK..."
  uv run -m euos25.cli prepare \
    --input "$TRAIN_FILE" \
    --output "$PREPARED_FILE" \
    --normalize --deduplicate

  # Create splits
  echo ""
  echo "Step 2b: Creating scaffold splits for $TASK..."
  uv run -m euos25.cli make-splits \
    --input "$PREPARED_FILE" \
    --output "$SPLIT_FILE" \
    --folds 5 \
    --seed 42 \
    --scaffold-min-size 10 \
    --label-col "$LABEL_COL"

  # Build training features
  echo ""
  echo "Step 2c: Building training features for $TASK..."
  uv run -m euos25.cli build-features \
    --input "$PREPARED_FILE" \
    --output "$FEATURE_FILE" \
    --config $CONF

  # Train models
  echo ""
  echo "Step 2d: Training models for $TASK..."
  uv run -m euos25.cli train \
    --features "$FEATURE_FILE" \
    --splits "$SPLIT_FILE" \
    --config $CONF \
    --outdir "$MODEL_DIR/$TASK" \
    --data "$PREPARED_FILE" \
    --label-col "$LABEL_COL"

  # Generate OOF predictions
  echo ""
  echo "Step 2e: Generating OOF predictions for $TASK..."
  uv run -m euos25.cli infer \
    --features "$FEATURE_FILE" \
    --splits "$SPLIT_FILE" \
    --config $CONF \
    --model-dir "$MODEL_DIR/$TASK" \
    --outdir "$PRED_DIR/$TASK" \
    --mode oof
done

# Step 3: Build test features (shared features for all tasks)
echo ""
echo "Step 3: Building test features..."
uv run -m euos25.cli build-features \
  --input $PREPARED_TEST \
  --output $FEATURES_TEST \
  --config $CONF

# Step 4: Generate test predictions for each task
echo ""
echo "Step 4: Generating test predictions..."
for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"
  SPLIT_FILE="${SPLIT_FILES[$i]}"

  echo ""
  echo "Step 4a: Generating test predictions for $TASK..."
  uv run -m euos25.cli infer \
    --features $FEATURES_TEST \
    --splits "$SPLIT_FILE" \
    --config $CONF \
    --model-dir "$MODEL_DIR/$TASK" \
    --outdir "$PRED_DIR/$TASK" \
    --mode test
done

# Step 5: Create submissions for each task
echo ""
echo "Step 5: Creating submissions..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for TASK in "${TASKS[@]}"; do
  echo ""
  echo "Step 5a: Creating submission for $TASK..."

  # Determine the output column name based on task
  if [[ "$TASK" == fluo_* ]]; then
    OUTPUT_COL="y_fluo_any"
  else
    OUTPUT_COL="y_trans_any"
  fi

  uv run -m euos25.cli submit \
    --pred "$PRED_DIR/$TASK/${OUTPUT_COL}_test.csv" \
    --out "$SUBMISSION_DIR/${TASK}_${TIMESTAMP}.csv"
done

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
echo ""

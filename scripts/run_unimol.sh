#!/bin/bash
set -euo pipefail

# EUOS25 Uni-Mol-2 Pipeline
# This script runs the complete pipeline with Uni-Mol-2 model for a single task

# Default values
FORCE=false
CONF=configs/full_unimol.yaml
PROCESSED_DIR=data/processed
MODEL_DIR=data/models/unimol
PRED_DIR=data/preds/unimol
SUBMISSION_DIR=data/submissions
TASK="trans_340"  # Default task
TASK_NAME="y_trans_any"  # Default task name in code
LABEL_COL="Transmittance"  # Default label column
FULL_ONLY=false  # Skip CV and train directly on full data
NO_STANDARDIZE=false  # Don't standardize SMILES (use normalization instead)
NO_REMOVE_SALTS=false  # Don't remove salts/solvents during standardization

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
    --task)
      TASK="$2"
      shift 2
      ;;
    --full-only)
      FULL_ONLY=true
      shift
      ;;
    --no-standardize)
      NO_STANDARDIZE=true
      shift
      ;;
    --no-remove-salts)
      NO_REMOVE_SALTS=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --force              Force regeneration of all outputs (default: false)"
      echo "  --config PATH        Configuration file path (default: configs/full_unimol.yaml)"
      echo "  --processed-dir DIR  Processed data directory (default: data/processed)"
      echo "  --model-dir DIR      Model output directory (default: data/models/unimol)"
      echo "  --pred-dir DIR       Prediction output directory (default: data/preds/unimol)"
      echo "  --submission-dir DIR Submission output directory (default: data/submissions)"
      echo "  --task TASK          Task name (default: trans_340)"
      echo "                       Available tasks: fluo_340_450, fluo_480, trans_340, trans_450"
      echo "  --full-only          Skip CV and train directly on full data with fold 0 validation"
      echo "  --no-standardize     Don't standardize SMILES (use normalization instead)"
      echo "  --no-remove-salts    Don't remove salts/solvents during SMILES standardization"
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

# Task mapping
case $TASK in
  fluo_340_450)
    TASK_NAME="y_fluo_340_450"
    LABEL_COL="Fluorescence"
    RAW_TRAIN=data/raw/euos25_challenge_train_fluorescence340_450_extended.csv
    ;;
  fluo_480)
    TASK_NAME="y_fluo_480"
    LABEL_COL="Fluorescence"
    RAW_TRAIN=data/raw/euos25_challenge_train_fluorescence480_extended.csv
    ;;
  trans_340)
    TASK_NAME="y_trans_340"
    LABEL_COL="Transmittance"
    RAW_TRAIN=data/raw/euos25_challenge_train_transmittance340_extended.csv
    ;;
  trans_450)
    TASK_NAME="y_trans_450"
    LABEL_COL="Transmittance"
    RAW_TRAIN=data/raw/euos25_challenge_train_transmittance450_extended.csv
    ;;
  *)
    echo "Error: Unknown task '$TASK'"
    echo "Available tasks: fluo_340_450, fluo_480, trans_340, trans_450"
    exit 1
    ;;
esac

# File paths
RAW_TEST=data/raw/euos25_challenge_test.csv
PREPARED_TEST=$PROCESSED_DIR/test_prepared.csv
PREPARED_TRAIN=$PROCESSED_DIR/train_${TASK}_prepared.csv
FEATURES_TEST=$PROCESSED_DIR/features_test_unimol.parquet
FEATURES_TRAIN=$PROCESSED_DIR/features_train_${TASK}_unimol.parquet
SPLIT_FILE=$PROCESSED_DIR/splits_${TASK}.json

echo "==================================="
echo "EUOS25 Uni-Mol-2 Pipeline"
if [ "$FORCE" = true ]; then
  echo "Mode: FORCE (will regenerate all outputs)"
else
  echo "Mode: REUSE (will skip existing outputs)"
fi
if [ "$FULL_ONLY" = true ]; then
  echo "Training: FULL-ONLY (skip CV, train on full data with fold 0 validation)"
else
  echo "Training: CROSS-VALIDATION (5-fold CV + full model)"
fi
echo "Task: $TASK (task_name: $TASK_NAME)"
echo "Config: $CONF"
echo "==================================="

# Step 1: Prepare test data
echo ""
echo "Step 1: Preparing test data..."
PREPARE_OPTS="--deduplicate"
if [ "$NO_STANDARDIZE" = true ]; then
  PREPARE_OPTS="$PREPARE_OPTS --normalize --no-standardize"
else
  PREPARE_OPTS="$PREPARE_OPTS --normalize"
  if [ "$NO_REMOVE_SALTS" = true ]; then
    PREPARE_OPTS="$PREPARE_OPTS --standardize --no-remove-salts"
  fi
fi
if [ "$FORCE" = true ] || [ ! -f "$PREPARED_TEST" ]; then
  uv run -m euos25.cli prepare \
    --input $RAW_TEST \
    --output $PREPARED_TEST \
    $PREPARE_OPTS
else
  echo "  Skipping: $PREPARED_TEST already exists"
fi

# Step 2: Prepare training data
echo ""
echo "Step 2: Preparing training data for $TASK..."
if [ "$FORCE" = true ] || [ ! -f "$PREPARED_TRAIN" ]; then
  uv run -m euos25.cli prepare \
    --input "$RAW_TRAIN" \
    --output "$PREPARED_TRAIN" \
    $PREPARE_OPTS
else
  echo "  Skipping: $PREPARED_TRAIN already exists"
fi

# Step 3: Create splits
echo ""
echo "Step 3: Creating scaffold splits for $TASK..."
if [ "$FORCE" = true ] || [ ! -f "$SPLIT_FILE" ]; then
  uv run -m euos25.cli make-splits \
    --input "$PREPARED_TRAIN" \
    --output "$SPLIT_FILE" \
    --folds 5 \
    --seed 42 \
    --scaffold-min-size 10 \
    --label-col "$LABEL_COL"
else
  echo "  Skipping: $SPLIT_FILE already exists"
fi

# Step 4: Build training features
# Note: Uni-Mol-2 doesn't need traditional features, but we need a Parquet file
# with SMILES column for compatibility with the pipeline
echo ""
echo "Step 4: Building training features for $TASK..."
if [ "$FORCE" = true ] || [ ! -f "$FEATURES_TRAIN" ]; then
  uv run -m euos25.cli build-features \
    --input "$PREPARED_TRAIN" \
    --output "$FEATURES_TRAIN" \
    --config $CONF
else
  echo "  File exists, but checking for missing feature groups..."
  uv run -m euos25.cli build-features \
    --input "$PREPARED_TRAIN" \
    --output "$FEATURES_TRAIN" \
    --config $CONF
fi

# Step 5: Train models
echo ""
echo "Step 5: Training Uni-Mol-2 models for $TASK..."

if [ "$FULL_ONLY" = true ]; then
  # Full-only mode: skip CV, train directly on full data with fold 0 validation
  echo "  Mode: Full-only (skip CV, train on full data with fold 0 validation)"

  # Check if full model exists
  FULL_MODEL_EXISTS=false
  if [ -d "$MODEL_DIR/$TASK/$TASK_NAME/unimol/full_model" ]; then
    # Check for checkpoint files or model files
    if [ -n "$(find "$MODEL_DIR/$TASK/$TASK_NAME/unimol/full_model" -name '*.ckpt' -o -name 'model.*' 2>/dev/null)" ]; then
      FULL_MODEL_EXISTS=true
    fi
  fi

  if [ "$FORCE" = true ] || [ "$FULL_MODEL_EXISTS" = false ]; then
    uv run -m euos25.cli train \
      --features "$FEATURES_TRAIN" \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --outdir "$MODEL_DIR/$TASK" \
      --data "$PREPARED_TRAIN" \
      --label-col "$LABEL_COL" \
      --task "$TASK_NAME" \
      --full-only
  else
    echo "  Skipping: Full model in $MODEL_DIR/$TASK/$TASK_NAME/unimol/full_model already exists"
  fi
else
  # Standard CV mode
  echo "  Mode: Cross-validation (5-fold CV + full model)"

  # Check if at least one model file exists
  MODEL_EXISTS=false
  if [ -d "$MODEL_DIR/$TASK/$TASK_NAME/unimol" ]; then
    # Check for checkpoint files
    if [ -n "$(find "$MODEL_DIR/$TASK/$TASK_NAME/unimol" -name '*.ckpt' 2>/dev/null)" ]; then
      MODEL_EXISTS=true
    fi
  fi

  if [ "$FORCE" = true ] || [ "$MODEL_EXISTS" = false ]; then
    uv run -m euos25.cli train \
      --features "$FEATURES_TRAIN" \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --outdir "$MODEL_DIR/$TASK" \
      --data "$PREPARED_TRAIN" \
      --label-col "$LABEL_COL" \
      --task "$TASK_NAME"
  else
    echo "  Skipping: Models in $MODEL_DIR/$TASK/$TASK_NAME/unimol already exist"
  fi
fi

# Step 6: Generate OOF predictions (skip in full-only mode)
if [ "$FULL_ONLY" = false ]; then
  echo ""
  echo "Step 6: Generating OOF predictions for $TASK..."

  OOF_OUTPUT="$PRED_DIR/$TASK/${TASK_NAME}_oof.csv"
  if [ "$FORCE" = true ] || [ ! -f "$OOF_OUTPUT" ]; then
    uv run -m euos25.cli infer \
      --features "$FEATURES_TRAIN" \
      --splits "$SPLIT_FILE" \
      --config $CONF \
      --model-dir "$MODEL_DIR/$TASK" \
      --outdir "$PRED_DIR/$TASK" \
      --mode oof \
      --task "$TASK_NAME"
  else
    echo "  Skipping: $OOF_OUTPUT already exists"
  fi
else
  echo ""
  echo "Step 6: Skipping OOF predictions (full-only mode)"
fi

# Step 7: Build test features
echo ""
echo "Step 7: Building test features..."
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

# Step 8: Generate test predictions
echo ""
echo "Step 8: Generating test predictions for $TASK..."

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

# Step 9: Create submission
echo ""
echo "Step 9: Creating submission for $TASK..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SUBMISSION_FILE="$SUBMISSION_DIR/${TASK}_unimol_${TIMESTAMP}.csv"
PRED_INPUT="$PRED_DIR/$TASK/${TASK_NAME}_test.csv"
if [ ! -f "$PRED_INPUT" ]; then
  echo "  Warning: Prediction file $PRED_INPUT not found, skipping submission"
else
  uv run -m euos25.cli submit \
    --pred "$PRED_INPUT" \
    --out "$SUBMISSION_FILE"
  echo "  Created: ${TASK}_unimol_${TIMESTAMP}.csv"
fi

echo ""
echo "==================================="
echo "Pipeline completed successfully!"
echo "==================================="
echo "Models saved to: $MODEL_DIR/$TASK"
echo "Predictions saved to: $PRED_DIR/$TASK"
echo "Submission saved to: $SUBMISSION_FILE"
echo ""


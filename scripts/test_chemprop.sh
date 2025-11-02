#!/bin/bash
set -euo pipefail

# Quick test script for ChemProp with small configuration
# Tests only fluo_340_450 task with minimal epochs

CONF=configs/small_chemprop.yaml
PROCESSED_DIR=data/processed
MODEL_DIR=data/models/test_chemprop
PRED_DIR=data/preds/test_chemprop

# Task configuration
TASK="fluo_340_450"
TASK_NAME="y_fluo_any"
RAW_TRAIN=data/raw/euos25_challenge_train_fluorescence340_450_extended.csv
PREPARED_FILE=$PROCESSED_DIR/train_fluo_340_450_prepared.csv
FEATURE_FILE=$PROCESSED_DIR/features_train_fluo_340_450_test.parquet
SPLIT_FILE=$PROCESSED_DIR/splits_fluo_340_450.json
LABEL_COL="Fluorescence"

echo "==================================="
echo "ChemProp Quick Test"
echo "Config: $CONF"
echo "Task: $TASK"
echo "==================================="

# Step 1: Prepare data (skip if exists)
echo ""
echo "Step 1: Preparing training data..."
if [ ! -f "$PREPARED_FILE" ]; then
  uv run -m euos25.cli prepare \
    --input "$RAW_TRAIN" \
    --output "$PREPARED_FILE" \
    --normalize --deduplicate
else
  echo "  Skipping: $PREPARED_FILE already exists"
fi

# Step 2: Create splits (skip if exists)
echo ""
echo "Step 2: Creating scaffold splits..."
if [ ! -f "$SPLIT_FILE" ]; then
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

# Step 3: Build features (skip if exists)
echo ""
echo "Step 3: Building training features..."
if [ ! -f "$FEATURE_FILE" ]; then
  uv run -m euos25.cli build-features \
    --input "$PREPARED_FILE" \
    --output "$FEATURE_FILE" \
    --config "$CONF"
else
  echo "  Skipping: $FEATURE_FILE already exists"
fi

# Step 4: Train models (test with 1 fold only)
echo ""
echo "Step 4: Training models with CV (testing fold 0 only)..."
uv run -m euos25.cli train \
  --features "$FEATURE_FILE" \
  --splits "$SPLIT_FILE" \
  --config "$CONF" \
  --outdir "$MODEL_DIR" \
  --data "$PREPARED_FILE" \
  --label-col "$LABEL_COL" \
  --task "$TASK_NAME"

echo ""
echo "==================================="
echo "Test completed!"
echo "==================================="


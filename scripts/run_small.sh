#!/bin/bash
set -euo pipefail

# EUOS25 Small-scale Pipeline Quickstart
# This script runs the complete pipeline from data preparation to submission

CONF=configs/small.yaml
RAW_TRAIN=data/raw/euos25_challenge_train_fluorescence340_450.csv
RAW_TEST=data/raw/euos25_challenge_test.csv
PREPARED_TRAIN=data/processed/train_prepared.csv
PREPARED_TEST=data/processed/test_prepared.csv
FEATURES_TRAIN=data/processed/features_train_small.parquet
FEATURES_TEST=data/processed/features_test_small.parquet
SPLITS=data/processed/splits_5.json
MODEL_DIR=data/models
PRED_DIR=data/preds
SUBMISSION_DIR=data/submissions

echo "==================================="
echo "EUOS25 Small-scale Pipeline"
echo "==================================="

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing training data..."
uv run -m euos25.cli prepare \
  --input $RAW_TRAIN \
  --output $PREPARED_TRAIN \
  --normalize --deduplicate

echo ""
echo "Step 1b: Preparing test data..."
uv run -m euos25.cli prepare \
  --input $RAW_TEST \
  --output $PREPARED_TEST \
  --normalize --deduplicate

# Step 2: Create splits
echo ""
echo "Step 2: Creating scaffold splits..."
uv run -m euos25.cli make-splits \
  --input $PREPARED_TRAIN \
  --output $SPLITS \
  --folds 5 \
  --seed 42 \
  --scaffold-min-size 10 \
  --label-col Fluorescence

# Step 3: Build features
echo ""
echo "Step 3: Building training features..."
uv run -m euos25.cli build-features \
  --input $PREPARED_TRAIN \
  --output $FEATURES_TRAIN \
  --config $CONF

echo ""
echo "Step 3b: Building test features..."
uv run -m euos25.cli build-features \
  --input $PREPARED_TEST \
  --output $FEATURES_TEST \
  --config $CONF

# Step 4: Train models
echo ""
echo "Step 4: Training models with CV..."
uv run -m euos25.cli train \
  --features $FEATURES_TRAIN \
  --splits $SPLITS \
  --config $CONF \
  --outdir $MODEL_DIR \
  --data $PREPARED_TRAIN \
  --label-col Fluorescence

# Step 5: Generate OOF predictions
echo ""
echo "Step 5: Generating OOF predictions..."
uv run -m euos25.cli infer \
  --features $FEATURES_TRAIN \
  --splits $SPLITS \
  --config $CONF \
  --model-dir $MODEL_DIR \
  --outdir $PRED_DIR \
  --mode oof

# Step 6: Generate test predictions
echo ""
echo "Step 6: Generating test predictions..."
uv run -m euos25.cli infer \
  --features $FEATURES_TEST \
  --splits $SPLITS \
  --config $CONF \
  --model-dir $MODEL_DIR \
  --outdir $PRED_DIR \
  --mode test

# Step 7: Create submission
echo ""
echo "Step 7: Creating submission..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
uv run -m euos25.cli submit \
  --pred $PRED_DIR/y_fluo_any_test.csv \
  --out $SUBMISSION_DIR/y_fluo_any_${TIMESTAMP}.csv

echo ""
echo "==================================="
echo "Pipeline completed successfully!"
echo "==================================="
echo "Models saved to: $MODEL_DIR"
echo "Predictions saved to: $PRED_DIR"
echo "Submission saved to: $SUBMISSION_DIR"

"""Inference pipeline for predictions."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from euos25.config import Config
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_json, load_parquet, save_csv

logger = logging.getLogger(__name__)


def load_fold_model(model_dir: Path, config: Config):
    """Load model for a fold.

    Args:
        model_dir: Directory containing model
        config: Pipeline configuration

    Returns:
        Loaded model
    """
    if config.model.name == "lgbm":
        return LGBMClassifier.load(str(model_dir))
    else:
        raise ValueError(f"Unknown model: {config.model.name}")


def predict_oof(
    features_path: str,
    splits_path: str,
    model_dir: str,
    config: Config,
    output_path: str,
    task_name: Optional[str] = None,
) -> pd.DataFrame:
    """Generate out-of-fold predictions.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        model_dir: Directory containing trained models
        config: Pipeline configuration
        output_path: Path to save predictions
        task_name: Task name override (defaults to config.task)

    Returns:
        DataFrame with OOF predictions
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Initialize predictions array
    predictions = np.zeros(len(features))
    prediction_counts = np.zeros(len(features))

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Load models directory
    models_path = Path(model_dir) / actual_task_name / config.model.name

    # Generate predictions for each fold
    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])
        logger.info(f"Predicting fold {fold_idx}")

        # Load model
        model_path = models_path / f"fold_{fold_idx}"
        model = load_fold_model(model_path, config)

        # Get validation indices (these are positional indices)
        valid_pos_indices = fold_data["valid"]

        # Convert positional indices to IDs
        valid_ids = features.index[valid_pos_indices]
        X_valid = features.loc[valid_ids]

        # Predict
        preds = model.predict_proba(X_valid)

        # Store predictions using positional indices (predictions array is positional)
        predictions[valid_pos_indices] = preds
        prediction_counts[valid_pos_indices] += 1

    # Verify all samples have predictions
    if not np.all(prediction_counts > 0):
        missing = np.sum(prediction_counts == 0)
        logger.warning(f"{missing} samples have no predictions")

    # Create DataFrame
    pred_df = pd.DataFrame({
        "mol_id": features.index,
        "prediction": predictions,
    })

    # Save predictions
    save_csv(pred_df, output_path, index=False)
    logger.info(f"Saved OOF predictions to {output_path}")

    return pred_df


def predict_test(
    features_path: str,
    model_dir: str,
    config: Config,
    output_path: str,
    task_name: Optional[str] = None,
    average: bool = True,
) -> pd.DataFrame:
    """Generate test predictions by averaging across folds.

    Args:
        features_path: Path to test features Parquet
        model_dir: Directory containing trained models
        config: Pipeline configuration
        output_path: Path to save predictions
        task_name: Task name override (defaults to config.task)
        average: Whether to average predictions across folds

    Returns:
        DataFrame with test predictions
    """
    # Load features
    features = load_parquet(features_path)

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Load models directory
    models_path = Path(model_dir) / actual_task_name / config.model.name

    # Find all fold models
    fold_dirs = sorted(models_path.glob("fold_*"))
    logger.info(f"Found {len(fold_dirs)} fold models")

    # Generate predictions for each fold
    all_predictions = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[1])
        logger.info(f"Predicting with fold {fold_idx} model")

        # Load model
        model = load_fold_model(fold_dir, config)

        # Predict
        preds = model.predict_proba(features)
        all_predictions.append(preds)

    # Average or stack predictions
    if average:
        predictions = np.mean(all_predictions, axis=0)
    else:
        predictions = np.array(all_predictions)

    # Create DataFrame
    if average:
        pred_df = pd.DataFrame({
            "mol_id": features.index,
            "prediction": predictions,
        })
    else:
        pred_df = pd.DataFrame({
            "mol_id": features.index,
            **{f"fold_{i}": preds for i, preds in enumerate(all_predictions)}
        })

    # Save predictions
    save_csv(pred_df, output_path, index=False)
    logger.info(f"Saved test predictions to {output_path}")

    return pred_df

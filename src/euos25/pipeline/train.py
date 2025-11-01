"""Training pipeline with cross-validation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from euos25.config import Config
from euos25.models.base import ClfModel
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics, save_fold_metrics

logger = logging.getLogger(__name__)


def create_model(config: Config) -> ClfModel:
    """Create model from configuration.

    Args:
        config: Pipeline configuration

    Returns:
        Model instance
    """
    model_name = config.model.name
    model_params = config.model.params.copy()

    # Add imbalance handling parameters
    if config.imbalance.use_focal_loss:
        # Use focal loss (overrides pos_weight)
        model_params["use_focal_loss"] = True
        model_params["focal_alpha"] = config.imbalance.focal_alpha
        model_params["focal_gamma"] = config.imbalance.focal_gamma
        # pos_weight is ignored when use_focal_loss is True
        model_params["pos_weight"] = None
    elif config.imbalance.use_pos_weight and config.imbalance.pos_weight_from_data:
        # Will be computed during training
        model_params["pos_weight"] = None
    elif config.imbalance.use_pos_weight and config.imbalance.pos_weight_value:
        model_params["pos_weight"] = config.imbalance.pos_weight_value
    else:
        model_params["use_focal_loss"] = False

    # Add early stopping parameters
    model_params["early_stopping_rounds"] = config.early_stopping_rounds

    if model_name == "lgbm":
        return LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    config: Config,
    fold_idx: int,
    output_dir: Optional[Path] = None,
) -> tuple[ClfModel, Dict[str, float]]:
    """Train model on single fold.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        config: Pipeline configuration
        fold_idx: Fold index
        output_dir: Directory to save model (optional)

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Train: {len(y_train)} samples, pos={y_train.sum()}")
    logger.info(f"  Valid: {len(y_valid)} samples, pos={y_valid.sum()}")

    # Create model
    model = create_model(config)

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
    )

    # Predict on validation
    y_pred = model.predict_proba(X_valid)

    # Calculate metrics
    metrics = calc_metrics(y_valid, y_pred, metrics=config.metrics)

    # Log metrics
    for metric_name, score in metrics.items():
        logger.info(f"  {metric_name}: {score:.6f}")

    # Save model
    if output_dir:
        model_dir = output_dir / f"fold_{fold_idx}"
        model.save(str(model_dir))

    return model, metrics


def train_cv(
    features_path: str,
    splits_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    task_name: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Train models with cross-validation.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save models and metrics
        task_name: Task name override (defaults to config.task)

    Returns:
        List of metrics dictionaries per fold
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Train each fold
    fold_metrics = []

    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])

        # Get train/valid indices (these are positional indices)
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        # Convert positional indices to IDs
        # Splits were created on a DataFrame in the same order as features
        train_ids = features.index[train_pos_indices]
        valid_ids = features.index[valid_pos_indices]

        # Get features and labels using IDs
        X_train = features.loc[train_ids]
        y_train = labels.loc[train_ids].values

        X_valid = features.loc[valid_ids]
        y_valid = labels.loc[valid_ids].values

        # Train fold
        model, metrics = train_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config,
            fold_idx,
            output_dir=output_path,
        )

        fold_metrics.append(metrics)

    # Save fold metrics
    metrics_path = output_path / "cv_metrics.csv"
    save_fold_metrics(fold_metrics, str(metrics_path))

    # Log aggregated metrics
    logger.info("=" * 50)
    logger.info("Cross-validation results:")
    for metric_name in config.metrics:
        values = [m[metric_name] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
    logger.info("=" * 50)

    return fold_metrics

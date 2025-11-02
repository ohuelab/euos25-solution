"""Training pipeline with cross-validation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from euos25.config import Config
from euos25.models.base import ClfModel
from euos25.models.lgbm import LGBMClassifier
from euos25.models import ChemPropModel, CHEMPROP_AVAILABLE
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics, save_fold_metrics

logger = logging.getLogger(__name__)


def create_model(config: Config, checkpoint_dir: Optional[str] = None) -> ClfModel:
    """Create model from configuration.

    Args:
        config: Pipeline configuration
        checkpoint_dir: Optional checkpoint directory override (for ChemProp)

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
        # Store multiplier for later use if set
        if config.imbalance.pos_weight_multiplier is not None:
            model_params["pos_weight_multiplier"] = config.imbalance.pos_weight_multiplier
    elif config.imbalance.use_pos_weight and config.imbalance.pos_weight_value:
        model_params["pos_weight"] = config.imbalance.pos_weight_value
    else:
        model_params["use_focal_loss"] = False

    # Add early stopping parameters
    model_params["early_stopping_rounds"] = config.early_stopping_rounds

    if model_name == "lgbm":
        return LGBMClassifier(**model_params)
    elif model_name == "chemprop":
        if not CHEMPROP_AVAILABLE:
            raise ImportError(
                "ChemProp is not available. Please install chemprop package."
            )

        # For ChemProp, focal loss parameters come from config.model.params or config.imbalance
        # Check if use_focal_loss is set in model params (takes precedence)
        if model_params.get("use_focal_loss", False):
            # Focal loss params are already in model_params from config
            pass
        elif config.imbalance.use_focal_loss:
            # Use focal loss from imbalance config
            model_params["use_focal_loss"] = True
            model_params["focal_alpha"] = config.imbalance.focal_alpha
            model_params["focal_gamma"] = config.imbalance.focal_gamma

        # Add random_seed from config
        model_params["random_seed"] = config.seed

        # Add early stopping parameters for ChemProp
        model_params["early_stopping_rounds"] = config.early_stopping_rounds
        model_params["early_stopping_metric"] = config.early_stopping_metric

        # Override checkpoint_dir if provided
        if checkpoint_dir is not None:
            model_params["checkpoint_dir"] = checkpoint_dir

        return ChemPropModel(**model_params)
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
    task_name: Optional[str] = None,
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
        task_name: Task name for checkpoint directory (optional)

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Train: {len(y_train)} samples, pos={y_train.sum()}")
    logger.info(f"  Valid: {len(y_valid)} samples, pos={y_valid.sum()}")

    # Build checkpoint directory for ChemProp with task name and fold number
    checkpoint_dir = None
    if config.model.name == "chemprop" and output_dir is not None:
        actual_task_name = task_name if task_name is not None else config.task
        # Use checkpoint_dir from config if specified, otherwise construct from output_dir
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir) / actual_task_name / f"fold_{fold_idx}")
        else:
            checkpoint_dir = str(output_dir / "checkpoints" / f"fold_{fold_idx}")
        # Ensure directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(config, checkpoint_dir=checkpoint_dir)

    # Train model - handle different model signatures
    if config.model.name == "chemprop":
        # ChemProp uses X_val and y_val instead of eval_set
        model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
    else:
        # LGBM and others use eval_set
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
) -> tuple[List[Dict[str, float]], List[int], List[int]]:
    """Train models with cross-validation.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save models and metrics
        task_name: Task name override (defaults to config.task)

    Returns:
        Tuple of (fold_metrics, best_iterations, train_sizes)
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
    best_iterations = []
    train_sizes = []

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
            task_name=actual_task_name,
        )

        fold_metrics.append(metrics)
        best_iterations.append(model.best_iteration)
        train_sizes.append(len(X_train))

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

    avg_best_iter = int(np.mean(best_iterations))
    logger.info(f"  Average best iteration: {avg_best_iter}")
    logger.info("=" * 50)

    return fold_metrics, best_iterations, train_sizes


def train_full(
    features_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    best_iterations: List[int],
    train_sizes: List[int],
    task_name: Optional[str] = None,
) -> ClfModel:
    """Train model on full training data.

    Args:
        features_path: Path to features Parquet
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save model
        best_iterations: List of best_iteration from CV folds
        train_sizes: List of training set sizes from CV folds
        task_name: Task name override (defaults to config.task)

    Returns:
        Trained model
    """
    # Load features
    features = load_parquet(features_path)

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Get full training data
    # Only use samples that have labels
    common_ids = features.index.intersection(labels.index)
    X_full = features.loc[common_ids]
    y_full = labels.loc[common_ids].values

    logger.info("=" * 50)
    logger.info("Training on full dataset")
    logger.info(f"  Full dataset: {len(y_full)} samples, pos={y_full.sum()}")

    # Build checkpoint directory for ChemProp with task name and "full" suffix
    checkpoint_dir = None
    if config.model.name == "chemprop":
        # Use checkpoint_dir from config if specified, otherwise construct from output_dir
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir) / actual_task_name / "full")
        else:
            checkpoint_dir = str(output_path / "checkpoints" / "full")
        # Ensure directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(config, checkpoint_dir=checkpoint_dir)

    # Handle model-specific full training adjustments
    if config.model.name == "lgbm":
        # Calculate average best_iteration from CV
        avg_best_iter = int(np.mean(best_iterations))
        logger.info(f"  Average best iteration from CV: {avg_best_iter}")

        # Calculate size ratio: full_size / avg_train_size
        avg_train_size = np.mean(train_sizes)
        size_ratio = len(X_full) / avg_train_size
        logger.info(f"  Size ratio (full / avg_train): {size_ratio:.3f}")

        # Adjust n_estimators: int(0.9 * size_ratio * avg_best_iter)
        adjusted_n_estimators = int(0.9 * size_ratio * avg_best_iter)
        # Ensure at least avg_best_iter
        adjusted_n_estimators = max(adjusted_n_estimators, avg_best_iter)
        logger.info(f"  Adjusted n_estimators: {adjusted_n_estimators}")

        # Override n_estimators for full training
        original_n_estimators = model.params["n_estimators"]
        model.params["n_estimators"] = adjusted_n_estimators

        # Train on full data (no validation set, no early stopping)
        logger.info(f"Training with {adjusted_n_estimators} rounds (no early stopping)")
        model.fit(X_full, y_full, eval_set=None)

        # Restore original n_estimators (for metadata consistency)
        model.params["n_estimators"] = original_n_estimators
    elif config.model.name == "chemprop":
        # For ChemProp, use max_epochs from config (no adjustment needed)
        logger.info(f"  Training with max_epochs={model.params.get('max_epochs', 'default')}")
        logger.info("  Training on full data (no validation set)")
        model.fit(X_full, y_full, eval_set=None)
    else:
        # Generic fallback
        logger.info("  Training on full data (no validation set)")
        model.fit(X_full, y_full, eval_set=None)

    # Save full model
    full_model_dir = output_path / "full_model"
    model.save(str(full_model_dir))

    logger.info(f"Saved full model to {full_model_dir}")
    logger.info("=" * 50)

    return model

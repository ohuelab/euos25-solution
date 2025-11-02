"""Training script for multi-task learning with ChemProp/Chemeleon."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from euos25.config import Config, load_config
from euos25.models import ChemPropModel, CHEMPROP_AVAILABLE
from euos25.pipeline.train import create_model
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics, save_fold_metrics

logger = logging.getLogger(__name__)


def load_multitask_labels(
    task_names: List[str], data_dir: str = "data/raw"
) -> Tuple[pd.DataFrame, List[str]]:
    """Load labels for multiple tasks.

    Args:
        task_names: List of task names (e.g., ["transmittance340", "transmittance570"])
        data_dir: Directory containing raw data files

    Returns:
        Tuple of (labels_df, column_names) where labels_df has columns for each task
    """
    data_dir = Path(data_dir)

    # Task name to file mapping
    task_to_file = {
        "transmittance340": "euos25_challenge_train_transmittance340.csv",
        "transmittance450": "euos25_challenge_train_transmittance450.csv",
        "fluorescence340_450": "euos25_challenge_train_fluorescence340_450.csv",
        "fluorescence480": "euos25_challenge_train_fluorescence480.csv",
    }

    # Load each task's labels
    all_labels = []
    common_ids = None

    for task_name in task_names:
        if task_name not in task_to_file:
            raise ValueError(f"Unknown task: {task_name}")

        file_path = data_dir / task_to_file[task_name]
        df = pd.read_csv(file_path)

        # Column names vary by file - handle different naming conventions
        # Check for label column (in order of preference)
        label_col = None
        for possible_label in [
            "Transmittance (qualitative)",
            "Transmittance",
            "Fluorescence (qualitative)",
            "Fluorescence",
        ]:
            if possible_label in df.columns:
                label_col = possible_label
                break

        if label_col is None:
            raise ValueError(f"Cannot find label column in {file_path}. Available columns: {df.columns.tolist()}")

        # ID column can be either "N" or "ID"
        id_col = None
        if "N" in df.columns:
            id_col = "N"
        elif "ID" in df.columns:
            id_col = "ID"
        else:
            raise ValueError(f"Cannot find ID column in {file_path}. Available columns: {df.columns.tolist()}")

        # Set index to ID column
        df = df.set_index(id_col)

        # Get labels
        labels = df[label_col].rename(task_name)

        all_labels.append(labels)

        # Find common IDs across all tasks
        if common_ids is None:
            common_ids = set(labels.index)
        else:
            common_ids &= set(labels.index)

    # Combine labels for common IDs
    common_ids = sorted(common_ids)
    labels_df = pd.DataFrame(
        {task: labels.loc[common_ids] for task, labels in zip(task_names, all_labels)}
    )

    logger.info(f"Loaded {len(labels_df)} samples with labels for {len(task_names)} tasks")
    logger.info(f"Task names: {task_names}")
    for task in task_names:
        pos_count = labels_df[task].sum()
        logger.info(f"  {task}: {pos_count}/{len(labels_df)} positive samples")

    return labels_df, task_names


def train_multitask_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    config: Config,
    fold_idx: int,
    output_dir: Optional[Path] = None,
) -> Tuple[ChemPropModel, Dict[str, float]]:
    """Train multi-task model on single fold.

    Args:
        X_train: Training features
        y_train: Training labels (n_samples, n_tasks)
        X_valid: Validation features
        y_valid: Validation labels (n_samples, n_tasks)
        config: Pipeline configuration
        fold_idx: Fold index
        output_dir: Directory to save model (optional)

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Train: {len(y_train)} samples")
    logger.info(f"  Valid: {len(y_valid)} samples")

    # Log per-task statistics
    for task_idx, task_name in enumerate(config.task_names):
        train_pos = y_train[:, task_idx].sum()
        valid_pos = y_valid[:, task_idx].sum()
        logger.info(f"  Task {task_name}: train pos={train_pos}, valid pos={valid_pos}")

    # Check if model already exists
    if output_dir:
        model_dir = output_dir / f"fold_{fold_idx}"
        if config.model.name == "chemprop":
            # For ChemProp, check if model exists (can be file or directory)
            model_exists = False
            if model_dir.exists():
                if model_dir.is_file():
                    # Model is saved as a single checkpoint file
                    model_exists = True
                elif model_dir.is_dir():
                    # Model directory exists, check for checkpoint files
                    ckpt_files = list(model_dir.glob("*.ckpt"))
                    if ckpt_files:
                        model_exists = True

            if model_exists:
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                # Load existing model
                model_params = config.model.params.copy()
                model_params["n_tasks"] = config.n_tasks
                model_params["random_seed"] = config.seed
                model_params["early_stopping_rounds"] = config.early_stopping_rounds
                model_params["early_stopping_metric"] = config.early_stopping_metric

                # Add imbalance handling parameters
                if config.imbalance.use_focal_loss:
                    model_params["use_focal_loss"] = True
                    model_params["focal_alpha"] = config.imbalance.focal_alpha
                    model_params["focal_gamma"] = config.imbalance.focal_gamma

                model = ChemPropModel.load_from_checkpoint(str(model_dir), **model_params)

                # Predict on validation to compute metrics
                y_pred = model.predict_proba(X_valid)

                # Calculate metrics per task
                metrics = {}
                for task_idx, task_name in enumerate(config.task_names):
                    # For multi-task, y_pred is (n_samples, n_tasks) - probabilities of positive class for each task
                    task_pred_proba = y_pred[:, task_idx]
                    task_metrics = calc_metrics(
                        y_valid[:, task_idx], task_pred_proba, metrics=config.metrics
                    )

                    for metric_name, score in task_metrics.items():
                        metrics[f"{task_name}_{metric_name}"] = score
                        logger.info(f"  {task_name} {metric_name}: {score:.6f}")

                # Calculate average metrics across tasks
                for metric_name in config.metrics:
                    task_scores = [
                        metrics[f"{task}_{metric_name}"] for task in config.task_names
                    ]
                    avg_score = np.mean(task_scores)
                    metrics[f"avg_{metric_name}"] = avg_score
                    logger.info(f"  Average {metric_name}: {avg_score:.6f}")

                return model, metrics

    # Build checkpoint directory
    checkpoint_dir = None
    if config.model.name == "chemprop" and output_dir is not None:
        task_str = "_".join(config.task_names)
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir) / task_str / f"fold_{fold_idx}")
        else:
            checkpoint_dir = str(output_dir / "checkpoints" / f"fold_{fold_idx}")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create model with n_tasks parameter
    model_params = config.model.params.copy()
    model_params["n_tasks"] = config.n_tasks
    model_params["random_seed"] = config.seed
    model_params["early_stopping_rounds"] = config.early_stopping_rounds
    model_params["early_stopping_metric"] = config.early_stopping_metric

    if checkpoint_dir is not None:
        model_params["checkpoint_dir"] = checkpoint_dir

    # Add imbalance handling parameters
    if config.imbalance.use_focal_loss:
        model_params["use_focal_loss"] = True
        model_params["focal_alpha"] = config.imbalance.focal_alpha
        model_params["focal_gamma"] = config.imbalance.focal_gamma

    model = ChemPropModel(**model_params)

    # Train model
    model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)

    # Predict on validation - returns (n_samples, n_tasks) for multi-task
    y_pred = model.predict_proba(X_valid)

    # Calculate metrics per task
    metrics = {}
    for task_idx, task_name in enumerate(config.task_names):
        # For multi-task, y_pred is (n_samples, n_tasks) - probabilities of positive class for each task
        task_pred_proba = y_pred[:, task_idx]
        task_metrics = calc_metrics(
            y_valid[:, task_idx], task_pred_proba, metrics=config.metrics
        )

        for metric_name, score in task_metrics.items():
            metrics[f"{task_name}_{metric_name}"] = score
            logger.info(f"  {task_name} {metric_name}: {score:.6f}")

    # Calculate average metrics across tasks
    for metric_name in config.metrics:
        task_scores = [
            metrics[f"{task}_{metric_name}"] for task in config.task_names
        ]
        avg_score = np.mean(task_scores)
        metrics[f"avg_{metric_name}"] = avg_score
        logger.info(f"  Average {metric_name}: {avg_score:.6f}")

    # Save model
    if output_dir:
        model_dir = output_dir / f"fold_{fold_idx}"
        model.save(str(model_dir))

    return model, metrics


def train_multitask_cv(
    features_path: str,
    splits_path: str,
    labels_df: pd.DataFrame,
    config: Config,
    output_dir: str,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """Train multi-task models with cross-validation.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels_df: DataFrame with labels for each task (columns = task names)
        config: Pipeline configuration
        output_dir: Directory to save models and metrics

    Returns:
        Tuple of (fold_metrics, best_iterations)
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Create output directory
    task_str = "_".join(config.task_names)
    output_path = Path(output_dir) / task_str / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Train each fold
    fold_metrics = []
    best_iterations = []

    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])

        # Get train/valid indices
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        # Convert positional indices to IDs
        train_ids = features.index[train_pos_indices]
        valid_ids = features.index[valid_pos_indices]

        # Get features
        X_train = features.loc[train_ids]
        X_valid = features.loc[valid_ids]

        # Verify SMILES column exists (required for ChemProp)
        if "SMILES" not in X_train.columns:
            raise ValueError(
                f"SMILES column not found in features. Available columns: {X_train.columns.tolist()}"
            )

        # Get labels - find common IDs between features and labels
        train_common = train_ids.intersection(labels_df.index)
        valid_common = valid_ids.intersection(labels_df.index)

        # Get multi-task labels as numpy array (n_samples, n_tasks)
        y_train = labels_df.loc[train_common, config.task_names].values
        y_valid = labels_df.loc[valid_common, config.task_names].values

        X_train = X_train.loc[train_common]
        X_valid = X_valid.loc[valid_common]

        # Train fold
        model, metrics = train_multitask_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config,
            fold_idx,
            output_dir=output_path,
        )

        fold_metrics.append(metrics)
        best_iterations.append(model.best_iteration)

    # Save fold metrics
    metrics_path = output_path / "cv_metrics.csv"
    save_fold_metrics(fold_metrics, str(metrics_path))

    # Log aggregated metrics
    logger.info("=" * 50)
    logger.info("Cross-validation results:")

    # Per-task metrics
    for task_name in config.task_names:
        logger.info(f"\nTask: {task_name}")
        for metric_name in config.metrics:
            key = f"{task_name}_{metric_name}"
            values = [m[key] for m in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")

    # Average metrics across tasks
    logger.info(f"\nAverage across tasks:")
    for metric_name in config.metrics:
        key = f"avg_{metric_name}"
        values = [m[key] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")

    avg_best_iter = int(np.mean(best_iterations))
    logger.info(f"\nAverage best iteration: {avg_best_iter}")
    logger.info("=" * 50)

    return fold_metrics, best_iterations


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-task ChemProp/Chemeleon model")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.parquet",
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/processed/splits.json",
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/multitask",
        help="Output directory for models",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check if ChemProp is available
    if not CHEMPROP_AVAILABLE:
        raise ImportError(
            "ChemProp is not available. Multi-task learning requires ChemProp."
        )

    # Check if multi-task configuration
    if not config.is_multitask:
        raise ValueError(
            "Configuration must specify multiple tasks using 'tasks' field. "
            f"Got: task={config.task}, tasks={config.tasks}"
        )

    logger.info(f"Starting multi-task training with {config.n_tasks} tasks")
    logger.info(f"Tasks: {config.task_names}")
    logger.info(f"Model: {config.model.name}")

    # Load multi-task labels
    labels_df, task_names = load_multitask_labels(config.task_names, args.data_dir)

    # Train with cross-validation
    fold_metrics, best_iterations = train_multitask_cv(
        args.features,
        args.splits,
        labels_df,
        config,
        args.output,
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

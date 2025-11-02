"""Inference script for multi-task learning with ChemProp/Chemeleon."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from euos25.config import Config, load_config
from euos25.models import ChemPropModel, CHEMPROP_AVAILABLE
from euos25.utils.io import load_json, load_parquet

logger = logging.getLogger(__name__)


def predict_multitask_oof(
    features_path: str,
    splits_path: str,
    model_dir: str,
    config: Config,
    output_dir: str,
) -> Dict[str, str]:
    """Generate out-of-fold predictions for multi-task model.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        model_dir: Directory containing trained models
        config: Pipeline configuration
        output_dir: Directory to save predictions

    Returns:
        Dictionary mapping task names to prediction file paths
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Task string for model directory
    task_str = "_".join(config.task_names)
    model_path = Path(model_dir) / task_str / config.model.name

    # Initialize predictions array (n_samples, n_tasks)
    n_samples = len(features)
    n_tasks = config.n_tasks
    predictions = np.zeros((n_samples, n_tasks))

    logger.info(f"Generating OOF predictions for {n_tasks} tasks")

    # Predict for each fold
    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])
        logger.info(f"Processing fold {fold_idx}")

        # Load model
        fold_model_dir = model_path / f"fold_{fold_idx}"
        model = ChemPropModel.load_from_checkpoint(
            str(fold_model_dir),
            n_tasks=config.n_tasks,
            **config.model.params,
        )

        # Get validation indices
        valid_pos_indices = fold_data["valid"]
        valid_ids = features.index[valid_pos_indices]
        X_valid = features.loc[valid_ids]

        # Predict - returns (n_samples, n_tasks)
        y_pred = model.predict_proba(X_valid)

        # Validate shape
        if y_pred.shape[0] != len(valid_pos_indices):
            raise ValueError(
                f"Shape mismatch: expected {len(valid_pos_indices)} samples, got {y_pred.shape[0]}. "
                f"Model may have incorrect n_tasks setting (expected {config.n_tasks})."
            )
        if y_pred.shape[1] != config.n_tasks:
            raise ValueError(
                f"Shape mismatch: expected {config.n_tasks} tasks, got {y_pred.shape[1]}. "
                f"Model may have incorrect n_tasks setting."
            )

        # Store predictions
        for i, idx in enumerate(valid_pos_indices):
            predictions[idx, :] = y_pred[i, :]

    # Save predictions for each task
    pred_files = {}
    for task_idx, task_name in enumerate(config.task_names):
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            "ID": features.index,
            "prediction": predictions[:, task_idx],
        })

        # Save to CSV
        pred_file = output_path / f"{task_name}_oof.csv"
        pred_df.to_csv(pred_file, index=False)
        pred_files[task_name] = str(pred_file)
        logger.info(f"  Saved {task_name} OOF predictions to: {pred_file}")

    return pred_files


def predict_multitask_test(
    features_path: str,
    model_dir: str,
    config: Config,
    output_dir: str,
    n_folds: int = 5,
) -> Dict[str, str]:
    """Generate test predictions for multi-task model.

    Args:
        features_path: Path to test features Parquet
        model_dir: Directory containing trained models
        config: Pipeline configuration
        output_dir: Directory to save predictions
        n_folds: Number of folds to average

    Returns:
        Dictionary mapping task names to prediction file paths
    """
    # Load features
    features = load_parquet(features_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Task string for model directory
    task_str = "_".join(config.task_names)
    model_path = Path(model_dir) / task_str / config.model.name

    # Initialize predictions array (n_samples, n_tasks)
    n_samples = len(features)
    n_tasks = config.n_tasks
    predictions = np.zeros((n_samples, n_tasks))

    logger.info(f"Generating test predictions for {n_tasks} tasks")
    logger.info(f"Averaging predictions from {n_folds} folds")

    # Predict with each fold and average
    for fold_idx in range(n_folds):
        logger.info(f"Processing fold {fold_idx}")

        # Load model
        fold_model_dir = model_path / f"fold_{fold_idx}"
        if not fold_model_dir.exists():
            logger.warning(f"Model directory not found: {fold_model_dir}, skipping")
            continue

        # Load model with n_tasks parameter to ensure correct multi-task handling
        model = ChemPropModel.load_from_checkpoint(
            str(fold_model_dir),
            n_tasks=config.n_tasks,
            **config.model.params,
        )

        # Predict - returns (n_samples, n_tasks)
        y_pred = model.predict_proba(features)

        # Validate shape
        if y_pred.shape[0] != n_samples:
            raise ValueError(
                f"Shape mismatch: expected {n_samples} samples, got {y_pred.shape[0]}. "
                f"Model may have incorrect n_tasks setting (expected {n_tasks})."
            )
        if y_pred.shape[1] != n_tasks:
            raise ValueError(
                f"Shape mismatch: expected {n_tasks} tasks, got {y_pred.shape[1]}. "
                f"Model may have incorrect n_tasks setting."
            )

        # Accumulate predictions
        predictions += y_pred

    # Average predictions
    predictions /= n_folds

    # Save predictions for each task
    pred_files = {}
    for task_idx, task_name in enumerate(config.task_names):
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            "ID": features.index,
            "prediction": predictions[:, task_idx],
        })

        # Save to CSV
        pred_file = output_path / f"{task_name}_test.csv"
        pred_df.to_csv(pred_file, index=False)
        pred_files[task_name] = str(pred_file)
        logger.info(f"  Saved {task_name} test predictions to: {pred_file}")

    return pred_files


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate predictions for multi-task model")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        help="Path to splits JSON file (required for OOF predictions)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["oof", "test"],
        required=True,
        help="Prediction mode: 'oof' or 'test'",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds (for test predictions)",
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
            "ChemProp is not available. Multi-task inference requires ChemProp."
        )

    # Check if multi-task configuration
    if not config.is_multitask:
        raise ValueError(
            "Configuration must specify multiple tasks using 'tasks' field. "
            f"Got: task={config.task}, tasks={config.tasks}"
        )

    logger.info(f"Starting multi-task inference with {config.n_tasks} tasks")
    logger.info(f"Tasks: {config.task_names}")
    logger.info(f"Mode: {args.mode}")

    # Generate predictions
    if args.mode == "oof":
        if not args.splits:
            raise ValueError("--splits is required for OOF predictions")

        pred_files = predict_multitask_oof(
            features_path=args.features,
            splits_path=args.splits,
            model_dir=args.model_dir,
            config=config,
            output_dir=args.output,
        )
    else:  # test
        pred_files = predict_multitask_test(
            features_path=args.features,
            model_dir=args.model_dir,
            config=config,
            output_dir=args.output,
            n_folds=args.n_folds,
        )

    logger.info("Inference completed successfully!")
    logger.info("Prediction files:")
    for task_name, file_path in pred_files.items():
        logger.info(f"  {task_name}: {file_path}")


if __name__ == "__main__":
    main()

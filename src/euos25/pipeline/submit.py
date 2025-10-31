"""Submission file generation."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from euos25.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def create_submission(
    predictions_path: str,
    output_path: str,
    id_col: str = "mol_id",
    pred_col: str = "prediction",
    submission_id_col: str = "ID",
    submission_pred_col: str = "prediction",
) -> pd.DataFrame:
    """Create submission file from predictions.

    Args:
        predictions_path: Path to predictions CSV
        output_path: Path to save submission CSV
        id_col: Column name for IDs in predictions
        pred_col: Column name for predictions
        submission_id_col: Column name for IDs in submission
        submission_pred_col: Column name for predictions in submission

    Returns:
        Submission DataFrame
    """
    logger.info(f"Creating submission from {predictions_path}")

    # Load predictions
    preds_df = load_csv(predictions_path)

    # Validate columns
    if id_col not in preds_df.columns:
        raise ValueError(f"Column {id_col} not found in predictions")
    if pred_col not in preds_df.columns:
        raise ValueError(f"Column {pred_col} not found in predictions")

    # Create submission
    submission = pd.DataFrame({
        submission_id_col: preds_df[id_col],
        submission_pred_col: preds_df[pred_col],
    })

    # Clip predictions to [0, 1]
    submission[submission_pred_col] = submission[submission_pred_col].clip(0, 1)

    # Save submission
    save_csv(submission, output_path, index=False)
    logger.info(f"Saved submission to {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Prediction stats: min={submission[submission_pred_col].min():.6f}, "
                f"max={submission[submission_pred_col].max():.6f}, "
                f"mean={submission[submission_pred_col].mean():.6f}")

    return submission


def generate_timestamped_submission(
    predictions_path: str,
    output_dir: str,
    task_name: str,
) -> str:
    """Generate submission with timestamp.

    Args:
        predictions_path: Path to predictions CSV
        output_dir: Directory to save submission
        task_name: Task name for filename

    Returns:
        Path to saved submission
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{timestamp}.csv"
    output_path = output_dir / filename

    # Create submission
    create_submission(predictions_path, str(output_path))

    return str(output_path)

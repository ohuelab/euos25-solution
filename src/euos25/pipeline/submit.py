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


def create_final_submission(
    trans_340_path: str,
    trans_450_path: str,
    fluo_480_path: str,
    fluo_340_450_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Create final submission file by combining all task predictions.

    The final submission should have the format:
    Transmittance(340),Transmittance(450),Fluorescence(340/480),Fluorescence(multiple)

    Args:
        trans_340_path: Path to trans_340 submission CSV (ID, prediction)
        trans_450_path: Path to trans_450 submission CSV (ID, prediction)
        fluo_480_path: Path to fluo_480 submission CSV (ID, prediction)
        fluo_340_450_path: Path to fluo_340_450 submission CSV (ID, prediction)
        output_path: Path to save final submission CSV

    Returns:
        Final submission DataFrame
    """
    logger.info("Creating final submission by combining all tasks")

    # Load all submission files
    trans_340_df = load_csv(trans_340_path)
    trans_450_df = load_csv(trans_450_path)
    fluo_480_df = load_csv(fluo_480_path)
    fluo_340_450_df = load_csv(fluo_340_450_path)

    # Validate columns
    for name, df in [
        ("trans_340", trans_340_df),
        ("trans_450", trans_450_df),
        ("fluo_480", fluo_480_df),
        ("fluo_340_450", fluo_340_450_df),
    ]:
        if "ID" not in df.columns:
            raise ValueError(f"Column 'ID' not found in {name}")
        if "prediction" not in df.columns:
            raise ValueError(f"Column 'prediction' not found in {name}")

    # Sort by ID to ensure consistent ordering
    trans_340_df = trans_340_df.sort_values("ID").reset_index(drop=True)
    trans_450_df = trans_450_df.sort_values("ID").reset_index(drop=True)
    fluo_480_df = fluo_480_df.sort_values("ID").reset_index(drop=True)
    fluo_340_450_df = fluo_340_450_df.sort_values("ID").reset_index(drop=True)

    # Verify all have the same IDs
    ids = trans_340_df["ID"].values
    if not (ids == trans_450_df["ID"].values).all():
        raise ValueError("trans_340 and trans_450 have different IDs")
    if not (ids == fluo_480_df["ID"].values).all():
        raise ValueError("trans_340 and fluo_480 have different IDs")
    if not (ids == fluo_340_450_df["ID"].values).all():
        raise ValueError("trans_340 and fluo_340_450 have different IDs")

    # Create final submission
    final_submission = pd.DataFrame({
        "Transmittance(340)": trans_340_df["prediction"].clip(0, 1),
        "Transmittance(450)": trans_450_df["prediction"].clip(0, 1),
        "Fluorescence(340/480)": fluo_480_df["prediction"].clip(0, 1),
        "Fluorescence(multiple)": fluo_340_450_df["prediction"].clip(0, 1),
    })

    # Save submission
    save_csv(final_submission, output_path, index=False)
    logger.info(f"Saved final submission to {output_path}")
    logger.info(f"Final submission shape: {final_submission.shape}")

    # Log statistics
    for col in final_submission.columns:
        logger.info(
            f"{col}: min={final_submission[col].min():.6f}, "
            f"max={final_submission[col].max():.6f}, "
            f"mean={final_submission[col].mean():.6f}"
        )

    return final_submission

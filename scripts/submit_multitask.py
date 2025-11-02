"""Submission script for multi-task learning predictions."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def create_multitask_submissions(
    pred_files: Dict[str, str],
    output_dir: str,
) -> Dict[str, str]:
    """Create submission files for each task.

    Args:
        pred_files: Dictionary mapping task names to prediction file paths
        output_dir: Directory to save submission files

    Returns:
        Dictionary mapping task names to submission file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    submission_files = {}

    for task_name, pred_file in pred_files.items():
        logger.info(f"Creating submission for task: {task_name}")

        # Load predictions
        pred_df = pd.read_csv(pred_file)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            "ID": pred_df["ID"],
            "Label": pred_df["prediction"],
        })

        # Save submission
        submission_file = output_path / f"{task_name}_submission.csv"
        submission_df.to_csv(submission_file, index=False)
        submission_files[task_name] = str(submission_file)
        logger.info(f"  Saved to: {submission_file}")

    return submission_files


def main():
    """Main submission function."""
    parser = argparse.ArgumentParser(
        description="Create submission files for multi-task predictions"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save submission files",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task names (e.g., transmittance340 transmittance450)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["oof", "test"],
        default="test",
        help="Prediction mode (default: test)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build prediction files dictionary
    pred_dir = Path(args.pred_dir)
    pred_files = {}
    for task_name in args.tasks:
        if args.mode == "oof":
            pred_file = pred_dir / f"{task_name}_oof.csv"
        else:
            pred_file = pred_dir / f"{task_name}_test.csv"

        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        pred_files[task_name] = str(pred_file)

    logger.info(f"Creating submissions for {len(pred_files)} tasks")
    logger.info(f"Tasks: {list(pred_files.keys())}")
    logger.info(f"Mode: {args.mode}")

    # Create submissions
    submission_files = create_multitask_submissions(
        pred_files=pred_files,
        output_dir=args.output_dir,
    )

    logger.info("Submission creation completed successfully!")
    logger.info("Submission files:")
    for task_name, file_path in submission_files.items():
        logger.info(f"  {task_name}: {file_path}")


if __name__ == "__main__":
    main()

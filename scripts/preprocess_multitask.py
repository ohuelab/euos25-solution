"""Preprocessing script for multi-task learning.

This script prepares data for multi-task learning by combining multiple task datasets
and ensuring data consistency across tasks.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from euos25.pipeline.prepare import prepare_data

logger = logging.getLogger(__name__)


def preprocess_multitask(
    task_files: Dict[str, str],
    output_dir: str,
    normalize: bool = True,
    deduplicate: bool = True,
) -> Dict[str, str]:
    """Preprocess data for multiple tasks.

    Args:
        task_files: Dictionary mapping task names to raw CSV file paths
        output_dir: Directory to save prepared files
        normalize: Whether to normalize SMILES
        deduplicate: Whether to remove duplicates

    Returns:
        Dictionary mapping task names to prepared file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prepared_files = {}

    for task_name, input_file in task_files.items():
        logger.info(f"Preparing data for task: {task_name}")

        # Output file
        output_file = output_path / f"train_{task_name}_prepared.csv"

        # Prepare data
        prepare_data(
            input_path=input_file,
            output_path=str(output_file),
            remove_duplicates=deduplicate,
            normalize=normalize,
        )

        prepared_files[task_name] = str(output_file)
        logger.info(f"  Saved to: {output_file}")

    return prepared_files


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description="Preprocess data for multi-task learning"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task names (e.g., transmittance340 transmittance450)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save prepared files",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable SMILES normalization",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Task name to file mapping
    task_to_file = {
        "transmittance340": "euos25_challenge_train_transmittance340.csv",
        "transmittance450": "euos25_challenge_train_transmittance450.csv",
        "fluorescence340_450": "euos25_challenge_train_fluorescence340_450.csv",
        "fluorescence480": "euos25_challenge_train_fluorescence480.csv",
    }

    # Build task files dictionary
    input_dir = Path(args.input_dir)
    task_files = {}
    for task_name in args.tasks:
        if task_name not in task_to_file:
            raise ValueError(f"Unknown task: {task_name}")
        task_files[task_name] = str(input_dir / task_to_file[task_name])

    logger.info(f"Preprocessing {len(task_files)} tasks")
    logger.info(f"Tasks: {list(task_files.keys())}")

    # Preprocess
    prepared_files = preprocess_multitask(
        task_files=task_files,
        output_dir=args.output_dir,
        normalize=not args.no_normalize,
        deduplicate=not args.no_deduplicate,
    )

    logger.info("Preprocessing completed successfully!")
    logger.info("Prepared files:")
    for task_name, file_path in prepared_files.items():
        logger.info(f"  {task_name}: {file_path}")


if __name__ == "__main__":
    main()

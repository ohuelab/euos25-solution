"""Split generation script for multi-task learning.

This script creates scaffold-based K-fold splits for multi-task learning,
ensuring samples are consistent across tasks.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from euos25.data.splits import create_scaffold_splits, splits_to_serializable
from euos25.utils.io import load_csv, save_json
from euos25.utils.seed import set_seed

logger = logging.getLogger(__name__)


def create_multitask_splits(
    prepared_files: Dict[str, str],
    output_dir: str,
    folds: int = 5,
    seed: int = 42,
    scaffold_min_size: int = 10,
) -> Dict[str, str]:
    """Create scaffold splits for multi-task learning.

    The splits are created based on the intersection of samples across all tasks.
    This ensures that the same samples are used for training/validation across tasks.

    Args:
        prepared_files: Dictionary mapping task names to prepared CSV file paths
        output_dir: Directory to save split files
        folds: Number of folds
        seed: Random seed
        scaffold_min_size: Minimum scaffold group size

    Returns:
        Dictionary mapping task names to split file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all task data
    task_dfs = {}
    for task_name, file_path in prepared_files.items():
        logger.info(f"Loading data for task: {task_name}")
        df = load_csv(file_path)
        task_dfs[task_name] = df
        logger.info(f"  Loaded {len(df)} samples")

    # Find common sample IDs across all tasks
    common_ids: Set[str] = None
    for task_name, df in task_dfs.items():
        ids = set(df["ID"].values)
        if common_ids is None:
            common_ids = ids
        else:
            common_ids &= ids

    logger.info(f"Found {len(common_ids)} common samples across all tasks")

    # Filter each task to common IDs
    filtered_dfs = {}
    for task_name, df in task_dfs.items():
        filtered_df = df[df["ID"].isin(common_ids)].reset_index(drop=True)
        filtered_dfs[task_name] = filtered_df
        logger.info(f"Task {task_name}: {len(filtered_df)} samples after filtering")

    # Create splits based on the first task (they should all have the same samples)
    first_task = list(filtered_dfs.keys())[0]
    first_df = filtered_dfs[first_task]

    logger.info(f"Creating scaffold splits based on task: {first_task}")
    set_seed(seed)

    # Determine label column
    label_col = None
    if "Transmittance (qualitative)" in first_df.columns:
        label_col = "Transmittance (qualitative)"
    elif "Fluorescence (qualitative)" in first_df.columns:
        label_col = "Fluorescence (qualitative)"

    splits = create_scaffold_splits(
        df=first_df,
        smiles_col="SMILES",
        label_col=label_col,
        n_splits=folds,
        scaffold_min_size=scaffold_min_size,
        seed=seed,
    )

    # Save splits for each task
    split_files = {}
    for task_name in filtered_dfs.keys():
        split_file = output_path / f"splits_{task_name}.json"
        serializable_splits = splits_to_serializable(splits)
        save_json(serializable_splits, str(split_file))
        split_files[task_name] = str(split_file)
        logger.info(f"  Saved splits for {task_name} to: {split_file}")

    return split_files


def main():
    """Main split generation function."""
    parser = argparse.ArgumentParser(
        description="Create scaffold splits for multi-task learning"
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
        default="data/processed",
        help="Directory containing prepared data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save split files",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--scaffold-min-size",
        type=int,
        default=10,
        help="Minimum scaffold group size",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build prepared files dictionary
    input_dir = Path(args.input_dir)
    prepared_files = {}
    for task_name in args.tasks:
        prepared_file = input_dir / f"train_{task_name}_prepared.csv"
        if not prepared_file.exists():
            raise FileNotFoundError(f"Prepared file not found: {prepared_file}")
        prepared_files[task_name] = str(prepared_file)

    logger.info(f"Creating splits for {len(prepared_files)} tasks")
    logger.info(f"Tasks: {list(prepared_files.keys())}")

    # Create splits
    split_files = create_multitask_splits(
        prepared_files=prepared_files,
        output_dir=args.output_dir,
        folds=args.folds,
        seed=args.seed,
        scaffold_min_size=args.scaffold_min_size,
    )

    logger.info("Split generation completed successfully!")
    logger.info("Split files:")
    for task_name, file_path in split_files.items():
        logger.info(f"  {task_name}: {file_path}")


if __name__ == "__main__":
    main()

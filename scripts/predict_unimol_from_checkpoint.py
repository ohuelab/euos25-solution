#!/usr/bin/env python3
"""Predict test data using UniMol checkpoint (last epoch).

This script loads a UniMol checkpoint from the last epoch and generates predictions
for test data. It supports both checkpoint files and directories containing checkpoints.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from euos25.config import load_config
from euos25.models.unimol import UniMolModel, UNIMOL_AVAILABLE
from euos25.utils.io import load_csv, save_csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def find_last_checkpoint(checkpoint_path: Path) -> Path:
    """Find the last checkpoint file in a directory.

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        Path to the last checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint file is found
    """
    if checkpoint_path.is_file():
        return checkpoint_path

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Look for last.ckpt first (Lightning's save_last=True)
    last_ckpt = checkpoint_path / "last.ckpt"
    if last_ckpt.exists():
        logger.info(f"Found last checkpoint: {last_ckpt}")
        return last_ckpt

    # If no last.ckpt, find all .ckpt files and use the most recent one
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

    # Sort by modification time (most recent first)
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    logger.info(f"Found {len(ckpt_files)} checkpoint files, using most recent: {ckpt_files[0]}")
    return ckpt_files[0]


def predict_from_checkpoint(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 32,
    smiles_column: str = "SMILES",
    id_column: str = "ID",
) -> pd.DataFrame:
    """Predict test data using a UniMol checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file or directory
        test_data_path: Path to test data CSV file (must contain SMILES column)
        output_path: Path to save predictions CSV
        config_path: Optional path to config YAML file (for model parameters)
        batch_size: Batch size for prediction
        smiles_column: Name of SMILES column in test data
        id_column: Name of ID column in test data (for output)

    Returns:
        DataFrame with predictions
    """
    if not UNIMOL_AVAILABLE:
        raise ImportError(
            "UniMol is not available. Please install unimol_tools package: "
            "pip install unimol_tools"
        )

    # Find checkpoint file
    checkpoint_file = find_last_checkpoint(Path(checkpoint_path))
    logger.info(f"Loading checkpoint from: {checkpoint_file}")

    # Load config if provided
    config = None
    if config_path:
        logger.info(f"Loading config from: {config_path}")
        config = load_config(config_path)
        logger.info(f"Config loaded: model={config.model.name}, task={config.task}")

    # Load test data
    logger.info(f"Loading test data from: {test_data_path}")
    test_df = load_csv(test_data_path)

    # Check for SMILES column
    if smiles_column not in test_df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in test data. "
            f"Available columns: {list(test_df.columns)}"
        )

    # Prepare features DataFrame (UniMol expects DataFrame with SMILES column)
    features_df = pd.DataFrame({"SMILES": test_df[smiles_column]})

    # Set index to ID column if available
    if id_column in test_df.columns:
        features_df.index = test_df[id_column]
        features_df.index.name = id_column
    elif test_df.index.name and test_df.index.name in [id_column, "ID", "mol_id"]:
        features_df.index = test_df.index
    else:
        # Use default index
        features_df.index = test_df.index if test_df.index.name else range(len(test_df))
        logger.warning("No ID column found, using default index")

    logger.info(f"Test data loaded: {len(features_df)} samples")

    # Load model from checkpoint
    logger.info("Loading UniMol model from checkpoint...")
    model_params = {}
    if config:
        model_params = config.model.params.copy()
        # Override batch_size if specified
        if batch_size is not None:
            model_params["batch_size"] = batch_size

    # Load model
    model = UniMolModel.load_from_checkpoint(
        str(checkpoint_file),
        **model_params,
    )

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict_proba(features_df)

    # Handle output shape
    # UniMol returns (n_samples, 2) for single task, extract positive class probabilities
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        pred_probs = predictions[:, 1]
    elif predictions.ndim == 1:
        pred_probs = predictions
    else:
        # Multi-task case
        pred_probs = predictions

    # Create output DataFrame
    output_df = pd.DataFrame({
        id_column: features_df.index,
        "prediction": pred_probs,
    })

    # Save predictions
    logger.info(f"Saving predictions to: {output_path}")
    save_csv(output_df, output_path, index=False)
    logger.info(f"Predictions saved: {len(output_df)} samples")

    return output_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predict test data using UniMol checkpoint (last epoch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to checkpoint file or directory containing checkpoints",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        type=str,
        help="Path to test data CSV file (must contain SMILES column)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config YAML file (for model parameters)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Name of SMILES column in test data",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="ID",
        help="Name of ID column in test data (for output)",
    )

    args = parser.parse_args()

    try:
        predict_from_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            output_path=args.output,
            config_path=args.config,
            batch_size=args.batch_size,
            smiles_column=args.smiles_column,
            id_column=args.id_column,
        )
        logger.info("Prediction completed successfully!")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


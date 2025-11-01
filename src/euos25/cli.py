"""Command-line interface for EUOS25 pipeline."""

import logging
import sys
from pathlib import Path

import click
import pandas as pd

from euos25.config import load_config
from euos25.data.splits import create_scaffold_splits, splits_to_serializable
from euos25.pipeline.ensemble import blend_predictions, ensemble_from_directory
from euos25.pipeline.features import build_features_from_config
from euos25.pipeline.infer import predict_oof, predict_test
from euos25.pipeline.prepare import prepare_data
from euos25.pipeline.submit import create_submission, create_final_submission, generate_timestamped_submission
from euos25.pipeline.train import train_cv
from euos25.utils.io import load_csv, save_json
from euos25.utils.seed import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """EUOS25 Pipeline - Absorption/Fluorescence Classification."""
    pass


@cli.command()
@click.option("--input", required=True, help="Input CSV file")
@click.option("--output", required=True, help="Output splits JSON file")
@click.option("--folds", default=5, help="Number of folds")
@click.option("--seed", default=42, help="Random seed")
@click.option("--scaffold-min-size", default=10, help="Minimum scaffold group size")
@click.option("--label-col", default=None, help="Label column for balancing")
def make_splits(input, output, folds, seed, scaffold_min_size, label_col):
    """Create scaffold-based K-fold splits."""
    logger.info("Creating scaffold splits")

    # Set seed
    set_seed(seed)

    # Load data
    df = load_csv(input)

    # Create splits
    splits = create_scaffold_splits(
        df,
        smiles_col="SMILES",
        label_col=label_col,
        n_splits=folds,
        scaffold_min_size=scaffold_min_size,
        seed=seed,
    )

    # Save splits
    serializable_splits = splits_to_serializable(splits)
    save_json(serializable_splits, output)

    logger.info(f"Saved splits to {output}")


@cli.command()
@click.option("--input", required=True, help="Input CSV file")
@click.option("--output", required=True, help="Output features Parquet file")
@click.option("--config", required=True, help="Configuration YAML file")
def build_features(input, output, config):
    """Build features from SMILES."""
    logger.info("Building features")

    # Load config
    cfg = load_config(config)

    # Set seed
    set_seed(cfg.seed)

    # Build features
    build_features_from_config(input, output, cfg)

    logger.info(f"Saved features to {output}")


@cli.command()
@click.option("--features", required=True, help="Features Parquet file")
@click.option("--splits", required=True, help="Splits JSON file")
@click.option("--config", required=True, help="Configuration YAML file")
@click.option("--outdir", required=True, help="Output directory for models")
@click.option("--label-col", default=None, help="Label column name")
@click.option("--data", default=None, help="Data CSV file with labels")
@click.option("--task", default=None, help="Task name override (e.g., 'y_fluo_any', 'y_trans_any')")
def train(features, splits, config, outdir, label_col, data, task):
    """Train models with cross-validation."""
    logger.info("Training models")

    # Load config
    cfg = load_config(config)

    # Override task name if provided
    task_name = task if task is not None else cfg.task

    # Set seed
    set_seed(cfg.seed)

    # Load labels
    if data:
        df = load_csv(data)
    else:
        # Try to infer from task name
        logger.info("No data file provided, attempting to load from raw data")
        df = load_csv("data/raw/train.csv")

    # Get label column
    if label_col is None:
        # Infer from task name
        task_to_col = {
            "y_abs_340": "Transmittance",
            "y_trans_any": "Transmittance",
            "y_fluo_any": "Fluorescence",
            "y_fluo_340_450": "Fluorescence",
        }
        label_col = task_to_col.get(task_name, "Fluorescence")

    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found")

    labels = df.set_index("ID")[label_col]

    # Train with CV
    fold_metrics = train_cv(
        features_path=features,
        splits_path=splits,
        labels=labels,
        config=cfg,
        output_dir=outdir,
        task_name=task_name,
    )

    logger.info("Training completed")


@cli.command()
@click.option("--features", required=True, help="Features Parquet file")
@click.option("--splits", required=True, help="Splits JSON file")
@click.option("--config", required=True, help="Configuration YAML file")
@click.option("--model-dir", required=True, help="Directory with trained models")
@click.option("--outdir", required=True, help="Output directory for predictions")
@click.option("--mode", default="oof", help="Prediction mode: 'oof' or 'test'")
@click.option("--task", default=None, help="Task name override (e.g., 'y_fluo_any', 'y_trans_any')")
def infer(features, splits, config, model_dir, outdir, mode, task):
    """Generate predictions."""
    logger.info(f"Generating {mode} predictions")

    # Load config
    cfg = load_config(config)

    # Override task name if provided
    task_name = task if task is not None else cfg.task

    # Set seed
    set_seed(cfg.seed)

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if mode == "oof":
        output_path = Path(outdir) / f"{task_name}_oof.csv"
        predict_oof(
            features_path=features,
            splits_path=splits,
            model_dir=model_dir,
            config=cfg,
            output_path=str(output_path),
            task_name=task_name,
        )
    elif mode == "test":
        output_path = Path(outdir) / f"{task_name}_test.csv"
        predict_test(
            features_path=features,
            model_dir=model_dir,
            config=cfg,
            output_path=str(output_path),
            task_name=task_name,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info(f"Saved predictions to {output_path}")


@cli.command()
@click.option("--pred-dir", required=True, help="Directory with prediction files")
@click.option("--out", required=True, help="Output blended predictions CSV")
@click.option("--pattern", default="*.csv", help="Glob pattern for prediction files")
@click.option("--method", default="rank_average", help="Blending method")
def ensemble(pred_dir, out, pattern, method):
    """Ensemble predictions using rank averaging."""
    logger.info("Ensembling predictions")

    ensemble_from_directory(
        pred_dir=pred_dir,
        output_path=out,
        pattern=pattern,
        method=method,
    )

    logger.info(f"Saved ensemble predictions to {out}")


@cli.command()
@click.option("--pred", required=True, help="Predictions CSV file")
@click.option("--out", required=True, help="Output submission CSV file")
def submit(pred, out):
    """Create submission file."""
    logger.info("Creating submission")

    create_submission(
        predictions_path=pred,
        output_path=out,
    )

    logger.info(f"Saved submission to {out}")


@cli.command()
@click.option("--trans-340", required=True, help="trans_340 submission CSV file")
@click.option("--trans-450", required=True, help="trans_450 submission CSV file")
@click.option("--fluo-480", required=True, help="fluo_480 submission CSV file")
@click.option("--fluo-340-450", required=True, help="fluo_340_450 submission CSV file")
@click.option("--out", required=True, help="Output final submission CSV file")
def submit_final(trans_340, trans_450, fluo_480, fluo_340_450, out):
    """Create final submission file by combining all task submissions."""
    logger.info("Creating final submission")

    create_final_submission(
        trans_340_path=trans_340,
        trans_450_path=trans_450,
        fluo_480_path=fluo_480,
        fluo_340_450_path=fluo_340_450,
        output_path=out,
    )

    logger.info(f"Saved final submission to {out}")


@cli.command()
@click.option("--input", required=True, help="Input CSV file")
@click.option("--output", required=True, help="Output prepared CSV file")
@click.option("--normalize/--no-normalize", default=True, help="Normalize SMILES")
@click.option("--deduplicate/--no-deduplicate", default=True, help="Remove duplicates")
def prepare(input, output, normalize, deduplicate):
    """Prepare and clean data."""
    logger.info("Preparing data")

    prepare_data(
        input_path=input,
        output_path=output,
        remove_duplicates=deduplicate,
        normalize=normalize,
    )

    logger.info(f"Saved prepared data to {output}")


if __name__ == "__main__":
    cli()

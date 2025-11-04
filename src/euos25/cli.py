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
from euos25.pipeline.optuna_tuning import tune_hyperparameters
from euos25.pipeline.prepare import prepare_data
from euos25.pipeline.submit import create_submission, create_final_submission, generate_timestamped_submission
from euos25.pipeline.features import FEATURE_GROUP_MAPPING
from euos25.pipeline.train import train_cv, train_full
from euos25.utils.io import load_csv, load_json, save_json
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

    # Check if using quantitative values
    use_quantitative = cfg.imbalance.use_quantitative if cfg.imbalance else False
    quantitative_normalize = cfg.imbalance.quantitative_normalize if cfg.imbalance else True

    if use_quantitative and label_col == "Transmittance":
        # Load quantitative column
        quantitative_col = "Transmittance.1"
        if quantitative_col not in df.columns:
            raise ValueError(f"Quantitative column {quantitative_col} not found")

        # Get quantitative values
        quantitative_values = df.set_index("ID")[quantitative_col]

        # Normalize quantitative values to 0-1 range
        if quantitative_normalize:
            q_min = quantitative_values.min()
            q_max = quantitative_values.max()
            if q_max > q_min:
                labels = (quantitative_values - q_min) / (q_max - q_min)
                logger.info(f"Normalized quantitative values: min={q_min:.4f}, max={q_max:.4f}")
            else:
                labels = quantitative_values
                logger.warning("Quantitative values are constant, using as-is")
        else:
            labels = quantitative_values

        # Store binary labels for ROC-AUC calculation
        binary_labels = df.set_index("ID")[label_col]
        # Store binary labels in config for later use
        cfg._binary_labels = binary_labels
    else:
        labels = df.set_index("ID")[label_col]
        cfg._binary_labels = None

    # Check if Optuna mode is enabled
    feature_group_settings = None
    if cfg.optuna.enable:
        logger.info("Optuna mode enabled - running hyperparameter tuning")
        best_params = tune_hyperparameters(
            features_path=features,
            splits_path=splits,
            labels=labels,
            config=cfg,
            output_dir=outdir,
            task_name=task_name,
        )

        # Extract feature group settings from best_params (if feature groups were tuned)
        if cfg.optuna.feature_groups.get("tune", False):
            valid_feature_groups = set(FEATURE_GROUP_MAPPING.values())
            feature_group_settings = {}
            for key, value in best_params.items():
                if key.startswith("use_"):
                    group_name = key[4:]  # Remove "use_" prefix
                    # Only include if it's an actual feature group name (not e.g., "focal_loss")
                    if group_name in valid_feature_groups:
                        feature_group_settings[group_name] = value
            if feature_group_settings:
                logger.info(f"Extracted feature group settings from Optuna: {feature_group_settings}")

        # Update config with best parameters
        logger.info("Updating config with best parameters from Optuna")
        for key, value in best_params.items():
            if key.startswith("use_"):
                # Skip feature group params - handled separately
                continue
            elif key in ["focal_alpha", "focal_gamma"]:
                setattr(cfg.imbalance, key, value)
            elif key == "pos_weight_multiplier":
                # Store multiplier for later use
                cfg.imbalance.pos_weight_multiplier = value
            else:
                cfg.model.params[key] = value

    # Train with CV
    fold_metrics, best_iterations, train_sizes = train_cv(
        features_path=features,
        splits_path=splits,
        labels=labels,
        config=cfg,
        output_dir=outdir,
        task_name=task_name,
    )

    # Train on full dataset
    logger.info("Training on full dataset")
    train_full(
        features_path=features,
        labels=labels,
        config=cfg,
        output_dir=outdir,
        best_iterations=best_iterations,
        train_sizes=train_sizes,
        task_name=task_name,
        feature_group_settings=feature_group_settings,
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

    # Try to load Optuna best_params if Optuna was used
    feature_group_settings = None
    if cfg.optuna.enable:
        # Try to load best_params.json from optuna output directory
        optuna_output_dir = Path(model_dir) / task_name / cfg.model.name / "optuna"
        best_params_path = optuna_output_dir / "best_params.json"

        if best_params_path.exists():
            logger.info(f"Loading Optuna best parameters from {best_params_path}")
            optuna_results = load_json(best_params_path)
            best_params = optuna_results.get("best_params", {})

            # Extract feature group settings if feature groups were tuned
            if cfg.optuna.feature_groups.get("tune", False):
                valid_feature_groups = set(FEATURE_GROUP_MAPPING.values())
                feature_group_settings = {}
                for key, value in best_params.items():
                    if key.startswith("use_"):
                        group_name = key[4:]  # Remove "use_" prefix
                        # Only include if it's an actual feature group name (not e.g., "focal_loss")
                        if group_name in valid_feature_groups:
                            feature_group_settings[group_name] = value
                if feature_group_settings:
                    logger.info(f"Using Optuna-optimized feature groups: {feature_group_settings}")

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
            feature_group_settings=feature_group_settings,
        )
    elif mode == "test":
        output_path = Path(outdir) / f"{task_name}_test.csv"
        predict_test(
            features_path=features,
            model_dir=model_dir,
            config=cfg,
            output_path=str(output_path),
            task_name=task_name,
            feature_group_settings=feature_group_settings,
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

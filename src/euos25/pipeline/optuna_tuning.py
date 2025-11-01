"""Optuna hyperparameter tuning for LGBM models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from euos25.config import Config
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics

logger = logging.getLogger(__name__)


def suggest_lgbm_params(trial: optuna.Trial, config: Config) -> Dict[str, Any]:
    """Suggest LGBM hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial
        config: Pipeline configuration

    Returns:
        Dictionary of suggested parameters
    """
    params = {}

    # Core LGBM parameters
    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        config.optuna.learning_rate_min,
        config.optuna.learning_rate_max,
        log=True,
    )
    params["num_leaves"] = trial.suggest_int(
        "num_leaves",
        config.optuna.num_leaves_min,
        config.optuna.num_leaves_max,
    )
    params["max_depth"] = trial.suggest_int(
        "max_depth",
        config.optuna.max_depth_min,
        config.optuna.max_depth_max,
    )
    params["subsample"] = trial.suggest_float(
        "subsample",
        config.optuna.subsample_min,
        config.optuna.subsample_max,
    )
    params["colsample_bytree"] = trial.suggest_float(
        "colsample_bytree",
        config.optuna.colsample_bytree_min,
        config.optuna.colsample_bytree_max,
    )
    params["min_child_samples"] = trial.suggest_int(
        "min_child_samples",
        config.optuna.min_child_samples_min,
        config.optuna.min_child_samples_max,
    )
    params["reg_alpha"] = trial.suggest_float(
        "reg_alpha",
        config.optuna.reg_alpha_min,
        config.optuna.reg_alpha_max,
    )
    params["reg_lambda"] = trial.suggest_float(
        "reg_lambda",
        config.optuna.reg_lambda_min,
        config.optuna.reg_lambda_max,
    )

    # Fixed parameters from config
    params["n_estimators"] = config.model.params.get("n_estimators", 1000)
    params["early_stopping_rounds"] = config.early_stopping_rounds

    return params


def suggest_imbalance_params(trial: optuna.Trial, config: Config) -> Dict[str, Any]:
    """Suggest imbalance handling parameters for Optuna trial.

    Args:
        trial: Optuna trial
        config: Pipeline configuration

    Returns:
        Dictionary of suggested imbalance parameters
    """
    params = {}

    if config.imbalance.use_focal_loss:
        # Focal loss parameters
        params["use_focal_loss"] = True
        params["focal_alpha"] = trial.suggest_float(
            "focal_alpha",
            config.optuna.focal_alpha_min,
            config.optuna.focal_alpha_max,
        )
        params["focal_gamma"] = trial.suggest_float(
            "focal_gamma",
            config.optuna.focal_gamma_min,
            config.optuna.focal_gamma_max,
        )
        params["pos_weight"] = None
    elif config.imbalance.use_pos_weight:
        # pos_weight multiplier (will be multiplied by computed pos_weight)
        params["use_focal_loss"] = False
        params["pos_weight_multiplier"] = trial.suggest_float(
            "pos_weight_multiplier",
            config.optuna.pos_weight_multiplier_min,
            config.optuna.pos_weight_multiplier_max,
        )
        params["pos_weight"] = None  # Will be computed and multiplied
    else:
        params["use_focal_loss"] = False
        params["pos_weight"] = None

    return params


def objective(
    trial: optuna.Trial,
    features: pd.DataFrame,
    splits: Dict,
    labels: pd.Series,
    config: Config,
) -> float:
    """Optuna objective function for hyperparameter tuning.

    Args:
        trial: Optuna trial
        features: Feature dataframe
        splits: CV splits dictionary
        labels: Label series
        config: Pipeline configuration

    Returns:
        Average validation score across folds
    """
    # Suggest parameters
    lgbm_params = suggest_lgbm_params(trial, config)
    imbalance_params = suggest_imbalance_params(trial, config)

    # Combine parameters
    all_params = {**lgbm_params, **imbalance_params}

    # Run CV with these parameters
    fold_scores = []

    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])

        # Get train/valid indices
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        # Convert to IDs
        train_ids = features.index[train_pos_indices]
        valid_ids = features.index[valid_pos_indices]

        # Get features and labels
        X_train = features.loc[train_ids]
        y_train = labels.loc[train_ids].values

        X_valid = features.loc[valid_ids]
        y_valid = labels.loc[valid_ids].values

        # Handle pos_weight_multiplier
        if "pos_weight_multiplier" in all_params:
            multiplier = all_params.pop("pos_weight_multiplier")
            # Compute base pos_weight
            n_pos = np.sum(y_train)
            n_neg = len(y_train) - n_pos
            base_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            all_params["pos_weight"] = base_pos_weight * multiplier

        # Create and train model
        model = LGBMClassifier(**all_params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        # Predict and evaluate
        y_pred = model.predict_proba(X_valid)
        metrics = calc_metrics(y_valid, y_pred, metrics=config.metrics)

        # Use early_stopping_metric as objective
        score = metrics[config.early_stopping_metric]
        fold_scores.append(score)

        logger.debug(
            f"Trial {trial.number}, Fold {fold_idx}, "
            f"{config.early_stopping_metric}: {score:.6f}"
        )

    # Return mean score across folds
    mean_score = np.mean(fold_scores)
    return mean_score


def tune_hyperparameters(
    features_path: str,
    splits_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Tune hyperparameters using Optuna.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels: Series with labels
        config: Pipeline configuration
        output_dir: Directory to save results
        task_name: Task name override

    Returns:
        Best parameters dictionary
    """
    if not config.optuna.enable:
        raise ValueError("Optuna is not enabled in config")

    logger.info("=" * 50)
    logger.info("Starting Optuna hyperparameter tuning")
    logger.info(f"  n_trials: {config.optuna.n_trials}")
    logger.info(f"  timeout: {config.optuna.timeout}")
    logger.info(f"  Optimizing: {config.early_stopping_metric}")

    # Load data
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Use task_name override if provided
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name / "optuna"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create study
    study_name = config.optuna.study_name or f"{actual_task_name}_tuning"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, features, splits, labels, config),
        n_trials=config.optuna.n_trials,
        timeout=config.optuna.timeout,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    logger.info("=" * 50)
    logger.info("Optuna tuning completed")
    logger.info(f"  Best {config.early_stopping_metric}: {best_score:.6f}")
    logger.info("  Best parameters:")
    for key, value in best_params.items():
        logger.info(f"    {key}: {value}")

    # Save results
    results = {
        "best_params": best_params,
        "best_score": best_score,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }

    results_path = output_path / "best_params.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"  Saved best parameters to {results_path}")

    # Save study
    study_path = output_path / "study.pkl"
    import pickle

    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    logger.info(f"  Saved study to {study_path}")
    logger.info("=" * 50)

    return best_params

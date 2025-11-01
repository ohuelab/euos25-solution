"""Optuna hyperparameter tuning for LGBM models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd

from euos25.config import Config
from euos25.models.lgbm import LGBMClassifier
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics

logger = logging.getLogger(__name__)


# Default values for parameters if not in config
DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "focal_scale": 100.0,
    "pos_weight_multiplier": 1.0,
}


def suggest_param(
    trial: optuna.Trial, param_name: str, param_config: Dict[str, Any]
) -> Any:
    """Suggest a single parameter value based on config.

    Args:
        trial: Optuna trial
        param_name: Name of the parameter
        param_config: Parameter configuration dict

    Returns:
        Suggested parameter value
    """
    param_type = param_config["type"]

    if param_type == "float":
        return trial.suggest_float(
            param_name,
            param_config["min"],
            param_config["max"],
            log=param_config.get("log", False),
        )
    elif param_type == "int":
        return trial.suggest_int(
            param_name,
            int(param_config["min"]),
            int(param_config["max"]),
            log=param_config.get("log", False),
        )
    elif param_type == "categorical":
        return trial.suggest_categorical(param_name, param_config["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_fixed_value(param_name: str, config: Config) -> Any:
    """Get fixed parameter value from config or defaults.

    Args:
        param_name: Name of the parameter
        config: Pipeline configuration

    Returns:
        Fixed parameter value
    """
    # Check model params first
    if param_name in config.model.params:
        return config.model.params[param_name]

    # Check imbalance params
    if hasattr(config.imbalance, param_name):
        value = getattr(config.imbalance, param_name)
        if value is not None:
            return value

    # Fall back to defaults
    if param_name in DEFAULT_PARAMS:
        return DEFAULT_PARAMS[param_name]

    raise ValueError(f"No fixed value found for parameter: {param_name}")


def suggest_all_params(trial: optuna.Trial, config: Config) -> Dict[str, Any]:
    """Suggest all hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial
        config: Pipeline configuration

    Returns:
        Dictionary of all parameters (tuned and fixed)
    """
    params = {}

    # Suggest LGBM parameters (tuned ones from config.optuna.lgbm_params)
    lgbm_param_names = [
        "learning_rate",
        "num_leaves",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
    ]
    for param_name in lgbm_param_names:
        if param_name in config.optuna.lgbm_params:
            # Tune this parameter
            params[param_name] = suggest_param(
                trial, param_name, config.optuna.lgbm_params[param_name]
            )
        else:
            # Use fixed value
            params[param_name] = get_fixed_value(param_name, config)

    # Check if use_pos_weight should be tuned
    if "use_pos_weight" in config.optuna.imbalance_params:
        use_pos_weight = suggest_param(
            trial, "use_pos_weight", config.optuna.imbalance_params["use_pos_weight"]
        )
    else:
        use_pos_weight = config.imbalance.use_pos_weight

    # Imbalance handling parameters based on use_pos_weight
    if not use_pos_weight:
        # Use focal loss
        params["use_focal_loss"] = True

        # Suggest focal loss parameters (tuned ones from config.optuna.focal_params)
        focal_param_names = ["focal_alpha", "focal_gamma", "focal_scale"]
        for param_name in focal_param_names:
            if param_name in config.optuna.focal_params:
                # Tune this parameter
                params[param_name] = suggest_param(
                    trial, param_name, config.optuna.focal_params[param_name]
                )
            else:
                # Use fixed value
                params[param_name] = get_fixed_value(param_name, config)

        params["pos_weight"] = None

    else:
        # Use pos_weight
        params["use_focal_loss"] = False

        # Suggest pos_weight parameters (tuned ones from config.optuna.pos_weight_params)
        if "pos_weight_multiplier" in config.optuna.pos_weight_params:
            # Tune this parameter
            params["pos_weight_multiplier"] = suggest_param(
                trial,
                "pos_weight_multiplier",
                config.optuna.pos_weight_params["pos_weight_multiplier"],
            )
        else:
            # Use fixed value
            params["pos_weight_multiplier"] = get_fixed_value(
                "pos_weight_multiplier", config
            )

        params["pos_weight"] = None  # Will be computed during training

    # Fixed parameters
    params["n_estimators"] = config.model.params.get("n_estimators", 1000)
    params["early_stopping_rounds"] = config.early_stopping_rounds

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
    # Suggest all parameters
    all_params = suggest_all_params(trial, config)

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

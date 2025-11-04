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
from euos25.models.catboost import CatBoostClassifier
from euos25.models.random_forest import RandomForestClassifierModel
from euos25.pipeline.features import (
    FEATURE_GROUP_MAPPING,
    filter_feature_groups,
    filter_low_quality_features,
    get_available_feature_groups,
    get_feature_groups_from_config,
)
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
    # CatBoost defaults
    "colsample_bylevel": 0.8,
    "min_data_in_leaf": 20,
    "l2_leaf_reg": 3.0,
    # RandomForest defaults
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_samples": None,
    "bootstrap": True,
    "class_weight": "balanced",
    "n_jobs": -1,
    # Common defaults
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


def suggest_all_params(
    trial: optuna.Trial, config: Config, available_groups: set[str] | None = None
) -> Dict[str, Any]:
    """Suggest all hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial
        config: Pipeline configuration
        available_groups: Set of available feature group names. If None, will be detected
            from features in the objective function.

    Returns:
        Dictionary of all parameters (tuned and fixed)
    """
    params = {}

    model_name = config.model.name

    # Suggest model-specific parameters
    if model_name == "lgbm":
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
    elif model_name == "catboost":
        # Suggest CatBoost parameters (tuned ones from config.optuna.catboost_params)
        catboost_param_names = [
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bylevel",
            "min_data_in_leaf",
            "l2_leaf_reg",
        ]
        for param_name in catboost_param_names:
            if param_name in config.optuna.catboost_params:
                # Tune this parameter
                params[param_name] = suggest_param(
                    trial, param_name, config.optuna.catboost_params[param_name]
                )
            else:
                # Use fixed value
                params[param_name] = get_fixed_value(param_name, config)
    elif model_name == "random_forest":
        # Suggest RandomForest parameters (tuned ones from config.optuna.randomforest_params)
        randomforest_param_names = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "max_samples",
        ]
        for param_name in randomforest_param_names:
            if param_name in config.optuna.randomforest_params:
                # Tune this parameter
                params[param_name] = suggest_param(
                    trial, param_name, config.optuna.randomforest_params[param_name]
                )
            else:
                # Use fixed value
                params[param_name] = get_fixed_value(param_name, config)
        # Fixed parameters for RandomForest
        if "n_jobs" not in params:
            params["n_jobs"] = get_fixed_value("n_jobs", config)
        if "bootstrap" not in params:
            params["bootstrap"] = get_fixed_value("bootstrap", config)
    else:
        # Generic parameters for other models
        common_param_names = ["learning_rate", "max_depth"]
        for param_name in common_param_names:
            if param_name in config.model.params:
                params[param_name] = config.model.params[param_name]
            else:
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

    # Feature group selection (if enabled)
    # Dynamically handle all available feature groups
    if config.optuna.feature_groups.get("tune", False):
        if available_groups is None:
            # If groups not provided, default to known groups (backward compatibility)
            available_groups = {"ecfp4", "rdkit2d", "mordred", "chemeleon", "chemberta", "custom"}

        # Suggest boolean values for each available group
        for group_name in sorted(available_groups):  # Sort for reproducibility
            param_name = f"use_{group_name}"
            params[param_name] = trial.suggest_categorical(param_name, [True, False])
    else:
        # Use all feature groups by default
        if available_groups is None:
            # If groups not provided, default to known groups (backward compatibility)
            available_groups = {"ecfp4", "rdkit2d", "mordred", "chemeleon", "chemberta", "custom"}

        for group_name in sorted(available_groups):
            param_name = f"use_{group_name}"
            params[param_name] = True

    # Fixed parameters (model-specific)
    if model_name != "random_forest":
        # RandomForest handles n_estimators differently (can be tuned)
        params["n_estimators"] = config.model.params.get("n_estimators", 1000)
        params["early_stopping_rounds"] = config.early_stopping_rounds
    else:
        # For RandomForest, n_estimators might be tuned, so check
        if "n_estimators" not in params:
            params["n_estimators"] = config.model.params.get("n_estimators", 100)
        # RandomForest doesn't have early stopping
        params["early_stopping_rounds"] = None

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
    # Get feature groups enabled in config
    config_groups_dict = get_feature_groups_from_config(config)
    # Extract only groups that are True (enabled in config)
    config_groups = {g for g, enabled in config_groups_dict.items() if enabled} if config_groups_dict else set()

    # If config specifies groups, use those. Otherwise, detect from features.
    if config_groups:
        # Use intersection of config groups and actual feature groups
        actual_feature_groups = get_available_feature_groups(features)
        available_groups = config_groups & actual_feature_groups

        logger.info(
            f"Trial {trial.number}: Config groups: {config_groups}, "
            f"Config featurizers: {[f.name for f in config.featurizers]}, "
            f"Actual groups in features: {actual_feature_groups}, "
            f"Using: {available_groups}"
        )

        # Warn if config groups are not available in features
        if available_groups != config_groups:
            missing_in_features = config_groups - actual_feature_groups
            if missing_in_features:
                logger.warning(
                    f"Trial {trial.number}: Config specifies groups {config_groups} but "
                    f"features only contain {actual_feature_groups}. "
                    f"Missing groups: {missing_in_features}. "
                    f"Will use only available groups: {available_groups}"
                )
    else:
        # No featurizers in config (e.g., ChemProp), detect from features
        available_groups = get_available_feature_groups(features)
        logger.info(
            f"Trial {trial.number}: No featurizers in config, "
            f"detected from features: {available_groups}"
        )

    # Suggest all parameters (pass available groups for dynamic handling)
    all_params = suggest_all_params(trial, config, available_groups=available_groups)

    # Extract feature group selection parameters dynamically
    # Only extract params that correspond to actual feature groups
    valid_feature_groups = set(FEATURE_GROUP_MAPPING.values())
    feature_group_params = {}
    feature_group_keys = [key for key in all_params.keys() if key.startswith("use_")]

    for key in feature_group_keys:
        group_name = key[4:]  # Remove "use_" prefix
        # Only include if it's an actual feature group name (not e.g., "focal_loss")
        if group_name in valid_feature_groups:
            feature_group_params[group_name] = all_params.pop(key)
        # Otherwise, it's a model parameter like "use_focal_loss", keep it in all_params

    # Convert to dict format expected by filter_feature_groups
    # (group_name -> bool mapping)
    group_settings = feature_group_params

    # Check if at least one feature group is selected
    # Handle both empty dict and all False cases
    if not group_settings or not any(group_settings.values()):
        # If no groups selected, this is an invalid configuration
        # Return a very poor score to discourage this
        logger.warning(f"Trial {trial.number}: No feature groups selected, skipping")
        return 0.0

    # Apply feature group filtering using dict format
    try:
        filtered_features = filter_feature_groups(features, group_settings=group_settings)
    except ValueError as e:
        # filter_feature_groups raises ValueError if no columns selected
        logger.warning(
            f"Trial {trial.number}: Feature filtering failed: {e}. "
            f"Group settings: {group_settings}, "
            f"Available groups: {get_available_feature_groups(features)}"
        )
        return 0.0
    # Check if filtering actually resulted in features
    # (group_settings may have True values but those groups might not exist in features)
    if len(filtered_features.columns) == 0:
        logger.warning(
            f"Trial {trial.number}: No features remain after filtering. "
            f"Group settings: {group_settings}, "
            f"Available groups in features: {get_available_feature_groups(features)}"
        )
        return 0.0

    # Filter low-quality features (high NaN, low variance, mostly constant)
    # Only apply to mordred and rdkit2d features, as sparse features like ECFP and conj_proxy
    # should not be filtered based on quality metrics
    filtered_features = filter_low_quality_features(
        filtered_features,
        max_nan_ratio=0.99,  # 99%以上NaNの特徴量を除外
        min_variance=1e-6,
        min_unique_ratio=0.01,  # ユニーク値が1%未満の特徴量を除外
        low_variance_threshold=0.99,  # 99%以上が同じ値の特徴量を除外
        feature_groups_to_filter={'mordred', 'rdkit2d'},  # 品質フィルタリングを適用する特徴量グループ
    )

    # Check again after low-quality filtering
    if len(filtered_features.columns) == 0:
        logger.warning(
            f"Trial {trial.number}: No features remain after low-quality filtering."
        )
        return 0.0

    # Run CV with these parameters
    fold_scores = []

    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])

        # Get train/valid indices
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        # Convert to IDs
        train_ids = filtered_features.index[train_pos_indices]
        valid_ids = filtered_features.index[valid_pos_indices]

        # Get features and labels
        X_train = filtered_features.loc[train_ids]
        y_train = labels.loc[train_ids].values

        X_valid = filtered_features.loc[valid_ids]
        y_valid = labels.loc[valid_ids].values

        # Filter to numeric columns only (exclude SMILES and other non-numeric columns)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            logger.warning(
                f"Trial {trial.number}, Fold {fold_idx}: No numeric features after filtering. "
                f"All columns: {list(X_train.columns)}"
            )
            # Skip this fold with a poor score
            fold_scores.append(0.0)
            continue

        X_train = X_train[numeric_cols]
        X_valid = X_valid[numeric_cols]

        # Handle pos_weight_multiplier
        if "pos_weight_multiplier" in all_params:
            multiplier = all_params.pop("pos_weight_multiplier")
            # Compute base pos_weight
            n_pos = np.sum(y_train)
            n_neg = len(y_train) - n_pos
            base_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            all_params["pos_weight"] = base_pos_weight * multiplier

        # Create and train model based on model type
        model_name = config.model.name
        if model_name == "lgbm":
            model = LGBMClassifier(**all_params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        elif model_name == "catboost":
            model = CatBoostClassifier(**all_params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        elif model_name == "random_forest":
            # RandomForest doesn't use eval_set for early stopping
            model = RandomForestClassifierModel(**all_params)
            model.fit(X_train, y_train, eval_set=None)
        else:
            raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

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

    # Setup storage if enabled
    storage = None
    load_if_exists = False
    if config.optuna.storage_enable:
        # Determine storage path
        if config.optuna.storage_path:
            storage_path = Path(config.optuna.storage_path)
        else:
            storage_path = output_path / f"{study_name}.db"

        storage = f"sqlite:///{storage_path}"
        load_if_exists = True
        logger.info(f"Using SQLite storage: {storage}")
        logger.info(f"  Existing study will be loaded if available")

    # Check if study already exists (only if storage is enabled)
    existing_trials = 0
    if config.optuna.storage_enable and storage:
        try:
            existing_study = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            existing_trials = len(existing_study.trials)
            if existing_trials > 0:
                logger.info(f"Found existing study with {existing_trials} completed trials")
                if existing_study.best_trial:
                    logger.info(
                        f"  Best trial so far: {existing_study.best_trial.number} "
                        f"(value: {existing_study.best_value:.6f})"
                    )
        except (ValueError, KeyError):
            # Study doesn't exist yet, will be created
            pass

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        storage=storage,
        load_if_exists=load_if_exists,
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

"""Training pipeline with cross-validation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from euos25.config import Config
from euos25.featurizers.categorical_encoding import (
    LabelEncodingFeaturizer,
    TargetEncodingFeaturizer,
)
from euos25.models.base import ClfModel
from euos25.models.lgbm import LGBMClassifier
from euos25.models.catboost import CatBoostClassifier
from euos25.models.random_forest import RandomForestClassifierModel
from euos25.models import ChemPropModel, CHEMPROP_AVAILABLE, UniMolModel, UNIMOL_AVAILABLE
from euos25.pipeline.features import (
    filter_feature_groups,
    filter_low_quality_features,
    get_feature_groups_from_config,
)
from euos25.utils.io import load_json, load_parquet
from euos25.utils.metrics import calc_metrics, save_fold_metrics

logger = logging.getLogger(__name__)


def apply_categorical_encoding(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    config: Config,
    nested_cv: bool = True,
    inner_cv_folds: int = 3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply categorical encoding to features.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        config: Pipeline configuration
        nested_cv: Whether to use nested CV for target encoding
        inner_cv_folds: Number of inner CV folds (if nested_cv=True)
        seed: Random seed

    Returns:
        Tuple of (X_train_encoded, X_valid_encoded)
    """
    encoding_config = config.categorical_encoding
    if not encoding_config.enable:
        return X_train, X_valid

    encoded_train_list = []
    encoded_valid_list = []

    # Determine descriptor columns
    descriptor_columns = encoding_config.descriptor_columns
    if descriptor_columns is None and encoding_config.auto_detect:
        from euos25.featurizers.categorical_encoding import detect_categorical_descriptors

        descriptor_columns = detect_categorical_descriptors(
            X_train, max_unique_values=encoding_config.max_unique_values
        )

    if descriptor_columns is None or len(descriptor_columns) == 0:
        logger.warning("No descriptor columns found for categorical encoding")
        return X_train, X_valid

    # Label encoding
    if encoding_config.use_label_encoding:
        label_encoder = LabelEncodingFeaturizer(
            descriptor_columns=descriptor_columns,
            max_unique_values=encoding_config.max_unique_values,
            auto_detect=False,
        )
        label_encoder.fit(X_train)
        X_train_label = label_encoder.transform(X_train)
        X_valid_label = label_encoder.transform(X_valid)
        encoded_train_list.append(X_train_label)
        encoded_valid_list.append(X_valid_label)
        logger.info(f"Added {len(X_train_label.columns)} label-encoded features")

    # Target encoding
    if encoding_config.use_target_encoding:
        if nested_cv:
            # Nested CV: create inner splits and fit encoders on inner train
            skf = StratifiedKFold(
                n_splits=inner_cv_folds, shuffle=True, random_state=seed
            )
            train_indices = np.arange(len(X_train))
            inner_splits = list(skf.split(train_indices, y_train))

            # For training set: use nested CV approach
            # For each sample in train, find which inner folds it belongs to
            # and use the encoder from the inner train folds it's NOT in
            train_target_encoded_dict = {}
            for inner_fold_idx, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits):
                X_inner_train = X_train.iloc[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_valid = X_train.iloc[inner_valid_idx]

                # Fit target encoder on inner train
                target_encoder = TargetEncodingFeaturizer(
                    descriptor_columns=descriptor_columns,
                    max_unique_values=encoding_config.max_unique_values,
                    auto_detect=False,
                    smoothing=encoding_config.target_encoding_smoothing,
                )
                target_encoder.fit(X_inner_train, pd.Series(y_inner_train, index=X_inner_train.index))

                # Transform inner valid (using inner train encoder) - this is the key point
                X_inner_valid_target = target_encoder.transform(X_inner_valid)

                # Store encoded values for inner valid samples
                for valid_idx, valid_id in enumerate(X_inner_valid.index):
                    if valid_id not in train_target_encoded_dict:
                        train_target_encoded_dict[valid_id] = []
                    train_target_encoded_dict[valid_id].append(X_inner_valid_target.iloc[valid_idx])

            # Average across inner folds for each training sample
            X_train_target_list = []
            for train_id in X_train.index:
                if train_id in train_target_encoded_dict:
                    # Average across inner folds
                    avg_encoded = pd.concat(train_target_encoded_dict[train_id], axis=0).mean()
                    X_train_target_list.append(avg_encoded)
                else:
                    # Sample was in all inner train sets, fit encoder on full train
                    target_encoder_full = TargetEncodingFeaturizer(
                        descriptor_columns=descriptor_columns,
                        max_unique_values=encoding_config.max_unique_values,
                        auto_detect=False,
                        smoothing=encoding_config.target_encoding_smoothing,
                    )
                    target_encoder_full.fit(X_train, pd.Series(y_train, index=X_train.index))
                    single_encoded = target_encoder_full.transform(X_train.loc[[train_id]])
                    X_train_target_list.append(single_encoded.iloc[0])

            X_train_target = pd.DataFrame(X_train_target_list, index=X_train.index)

            # For validation set, fit on full train and transform
            target_encoder_full = TargetEncodingFeaturizer(
                descriptor_columns=descriptor_columns,
                max_unique_values=encoding_config.max_unique_values,
                auto_detect=False,
                smoothing=encoding_config.target_encoding_smoothing,
            )
            target_encoder_full.fit(X_train, pd.Series(y_train, index=X_train.index))
            X_valid_target = target_encoder_full.transform(X_valid)

            encoded_train_list.append(X_train_target)
            encoded_valid_list.append(X_valid_target)
        else:
            # Standard: fit on train, transform valid
            target_encoder = TargetEncodingFeaturizer(
                descriptor_columns=descriptor_columns,
                max_unique_values=encoding_config.max_unique_values,
                auto_detect=False,
                smoothing=encoding_config.target_encoding_smoothing,
            )
            target_encoder.fit(X_train, pd.Series(y_train, index=X_train.index))
            X_train_target = target_encoder.transform(X_train)
            X_valid_target = target_encoder.transform(X_valid)
            encoded_train_list.append(X_train_target)
            encoded_valid_list.append(X_valid_target)

        logger.info(f"Added {len(encoded_valid_list[-1].columns)} target-encoded features")

    # Concatenate all encoded features
    if encoded_train_list:
        X_train_encoded = pd.concat([X_train] + encoded_train_list, axis=1)
        X_valid_encoded = pd.concat([X_valid] + encoded_valid_list, axis=1)
        return X_train_encoded, X_valid_encoded
    else:
        return X_train, X_valid


def create_model(config: Config, checkpoint_dir: Optional[str] = None) -> ClfModel:
    """Create model from configuration.

    Args:
        config: Pipeline configuration
        checkpoint_dir: Optional checkpoint directory override (for ChemProp)

    Returns:
        Model instance
    """
    model_name = config.model.name
    model_params = config.model.params.copy()

    # Add imbalance handling parameters
    if config.imbalance.use_focal_loss:
        # Use focal loss (overrides pos_weight)
        model_params["use_focal_loss"] = True
        model_params["focal_alpha"] = config.imbalance.focal_alpha
        model_params["focal_gamma"] = config.imbalance.focal_gamma
        # pos_weight is ignored when use_focal_loss is True
        model_params["pos_weight"] = None
    elif config.imbalance.use_pos_weight and config.imbalance.pos_weight_from_data:
        # Will be computed during training
        model_params["pos_weight"] = None
        # Store multiplier for later use if set
        if config.imbalance.pos_weight_multiplier is not None:
            model_params["pos_weight_multiplier"] = config.imbalance.pos_weight_multiplier
    elif config.imbalance.use_pos_weight and config.imbalance.pos_weight_value:
        model_params["pos_weight"] = config.imbalance.pos_weight_value
    else:
        model_params["use_focal_loss"] = False

    # Add early stopping parameters
    model_params["early_stopping_rounds"] = config.early_stopping_rounds

    # Add objective_type if specified in config
    if config.model.objective_type is not None:
        model_params["objective_type"] = config.model.objective_type

    if model_name == "lgbm":
        return LGBMClassifier(**model_params)
    elif model_name == "catboost":
        return CatBoostClassifier(**model_params)
    elif model_name == "random_forest":
        return RandomForestClassifierModel(**model_params)
    elif model_name == "chemprop":
        if not CHEMPROP_AVAILABLE:
            raise ImportError(
                "ChemProp is not available. Please install chemprop package."
            )

        # For ChemProp, focal loss parameters come from config.model.params or config.imbalance
        # Check if use_focal_loss is set in model params (takes precedence)
        if model_params.get("use_focal_loss", False):
            # Focal loss params are already in model_params from config
            pass
        elif config.imbalance.use_focal_loss:
            # Use focal loss from imbalance config
            model_params["use_focal_loss"] = True
            model_params["focal_alpha"] = config.imbalance.focal_alpha
            model_params["focal_gamma"] = config.imbalance.focal_gamma

        # Add objective_type from config
        if config.model.objective_type is not None:
            model_params["objective_type"] = config.model.objective_type

        # Add random_seed from config
        model_params["random_seed"] = config.seed

        # Add early stopping parameters for ChemProp
        model_params["early_stopping_rounds"] = config.early_stopping_rounds
        model_params["early_stopping_metric"] = config.early_stopping_metric

        # Override checkpoint_dir if provided
        if checkpoint_dir is not None:
            model_params["checkpoint_dir"] = checkpoint_dir

        return ChemPropModel(**model_params)
    elif model_name == "unimol":
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "Uni-Mol is not available. Please install 'unimol_tools' package. "
                "Install with: pip install unimol_tools"
            )

        # For UniMol, focal loss parameters come from config.model.params or config.imbalance
        if model_params.get("use_focal_loss", False):
            pass
        elif config.imbalance.use_focal_loss:
            model_params["use_focal_loss"] = True
            model_params["focal_alpha"] = config.imbalance.focal_alpha
            model_params["focal_gamma"] = config.imbalance.focal_gamma

        # Add objective_type from config
        if config.model.objective_type is not None:
            model_params["objective_type"] = config.model.objective_type

        # Add random_seed from config
        model_params["random_seed"] = config.seed

        # Add early stopping parameters for UniMol
        model_params["early_stopping_rounds"] = config.early_stopping_rounds
        model_params["early_stopping_metric"] = config.early_stopping_metric

        # Override checkpoint_dir if provided
        if checkpoint_dir is not None:
            model_params["checkpoint_dir"] = checkpoint_dir

        # Automatically set cache_dir if not specified in config
        if "cache_dir" not in model_params or model_params["cache_dir"] is None:
            model_params["cache_dir"] = "data/processed/unimol_3d_cache"
            logger.info(f"Automatically setting cache_dir to: {model_params['cache_dir']}")

        return UniMolModel(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    config: Config,
    fold_idx: int,
    output_dir: Optional[Path] = None,
    task_name: Optional[str] = None,
    binary_labels_train: Optional[np.ndarray] = None,
    binary_labels_valid: Optional[np.ndarray] = None,
) -> tuple[ClfModel, Dict[str, float]]:
    """Train model on single fold.

    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        config: Pipeline configuration
        fold_idx: Fold index
        output_dir: Directory to save model (optional)
        task_name: Task name for checkpoint directory (optional)

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info(f"Training fold {fold_idx}")
    logger.info(f"  Train: {len(y_train)} samples, pos={y_train.sum()}")
    logger.info(f"  Valid: {len(y_valid)} samples, pos={y_valid.sum()}")

    # Check if model already exists
    if output_dir:
        model_dir = output_dir / f"fold_{fold_idx}"
        if config.model.name == "lgbm":
            model_path = model_dir / "model.txt"
            if model_path.exists():
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                from euos25.models.lgbm import LGBMClassifier
                model = LGBMClassifier.load(str(model_dir))
                # Predict on validation to compute metrics
                y_pred = model.predict_proba(X_valid)
                y_pred_proba = y_pred
                metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
                for metric_name, score in metrics.items():
                    logger.info(f"  {metric_name}: {score:.6f}")
                return model, metrics
        elif config.model.name == "catboost":
            model_path = model_dir / "model.cbm"
            if model_path.exists():
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                from euos25.models.catboost import CatBoostClassifier
                model = CatBoostClassifier.load(str(model_dir))
                # Predict on validation to compute metrics
                y_pred = model.predict_proba(X_valid)
                y_pred_proba = y_pred
                metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
                for metric_name, score in metrics.items():
                    logger.info(f"  {metric_name}: {score:.6f}")
                return model, metrics
        elif config.model.name == "random_forest":
            model_path = model_dir / "model.pkl"
            if model_path.exists():
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                from euos25.models.random_forest import RandomForestClassifierModel
                model = RandomForestClassifierModel.load(str(model_dir))
                # Predict on validation to compute metrics
                y_pred = model.predict_proba(X_valid)
                y_pred_proba = y_pred
                metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
                for metric_name, score in metrics.items():
                    logger.info(f"  {metric_name}: {score:.6f}")
                return model, metrics
        elif config.model.name == "chemprop":
            # For ChemProp, check if model exists (can be file or directory)
            # trainer.save_checkpoint() saves as a file
            if model_dir.exists() and (model_dir.is_file() or model_dir.is_dir()):
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                from euos25.models.chemprop import ChemPropModel
                # Load model parameters from config to reconstruct
                model = ChemPropModel.load_from_checkpoint(
                    str(model_dir),
                    **config.model.params,
                    random_seed=config.seed,
                    early_stopping_rounds=config.early_stopping_rounds,
                    early_stopping_metric=config.early_stopping_metric,
                )
                # Predict on validation to compute metrics
                y_pred = model.predict_proba(X_valid)
                y_pred_proba = y_pred[:, 1] if y_pred.ndim == 2 else y_pred
                metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
                for metric_name, score in metrics.items():
                    logger.info(f"  {metric_name}: {score:.6f}")
                return model, metrics
        elif config.model.name == "unimol":
            # For UniMol, check if model exists (similar to ChemProp)
            if model_dir.exists() and (model_dir.is_file() or model_dir.is_dir()):
                logger.info(f"  Model already exists at {model_dir}, skipping training")
                # Note: UniMolModel.load_from_checkpoint is not fully implemented yet
                # For now, we'll skip loading and continue training
                logger.warning("UniMolModel checkpoint loading not fully implemented. Skipping checkpoint load.")

    # Build checkpoint directory for ChemProp/UniMol
    # Note: task_name and fold_name will be added in fit() method to ensure proper separation
    checkpoint_dir = None
    resume_ckpt = None
    if config.model.name in ["chemprop", "unimol"] and output_dir is not None:
        actual_task_name = task_name if task_name is not None else config.task
        # Use checkpoint_dir from config if specified, otherwise construct from output_dir
        # Don't add task_name and fold_name here - let fit() method handle it
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir))
        else:
            checkpoint_dir = str(output_dir / "checkpoints")
        # Ensure base directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Check if there's a checkpoint to resume from (in task/fold subdirectory)
        checkpoint_path = Path(checkpoint_dir) / actual_task_name / f"fold_{fold_idx}"
        last_ckpt = checkpoint_path / "last.ckpt"
        if last_ckpt.exists():
            resume_ckpt = str(last_ckpt)
            logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")
        else:
            # Check for any .ckpt file in the directory
            ckpt_files = list(checkpoint_path.glob("*.ckpt"))
            if ckpt_files:
                # Use the most recent checkpoint
                resume_ckpt = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
                logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")

    # Apply categorical encoding if enabled
    encoding_config = config.categorical_encoding
    if encoding_config.enable:
        logger.info("Applying categorical encoding...")
        use_nested_cv = encoding_config.use_target_encoding
        X_train, X_valid = apply_categorical_encoding(
            X_train,
            y_train,
            X_valid,
            config,
            nested_cv=use_nested_cv,
            inner_cv_folds=encoding_config.nested_cv_folds,
            seed=config.seed,
        )

    # Create model
    model = create_model(config, checkpoint_dir=checkpoint_dir)

    # Set binary_labels for regression/ranking if provided
    if binary_labels_train is not None and config.model.name in ["lgbm", "catboost"]:
        if hasattr(model, '_binary_labels'):
            model._binary_labels = binary_labels_train

    # Train model - handle different model signatures
    if config.model.name in ["chemprop", "unimol"]:
        # ChemProp and UniMol use X_val and y_val instead of eval_set
        # Pass binary_labels_val for regression/ranking
        actual_task_name = task_name if task_name is not None else config.task
        fit_kwargs = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_valid,
            "y_val": y_valid,
            "resume_from_checkpoint": resume_ckpt,
            "task_name": actual_task_name,
            "fold_name": f"fold_{fold_idx}",
        }
        if binary_labels_valid is not None:
            fit_kwargs["binary_labels_val"] = binary_labels_valid
        model.fit(**fit_kwargs)
    elif config.model.name == "random_forest":
        # RandomForest doesn't use eval_set (no early stopping)
        model.fit(X_train, y_train, eval_set=None)
    else:
        # LGBM and CatBoost use eval_set
        # Pass binary_labels_valid for regression/ranking
        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "eval_set": (X_valid, y_valid),
        }
        if binary_labels_valid is not None and config.model.name in ["lgbm", "catboost"]:
            fit_kwargs["binary_labels_valid"] = binary_labels_valid
        model.fit(**fit_kwargs)

    # Predict on validation
    y_pred = model.predict_proba(X_valid)

    # Handle different output shapes: chemprop/unimol return (n_samples, 2), lgbm returns (n_samples,)
    if config.model.name in ["chemprop", "unimol"] and y_pred.ndim == 2:
        # Extract positive class probabilities for metrics calculation
        y_pred_proba = y_pred[:, 1]
    else:
        # LGBM and other models return 1D array directly
        y_pred_proba = y_pred

    # Calculate metrics
    # Use binary_labels_valid for regression/ranking (ROC-AUC calculation)
    y_valid_for_metrics = binary_labels_valid if binary_labels_valid is not None else y_valid
    metrics = calc_metrics(y_valid_for_metrics, y_pred_proba, metrics=config.metrics)

    # Log metrics
    for metric_name, score in metrics.items():
        logger.info(f"  {metric_name}: {score:.6f}")

    # Save model
    if output_dir:
        model_dir = output_dir / f"fold_{fold_idx}"
        model.save(str(model_dir))

    return model, metrics


def train_cv(
    features_path: str,
    splits_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    task_name: Optional[str] = None,
) -> tuple[List[Dict[str, float]], List[int], List[int]]:
    """Train models with cross-validation.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save models and metrics
        task_name: Task name override (defaults to config.task)

    Returns:
        Tuple of (fold_metrics, best_iterations, train_sizes)
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Filter features based on config (only use features specified in config)
    group_settings = get_feature_groups_from_config(config)
    if group_settings and len(group_settings) > 0:
        # Check if at least one group is enabled
        if any(group_settings.values()):
            features = filter_feature_groups(features, group_settings=group_settings)
        else:
            raise ValueError(
                f"All feature groups are disabled in config. "
                f"At least one group must be enabled. Settings: {group_settings}"
            )

    # Filter low-quality features (high NaN, low variance, mostly constant)
    # Only apply to mordred and rdkit2d features, as sparse features like ECFP and conj_proxy
    # should not be filtered based on quality metrics
    features = filter_low_quality_features(
        features,
        max_nan_ratio=0.99,  # 99%以上NaNの特徴量を除外
        min_variance=1e-6,
        min_unique_ratio=0.01,  # ユニーク値が1%未満の特徴量を除外
        low_variance_threshold=0.99,  # 99%以上が同じ値の特徴量を除外
        feature_groups_to_filter={'mordred', 'rdkit2d'},  # 品質フィルタリングを適用する特徴量グループ
    )

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Train each fold
    fold_metrics = []
    best_iterations = []
    train_sizes = []

    for fold_name, fold_data in splits.items():
        fold_idx = int(fold_name.split("_")[1])

        # Get train/valid indices (these are positional indices)
        train_pos_indices = fold_data["train"]
        valid_pos_indices = fold_data["valid"]

        # Convert positional indices to IDs
        # Splits were created on a DataFrame in the same order as features
        train_ids = features.index[train_pos_indices]
        valid_ids = features.index[valid_pos_indices]

        # Get features and labels using IDs
        X_train = features.loc[train_ids]
        y_train = labels.loc[train_ids].values

        X_valid = features.loc[valid_ids]
        y_valid = labels.loc[valid_ids].values

        # Get binary labels if available (for regression/ranking)
        binary_labels_train = None
        binary_labels_valid = None
        if hasattr(config, '_binary_labels') and config._binary_labels is not None:
            binary_labels_train = config._binary_labels.loc[train_ids].values
            binary_labels_valid = config._binary_labels.loc[valid_ids].values

        # Train fold
        model, metrics = train_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            config,
            fold_idx,
            output_dir=output_path,
            task_name=actual_task_name,
            binary_labels_train=binary_labels_train,
            binary_labels_valid=binary_labels_valid,
        )

        fold_metrics.append(metrics)
        best_iterations.append(model.best_iteration)
        train_sizes.append(len(X_train))

    # Save fold metrics
    metrics_path = output_path / "cv_metrics.csv"
    save_fold_metrics(fold_metrics, str(metrics_path))

    # Log aggregated metrics
    logger.info("=" * 50)
    logger.info("Cross-validation results:")
    for metric_name in config.metrics:
        values = [m[metric_name] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")

    avg_best_iter = int(np.mean(best_iterations))
    logger.info(f"  Average best iteration: {avg_best_iter}")
    logger.info("=" * 50)

    return fold_metrics, best_iterations, train_sizes


def train_full(
    features_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    best_iterations: List[int],
    train_sizes: List[int],
    task_name: Optional[str] = None,
    feature_group_settings: Optional[dict[str, bool]] = None,
) -> ClfModel:
    """Train model on full training data.

    Args:
        features_path: Path to features Parquet
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save model
        best_iterations: List of best_iteration from CV folds
        train_sizes: List of training set sizes from CV folds
        task_name: Task name override (defaults to config.task)
        feature_group_settings: Optional feature group settings dict (from Optuna).
            If provided, overrides config-based feature filtering.

    Returns:
        Trained model
    """
    # Load features
    features = load_parquet(features_path)

    # Filter features based on feature_group_settings or config
    if feature_group_settings and len(feature_group_settings) > 0:
        # Check if at least one group is enabled
        if any(feature_group_settings.values()):
            # Use Optuna-optimized feature groups
            logger.info("Using Optuna-optimized feature groups for full model training")
            features = filter_feature_groups(features, group_settings=feature_group_settings)
        else:
            raise ValueError(
                f"All Optuna-optimized feature groups are disabled. "
                f"At least one group must be enabled. Settings: {feature_group_settings}"
            )
    else:
        # Filter features based on config (only use features specified in config)
        group_settings = get_feature_groups_from_config(config)
        if group_settings and len(group_settings) > 0:
            # Check if at least one group is enabled
            if any(group_settings.values()):
                features = filter_feature_groups(features, group_settings=group_settings)
            else:
                raise ValueError(
                    f"All feature groups are disabled in config. "
                    f"At least one group must be enabled. Settings: {group_settings}"
                )

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if full model already exists
    full_model_dir = output_path / "full_model"
    if config.model.name == "lgbm":
        model_path = full_model_dir / "model.txt"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.lgbm import LGBMClassifier
            return LGBMClassifier.load(str(full_model_dir))
    elif config.model.name == "catboost":
        model_path = full_model_dir / "model.cbm"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.catboost import CatBoostClassifier
            return CatBoostClassifier.load(str(full_model_dir))
    elif config.model.name == "random_forest":
        model_path = full_model_dir / "model.pkl"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.random_forest import RandomForestClassifierModel
            return RandomForestClassifierModel.load(str(full_model_dir))
    elif config.model.name == "chemprop":
        # For ChemProp, check if model exists (can be file or directory)
        if full_model_dir.exists() and (full_model_dir.is_file() or full_model_dir.is_dir()):
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.chemprop import ChemPropModel
            model = ChemPropModel.load_from_checkpoint(
                str(full_model_dir),
                **config.model.params,
                random_seed=config.seed,
                early_stopping_rounds=config.early_stopping_rounds,
                early_stopping_metric=config.early_stopping_metric,
            )
            return model
    elif config.model.name == "unimol":
        # For UniMol, check if model exists (similar to ChemProp)
        if full_model_dir.exists() and (full_model_dir.is_file() or full_model_dir.is_dir()):
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            # Note: UniMolModel.load_from_checkpoint is not fully implemented yet
            logger.warning("UniMolModel checkpoint loading not fully implemented. Skipping checkpoint load.")

    # Get full training data
    # Only use samples that have labels
    common_ids = features.index.intersection(labels.index)
    X_full = features.loc[common_ids]
    y_full = labels.loc[common_ids].values

    # Apply categorical encoding if enabled
    # For full model without split, we fit on all data (no validation set)
    encoding_config = config.categorical_encoding
    if encoding_config.enable:
        logger.info("Applying categorical encoding for full model...")
        # Create dummy valid set for encoding (will not be used for training)
        # We still need to fit encoders on full data
        X_full, _ = apply_categorical_encoding(
            X_full,
            y_full,
            X_full.copy(),  # Dummy valid set
            config,
            nested_cv=True,  # Nested CV for full model
            inner_cv_folds=encoding_config.nested_cv_folds,
            seed=config.seed,
        )

    logger.info("=" * 50)
    logger.info("Training on full dataset")
    logger.info(f"  Full dataset: {len(y_full)} samples, pos={y_full.sum()}")

    # Build checkpoint directory for ChemProp/UniMol
    # Note: task_name and fold_name will be added in fit() method to ensure proper separation
    checkpoint_dir = None
    resume_ckpt = None
    if config.model.name in ["chemprop", "unimol"]:
        # Use checkpoint_dir from config if specified, otherwise construct from output_dir
        # Don't add task_name and fold_name here - let fit() method handle it
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir))
        else:
            checkpoint_dir = str(output_path / "checkpoints")
        # Ensure base directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Check if there's a checkpoint to resume from (in task/full subdirectory)
        checkpoint_path = Path(checkpoint_dir) / actual_task_name / "full"
        last_ckpt = checkpoint_path / "last.ckpt"
        if last_ckpt.exists():
            resume_ckpt = str(last_ckpt)
            logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")
        else:
            # Check for any .ckpt file in the directory
            ckpt_files = list(checkpoint_path.glob("*.ckpt"))
            if ckpt_files:
                # Use the most recent checkpoint
                resume_ckpt = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
                logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")

    # Create model
    model = create_model(config, checkpoint_dir=checkpoint_dir)

    # Handle model-specific full training adjustments
    if config.model.name == "lgbm":
        # Calculate average best_iteration from CV
        avg_best_iter = int(np.mean(best_iterations))
        logger.info(f"  Average best iteration from CV: {avg_best_iter}")

        # Calculate size ratio: full_size / avg_train_size
        avg_train_size = np.mean(train_sizes)
        size_ratio = len(X_full) / avg_train_size
        logger.info(f"  Size ratio (full / avg_train): {size_ratio:.3f}")

        # Adjust n_estimators: int(0.9 * size_ratio * avg_best_iter)
        adjusted_n_estimators = int(0.9 * size_ratio * avg_best_iter)
        # Ensure at least avg_best_iter
        adjusted_n_estimators = max(adjusted_n_estimators, avg_best_iter)
        logger.info(f"  Adjusted n_estimators: {adjusted_n_estimators}")

        # Override n_estimators for full training
        original_n_estimators = model.params["n_estimators"]
        model.params["n_estimators"] = adjusted_n_estimators

        # Train on full data (no validation set, no early stopping)
        logger.info(f"Training with {adjusted_n_estimators} rounds (no early stopping)")
        model.fit(X_full, y_full, eval_set=None)

        # Restore original n_estimators (for metadata consistency)
        model.params["n_estimators"] = original_n_estimators
    elif config.model.name == "catboost":
        # Calculate average best_iteration from CV
        avg_best_iter = int(np.mean(best_iterations))
        logger.info(f"  Average best iteration from CV: {avg_best_iter}")

        # Calculate size ratio: full_size / avg_train_size
        avg_train_size = np.mean(train_sizes)
        size_ratio = len(X_full) / avg_train_size
        logger.info(f"  Size ratio (full / avg_train): {size_ratio:.3f}")

        # Adjust n_estimators: int(0.9 * size_ratio * avg_best_iter)
        adjusted_n_estimators = int(0.9 * size_ratio * avg_best_iter)
        # Ensure at least avg_best_iter
        adjusted_n_estimators = max(adjusted_n_estimators, avg_best_iter)
        logger.info(f"  Adjusted n_estimators: {adjusted_n_estimators}")

        # Override n_estimators for full training
        original_n_estimators = model.params["n_estimators"]
        model.params["n_estimators"] = adjusted_n_estimators

        # Train on full data (no validation set, no early stopping)
        logger.info(f"Training with {adjusted_n_estimators} rounds (no early stopping)")
        model.fit(X_full, y_full, eval_set=None)

        # Restore original n_estimators (for metadata consistency)
        model.params["n_estimators"] = original_n_estimators
    elif config.model.name == "random_forest":
        # RandomForest doesn't adjust n_estimators based on data size
        # Just train on full data
        logger.info(f"Training with {model.params['n_estimators']} trees")
        logger.info("  Training on full data (no validation set)")
        model.fit(X_full, y_full, eval_set=None)
    elif config.model.name in ["chemprop", "unimol"]:
        # For ChemProp/UniMol, use max_epochs from config (no adjustment needed)
        logger.info(f"  Training with max_epochs={model.params.get('max_epochs', 'default')}")
        logger.info("  Training on full data (no validation set)")
        # ChemProp/UniMol use X_val and y_val instead of eval_set
        model.fit(
            X_full,
            y_full,
            X_val=None,
            y_val=None,
            resume_from_checkpoint=resume_ckpt,
            task_name=actual_task_name,
            fold_name="full",
        )
    else:
        # Generic fallback
        logger.info("  Training on full data (no validation set)")
        model.fit(X_full, y_full, eval_set=None)

    # Save full model
    full_model_dir = output_path / "full_model"
    model.save(str(full_model_dir))

    logger.info(f"Saved full model to {full_model_dir}")
    logger.info("=" * 50)

    return model


def train_full_with_split(
    features_path: str,
    splits_path: str,
    labels: pd.Series,
    config: Config,
    output_dir: str,
    task_name: Optional[str] = None,
    feature_group_settings: Optional[dict[str, bool]] = None,
) -> ClfModel:
    """Train model on full data using fold 0 split for validation.

    This function trains on all data except fold 0 valid set, using fold 0 valid set
    for early stopping. The resulting model is saved as the final model.

    Args:
        features_path: Path to features Parquet
        splits_path: Path to splits JSON
        labels: Series with labels (indexed by ID)
        config: Pipeline configuration
        output_dir: Directory to save model
        task_name: Task name override (defaults to config.task)
        feature_group_settings: Optional feature group settings dict (from Optuna).
            If provided, overrides config-based feature filtering.

    Returns:
        Trained model
    """
    # Load features and splits
    features = load_parquet(features_path)
    splits = load_json(splits_path)

    # Filter features based on feature_group_settings or config
    if feature_group_settings and len(feature_group_settings) > 0:
        # Check if at least one group is enabled
        if any(feature_group_settings.values()):
            # Use Optuna-optimized feature groups
            logger.info("Using Optuna-optimized feature groups for full model training")
            features = filter_feature_groups(features, group_settings=feature_group_settings)
        else:
            raise ValueError(
                f"All Optuna-optimized feature groups are disabled. "
                f"At least one group must be enabled. Settings: {feature_group_settings}"
            )
    else:
        # Filter features based on config (only use features specified in config)
        group_settings = get_feature_groups_from_config(config)
        if group_settings and len(group_settings) > 0:
            # Check if at least one group is enabled
            if any(group_settings.values()):
                features = filter_feature_groups(features, group_settings=group_settings)
            else:
                raise ValueError(
                    f"All feature groups are disabled in config. "
                    f"At least one group must be enabled. Settings: {group_settings}"
                )

    # Filter low-quality features
    features = filter_low_quality_features(
        features,
        max_nan_ratio=0.99,
        min_variance=1e-6,
        min_unique_ratio=0.01,
        low_variance_threshold=0.99,
        feature_groups_to_filter={'mordred', 'rdkit2d'},
    )

    # Use task_name override if provided, otherwise use config.task
    actual_task_name = task_name if task_name is not None else config.task

    # Create output directory
    output_path = Path(output_dir) / actual_task_name / config.model.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if full model already exists
    full_model_dir = output_path / "full_model"
    if config.model.name == "lgbm":
        model_path = full_model_dir / "model.txt"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.lgbm import LGBMClassifier
            return LGBMClassifier.load(str(full_model_dir))
    elif config.model.name == "catboost":
        model_path = full_model_dir / "model.cbm"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.catboost import CatBoostClassifier
            return CatBoostClassifier.load(str(full_model_dir))
    elif config.model.name == "random_forest":
        model_path = full_model_dir / "model.pkl"
        if model_path.exists():
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.random_forest import RandomForestClassifierModel
            return RandomForestClassifierModel.load(str(full_model_dir))
    elif config.model.name == "chemprop":
        # For ChemProp, check if model exists (can be file or directory)
        if full_model_dir.exists() and (full_model_dir.is_file() or full_model_dir.is_dir()):
            logger.info("=" * 50)
            logger.info("Full model already exists, skipping training")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            from euos25.models.chemprop import ChemPropModel
            model = ChemPropModel.load_from_checkpoint(
                str(full_model_dir),
                **config.model.params,
                random_seed=config.seed,
                early_stopping_rounds=config.early_stopping_rounds,
                early_stopping_metric=config.early_stopping_metric,
            )
            return model
    elif config.model.name == "unimol":
        # For UniMol, check if model exists (similar to ChemProp)
        if full_model_dir.exists() and (full_model_dir.is_file() or full_model_dir.is_dir()):
            logger.info("=" * 50)
            logger.info("Full model already exists, loading model for validation metrics calculation")
            logger.info(f"  Model path: {full_model_dir}")
            logger.info("=" * 50)
            # Load model to calculate validation metrics
            from euos25.models.unimol import UniMolModel
            try:
                model = UniMolModel.load_from_checkpoint(
                    str(full_model_dir),
                    **config.model.params,
                    random_seed=config.seed,
                    early_stopping_rounds=config.early_stopping_rounds,
                    early_stopping_metric=config.early_stopping_metric,
                )
                # Calculate and save validation metrics for existing model
                # Get fold 0 split for validation
                fold_0_data = splits.get("fold_0")
                if fold_0_data is not None:
                    # Prepare validation data
                    fold_0_valid_pos_indices = fold_0_data["valid"]
                    fold_0_valid_ids = features.index[fold_0_valid_pos_indices]
                    X_valid = features.loc[fold_0_valid_ids]
                    y_valid = labels.loc[fold_0_valid_ids].values

                    # Calculate metrics on validation set
                    y_pred = model.predict_proba(X_valid)
                    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
                        y_pred_proba = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
                    else:
                        y_pred_proba = y_pred

                    metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
                    logger.info("Validation metrics (recalculated):")
                    for metric_name, score in metrics.items():
                        logger.info(f"  {metric_name}: {score:.6f}")

                    # Save validation metrics to file
                    if full_model_dir.is_file():
                        metrics_dir = full_model_dir.parent
                    else:
                        metrics_dir = full_model_dir

                    metrics_path = metrics_dir / "validation_metrics.json"
                    import json
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"Saved validation metrics to {metrics_path}")

                    # Also save as CSV for consistency with other models
                    metrics_csv_path = metrics_dir / "validation_metrics.csv"
                    metrics_df = pd.DataFrame([metrics])
                    metrics_df.to_csv(metrics_csv_path, index=False)
                    logger.info(f"Saved validation metrics to {metrics_csv_path}")
                else:
                    logger.warning("fold_0 not found in splits, skipping validation metrics calculation")

                logger.info("=" * 50)
                return model
            except Exception as e:
                logger.warning(f"Failed to load UniMol model: {e}. Will train new model.")
                # Continue to train new model

    # Get fold 0 split for validation
    fold_0_data = splits.get("fold_0")
    if fold_0_data is None:
        raise ValueError("fold_0 not found in splits file")

    # Combine all fold train sets (excluding fold 0 valid)
    all_train_pos_indices = []
    fold_0_valid_pos_indices = fold_0_data["valid"]

    for fold_name, fold_data in splits.items():
        # Add all train indices from all folds
        all_train_pos_indices.extend(fold_data["train"])

    # Convert positional indices to IDs
    all_train_ids = features.index[all_train_pos_indices]
    fold_0_valid_ids = features.index[fold_0_valid_pos_indices]

    # Get features and labels
    X_train = features.loc[all_train_ids]
    y_train = labels.loc[all_train_ids].values

    X_valid = features.loc[fold_0_valid_ids]
    y_valid = labels.loc[fold_0_valid_ids].values

    # Apply categorical encoding if enabled
    # For full model, we use CV splits: fit encoders on each fold's train, transform on valid
    encoding_config = config.categorical_encoding
    if encoding_config.enable:
        logger.info("Applying categorical encoding for full model...")
        # For full model with split, we fit on train and transform on valid
        # (similar to standard CV, but using fold 0 split)
        X_train, X_valid = apply_categorical_encoding(
            X_train,
            y_train,
            X_valid,
            config,
            nested_cv=True,  # Nested CV for full model
            inner_cv_folds=encoding_config.nested_cv_folds,
            seed=config.seed,
        )

    # Get binary labels if available (for regression/ranking)
    binary_labels_train = None
    binary_labels_valid = None
    if hasattr(config, '_binary_labels') and config._binary_labels is not None:
        binary_labels_train = config._binary_labels.loc[all_train_ids].values
        binary_labels_valid = config._binary_labels.loc[fold_0_valid_ids].values

    logger.info("=" * 50)
    logger.info("Training on full data with fold 0 validation split")
    logger.info(f"  Train: {len(y_train)} samples, pos={y_train.sum()}")
    logger.info(f"  Valid: {len(y_valid)} samples, pos={y_valid.sum()}")

    # Build checkpoint directory for ChemProp/UniMol
    # Note: task_name and fold_name will be added in fit() method to ensure proper separation
    checkpoint_dir = None
    resume_ckpt = None
    if config.model.name in ["chemprop", "unimol"]:
        # Use checkpoint_dir from config if specified, otherwise construct from output_dir
        # Don't add task_name and fold_name here - let fit() method handle it
        base_checkpoint_dir = config.model.params.get("checkpoint_dir")
        if base_checkpoint_dir:
            checkpoint_dir = str(Path(base_checkpoint_dir))
        else:
            checkpoint_dir = str(output_path / "checkpoints")
        # Ensure base directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Check if there's a checkpoint to resume from (in task/full subdirectory)
        checkpoint_path = Path(checkpoint_dir) / actual_task_name / "full"
        last_ckpt = checkpoint_path / "last.ckpt"
        if last_ckpt.exists():
            resume_ckpt = str(last_ckpt)
            logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")
        else:
            # Check for any .ckpt file in the directory
            ckpt_files = list(checkpoint_path.glob("*.ckpt"))
            if ckpt_files:
                # Use the most recent checkpoint
                resume_ckpt = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
                logger.info(f"  Found checkpoint to resume from: {resume_ckpt}")

    # Create model
    model = create_model(config, checkpoint_dir=checkpoint_dir)

    # Train model with validation set for early stopping
    if config.model.name == "lgbm":
        logger.info("  Training with validation set for early stopping")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    elif config.model.name == "catboost":
        logger.info("  Training with validation set for early stopping")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    elif config.model.name == "random_forest":
        # RandomForest doesn't support early stopping, just train on full data
        logger.info(f"  Training with {model.params['n_estimators']} trees")
        logger.info("  Note: RandomForest doesn't support early stopping")
        model.fit(X_train, y_train, eval_set=None)
    elif config.model.name in ["chemprop", "unimol"]:
        logger.info(f"  Training with max_epochs={model.params.get('max_epochs', 'default')}")
        logger.info("  Training with validation set for early stopping")
        # ChemProp/UniMol use X_val and y_val instead of eval_set
        model.fit(
            X_train,
            y_train,
            X_val=X_valid,
            y_val=y_valid,
            resume_from_checkpoint=resume_ckpt,
            task_name=actual_task_name,
            fold_name="full",
            output_dir=str(full_model_dir.parent),  # Pass parent directory (task/model directory)
        )
    else:
        # Generic fallback
        logger.info("  Training with validation set for early stopping")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    # Calculate metrics on validation set
    y_pred = model.predict_proba(X_valid)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred_proba = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
    else:
        y_pred_proba = y_pred

    metrics = calc_metrics(y_valid, y_pred_proba, metrics=config.metrics)
    logger.info("Validation metrics:")
    for metric_name, score in metrics.items():
        logger.info(f"  {metric_name}: {score:.6f}")

    # Save validation metrics to file (for full-only mode)
    if config.model.name == "unimol":
        # For UniMol, save metrics to models directory
        # full_model_dir might be a file (for .ckpt) or directory, so use parent if it's a file
        if full_model_dir.is_file():
            metrics_dir = full_model_dir.parent
        else:
            metrics_dir = full_model_dir

        metrics_path = metrics_dir / "validation_metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved validation metrics to {metrics_path}")

        # Also save as CSV for consistency with other models
        metrics_csv_path = metrics_dir / "validation_metrics.csv"
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_csv_path, index=False)
        logger.info(f"Saved validation metrics to {metrics_csv_path}")

    # Save full model (full_model_dir already defined above)
    model.save(str(full_model_dir))

    logger.info(f"Saved full model to {full_model_dir}")
    logger.info("=" * 50)

    return model

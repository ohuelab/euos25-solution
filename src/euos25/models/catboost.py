"""CatBoost classifier implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)


class CatBoostClassifier(BaseClfModel):
    """CatBoost binary classifier with pos_weight support."""

    def __init__(
        self,
        name: str = "catboost",
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bylevel: float = 0.8,
        min_data_in_leaf: int = 20,
        l2_leaf_reg: float = 3.0,
        pos_weight: Optional[float] = None,
        pos_weight_multiplier: Optional[float] = None,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
        objective_type: Optional[str] = None,
        binary_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        """Initialize CatBoost classifier.

        Args:
            name: Model name
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training data
            colsample_bylevel: Subsample ratio of features per level
            min_data_in_leaf: Minimum samples in leaf
            l2_leaf_reg: L2 regularization coefficient
            pos_weight: Weight for positive class (auto-computed if None)
            pos_weight_multiplier: Multiplier for auto-computed pos_weight
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level
            objective_type: Objective type ("regression" or None for binary)
            binary_labels: Binary labels for ROC-AUC calculation (used with regression)
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(
            name=name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            min_data_in_leaf=min_data_in_leaf,
            l2_leaf_reg=l2_leaf_reg,
            pos_weight=pos_weight,
            pos_weight_multiplier=pos_weight_multiplier,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            **kwargs,
        )

        self.model: Optional[cb.CatBoostClassifier | cb.CatBoostRegressor] = None
        self.feature_names: Optional[list] = None
        self.best_iteration: int = 0
        self._objective_type: Optional[str] = objective_type
        self._binary_labels: Optional[np.ndarray] = binary_labels

    def _compute_pos_weight(self, y: np.ndarray) -> float:
        """Compute pos_weight as (N-P)/P.

        Args:
            y: Binary labels

        Returns:
            Positive class weight
        """
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Auto-computed pos_weight: {weight:.4f} (N={n_neg}, P={n_pos})")
        return weight

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[tuple] = None,
        binary_labels_valid: Optional[np.ndarray] = None,
    ) -> "CatBoostClassifier":
        """Fit CatBoost model.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            eval_set: Optional validation set tuple (X_val, y_val)
            binary_labels_valid: Binary labels (0/1) for validation set ROC-AUC calculation

        Returns:
            Self
        """
        # Filter to numeric columns only (exclude SMILES and other non-numeric columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(
                f"Training data has 0 numeric features. "
                f"All columns: {list(X.columns)}, X.shape={X.shape}"
            )

        # Use only numeric columns
        X = X[numeric_cols]

        # Store feature names
        self.feature_names = numeric_cols

        # Check objective type
        objective_type = self._objective_type or self.params.get("objective_type")

        # Prepare CatBoost parameters
        cat_params = {
            "iterations": self.params["n_estimators"],
            "learning_rate": self.params["learning_rate"],
            "depth": self.params["max_depth"],
            "subsample": self.params["subsample"],
            "colsample_bylevel": self.params["colsample_bylevel"],
            "min_data_in_leaf": self.params["min_data_in_leaf"],
            "l2_leaf_reg": self.params["l2_leaf_reg"],
            "random_seed": 42,
            "verbose": self.params["verbose"],
        }

        # Set loss function and eval metric based on objective type
        # Store binary_labels_valid for use in custom metric
        self._binary_labels_valid = binary_labels_valid

        if objective_type == "regression":
            logger.info("Using regression (RMSE) objective")
            cat_params["loss_function"] = "RMSE"
            # For regression, we'll use custom eval metric for ROC-AUC if binary labels are available
            if eval_set is not None and binary_labels_valid is not None:
                # Create custom ROC-AUC eval metric
                # CatBoost custom metric signature: (approx, target, weight) -> (metric_name, metric_value, is_higher_better)
                def roc_auc_metric(approx: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[str, float, bool]:
                    """Custom ROC-AUC metric for CatBoost regression.

                    Args:
                        approx: Predicted values (quantitative)
                        target: True values (quantitative, not used for ROC-AUC)
                        weight: Sample weights (not used for ROC-AUC)

                    Returns:
                        Tuple of (metric_name, metric_value, is_higher_better)
                    """
                    # Apply sigmoid to get probabilities from regression predictions
                    probs = 1.0 / (1.0 + np.exp(-approx))
                    try:
                        auc = roc_auc_score(binary_labels_valid, probs)
                        return 'ROC-AUC', auc, True  # True means higher is better
                    except ValueError:
                        # Handle case where only one class is present
                        return 'ROC-AUC', 0.0, True

                cat_params["eval_metric"] = roc_auc_metric
                logger.info("Using custom ROC-AUC evaluation metric for regression")
            else:
                # Use RMSE for early stopping if no binary labels
                cat_params["eval_metric"] = "RMSE"
                if eval_set is not None:
                    logger.warning("Binary labels for validation set not provided. Using RMSE metric for early stopping instead of ROC-AUC.")
        else:
            # Binary classification
            cat_params["loss_function"] = "Logloss"
            cat_params["eval_metric"] = "AUC"

            # Handle pos_weight
            pos_weight = self.params.get("pos_weight")
            if pos_weight is None:
                pos_weight = self._compute_pos_weight(y)

            # Apply multiplier if specified
            pos_weight_multiplier = self.params.get("pos_weight_multiplier")
            if pos_weight_multiplier is not None:
                logger.info(f"Applying pos_weight_multiplier: {pos_weight_multiplier:.4f}")
                pos_weight = pos_weight * pos_weight_multiplier
                logger.info(f"Final pos_weight: {pos_weight:.4f}")

            # CatBoost uses class_weights parameter
            # class_weights = [weight_negative, weight_positive]
            # For binary classification with pos_weight, we set:
            # class_weights = [1.0, pos_weight]
            cat_params["class_weights"] = [1.0, pos_weight]

        # Handle verbose parameter: CatBoost requires non-negative integer
        # Convert -1 (silent) to False or 0
        if cat_params.get("verbose") == -1:
            cat_params["verbose"] = False

        # Add any additional kwargs
        excluded_keys = [
            "name",
            "n_estimators",
            "max_depth",  # Already converted to 'depth' above
            "pos_weight",
            "pos_weight_multiplier",
            "early_stopping_rounds",
            "use_focal_loss",
            "focal_alpha",
            "focal_gamma",
            "focal_scale",
            "verbose",  # Already handled above
            "objective_type",
            "binary_labels",
        ]
        for key, value in self.params.items():
            if key not in cat_params and key not in excluded_keys:
                cat_params[key] = value

        # Debug: log final parameters (excluding sensitive info)
        logger.debug(f"CatBoost parameters: { {k: v for k, v in cat_params.items() if k not in ['class_weights']} }")

        # Prepare training data
        X_train = X.values
        if objective_type == "regression":
            y_train = y.astype(float)
        else:
            y_train = y.astype(int)

        # Prepare validation set
        eval_set_cb = None
        if eval_set is not None:
            X_val, y_val = eval_set
            # Ensure validation set uses same numeric columns as training
            X_val = X_val[numeric_cols]
            if objective_type == "regression":
                y_val_for_pool = y_val.astype(float)
            else:
                y_val_for_pool = y_val.astype(int)
            eval_set_cb = cb.Pool(
                X_val.values,
                label=y_val_for_pool,
                feature_names=self.feature_names,
            )

        # Create and train model
        train_pool = cb.Pool(
            X_train,
            label=y_train,
            weight=sample_weight,
            feature_names=self.feature_names,
        )

        logger.info(f"Training CatBoost with {self.params['n_estimators']} rounds")

        # Use CatBoostRegressor for regression, CatBoostClassifier for binary classification
        if objective_type == "regression":
            self.model = cb.CatBoostRegressor(**cat_params)
        else:
            self.model = cb.CatBoostClassifier(**cat_params)

        # Train with early stopping if validation set is provided
        # Convert verbose for fit method (CatBoost requires non-negative integer)
        verbose_fit = False if self.params["verbose"] == -1 else self.params["verbose"]
        if eval_set is not None:
            self.model.fit(
                train_pool,
                eval_set=eval_set_cb,
                early_stopping_rounds=self.params["early_stopping_rounds"],
                verbose=verbose_fit,
            )
            # Get best iteration from the model
            self.best_iteration = self.model.get_best_iteration()
            if self.best_iteration == -1:
                # Early stopping didn't trigger, use all iterations
                self.best_iteration = self.params["n_estimators"]
        else:
            self.model.fit(train_pool, verbose=verbose_fit)
            self.best_iteration = self.params["n_estimators"]

        logger.info(f"Training completed. Best iteration: {self.best_iteration}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.

        Args:
            X: Features

        Returns:
            Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        # Ensure feature order matches training
        if self.feature_names:
            # Check which features are missing
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(
                    f"Missing {len(missing_features)} features in prediction data. "
                    f"First few missing: {list(missing_features)[:5]}. "
                    "Filling with zeros."
                )
                # Create DataFrame with all required features, filling missing ones with zeros
                X = X.reindex(columns=self.feature_names, fill_value=0.0)
            else:
                # All features present, just reorder
                X = X[self.feature_names]

        objective_type = self._objective_type or self.params.get("objective_type")

        if objective_type == "regression":
            # For regression, predictions are raw values, apply sigmoid to get probabilities
            preds = self.model.predict(X.values)
            preds = 1.0 / (1.0 + np.exp(-preds))
        else:
            # CatBoost returns probabilities for both classes, extract positive class
            preds = self.model.predict_proba(X.values)[:, 1]

        return preds

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / "model.cbm"
        self.model.save_model(str(model_path))

        # Save metadata
        metadata = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "binary_labels": self._binary_labels,
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "CatBoostClassifier":
        """Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(name=metadata["name"], **metadata["params"])

        # Load model
        model_path = path / "model.cbm"
        # Determine model type from metadata or try to load as classifier first
        objective_type = metadata["params"].get("objective_type")
        if objective_type == "regression":
            instance.model = cb.CatBoostRegressor()
        else:
            instance.model = cb.CatBoostClassifier()
        instance.model.load_model(str(model_path))

        instance.feature_names = metadata["feature_names"]
        instance.best_iteration = metadata["best_iteration"]
        instance._objective_type = objective_type
        instance._binary_labels = metadata.get("binary_labels")

        logger.info(f"Loaded model from {path}")
        return instance

    def get_feature_importance(self, importance_type: str = "PredictionValuesChange") -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance
                ("PredictionValuesChange", "LossFunctionChange", "FeatureImportance")

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        importance = self.model.get_feature_importance(importance_type=importance_type)
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return df


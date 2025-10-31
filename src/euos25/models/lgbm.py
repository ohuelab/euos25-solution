"""LightGBM classifier implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)


class LGBMClassifier(BaseClfModel):
    """LightGBM binary classifier with pos_weight support."""

    def __init__(
        self,
        name: str = "lgbm",
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = -1,
        num_leaves: int = 127,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
        **kwargs: Any,
    ):
        """Initialize LightGBM classifier.

        Args:
            name: Model name
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth (-1 = no limit)
            num_leaves: Maximum number of leaves
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of features
            min_child_samples: Minimum samples in leaf
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            pos_weight: Weight for positive class (auto-computed if None)
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level
            **kwargs: Additional LightGBM parameters
        """
        super().__init__(
            name=name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            pos_weight=pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            **kwargs,
        )

        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[list] = None
        self.best_iteration: int = 0

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
    ) -> "LGBMClassifier":
        """Fit LightGBM model.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            eval_set: Optional validation set tuple (X_val, y_val)

        Returns:
            Self
        """
        # Store feature names
        self.feature_names = list(X.columns)

        # Compute pos_weight if not provided
        pos_weight = self.params.get("pos_weight")
        if pos_weight is None:
            pos_weight = self._compute_pos_weight(y)

        # Prepare LightGBM parameters
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": self.params["learning_rate"],
            "num_leaves": self.params["num_leaves"],
            "max_depth": self.params["max_depth"],
            "subsample": self.params["subsample"],
            "subsample_freq": 1,
            "colsample_bytree": self.params["colsample_bytree"],
            "min_child_samples": self.params["min_child_samples"],
            "reg_alpha": self.params["reg_alpha"],
            "reg_lambda": self.params["reg_lambda"],
            "scale_pos_weight": pos_weight,
            "verbose": self.params["verbose"],
            "seed": 42,
        }

        # Add any additional kwargs
        for key, value in self.params.items():
            if key not in lgb_params and key not in [
                "name", "n_estimators", "pos_weight", "early_stopping_rounds"
            ]:
                lgb_params[key] = value

        # Create dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            feature_name=self.feature_names,
        )

        # Prepare validation set
        valid_sets = [train_data]
        valid_names = ["train"]

        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=self.feature_names,
            )
            valid_sets.append(valid_data)
            valid_names.append("valid")

        # Train model
        logger.info(f"Training LightGBM with {self.params['n_estimators']} rounds")
        callbacks = []
        if eval_set is not None:
            callbacks.append(lgb.early_stopping(self.params["early_stopping_rounds"]))

        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration
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
            X = X[self.feature_names]

        preds = self.model.predict(X, num_iteration=self.best_iteration)
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
        model_path = path / "model.txt"
        self.model.save_model(str(model_path))

        # Save metadata
        metadata = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMClassifier":
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
        model_path = path / "model.txt"
        instance.model = lgb.Booster(model_file=str(model_path))

        instance.feature_names = metadata["feature_names"]
        instance.best_iteration = metadata["best_iteration"]

        logger.info(f"Loaded model from {path}")
        return instance

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance ("gain" or "split")

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        importance = self.model.feature_importance(importance_type=importance_type)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df

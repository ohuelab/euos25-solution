"""Random Forest classifier implementation with large-scale data optimizations."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)


class RandomForestClassifierModel(BaseClfModel):
    """Random Forest binary classifier with pos_weight support and large-scale optimizations.

    Optimizations for large-scale data:
    - Parallel processing via n_jobs
    - max_samples for subsampling training data
    - max_features for feature subsampling
    - class_weight for handling imbalanced data
    """

    def __init__(
        self,
        name: str = "random_forest",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = "sqrt",
        max_samples: Optional[float] = None,
        bootstrap: bool = True,
        class_weight: Optional[str] = "balanced",
        pos_weight: Optional[float] = None,
        pos_weight_multiplier: Optional[float] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """Initialize Random Forest classifier.

        Args:
            name: Model name
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = no limit)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
                ("sqrt", "log2", int, float, None)
            max_samples: Proportion or number of samples to draw for each tree
                (None = use all samples, float = proportion, int = number)
            bootstrap: Whether to use bootstrap samples
            class_weight: Class weight mode ("balanced", "balanced_subsample", dict, None)
                If pos_weight is provided, will be converted to dict
            pos_weight: Weight for positive class (auto-computed if None)
            pos_weight_multiplier: Multiplier for auto-computed pos_weight
            n_jobs: Number of parallel jobs (-1 = use all cores)
            random_state: Random seed
            verbose: Verbosity level (0 = silent, 1 = progress bar)
            **kwargs: Additional RandomForestClassifier parameters
        """
        super().__init__(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap=bootstrap,
            class_weight=class_weight,
            pos_weight=pos_weight,
            pos_weight_multiplier=pos_weight_multiplier,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[list] = None
        self.best_iteration: int = n_estimators  # RandomForest doesn't have early stopping

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
    ) -> "RandomForestClassifierModel":
        """Fit Random Forest model.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights (passed to RandomForest)
            eval_set: Optional validation set tuple (ignored, RandomForest doesn't use it)

        Returns:
            Self
        """
        # Filter to numeric columns only
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

        # Prepare RandomForest parameters
        rf_params = {
            "n_estimators": self.params["n_estimators"],
            "max_depth": self.params["max_depth"],
            "min_samples_split": self.params["min_samples_split"],
            "min_samples_leaf": self.params["min_samples_leaf"],
            "max_features": self.params["max_features"],
            "max_samples": self.params["max_samples"],
            "bootstrap": self.params["bootstrap"],
            "n_jobs": self.params["n_jobs"],
            "random_state": self.params["random_state"],
            "verbose": self.params["verbose"],
            "class_weight": self.params.get("class_weight", None),
        }

        # Handle pos_weight
        pos_weight = self.params.get("pos_weight")
        if pos_weight is not None:
            # Apply multiplier if specified
            pos_weight_multiplier = self.params.get("pos_weight_multiplier")
            if pos_weight_multiplier is not None:
                logger.info(f"Applying pos_weight_multiplier: {pos_weight_multiplier:.4f}")
                pos_weight = pos_weight * pos_weight_multiplier
                logger.info(f"Final pos_weight: {pos_weight:.4f}")

            # Convert pos_weight to class_weight dict
            # class_weight = {0: 1.0, 1: pos_weight}
            rf_params["class_weight"] = {0: 1.0, 1: pos_weight}
        elif self.params.get("pos_weight_multiplier") is not None:
            # Compute pos_weight from data and apply multiplier
            pos_weight = self._compute_pos_weight(y)
            multiplier = self.params["pos_weight_multiplier"]
            logger.info(f"Applying pos_weight_multiplier: {multiplier:.4f}")
            pos_weight = pos_weight * multiplier
            logger.info(f"Final pos_weight: {pos_weight:.4f}")
            rf_params["class_weight"] = {0: 1.0, 1: pos_weight}
        elif rf_params["class_weight"] is None:
            # Use balanced class_weight as default
            rf_params["class_weight"] = "balanced"

        # Log max_samples if set (important for large-scale data)
        if rf_params["max_samples"] is not None:
            if isinstance(rf_params["max_samples"], float):
                logger.info(f"Using max_samples={rf_params['max_samples']:.2f} (proportion)")
            else:
                logger.info(f"Using max_samples={rf_params['max_samples']} (absolute number)")

        # Add any additional kwargs
        excluded_keys = [
            "name",
            "pos_weight",
            "pos_weight_multiplier",
            "early_stopping_rounds",
            "use_focal_loss",
            "focal_alpha",
            "focal_gamma",
            "focal_scale",
        ]
        for key, value in self.params.items():
            if key not in rf_params and key not in excluded_keys:
                rf_params[key] = value

        # Create and train model
        logger.info(
            f"Training RandomForest with {self.params['n_estimators']} trees, "
            f"n_jobs={rf_params['n_jobs']}"
        )
        if rf_params["max_samples"] is not None:
            logger.info(f"  max_samples: {rf_params['max_samples']}")
        logger.info(f"  max_features: {rf_params['max_features']}")
        logger.info(f"  class_weight: {rf_params['class_weight']}")

        self.model = RandomForestClassifier(**rf_params)
        self.model.fit(X.values, y.astype(int), sample_weight=sample_weight)

        logger.info("Training completed")

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

        # RandomForest returns probabilities for both classes, extract positive class
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

        # Save model using pickle (sklearn models use pickle)
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

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
    def load(cls, path: str) -> "RandomForestClassifierModel":
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
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            instance.model = pickle.load(f)

        instance.feature_names = metadata["feature_names"]
        instance.best_iteration = metadata["best_iteration"]

        logger.info(f"Loaded model from {path}")
        return instance

    def get_feature_importance(self, importance_type: str = "gini") -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance ("gini" or "permutation")

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        if importance_type == "gini":
            importance = self.model.feature_importances_
        elif importance_type == "permutation":
            # Note: permutation importance requires validation data
            raise NotImplementedError(
                "Permutation importance requires validation data. Use sklearn.inspection.permutation_importance instead."
            )
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return df


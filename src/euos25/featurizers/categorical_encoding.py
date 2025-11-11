"""Categorical encoding featurizers (Label Encoding and Target Encoding)."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


def detect_categorical_descriptors(
    features: pd.DataFrame,
    max_unique_values: int = 100,
    descriptor_columns: Optional[List[str]] = None,
) -> List[str]:
    """Detect categorical descriptor columns.

    Args:
        features: DataFrame with features
        max_unique_values: Maximum number of unique values to consider as categorical
        descriptor_columns: Optional list of descriptor column names to check.
            If None, automatically detects columns with 'rdkit2d__' or 'mordred__' prefix.

    Returns:
        List of categorical descriptor column names
    """
    if descriptor_columns is None:
        # Auto-detect descriptor columns
        descriptor_columns = [
            col
            for col in features.columns
            if col.startswith("rdkit2d__") or col.startswith("mordred__")
        ]

    categorical_cols = []
    for col in descriptor_columns:
        if col not in features.columns:
            continue

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue

        # Check unique values
        unique_count = features[col].nunique()
        if unique_count <= max_unique_values:
            categorical_cols.append(col)
            logger.debug(f"Detected categorical descriptor: {col} ({unique_count} unique values)")

    logger.info(
        f"Detected {len(categorical_cols)} categorical descriptors "
        f"(max_unique_values={max_unique_values})"
    )
    return categorical_cols


class LabelEncoder:
    """Label encoder for categorical features."""

    def __init__(self):
        """Initialize label encoder."""
        self.mapping: Dict[Union[str, float], int] = {}
        self.inverse_mapping: Dict[int, Union[str, float]] = {}

    def fit(self, values: pd.Series) -> "LabelEncoder":
        """Fit encoder on values.

        Args:
            values: Series of categorical values

        Returns:
            Self
        """
        unique_values = sorted(values.dropna().unique())
        self.mapping = {val: idx for idx, val in enumerate(unique_values)}
        self.inverse_mapping = {idx: val for val, idx in self.mapping.items()}
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        """Transform values to encoded labels.

        Args:
            values: Series of categorical values

        Returns:
            Series of encoded labels (NaN preserved)
        """
        encoded = values.map(self.mapping)
        return encoded

    def fit_transform(self, values: pd.Series) -> pd.Series:
        """Fit and transform in one step.

        Args:
            values: Series of categorical values

        Returns:
            Series of encoded labels
        """
        return self.fit(values).transform(values)


class TargetEncoder:
    """Target encoder for categorical features."""

    def __init__(self, smoothing: float = 1.0):
        """Initialize target encoder.

        Args:
            smoothing: Smoothing parameter for target encoding
        """
        self.smoothing = smoothing
        self.mapping: Dict[Union[str, float], float] = {}
        self.global_mean: float = 0.0

    def fit(self, values: pd.Series, target: pd.Series) -> "TargetEncoder":
        """Fit encoder on values and target.

        Args:
            values: Series of categorical values
            target: Series of target values

        Returns:
            Self
        """
        # Calculate global mean
        self.global_mean = target.mean()

        # Calculate category means
        df = pd.DataFrame({"value": values, "target": target})
        category_stats = df.groupby("value")["target"].agg(["mean", "count"])

        # Apply smoothing
        for category, row in category_stats.iterrows():
            category_mean = row["mean"]
            category_count = row["count"]
            # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = (
                category_count * category_mean + self.smoothing * self.global_mean
            ) / (category_count + self.smoothing)
            self.mapping[category] = smoothed_mean

        return self

    def transform(self, values: pd.Series) -> pd.Series:
        """Transform values to target-encoded values.

        Args:
            values: Series of categorical values

        Returns:
            Series of target-encoded values (unknown categories get global_mean)
        """
        encoded = values.map(self.mapping)
        # Fill unknown categories with global mean
        encoded = encoded.fillna(self.global_mean)
        return encoded

    def fit_transform(self, values: pd.Series, target: pd.Series) -> pd.Series:
        """Fit and transform in one step.

        Args:
            values: Series of categorical values
            target: Series of target values

        Returns:
            Series of target-encoded values
        """
        return self.fit(values, target).transform(values)


class LabelEncodingFeaturizer(BaseFeaturizer):
    """Label encoding featurizer for categorical descriptors."""

    def __init__(
        self,
        name: str = "label_encoding",
        descriptor_columns: Optional[List[str]] = None,
        max_unique_values: int = 100,
        auto_detect: bool = True,
    ):
        """Initialize label encoding featurizer.

        Args:
            name: Feature name prefix
            descriptor_columns: List of descriptor column names to encode.
                If None and auto_detect=True, automatically detects categorical descriptors.
            max_unique_values: Maximum number of unique values to consider as categorical
            auto_detect: Whether to auto-detect categorical descriptors if descriptor_columns is None
        """
        super().__init__(name=name)
        self.descriptor_columns = descriptor_columns
        self.max_unique_values = max_unique_values
        self.auto_detect = auto_detect
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(self, features: pd.DataFrame) -> "LabelEncodingFeaturizer":
        """Fit encoders on features.

        Args:
            features: DataFrame with features

        Returns:
            Self
        """
        # Determine columns to encode
        if self.descriptor_columns is None and self.auto_detect:
            self.descriptor_columns = detect_categorical_descriptors(
                features, max_unique_values=self.max_unique_values
            )

        if self.descriptor_columns is None:
            logger.warning("No descriptor columns specified for label encoding")
            return self

        # Filter to columns that exist
        available_cols = [col for col in self.descriptor_columns if col in features.columns]
        if not available_cols:
            logger.warning("No available descriptor columns found for label encoding")
            return self

        # Fit encoders for each column
        self.encoders = {}
        for col in available_cols:
            encoder = LabelEncoder()
            encoder.fit(features[col])
            self.encoders[col] = encoder

        logger.info(f"Fitted label encoders for {len(self.encoders)} columns")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted encoders.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with label-encoded features
        """
        if not self.encoders:
            logger.warning("No encoders fitted, returning empty DataFrame")
            return pd.DataFrame(index=features.index)

        encoded_features = {}
        for col, encoder in self.encoders.items():
            if col not in features.columns:
                logger.warning(f"Column {col} not found in features, skipping")
                continue

            encoded_col = encoder.transform(features[col])
            encoded_features[f"{self.name}__{col}"] = encoded_col

        if not encoded_features:
            return pd.DataFrame(index=features.index)

        result_df = pd.DataFrame(encoded_features, index=features.index)
        logger.info(f"Generated {len(result_df.columns)} label-encoded features")
        return result_df

    def save(self, path: str) -> None:
        """Save encoders to file.

        Args:
            path: Path to save encoders
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "wb") as f:
            pickle.dump(
                {
                    "encoders": self.encoders,
                    "descriptor_columns": self.descriptor_columns,
                    "max_unique_values": self.max_unique_values,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "LabelEncodingFeaturizer":
        """Load encoders from file.

        Args:
            path: Path to load encoders from

        Returns:
            LabelEncodingFeaturizer instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        featurizer = cls(
            descriptor_columns=data["descriptor_columns"],
            max_unique_values=data["max_unique_values"],
        )
        featurizer.encoders = data["encoders"]
        return featurizer


class TargetEncodingFeaturizer(BaseFeaturizer):
    """Target encoding featurizer for categorical descriptors."""

    def __init__(
        self,
        name: str = "target_encoding",
        descriptor_columns: Optional[List[str]] = None,
        max_unique_values: int = 100,
        auto_detect: bool = True,
        smoothing: float = 1.0,
    ):
        """Initialize target encoding featurizer.

        Args:
            name: Feature name prefix
            descriptor_columns: List of descriptor column names to encode.
                If None and auto_detect=True, automatically detects categorical descriptors.
            max_unique_values: Maximum number of unique values to consider as categorical
            auto_detect: Whether to auto-detect categorical descriptors if descriptor_columns is None
            smoothing: Smoothing parameter for target encoding
        """
        super().__init__(name=name)
        self.descriptor_columns = descriptor_columns
        self.max_unique_values = max_unique_values
        self.auto_detect = auto_detect
        self.smoothing = smoothing
        self.encoders: Dict[str, TargetEncoder] = {}

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "TargetEncodingFeaturizer":
        """Fit encoders on features and target.

        Args:
            features: DataFrame with features
            target: Series with target values (indexed by same index as features)

        Returns:
            Self
        """
        # Determine columns to encode
        if self.descriptor_columns is None and self.auto_detect:
            self.descriptor_columns = detect_categorical_descriptors(
                features, max_unique_values=self.max_unique_values
            )

        if self.descriptor_columns is None:
            logger.warning("No descriptor columns specified for target encoding")
            return self

        # Filter to columns that exist
        available_cols = [col for col in self.descriptor_columns if col in features.columns]
        if not available_cols:
            logger.warning("No available descriptor columns found for target encoding")
            return self

        # Align features and target by index
        common_idx = features.index.intersection(target.index)
        if len(common_idx) == 0:
            raise ValueError("No common indices between features and target")

        features_aligned = features.loc[common_idx]
        target_aligned = target.loc[common_idx]

        # Fit encoders for each column
        self.encoders = {}
        for col in available_cols:
            encoder = TargetEncoder(smoothing=self.smoothing)
            encoder.fit(features_aligned[col], target_aligned)
            self.encoders[col] = encoder

        logger.info(f"Fitted target encoders for {len(self.encoders)} columns")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted encoders.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with target-encoded features
        """
        if not self.encoders:
            logger.warning("No encoders fitted, returning empty DataFrame")
            return pd.DataFrame(index=features.index)

        encoded_features = {}
        for col, encoder in self.encoders.items():
            if col not in features.columns:
                logger.warning(f"Column {col} not found in features, skipping")
                continue

            encoded_col = encoder.transform(features[col])
            encoded_features[f"{self.name}__{col}"] = encoded_col

        if not encoded_features:
            return pd.DataFrame(index=features.index)

        result_df = pd.DataFrame(encoded_features, index=features.index)
        logger.info(f"Generated {len(result_df.columns)} target-encoded features")
        return result_df

    def save(self, path: str) -> None:
        """Save encoders to file.

        Args:
            path: Path to save encoders
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "wb") as f:
            pickle.dump(
                {
                    "encoders": self.encoders,
                    "descriptor_columns": self.descriptor_columns,
                    "max_unique_values": self.max_unique_values,
                    "smoothing": self.smoothing,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "TargetEncodingFeaturizer":
        """Load encoders from file.

        Args:
            path: Path to load encoders from

        Returns:
            TargetEncodingFeaturizer instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        featurizer = cls(
            descriptor_columns=data["descriptor_columns"],
            max_unique_values=data["max_unique_values"],
            smoothing=data["smoothing"],
        )
        featurizer.encoders = data["encoders"]
        return featurizer


"""Feature generation pipeline."""

import logging
from pathlib import Path
from typing import List

import pandas as pd

from euos25.config import Config, FeaturizerConfig
from euos25.featurizers.base import Featurizer
from euos25.featurizers.conj_proxy import ConjugationProxyFeaturizer
from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.mordred import MordredFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer
from euos25.utils.io import load_csv, load_parquet, save_parquet

try:
    from euos25.featurizers.chemeleon import ChemeleonFeaturizer
except ImportError:
    ChemeleonFeaturizer = None

try:
    from euos25.featurizers.chemberta import ChemBERTaFeaturizer
except ImportError:
    ChemBERTaFeaturizer = None

logger = logging.getLogger(__name__)


# Define feature group mappings
FEATURE_GROUP_MAPPING = {
    "ecfp": "ecfp4",
    "rdkit2d": "rdkit2d",
    "mordred": "mordred",
    "chemeleon": "chemeleon",
    "chemberta": "chemberta",
    "conj_proxy": "custom",
}

# Map of group name variants (with aggregation suffixes) to base group names
# This handles cases like 'chemeleon_mean' -> 'chemeleon' when features were
# created with aggregation-specific names
GROUP_NAME_VARIANTS = {
    "chemeleon_mean": "chemeleon",
    "chemeleon_sum": "chemeleon",
    "chemeleon_norm": "chemeleon",
    "chemberta_mean": "chemberta",
    "chemberta_sum": "chemberta",
    "chemberta_norm": "chemberta",
}


def create_featurizer(config: FeaturizerConfig) -> Featurizer:
    """Create featurizer from configuration.

    Args:
        config: Featurizer configuration

    Returns:
        Featurizer instance
    """
    if config.name == "ecfp":
        return ECFPFeaturizer(**config.params)
    elif config.name == "rdkit2d":
        return RDKit2DFeaturizer(**config.params)
    elif config.name == "mordred":
        return MordredFeaturizer(**config.params)
    elif config.name == "conj_proxy":
        return ConjugationProxyFeaturizer(**config.params)
    elif config.name == "chemeleon":
        if ChemeleonFeaturizer is None:
            raise ImportError(
                "ChemeleonFeaturizer is not available. "
                "Make sure chemprop is installed."
            )
        return ChemeleonFeaturizer(**config.params)
    elif config.name == "chemberta":
        if ChemBERTaFeaturizer is None:
            raise ImportError(
                "ChemBERTaFeaturizer is not available. "
                "Make sure transformers is installed."
            )
        return ChemBERTaFeaturizer(**config.params)
    else:
        raise ValueError(f"Unknown featurizer: {config.name}")


def build_features(
    df: pd.DataFrame,
    featurizers: List[Featurizer],
    smiles_col: str = "SMILES",
    id_col: str = "ID",
) -> pd.DataFrame:
    """Build features from SMILES using multiple featurizers.

    Args:
        df: DataFrame with SMILES
        featurizers: List of featurizers
        smiles_col: Name of SMILES column
        id_col: Name of ID column

    Returns:
        DataFrame with all features (indexed by ID)
    """
    logger.info(f"Building features with {len(featurizers)} featurizers")

    # Validate input DataFrame
    if df.empty:
        raise ValueError(
            f"Input DataFrame is empty. Cannot generate features from empty data."
        )

    if smiles_col not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(
            f"Required column '{smiles_col}' not found in input DataFrame. "
            f"Available columns: {available_cols}. "
            f"DataFrame shape: {df.shape}"
        )

    logger.info(
        f"Input DataFrame: {len(df)} rows, {len(df.columns)} columns. "
        f"Columns: {list(df.columns)}"
    )

    # If no featurizers, return DataFrame with just SMILES for models like ChemProp
    if not featurizers:
        logger.info("No featurizers specified - creating SMILES-only DataFrame for graph-based models")
        features = df[[smiles_col]].copy()

        # Set ID as index if available
        if id_col in df.columns:
            features.index = df[id_col].values

        logger.info(f"Created SMILES-only DataFrame with {len(features)} samples")
        return features

    # Start with ID index
    feature_dfs = []
    failed_featurizers = []

    # Generate features from each featurizer
    for featurizer in featurizers:
        logger.info(f"Applying featurizer: {featurizer.name}")
        try:
            feat_df = featurizer.transform(df, smiles_col=smiles_col)

            # Check both empty and column count explicitly
            # A DataFrame with no columns but with rows would have empty=False
            # but would still not have any features
            if not feat_df.empty and len(feat_df.columns) > 0:
                # Add group prefix to column names for identification
                group_name = FEATURE_GROUP_MAPPING.get(featurizer.name, featurizer.name)
                feat_df.columns = [f"{group_name}__{col}" for col in feat_df.columns]
                feature_dfs.append(feat_df)
                logger.info(
                    f"Featurizer {featurizer.name} generated {len(feat_df.columns)} features "
                    f"for {len(feat_df)} samples"
                )
            else:
                failed_featurizers.append(featurizer.name)
                logger.warning(
                    f"Featurizer {featurizer.name} produced no features. "
                    f"Result shape: {feat_df.shape}, empty: {feat_df.empty}, "
                    f"columns: {len(feat_df.columns)}"
                )
        except Exception as e:
            failed_featurizers.append(featurizer.name)
            logger.error(
                f"Featurizer {featurizer.name} failed with error: {e}",
                exc_info=True
            )

    if not feature_dfs:
        error_msg = (
            f"No features were generated. All {len(featurizers)} featurizers failed or produced empty results.\n"
            f"Failed featurizers: {failed_featurizers}\n"
            f"Input DataFrame: {len(df)} rows, columns: {list(df.columns)}\n"
            f"SMILES column: {smiles_col}, first few SMILES: {df[smiles_col].head(3).tolist()}"
        )
        raise ValueError(error_msg)

    # Log summary of successful featurizers
    if failed_featurizers:
        logger.warning(
            f"Some featurizers failed: {failed_featurizers}. "
            f"Successfully generated features from {len(feature_dfs)} out of {len(featurizers)} featurizers."
        )

    # Concatenate all features
    features = pd.concat(feature_dfs, axis=1)

    # Set ID as index
    if id_col in df.columns:
        features.index = df[id_col].values

    logger.info(f"Total features generated: {len(features.columns)} from {len(feature_dfs)} featurizers")
    return features


def get_available_feature_groups(features: pd.DataFrame) -> set[str]:
    """Extract available feature groups from features DataFrame.

    Args:
        features: DataFrame with features (columns have group prefixes like 'ecfp4__...')

    Returns:
        Set of available feature group names (normalized to base group names)
    """
    groups = set()
    for col in features.columns:
        if "__" in col:
            group = col.split("__")[0]
            # Normalize group name if it has a known variant
            normalized_group = GROUP_NAME_VARIANTS.get(group, group)
            groups.add(normalized_group)
        else:
            # If no prefix, treat as a special group
            groups.add("unknown")
    return groups


def get_feature_groups_from_config(config: Config) -> dict[str, bool]:
    """Get feature group settings from config.

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary mapping group names to boolean values (True if enabled).
        Returns empty dict if no featurizers specified (will use all features).
    """
    # Get feature group names from config featurizers
    enabled_groups = set()
    for feat_cfg in config.featurizers:
        group_name = FEATURE_GROUP_MAPPING.get(feat_cfg.name)
        if group_name:
            enabled_groups.add(group_name)

    # If no featurizers specified (e.g., for ChemProp), return empty dict (will use all)
    if not enabled_groups:
        return {}

    # Build settings dict: all known groups, enabled ones set to True
    all_groups = set(FEATURE_GROUP_MAPPING.values())
    group_settings = {group: group in enabled_groups for group in all_groups}

    return group_settings


def filter_feature_groups(
    features: pd.DataFrame,
    group_settings: dict[str, bool] | None = None,
    use_ecfp4: bool = True,
    use_rdkit2d: bool = True,
    use_mordred: bool = True,
    use_chemeleon: bool = True,
    use_chemberta: bool = True,
    use_custom: bool = True,
    **kwargs: bool,
) -> pd.DataFrame:
    """Filter features by group selection.

    Args:
        features: DataFrame with features (columns have group prefixes)
        group_settings: Dictionary mapping group names to boolean values. If provided,
            this takes precedence over individual use_* parameters.
        use_ecfp4: Whether to include ECFP4 features (used if group_settings not provided)
        use_rdkit2d: Whether to include RDKit 2D features (used if group_settings not provided)
        use_mordred: Whether to include Mordred features (used if group_settings not provided)
        use_chemeleon: Whether to include Chemeleon features (used if group_settings not provided)
        use_chemberta: Whether to include ChemBERTa features (used if group_settings not provided)
        use_custom: Whether to include custom features (used if group_settings not provided)
        **kwargs: Additional group settings (e.g., use_new_group=True)

    Returns:
        Filtered DataFrame with selected feature groups
    """
    # If group_settings dict is provided, use it directly
    if group_settings is not None:
        settings = group_settings.copy()
        # Add any additional kwargs (remove 'use_' prefix)
        for key, value in kwargs.items():
            if key.startswith("use_"):
                group_name = key[4:]  # Remove 'use_' prefix
                settings[group_name] = value
    else:
        # Build from individual parameters (backward compatibility)
        settings = {
            "ecfp4": use_ecfp4,
            "rdkit2d": use_rdkit2d,
            "mordred": use_mordred,
            "chemeleon": use_chemeleon,
            "chemberta": use_chemberta,
            "custom": use_custom,
        }
        # Add any additional kwargs (remove 'use_' prefix)
        for key, value in kwargs.items():
            if key.startswith("use_"):
                group_name = key[4:]  # Remove 'use_' prefix
                settings[group_name] = value

    # Select columns based on group settings
    selected_cols = []
    for col in features.columns:
        # Extract group prefix
        if "__" in col:
            group = col.split("__")[0]
            # Normalize group name if it has a known variant
            normalized_group = GROUP_NAME_VARIANTS.get(group, group)
            # Check both the original group name and normalized name
            if settings.get(normalized_group, settings.get(group, True)):  # Default to True if group not found
                selected_cols.append(col)
        else:
            # If no prefix, include by default
            selected_cols.append(col)

    if not selected_cols:
        raise ValueError("No feature groups selected - at least one must be enabled")

    filtered_features = features[selected_cols]

    logger.info(f"Filtered features: {len(filtered_features.columns)} features from {len(features.columns)} total")
    logger.info(f"Active groups: {[g for g, active in settings.items() if active]}")

    return filtered_features


def filter_low_quality_features(
    features: pd.DataFrame,
    max_nan_ratio: float = 0.99,
    min_variance: float = 1e-6,
    min_unique_ratio: float = 0.01,
    low_variance_threshold: float = 0.99,
) -> pd.DataFrame:
    """Filter out low-quality feature columns.

    Removes features that are:
    - Mostly NaN (above max_nan_ratio)
    - Low variance (below min_variance)
    - Mostly the same value (unique ratio below min_unique_ratio or
      one value accounts for more than low_variance_threshold of samples)

    Args:
        features: DataFrame with features
        max_nan_ratio: Maximum ratio of NaN values allowed (default: 0.99)
        min_variance: Minimum variance threshold (default: 1e-6)
        min_unique_ratio: Minimum ratio of unique values (default: 0.01)
        low_variance_threshold: Threshold for detecting mostly constant features.
            If a single value accounts for more than this ratio, the feature is removed.
            (default: 0.99)

    Returns:
        DataFrame with low-quality features removed
    """
    logger.info(f"Filtering low-quality features from {len(features.columns)} columns")

    initial_count = len(features.columns)
    columns_to_keep = []
    stats = {
        'high_nan': [],
        'low_variance': [],
        'low_unique_ratio': [],
        'mostly_constant': [],
    }

    for col in features.columns:
        col_data = features[col]

        # Check NaN ratio
        nan_ratio = col_data.isna().sum() / len(col_data)
        if nan_ratio > max_nan_ratio:
            stats['high_nan'].append(col)
            continue

        # Drop NaN for variance calculation
        col_data_no_nan = col_data.dropna()

        if len(col_data_no_nan) == 0:
            # All NaN after dropping
            stats['high_nan'].append(col)
            continue

        # Check variance
        variance = col_data_no_nan.var()
        if variance < min_variance:
            stats['low_variance'].append(col)
            continue

        # Check unique value ratio
        unique_ratio = col_data_no_nan.nunique() / len(col_data_no_nan)
        if unique_ratio < min_unique_ratio:
            stats['low_unique_ratio'].append(col)
            continue

        # Check if mostly constant (one value dominates)
        value_counts = col_data_no_nan.value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] > low_variance_threshold:
            stats['mostly_constant'].append(col)
            continue

        columns_to_keep.append(col)

    # Log statistics
    filtered_count = initial_count - len(columns_to_keep)
    logger.info(
        f"Filtered {filtered_count} low-quality features: "
        f"high NaN: {len(stats['high_nan'])}, "
        f"low variance: {len(stats['low_variance'])}, "
        f"low unique ratio: {len(stats['low_unique_ratio'])}, "
        f"mostly constant: {len(stats['mostly_constant'])}"
    )

    if filtered_count > 0:
        logger.info(f"Kept {len(columns_to_keep)}/{initial_count} features")
        # Log a few examples of filtered features (first 5 of each type)
        for stat_type, cols in stats.items():
            if cols:
                logger.debug(
                    f"Examples of {stat_type} features: {cols[:5]}"
                )

    filtered_features = features[columns_to_keep]

    return filtered_features


def build_features_from_config(
    input_path: str,
    output_path: str,
    config: Config,
) -> pd.DataFrame:
    """Build features from configuration.

    Args:
        input_path: Path to input CSV
        output_path: Path to save features Parquet
        config: Pipeline configuration

    Returns:
        DataFrame with features
    """
    output_path_obj = Path(output_path)

    # Check if output parquet already exists
    if output_path_obj.exists():
        logger.info(f"Features parquet already exists at {output_path}, loading from cache")
        existing_features = load_parquet(output_path)

        # Check which feature groups are required from config
        required_groups = get_feature_groups_from_config(config)
        # Extract only enabled groups (True values)
        required_group_names = {group for group, enabled in required_groups.items() if enabled}

        # Get available groups from existing features
        available_groups = get_available_feature_groups(existing_features)

        # Log for debugging
        logger.info(
            f"Checking feature groups: required={required_group_names}, "
            f"available={available_groups}, "
            f"config featurizers={[f.name for f in config.featurizers]}"
        )

        # Find missing groups
        missing_groups = required_group_names - available_groups

        if missing_groups:
            logger.info(f"Found missing feature groups: {missing_groups}. Adding them...")

            # Load input data to generate missing features
            df = load_csv(input_path)

            # Find featurizers for missing groups
            missing_featurizers = []
            for feat_cfg in config.featurizers:
                group_name = FEATURE_GROUP_MAPPING.get(feat_cfg.name)
                if group_name in missing_groups:
                    missing_featurizers.append(create_featurizer(feat_cfg))

            if missing_featurizers:
                # Generate only missing features
                missing_features = build_features(df, missing_featurizers)

                # Merge with existing features (on index/ID)
                # Ensure both have the same index alignment
                features = pd.concat([existing_features, missing_features], axis=1)

                # Save updated features
                save_parquet(features, output_path)
                logger.info(f"Added {len(missing_groups)} missing feature groups and saved to {output_path}")

                return features
            else:
                logger.warning(f"No featurizers found for missing groups: {missing_groups}")
                return existing_features
        else:
            logger.info("All required feature groups are present in cached features")
            return existing_features

    # No existing parquet file - build all features from scratch
    # Load data
    df = load_csv(input_path)

    # Create featurizers
    featurizers = [create_featurizer(feat_cfg) for feat_cfg in config.featurizers]

    # Build features
    features = build_features(df, featurizers)

    # Save features
    save_parquet(features, output_path)

    return features

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

    # Generate features from each featurizer
    for featurizer in featurizers:
        logger.info(f"Applying featurizer: {featurizer.name}")
        feat_df = featurizer.transform(df, smiles_col=smiles_col)

        if not feat_df.empty:
            # Add group prefix to column names for identification
            group_name = FEATURE_GROUP_MAPPING.get(featurizer.name, featurizer.name)
            feat_df.columns = [f"{group_name}__{col}" for col in feat_df.columns]
            feature_dfs.append(feat_df)
        else:
            logger.warning(f"Featurizer {featurizer.name} produced no features")

    if not feature_dfs:
        raise ValueError("No features were generated")

    # Concatenate all features
    features = pd.concat(feature_dfs, axis=1)

    # Set ID as index
    if id_col in df.columns:
        features.index = df[id_col].values

    logger.info(f"Total features generated: {len(features.columns)}")
    return features


def get_available_feature_groups(features: pd.DataFrame) -> set[str]:
    """Extract available feature groups from features DataFrame.

    Args:
        features: DataFrame with features (columns have group prefixes like 'ecfp4__...')

    Returns:
        Set of available feature group names
    """
    groups = set()
    for col in features.columns:
        if "__" in col:
            group = col.split("__")[0]
            groups.add(group)
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
            if settings.get(group, True):  # Default to True if group not found
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

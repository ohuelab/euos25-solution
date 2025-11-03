"""Feature generation pipeline."""

import logging
from typing import List

import pandas as pd

from euos25.config import Config, FeaturizerConfig
from euos25.featurizers.base import Featurizer
from euos25.featurizers.chemeleon import ChemeleonFeaturizer
from euos25.featurizers.conj_proxy import ConjugationProxyFeaturizer
from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.mordred import MordredFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer
from euos25.utils.io import load_csv, save_parquet

logger = logging.getLogger(__name__)


# Define feature group mappings
FEATURE_GROUP_MAPPING = {
    "ecfp": "ecfp4",
    "rdkit2d": "rdkit2d",
    "mordred": "mordred",
    "chemeleon": "chemeleon",
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
        return ChemeleonFeaturizer(**config.params)
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


def filter_feature_groups(
    features: pd.DataFrame,
    use_ecfp4: bool = True,
    use_rdkit2d: bool = True,
    use_mordred: bool = True,
    use_chemeleon: bool = True,
    use_custom: bool = True,
) -> pd.DataFrame:
    """Filter features by group selection.

    Args:
        features: DataFrame with features (columns have group prefixes)
        use_ecfp4: Whether to include ECFP4 features
        use_rdkit2d: Whether to include RDKit 2D features
        use_mordred: Whether to include Mordred features
        use_chemeleon: Whether to include Chemeleon features
        use_custom: Whether to include custom features (e.g., conjugation proxy)

    Returns:
        Filtered DataFrame with selected feature groups
    """
    group_settings = {
        "ecfp4": use_ecfp4,
        "rdkit2d": use_rdkit2d,
        "mordred": use_mordred,
        "chemeleon": use_chemeleon,
        "custom": use_custom,
    }

    # Select columns based on group settings
    selected_cols = []
    for col in features.columns:
        # Extract group prefix
        if "__" in col:
            group = col.split("__")[0]
            if group_settings.get(group, True):  # Default to True if group not found
                selected_cols.append(col)
        else:
            # If no prefix, include by default
            selected_cols.append(col)

    if not selected_cols:
        raise ValueError("No feature groups selected - at least one must be enabled")

    filtered_features = features[selected_cols]

    logger.info(f"Filtered features: {len(filtered_features.columns)} features from {len(features.columns)} total")
    logger.info(f"Active groups: {[g for g, active in group_settings.items() if active]}")

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
    # Load data
    df = load_csv(input_path)

    # Create featurizers
    featurizers = [create_featurizer(feat_cfg) for feat_cfg in config.featurizers]

    # Build features
    features = build_features(df, featurizers)

    # Save features
    save_parquet(features, output_path)

    return features

"""Feature generation pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from euos25.config import Config, FeaturizerConfig
from euos25.featurizers.base import Featurizer
from euos25.featurizers.conj_proxy import ConjugationProxyFeaturizer
from euos25.featurizers.ecfp import ECFPFeaturizer
from euos25.featurizers.rdkit2d import RDKit2DFeaturizer
from euos25.utils.io import load_csv, save_parquet

logger = logging.getLogger(__name__)


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
    elif config.name == "conj_proxy":
        return ConjugationProxyFeaturizer(**config.params)
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

    # Start with ID index
    feature_dfs = []

    # Generate features from each featurizer
    for featurizer in featurizers:
        logger.info(f"Applying featurizer: {featurizer.name}")
        feat_df = featurizer.transform(df, smiles_col=smiles_col)

        if not feat_df.empty:
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

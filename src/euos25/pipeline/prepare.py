"""Data preparation pipeline."""

import logging
from pathlib import Path
from typing import Optional

import datamol as dm
import pandas as pd
from rdkit import Chem

from euos25.data.schema import clean_smiles, standardize_columns, validate_train_data
from euos25.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def normalize_smiles(smiles: str) -> Optional[str]:
    """Normalize SMILES string using RDKit.

    Args:
        smiles: Input SMILES string

    Returns:
        Normalized SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def standardize_smiles_dm(smiles: str, remove_salts: bool = True) -> Optional[str]:
    """Standardize SMILES string using datamol.

    This function applies a comprehensive standardization procedure including:
    - Multiple fragment handling (keeps largest fragment if multiple components)
    - Salt/solvent removal (e.g., .Cl, .HCl, .O=C(O)C(F)(F)F) [optional]
    - RDKit molecule sanitization
    - Charge neutralization
    - Tautomer canonicalization
    - Stereochemistry handling

    Args:
        smiles: Input SMILES string (may contain salts/solvents like .Cl, .HCl,
                or larger fragments like .O=C(O)C(F)(F)F)
        remove_salts: Whether to remove salts and solvents (default: True)

    Returns:
        Standardized SMILES or None if invalid
    """
    try:
        # First, parse the SMILES to a molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Handle multiple fragments: if there are multiple disconnected components,
        # keep only the largest one (by number of atoms)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            # Select the largest fragment
            mol = max(frags, key=lambda x: x.GetNumAtoms())
            logger.debug(f"Multiple fragments detected, keeping largest fragment ({mol.GetNumAtoms()} atoms)")

        # Remove salts and solvents from the selected fragment (e.g., .Cl, .HCl, etc.)
        if remove_salts:
            mol_no_salt = dm.remove_salts_solvents(mol)
            if mol_no_salt is None:
                return None
            # Convert back to SMILES before standardizing
            smiles_no_salt = Chem.MolToSmiles(mol_no_salt)
        else:
            # Skip salt/solvent removal
            smiles_no_salt = Chem.MolToSmiles(mol)

        # Apply comprehensive standardization
        standardized = dm.standardize_smiles(smiles_no_salt)
        return standardized if standardized else None
    except Exception as e:
        logger.debug(f"Failed to standardize SMILES: {smiles}, error: {e}")
        return None


def sanitize_smiles_dm(smiles: str, isomeric: bool = True) -> Optional[str]:
    """Sanitize SMILES string using datamol.

    This function sanitizes the SMILES representation.

    Args:
        smiles: Input SMILES string
        isomeric: Whether to include stereochemistry information

    Returns:
        Sanitized SMILES or None if invalid
    """
    try:
        sanitized = dm.sanitize_smiles(smiles, isomeric=isomeric)
        return sanitized
    except Exception as e:
        logger.debug(f"Failed to sanitize SMILES: {smiles}, error: {e}")
        return None


def get_inchikey_block(smiles: str) -> Optional[str]:
    """Get first block of InChIKey for duplicate detection.

    Args:
        smiles: SMILES string

    Returns:
        First block of InChIKey or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi_key = Chem.MolToInchiKey(mol)
        return inchi_key.split("-")[0] if inchi_key else None
    except Exception:
        return None


def prepare_data(
    input_path: str,
    output_path: Optional[str] = None,
    remove_duplicates: bool = True,
    normalize: bool = True,
    standardize: bool = True,
    remove_salts: bool = True,
) -> pd.DataFrame:
    """Prepare training data.

    Steps:
    1. Load data
    2. Standardize columns
    3. Clean SMILES
    4. Standardize/Normalize SMILES (optional)
    5. Remove duplicates based on InChIKey (optional)
    6. Save prepared data (optional)

    Args:
        input_path: Path to input CSV
        output_path: Path to save prepared CSV (optional)
        remove_duplicates: Whether to remove duplicates
        normalize: Whether to normalize SMILES (basic RDKit normalization)
        standardize: Whether to standardize SMILES (comprehensive datamol standardization,
                     takes precedence over normalize if both are True)
        remove_salts: Whether to remove salts and solvents during standardization
                     (only applies when standardize=True)

    Returns:
        Prepared DataFrame
    """
    logger.info(f"Loading data from {input_path}")
    df = load_csv(input_path)

    # Standardize columns
    df = standardize_columns(df)

    # Validate schema
    df = validate_train_data(df)

    # Clean SMILES
    df = clean_smiles(df)

    # Standardize or Normalize SMILES
    if standardize:
        salt_msg = "with salt/solvent removal" if remove_salts else "without salt/solvent removal"
        logger.info(f"Standardizing SMILES with datamol ({salt_msg}, includes tautomer canonicalization)")
        df["SMILES_processed"] = df["SMILES"].apply(lambda x: standardize_smiles_dm(x, remove_salts=remove_salts))

        # Remove rows where standardization failed
        before = len(df)
        df = df[df["SMILES_processed"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid SMILES during standardization")

        # Replace original SMILES
        df["SMILES"] = df["SMILES_processed"]
        df = df.drop(columns=["SMILES_processed"])
    elif normalize:
        logger.info("Normalizing SMILES with RDKit")
        df["SMILES_processed"] = df["SMILES"].apply(normalize_smiles)

        # Remove rows where normalization failed
        before = len(df)
        df = df[df["SMILES_processed"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid SMILES during normalization")

        # Replace original SMILES
        df["SMILES"] = df["SMILES_processed"]
        df = df.drop(columns=["SMILES_processed"])

    # Remove duplicates
    if remove_duplicates:
        logger.info("Checking for duplicates using InChIKey")
        df["inchikey_block"] = df["SMILES"].apply(get_inchikey_block)

        # Remove rows where InChIKey generation failed
        before = len(df)
        df = df[df["inchikey_block"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid InChIKey")

        # Find and handle duplicates
        duplicates = df[df.duplicated(subset=["inchikey_block"], keep=False)]
        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate molecules")

            # For duplicates with labels, keep majority vote or first occurrence
            if "Transmittance" in df.columns or "Fluorescence" in df.columns:
                # Group by InChIKey and aggregate labels
                def aggregate_labels(group):
                    result = group.iloc[0].copy()
                    for col in ["Transmittance", "Fluorescence"]:
                        if col in group.columns:
                            # Take majority vote
                            result[col] = group[col].mode()[0] if len(group[col].mode()) > 0 else group[col].iloc[0]
                    return result

                df = df.groupby("inchikey_block", as_index=False).apply(aggregate_labels)
                df = df.reset_index(drop=True)
            else:
                # No labels, just keep first occurrence
                df = df.drop_duplicates(subset=["inchikey_block"], keep="first")

            logger.info(f"After deduplication: {len(df)} unique molecules")

        df = df.drop(columns=["inchikey_block"])

    # Log final statistics
    logger.info(f"Final dataset: {len(df)} samples")
    if "Transmittance" in df.columns:
        logger.info(f"  Transmittance positive: {df['Transmittance'].sum()}")
    if "Fluorescence" in df.columns:
        logger.info(f"  Fluorescence positive: {df['Fluorescence'].sum()}")

    # Save prepared data
    if output_path:
        save_csv(df, output_path, index=False)

    return df

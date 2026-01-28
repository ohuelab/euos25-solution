"""Data preparation pipeline."""

import logging
from typing import Optional

import datamol as dm
import pandas as pd
from rdkit import Chem

from euos25.data.schema import clean_smiles, standardize_columns, validate_train_data
from euos25.utils.io import load_csv, save_csv

logger = logging.getLogger(__name__)


def normalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize via RDKit: MolFromSmiles → MolToSmiles(isomericSmiles=True)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def standardize_smiles_dm(smiles: str, remove_salts: bool = True) -> Optional[str]:
    """Standardize SMILES: keep largest fragment → (opt) dm.remove_salts_solvents
    → dm.standardize_smiles (RDKit cleanup).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Keep largest fragment when multiple (RDKit)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumAtoms())
            logger.debug(f"Multiple fragments detected, keeping largest fragment ({mol.GetNumAtoms()} atoms)")

        if remove_salts:
            mol_no_salt = dm.remove_salts_solvents(mol)
            if mol_no_salt is None:
                return None
            smiles_no_salt = Chem.MolToSmiles(mol_no_salt)
        else:
            smiles_no_salt = Chem.MolToSmiles(mol)

        standardized = dm.standardize_smiles(smiles_no_salt)
        return standardized if standardized else None
    except Exception as e:
        logger.debug(f"Failed to standardize SMILES: {smiles}, error: {e}")
        return None


def sanitize_smiles_dm(smiles: str, isomeric: bool = True) -> Optional[str]:
    try:
        sanitized = dm.sanitize_smiles(smiles, isomeric=isomeric)
        return sanitized
    except Exception as e:
        logger.debug(f"Failed to sanitize SMILES: {smiles}, error: {e}")
        return None


def get_inchikey_block(smiles: str) -> Optional[str]:
    """Return first block of InChIKey for duplicate detection."""
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
    """Prepare training data: standardize columns, validate schema, clean SMILES,
    then standardize or normalize, dedupe by InChIKey (majority vote when labels exist).
    Saves CSV if output_path is set.
    """
    logger.info(f"Loading data from {input_path}")
    df = load_csv(input_path)

    df = standardize_columns(df)
    df = validate_train_data(df)
    df = clean_smiles(df)

    if standardize:
        salt_msg = "with salt/solvent removal" if remove_salts else "without salt/solvent removal"
        logger.info(f"Standardizing SMILES with datamol ({salt_msg})")
        df["SMILES_processed"] = df["SMILES"].apply(lambda x: standardize_smiles_dm(x, remove_salts=remove_salts))

        before = len(df)
        df = df[df["SMILES_processed"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid SMILES during standardization")

        df["SMILES"] = df["SMILES_processed"]
        df = df.drop(columns=["SMILES_processed"])
    elif normalize:
        logger.info("Normalizing SMILES with RDKit")
        df["SMILES_processed"] = df["SMILES"].apply(normalize_smiles)

        before = len(df)
        df = df[df["SMILES_processed"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid SMILES during normalization")

        df["SMILES"] = df["SMILES_processed"]
        df = df.drop(columns=["SMILES_processed"])

    if remove_duplicates:
        logger.info("Checking for duplicates using InChIKey")
        df["inchikey_block"] = df["SMILES"].apply(get_inchikey_block)

        before = len(df)
        df = df[df["inchikey_block"].notna()]
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} rows with invalid InChIKey")

        # Find and handle duplicates
        duplicates = df[df.duplicated(subset=["inchikey_block"], keep=False)]
        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate molecules")

            if "Transmittance" in df.columns or "Fluorescence" in df.columns:
                def aggregate_labels(group):
                    result = group.iloc[0].copy()
                    for col in ["Transmittance", "Fluorescence"]:
                        if col in group.columns:
                            result[col] = group[col].mode()[0] if len(group[col].mode()) > 0 else group[col].iloc[0]
                    return result

                df = df.groupby("inchikey_block", as_index=False).apply(aggregate_labels)
                df = df.reset_index(drop=True)
            else:
                df = df.drop_duplicates(subset=["inchikey_block"], keep="first")

            logger.info(f"After deduplication: {len(df)} unique molecules")

        df = df.drop(columns=["inchikey_block"])

    logger.info(f"Final dataset: {len(df)} samples")
    if "Transmittance" in df.columns:
        logger.info(f"  Transmittance positive: {df['Transmittance'].sum()}")
    if "Fluorescence" in df.columns:
        logger.info(f"  Fluorescence positive: {df['Fluorescence'].sum()}")

    if output_path:
        save_csv(df, output_path, index=False)

    return df

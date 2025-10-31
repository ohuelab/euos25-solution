"""Conjugation proxy features for optical properties."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class ConjugationProxyFeaturizer(BaseFeaturizer):
    """Featurizer for conjugation length and donor-acceptor proxies.

    These features are relevant for optical properties like absorption/fluorescence.
    """

    def __init__(
        self,
        name: str = "conj_proxy",
        L_cut: int = 4,
    ):
        """Initialize conjugation proxy featurizer.

        Args:
            name: Feature name prefix
            L_cut: Cutoff for long conjugation indicator
        """
        super().__init__(name=name, L_cut=L_cut)
        self.L_cut = L_cut

    def _calc_conjugation_features(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate conjugation-related features.

        Args:
            smiles: SMILES string

        Returns:
            Feature array or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Basic conjugation proxies
            features = []

            # 1. Number of aromatic bonds
            n_aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
            features.append(n_aromatic_bonds)

            # 2. Number of double bonds (non-aromatic)
            n_double_bonds = sum(
                1 for bond in mol.GetBonds()
                if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic()
            )
            features.append(n_double_bonds)

            # 3. Number of aromatic atoms
            n_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            features.append(n_aromatic_atoms)

            # 4. Aromatic proportion
            aromatic_prop = n_aromatic_atoms / mol.GetNumHeavyAtoms() if mol.GetNumHeavyAtoms() > 0 else 0
            features.append(aromatic_prop)

            # 5. Number of aromatic rings
            n_aromatic_rings = Descriptors.NumAromaticRings(mol)
            features.append(n_aromatic_rings)

            # 6. Long conjugation indicator (aromatic rings >= L_cut)
            long_conj = 1 if n_aromatic_rings >= self.L_cut else 0
            features.append(long_conj)

            # 7. Donor/Acceptor proxies
            # NH2, OH groups as donors
            n_donors = Lipinski.NumHDonors(mol)
            features.append(n_donors)

            # NO2, C=O, CN groups as acceptors
            n_acceptors = Lipinski.NumHAcceptors(mol)
            features.append(n_acceptors)

            # 8. Donor-Acceptor indicator
            has_donor_acceptor = 1 if n_donors > 0 and n_acceptors > 0 else 0
            features.append(has_donor_acceptor)

            # 9. Number of heteroatoms in aromatic rings (N, O, S)
            hetero_aromatic = sum(
                1 for atom in mol.GetAtoms()
                if atom.GetIsAromatic() and atom.GetSymbol() in ["N", "O", "S"]
            )
            features.append(hetero_aromatic)

            # 10. Ratio of conjugated to total bonds
            total_bonds = mol.GetNumBonds()
            conj_ratio = (n_aromatic_bonds + n_double_bonds) / total_bonds if total_bonds > 0 else 0
            features.append(conj_ratio)

            # 11. Longest aromatic chain (approximate)
            # Use aromatic ring count as proxy
            features.append(n_aromatic_rings)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Error calculating conjugation features: {smiles}, {e}")
            return None

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Transform SMILES to conjugation proxy features.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with conjugation feature columns
        """
        logger.info(f"Generating {self.name} features (L_cut={self.L_cut})")

        # Feature names
        feature_names = [
            "n_aromatic_bonds",
            "n_double_bonds",
            "n_aromatic_atoms",
            "aromatic_proportion",
            "n_aromatic_rings",
            "long_conjugation",
            "n_donors",
            "n_acceptors",
            "has_donor_acceptor",
            "hetero_aromatic",
            "conjugation_ratio",
            "longest_aromatic_chain",
        ]

        # Generate features
        feat_list = []
        valid_indices = []

        for idx, smiles in enumerate(df[smiles_col]):
            feat = self._calc_conjugation_features(smiles)
            if feat is not None:
                feat_list.append(feat)
                valid_indices.append(idx)

        if not feat_list:
            logger.warning("No valid conjugation features generated")
            return pd.DataFrame(index=df.index)

        # Create DataFrame
        feat_array = np.vstack(feat_list)
        columns = [f"{self.name}_{fname}" for fname in feature_names]

        result_df = pd.DataFrame(
            feat_array,
            columns=columns,
            index=df.index[valid_indices],
        )

        # Reindex to match input (fill missing with zeros)
        result_df = result_df.reindex(df.index, fill_value=0.0)

        logger.info(f"Generated {len(columns)} {self.name} features for {len(result_df)} samples")
        return result_df

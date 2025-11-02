"""Chemeleon embedding featurizer for extracting molecular representations."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, nn
from rdkit import Chem

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class ChemeleonFeaturizer(BaseFeaturizer):
    """Extract molecular embeddings using Chemeleon foundation model.

    This featurizer uses the pretrained Chemeleon model to generate
    fixed-dimensional molecular representations that can be used with
    traditional ML models like LightGBM.
    """

    def __init__(
        self,
        chemeleon_path: str = "chemeleon_mp.pt",
        aggregation: str = "mean",
        device: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize Chemeleon featurizer.

        Args:
            chemeleon_path: Path to Chemeleon pretrained weights
            aggregation: Aggregation method ('mean', 'sum', 'norm')
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            name: Featurizer name. If None, will be set to 'chemeleon_{aggregation}'
        """
        # Set name based on aggregation if not provided
        if name is None:
            name = f"chemeleon_{aggregation}"

        super().__init__(name=name, chemeleon_path=chemeleon_path, aggregation=aggregation, device=device)
        self.chemeleon_path = chemeleon_path
        self.aggregation = aggregation

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load Chemeleon model
        self.mp = None
        self.agg = None
        self.featurizer = None
        self._load_chemeleon()

    def _load_chemeleon(self):
        """Load pretrained Chemeleon model."""
        chemeleon_path = Path(self.chemeleon_path)

        if not chemeleon_path.exists():
            raise FileNotFoundError(
                f"Chemeleon weights not found at {chemeleon_path}. "
                "Please download from: "
                "https://chemprop.readthedocs.io/en/latest/chemeleon_foundation_finetuning.html"
            )

        logger.info(f"Loading Chemeleon from {chemeleon_path}")

        # Load pretrained weights
        chemeleon_mp = torch.load(chemeleon_path, map_location=self.device, weights_only=True)

        # Create message passing layer
        self.mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        self.mp.load_state_dict(chemeleon_mp["state_dict"])
        self.mp.to(self.device)
        self.mp.eval()  # Set to evaluation mode

        # Create aggregation layer
        if self.aggregation == "mean":
            self.agg = nn.MeanAggregation()
        elif self.aggregation == "sum":
            self.agg = nn.SumAggregation()
        elif self.aggregation == "norm":
            self.agg = nn.NormAggregation()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Create featurizer for molecular graphs
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        logger.info(f"Chemeleon loaded. Output dimension: {self.mp.output_dim}")


    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        if self.mp is None:
            return []
        return [f"{self.name}_{i}" for i in range(self.mp.output_dim)]

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Extract Chemeleon embeddings from SMILES.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with Chemeleon embedding feature columns
        """
        if smiles_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{smiles_col}' column")

        smiles_list = df[smiles_col].tolist()
        logger.info(f"Extracting Chemeleon embeddings for {len(smiles_list)} molecules")

        # Create datapoints (without labels)
        datapoints = []
        valid_indices = []

        for idx, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    datapoints.append(data.MoleculeDatapoint.from_smi(smi))
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Invalid SMILES at index {idx}: {smi}")
            except Exception as e:
                logger.warning(f"Error processing SMILES at index {idx}: {smi}, {e}")

        if len(datapoints) == 0:
            logger.warning("No valid SMILES found")
            return pd.DataFrame(index=df.index)

        # Create dataset
        dataset = data.MoleculeDataset(datapoints, self.featurizer)

        # Create dataloader
        dataloader = data.build_dataloader(
            dataset, batch_size=32, shuffle=False, num_workers=0
        )

        # Extract embeddings
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                bmg = batch.bmg.to(self.device)

                # Get message passing output
                H_v = self.mp(bmg)

                # Aggregate to get molecular embedding
                Z = self.agg(H_v, bmg.batch)

                # Move to CPU and convert to numpy
                embeddings.append(Z.cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)

        # Create feature names
        feature_names = self.feature_names

        # Create DataFrame with valid indices
        result_df = pd.DataFrame(
            embeddings,
            columns=feature_names,
            index=df.index[valid_indices],
        )

        # Handle invalid SMILES by filling with zeros and reindexing
        if len(valid_indices) < len(smiles_list):
            logger.warning(
                f"Filled {len(smiles_list) - len(valid_indices)} invalid SMILES with zeros"
            )
            result_df = result_df.reindex(df.index, fill_value=0)

        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        logger.info(f"Generated {len(feature_names)} {self.name} features for {len(result_df)} samples")
        return result_df


def create_chemeleon_featurizer(
    chemeleon_path: str = "chemeleon_mp.pt",
    aggregation: str = "mean",
    device: Optional[str] = None,
) -> ChemeleonFeaturizer:
    """Factory function to create Chemeleon featurizer.

    Args:
        chemeleon_path: Path to Chemeleon pretrained weights
        aggregation: Aggregation method
        device: Device to use

    Returns:
        ChemeleonFeaturizer instance
    """
    return ChemeleonFeaturizer(
        chemeleon_path=chemeleon_path, aggregation=aggregation, device=device
    )

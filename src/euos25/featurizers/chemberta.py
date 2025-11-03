"""ChemBERTa-2 embedding featurizer for extracting molecular representations."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from euos25.featurizers.base import BaseFeaturizer

logger = logging.getLogger(__name__)


class ChemBERTaFeaturizer(BaseFeaturizer):
    """Extract molecular embeddings using ChemBERTa-2 foundation model.

    This featurizer uses the pretrained ChemBERTa-2 model to generate
    fixed-dimensional molecular representations that can be used with
    traditional ML models like LightGBM.

    ChemBERTa-2 models are available on Hugging Face:
    - DeepChem/ChemBERTa-77M-MLM (77M parameters, masked language modeling)
    - DeepChem/ChemBERTa-10M-MTR (10M parameters, multi-task regression)
    """

    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        pooling: str = "mean",
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 32,
        name: Optional[str] = None,
    ):
        """Initialize ChemBERTa featurizer.

        Args:
            model_name: Hugging Face model name or path
            pooling: Pooling method ('cls', 'mean', 'max')
            max_length: Maximum sequence length for tokenization
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detect if None.
            batch_size: Batch size for inference
            name: Featurizer name. If None, will be set to 'chemberta_{pooling}'
        """
        # Set name based on pooling if not provided
        if name is None:
            name = f"chemberta_{pooling}"

        super().__init__(
            name=name,
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            device=device,
            batch_size=batch_size,
        )
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size

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

        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load pretrained ChemBERTa model and tokenizer."""
        logger.info(f"Loading ChemBERTa model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Get hidden size for feature names
            self.hidden_size = self.model.config.hidden_size

            logger.info(f"ChemBERTa loaded. Hidden size: {self.hidden_size}, Pooling: {self.pooling}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ChemBERTa model {self.model_name}. "
                f"Make sure transformers is installed and the model is available. Error: {e}"
            )

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        if self.model is None:
            return []
        return [f"{self.name}_{i}" for i in range(self.hidden_size)]

    def _pool_embeddings(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings to get molecular representation.

        Args:
            last_hidden_state: Token embeddings (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        if self.pooling == "cls":
            # Use CLS token (first token)
            return last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling over sequence length (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == "max":
            # Max pooling over sequence length (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Set padding tokens to very negative value so they don't affect max
            last_hidden_state_masked = last_hidden_state.clone()
            last_hidden_state_masked[~mask_expanded.bool()] = -1e9
            return torch.max(last_hidden_state_masked, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def transform(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
        """Extract ChemBERTa embeddings from SMILES.

        Args:
            df: DataFrame with SMILES column
            smiles_col: Name of SMILES column

        Returns:
            DataFrame with ChemBERTa embedding feature columns
        """
        if smiles_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{smiles_col}' column")

        smiles_list = df[smiles_col].tolist()
        logger.info(f"Extracting ChemBERTa embeddings for {len(smiles_list)} molecules")

        # Tokenize SMILES
        valid_indices = []
        tokenized_inputs = []

        for idx, smi in enumerate(smiles_list):
            try:
                # ChemBERTa tokenizer expects SMILES as input
                # Some models may need special formatting
                encoded = self.tokenizer(
                    smi,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokenized_inputs.append(encoded)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Error tokenizing SMILES at index {idx}: {smi}, {e}")

        if len(tokenized_inputs) == 0:
            logger.warning("No valid SMILES found")
            return pd.DataFrame(index=df.index)

        # Extract embeddings in batches
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(tokenized_inputs), self.batch_size):
                batch_inputs = tokenized_inputs[i : i + self.batch_size]

                # Concatenate batch
                input_ids = torch.cat([inp["input_ids"] for inp in batch_inputs], dim=0).to(self.device)
                attention_mask = torch.cat([inp["attention_mask"] for inp in batch_inputs], dim=0).to(self.device)

                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state

                # Pool embeddings
                pooled = self._pool_embeddings(last_hidden_state, attention_mask)

                # Move to CPU and convert to numpy
                embeddings.append(pooled.cpu().numpy())

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


def create_chemberta_featurizer(
    model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    pooling: str = "mean",
    max_length: int = 512,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> ChemBERTaFeaturizer:
    """Factory function to create ChemBERTa featurizer.

    Args:
        model_name: Hugging Face model name or path
        pooling: Pooling method
        max_length: Maximum sequence length
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        ChemBERTaFeaturizer instance
    """
    return ChemBERTaFeaturizer(
        model_name=model_name,
        pooling=pooling,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )


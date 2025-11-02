"""ChemProp model wrapper for binary classification."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)


class ChemPropModel(BaseClfModel):
    """ChemProp MPNN model wrapper for binary classification.

    This model uses Message Passing Neural Networks (MPNN) to learn
    molecular representations directly from SMILES strings.
    """

    def __init__(
        self,
        max_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        aggregation: str = "mean",
        batch_norm: bool = True,
        ffn_num_layers: int = 2,
        use_foundation: bool = False,
        foundation_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        random_seed: int = 42,
        **kwargs,
    ):
        """Initialize ChemProp model.

        Args:
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            hidden_size: Hidden dimension size
            depth: Number of message passing steps
            dropout: Dropout probability
            aggregation: Aggregation method ('mean', 'sum', 'norm')
            batch_norm: Whether to use batch normalization
            ffn_num_layers: Number of feed-forward network layers
            use_foundation: Whether to use foundation model (e.g., Chemeleon)
            foundation_name: Name of foundation model ('chemeleon')
            checkpoint_dir: Directory to save model checkpoints
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(name="chemprop", **kwargs)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.aggregation = aggregation
        self.batch_norm = batch_norm
        self.ffn_num_layers = ffn_num_layers
        self.use_foundation = use_foundation
        self.foundation_name = foundation_name
        self.checkpoint_dir = checkpoint_dir or "checkpoints/chemprop"
        self.random_seed = random_seed

        self.model = None
        self.trainer = None
        self.featurizer = None
        self.scaler = None

        # Set random seed
        pl.seed_everything(self.random_seed)

    def _prepare_data(
        self, smiles: List[str], labels: Optional[np.ndarray] = None
    ) -> List[data.MoleculeDatapoint]:
        """Prepare data for ChemProp.

        Args:
            smiles: List of SMILES strings
            labels: Optional labels

        Returns:
            List of MoleculeDatapoint objects
        """
        if labels is not None:
            # For training/validation with labels
            datapoints = [
                data.MoleculeDatapoint.from_smi(smi, [float(label)])
                for smi, label in zip(smiles, labels)
            ]
        else:
            # For prediction without labels
            datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]

        return datapoints

    def _create_dataloader(
        self,
        smiles: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = True,
        normalize: bool = False,
    ) -> Any:
        """Create ChemProp dataloader.

        Args:
            smiles: List of SMILES strings
            labels: Optional labels
            shuffle: Whether to shuffle data
            normalize: Whether to normalize targets

        Returns:
            ChemProp dataloader
        """
        datapoints = self._prepare_data(smiles, labels)
        dataset = data.MoleculeDataset(datapoints, self.featurizer)

        if normalize and labels is not None:
            if self.scaler is None:
                self.scaler = dataset.normalize_targets()
            else:
                dataset.normalize_targets(self.scaler)

        return data.build_dataloader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
        )

    def _build_model(self) -> models.MPNN:
        """Build ChemProp MPNN model.

        Returns:
            MPNN model
        """
        # Message passing
        if self.use_foundation and self.foundation_name == "chemeleon":
            logger.info("Loading Chemeleon foundation model")
            # Load pretrained Chemeleon weights
            try:
                chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
                mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
                mp.load_state_dict(chemeleon_mp["state_dict"])
            except FileNotFoundError:
                logger.warning(
                    "Chemeleon weights not found. Using standard message passing."
                )
                mp = nn.BondMessagePassing(
                    d_h=self.hidden_size, depth=self.depth, dropout=self.dropout
                )
        else:
            mp = nn.BondMessagePassing(
                d_h=self.hidden_size, depth=self.depth, dropout=self.dropout
            )

        # Aggregation
        if self.aggregation == "mean":
            agg = nn.MeanAggregation()
        elif self.aggregation == "sum":
            agg = nn.SumAggregation()
        elif self.aggregation == "norm":
            agg = nn.NormAggregation()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Output transform
        if self.scaler is not None:
            output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler)
        else:
            output_transform = None

        # Feed-forward network
        # For binary classification, we use n_tasks=1
        # input_dim should match the output dimension of message passing
        ffn = nn.BinaryClassificationFFN(
            n_tasks=1,
            input_dim=mp.output_dim,
            hidden_dim=self.hidden_size,
            n_layers=self.ffn_num_layers,
            dropout=self.dropout,
            output_transform=output_transform,
        )

        # Metrics
        metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy()]

        # Build MPNN
        mpnn = models.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            batch_norm=self.batch_norm,
            metrics=metric_list,
        )

        return mpnn

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ChemPropModel":
        """Train the ChemProp model.

        Args:
            X_train: Training features (must contain 'SMILES' column)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Self
        """
        logger.info("Training ChemProp model")

        # Initialize featurizer
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Create dataloaders
        train_smiles = X_train["SMILES"].tolist()
        train_loader = self._create_dataloader(
            train_smiles, y_train, shuffle=True, normalize=False
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_smiles = X_val["SMILES"].tolist()
            val_loader = self._create_dataloader(
                val_smiles, y_val, shuffle=False, normalize=False
            )

        # Build model
        self.model = self._build_model()

        # Setup checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss" if val_loader else "train_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        )

        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            logger=False,
        )

        # Train
        self.trainer.fit(self.model, train_loader, val_loader)

        logger.info("Training completed")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features (must contain 'SMILES' column)

        Returns:
            Predicted probabilities of shape (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info(f"Predicting {len(X)} samples")

        # Create dataloader
        smiles = X["SMILES"].tolist()
        test_loader = self._create_dataloader(smiles, labels=None, shuffle=False)

        # Predict
        self.model.eval()
        predictions = self.trainer.predict(self.model, test_loader)

        # Convert to numpy array
        probs = torch.cat(predictions).cpu().numpy()

        # For binary classification, convert to (n_samples, 2) shape
        probs_positive = probs.reshape(-1)
        probs_negative = 1 - probs_positive
        probs_binary = np.column_stack([probs_negative, probs_positive])

        return probs_binary

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features (must contain 'SMILES' column)

        Returns:
            Predicted labels
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        if self.trainer is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.trainer.save_checkpoint(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        # This is a simplified load - in practice you'd need to reconstruct
        # the model architecture and load the checkpoint
        logger.warning("Loading ChemProp models requires manual reconstruction")
        raise NotImplementedError(
            "ChemProp model loading not fully implemented. "
            "Use trainer.save_checkpoint() and models.MPNN.load_from_checkpoint()"
        )

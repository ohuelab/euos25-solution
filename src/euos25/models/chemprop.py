"""ChemProp model wrapper for binary classification."""

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)

# Set OpenMP environment variables to prevent segmentation faults on macOS
# These need to be set before importing torch or any libraries that use OpenMP
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"


class BinaryROCAUCMetric(Metric):
    """Custom ROC AUC metric that uses binary labels for regression/ranking.

    This metric computes ROC AUC using binary labels (0/1) even when
    the model is trained with regression or ranking objectives.
    """

    def __init__(self, binary_labels: Optional[torch.Tensor] = None):
        """Initialize Binary ROC AUC Metric.

        Args:
            binary_labels: Binary labels (0/1) for ROC AUC calculation.
                          If None, uses the target values directly.
        """
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("indices", default=[], dist_reduce_fx="cat")
        self.binary_labels = binary_labels
        self.current_idx = 0

    def reset(self):
        """Reset metric state."""
        super().reset()
        self.current_idx = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metric state.

        Args:
            preds: Predictions (probabilities after sigmoid)
            targets: Target values (quantitative values for regression/ranking)
        """
        # Flatten if multi-dimensional
        if preds.dim() > 1:
            preds = preds.flatten()
        if targets.dim() > 1:
            targets = targets.flatten()

        batch_size = len(preds)
        indices = torch.arange(self.current_idx, self.current_idx + batch_size, device=preds.device)
        self.current_idx += batch_size

        self.preds.append(preds)
        self.indices.append(indices)

    def compute(self) -> torch.Tensor:
        """Compute ROC AUC.

        Returns:
            ROC AUC score
        """
        from torchmetrics.functional.classification import (
            binary_auroc,
        )

        preds = dim_zero_cat(self.preds)
        indices = dim_zero_cat(self.indices)

        # Use binary labels if provided, otherwise use targets directly
        if self.binary_labels is not None:
            # Map indices to binary labels
            indices_np = indices.cpu().numpy().astype(int)
            if len(self.binary_labels) > indices_np.max():
                targets = torch.from_numpy(self.binary_labels[indices_np]).to(preds.device)
            else:
                logger.warning(
                    f"Binary labels length ({len(self.binary_labels)}) "
                    f"is less than max index ({indices_np.max()}). "
                    "Using 0.5 threshold on predictions."
                )
                targets = (preds > 0.5).long()
        else:
            # Fallback: use 0.5 threshold
            targets = (preds > 0.5).long()

        # Ensure targets are binary
        if targets.dtype != torch.bool and targets.dtype != torch.long:
            targets = (targets > 0.5).long()

        try:
            return binary_auroc(preds, targets)
        except Exception as e:
            logger.warning(f"Error computing ROC AUC: {e}. Returning 0.5")
            return torch.tensor(0.5, device=preds.device)


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
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        objective_type: Optional[str] = None,  # "regression", "listmle", or None (binary)
        checkpoint_dir: Optional[str] = None,
        random_seed: int = 42,
        accelerator: Optional[str] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: str = "roc_auc",
        n_tasks: int = 1,
        binary_labels: Optional[np.ndarray] = None,  # Binary labels for ROC AUC (regression/ranking)
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
            use_focal_loss: Whether to use Focal loss for imbalanced data
            focal_alpha: Alpha parameter for focal loss (weighting factor)
            focal_gamma: Gamma parameter for focal loss (focusing parameter)
            objective_type: Objective type ("regression", "listmle", or None for binary)
            checkpoint_dir: Directory to save model checkpoints
            random_seed: Random seed for reproducibility
            accelerator: Accelerator to use ('auto', 'cpu', 'cuda', 'mps').
                        If None or 'auto', automatically selects based on availability.
                        On macOS, 'auto' will select 'mps' if available, else 'cpu'.
            early_stopping_rounds: Number of epochs to wait before early stopping (patience).
                                  If None, early stopping is disabled.
            early_stopping_metric: Metric to monitor for early stopping ('roc_auc', 'pr_auc', etc.)
            n_tasks: Number of tasks for multi-task learning (default 1 for single task)
            binary_labels: Binary labels (0/1) for ROC AUC calculation in regression/ranking mode
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
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.objective_type = objective_type
        self.checkpoint_dir = checkpoint_dir or "checkpoints/chemprop"
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.n_tasks = n_tasks
        self.binary_labels = binary_labels

        # Determine accelerator
        if accelerator is None or accelerator == "auto":
            # Auto-detect: CUDA > MPS > CPU
            if torch.cuda.is_available():
                self.accelerator = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Enable MPS fallback to CPU for unsupported operations
                if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    logger.info("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS stability")

                # Set additional MPS stability environment variables
                # These help prevent segmentation faults on macOS
                if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
                    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

                # Test MPS availability with a simple operation to avoid segfault during training
                # Use try-except to catch any RuntimeError that might occur
                try:
                    # Test basic MPS operations
                    test_tensor = torch.randn(2, 2, device="mps")
                    _ = test_tensor @ test_tensor.T
                    torch.mps.synchronize()  # Ensure MPS operations complete
                    # Clear any cached operations
                    torch.mps.empty_cache()
                    self.accelerator = "mps"
                    logger.info("MPS accelerator validated and selected")
                except (RuntimeError, Exception) as e:
                    logger.warning(
                        f"MPS test failed: {e}. Falling back to CPU to avoid segmentation faults."
                    )
                    self.accelerator = "cpu"
            else:
                self.accelerator = "cpu"
        else:
            self.accelerator = accelerator

        logger.info(f"Using accelerator: {self.accelerator}")

        self.model = None
        self.trainer = None
        self.featurizer = None
        self.scaler = None
        self.best_iteration: int = 0  # For compatibility with training pipeline

        # Set random seed
        pl.seed_everything(self.random_seed)

    def _prepare_data(
        self, smiles: List[str], labels: Optional[np.ndarray] = None
    ) -> List[data.MoleculeDatapoint]:
        """Prepare data for ChemProp.

        Args:
            smiles: List of SMILES strings
            labels: Optional labels. Shape (n_samples,) for single task or (n_samples, n_tasks) for multi-task

        Returns:
            List of MoleculeDatapoint objects
        """
        if labels is not None:
            # Handle both single-task and multi-task labels
            if labels.ndim == 1:
                # Single task: shape (n_samples,)
                datapoints = [
                    data.MoleculeDatapoint.from_smi(smi, [float(label)])
                    for smi, label in zip(smiles, labels)
                ]
            else:
                # Multi-task: shape (n_samples, n_tasks)
                datapoints = [
                    data.MoleculeDatapoint.from_smi(smi, [float(l) for l in label_row])
                    for smi, label_row in zip(smiles, labels)
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

        # Always use num_workers=0 to avoid multiprocessing issues
        # MPS and some PyTorch operations don't work well with multiprocessing
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
                chemeleon_mp = torch.load("chemeleon_mp.pt", map_location="cpu", weights_only=True)
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
        # For binary classification, we use n_tasks (1 for single task, >1 for multi-task)
        # input_dim should match the output dimension of message passing
        if self.objective_type == "regression":
            logger.info("Using RegressionFFN for regression objective")
            ffn = nn.RegressionFFN(
                n_tasks=self.n_tasks,
                input_dim=mp.output_dim,
                hidden_dim=self.hidden_size,
                n_layers=self.ffn_num_layers,
                dropout=self.dropout,
                output_transform=output_transform,
            )
        elif self.objective_type == "listmle":
            logger.info("Using ListMLE FFN for ranking objective")
            from euos25.models.chemprop_listmle import create_listmle_ffn

            ffn = create_listmle_ffn(
                n_tasks=self.n_tasks,
                input_dim=mp.output_dim,
                hidden_dim=self.hidden_size,
                n_layers=self.ffn_num_layers,
                dropout=self.dropout,
                output_transform=output_transform,
            )
        elif self.use_focal_loss:
            from euos25.models.chemprop_focal import create_focal_loss_ffn

            ffn = create_focal_loss_ffn(
                n_tasks=self.n_tasks,
                input_dim=mp.output_dim,
                hidden_dim=self.hidden_size,
                n_layers=self.ffn_num_layers,
                dropout=self.dropout,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                output_transform=output_transform,
            )
        else:
            ffn = nn.BinaryClassificationFFN(
                n_tasks=self.n_tasks,
                input_dim=mp.output_dim,
                hidden_dim=self.hidden_size,
                n_layers=self.ffn_num_layers,
                dropout=self.dropout,
                output_transform=output_transform,
            )

        # Metrics
        # For regression/ranking, use binary labels for ROC AUC if available
        if self.objective_type in ["regression", "listmle"] and self.binary_labels is not None:
            # Convert binary_labels to torch tensor for metric
            binary_labels_tensor = torch.from_numpy(self.binary_labels).float()
            # Create metric with binary labels
            roc_metric = BinaryROCAUCMetric(binary_labels=binary_labels_tensor)
            # Reset metric state for fresh computation
            roc_metric.reset()
            metric_list = [
                roc_metric,
                nn.metrics.BinaryAccuracy(),
            ]
            logger.info("Using BinaryROCAUCMetric with binary labels for regression/ranking")
        else:
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
        binary_labels_val: Optional[np.ndarray] = None,
        resume_from_checkpoint: Optional[str] = None,
        task_name: Optional[str] = None,
        fold_name: Optional[str] = None,
    ) -> "ChemPropModel":
        """Train the ChemProp model.

        Args:
            X_train: Training features (must contain 'SMILES' column)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            binary_labels_val: Binary labels (0/1) for validation set ROC AUC calculation
                              (used with regression/ranking objectives)
            resume_from_checkpoint: Path to checkpoint file to resume training from.
                                   If None, training starts from scratch.
            task_name: Task name for organizing checkpoints (e.g., "transmittance340")
            fold_name: Fold name for organizing checkpoints (e.g., "fold_0", "full")

        Returns:
            Self
        """
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("Training ChemProp model")

        # Store validation binary labels for metric
        if binary_labels_val is not None:
            self.binary_labels_val = binary_labels_val
        else:
            self.binary_labels_val = None

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

        # Build model (use validation binary labels if available)
        if self.binary_labels_val is not None:
            original_binary_labels = self.binary_labels
            self.binary_labels = self.binary_labels_val
        self.model = self._build_model()
        if self.binary_labels_val is not None:
            self.binary_labels = original_binary_labels

        # Map early_stopping_metric to PyTorch Lightning metric name
        metric_name_map = {
            "roc_auc": "val/roc",
            "pr_auc": "val/pr",  # Assuming PR-AUC is logged as pr
        }
        monitor_metric = metric_name_map.get(
            self.early_stopping_metric.lower(), "val_loss"
        )

        # Setup checkpointing with task and fold separation
        checkpoint_dir = Path(self.checkpoint_dir)

        # Add task_name and fold_name to checkpoint_dir if provided
        # This ensures checkpoints are separated by task and fold to avoid conflicts
        if task_name is not None:
            checkpoint_dir = checkpoint_dir / task_name
            if fold_name is not None:
                checkpoint_dir = checkpoint_dir / fold_name

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create filename suffix for checkpoint files
        task_fold_suffix = ""
        if task_name is not None and fold_name is not None:
            task_fold_suffix = f"-{task_name}-{fold_name}"
        elif task_name is not None:
            task_fold_suffix = f"-{task_name}"
        elif fold_name is not None:
            task_fold_suffix = f"-{fold_name}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"best{task_fold_suffix}-{{epoch}}-{{val_loss:.4f}}",
            monitor="val_loss" if val_loader else "train_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        )

        # Setup callbacks
        callbacks = [checkpoint_callback]

        # Setup early stopping if enabled
        if (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and val_loader is not None
        ):
            # Determine mode: 'max' for metrics to maximize (AUC), 'min' for loss
            early_stopping_mode = (
                "max" if self.early_stopping_metric.lower() in ["roc_auc", "pr_auc"] else "min"
            )
            early_stopping_callback = EarlyStopping(
                monitor=monitor_metric,
                patience=self.early_stopping_rounds,
                mode=early_stopping_mode,
                verbose=True,
            )
            callbacks.append(early_stopping_callback)
            logger.info(
                f"Early stopping enabled: patience={self.early_stopping_rounds}, "
                f"monitor={monitor_metric}, mode={early_stopping_mode}"
            )

        # Setup trainer with MPS-specific optimizations
        trainer_kwargs = {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "devices": 1,
            "callbacks": callbacks,
            "enable_progress_bar": True,
            "logger": False,
            "num_sanity_val_steps": 0,  # Skip sanity check to avoid segfault on macOS MPS
            "enable_model_summary": False,  # Disable model summary to avoid MPS issues
        }

        # MPS-specific settings to prevent segmentation faults
        # Based on known issues: OpenMP conflicts, Sanity Checking crashes,
        # and PyTorch Lightning's MPS backend stability issues on macOS
        if self.accelerator == "mps":
            # Use float32 precision (MPS doesn't fully support float16)
            # "32" is recommended over "32-true" for better MPS compatibility
            trainer_kwargs["precision"] = "32"
            # Disable gradient accumulation issues on MPS
            trainer_kwargs["accumulate_grad_batches"] = 1
            # Set deterministic mode for reproducibility
            trainer_kwargs["deterministic"] = False  # MPS doesn't support deterministic mode
            # Disable gradient clipping on MPS (can cause crashes)
            trainer_kwargs["gradient_clip_val"] = None
            # Disable profiling which can cause issues
            trainer_kwargs["profiler"] = None
            logger.info(
                "Applied MPS-specific optimizations to prevent segmentation faults: "
                "32-bit precision, no gradient clipping, no profiling"
            )

        self.trainer = pl.Trainer(**trainer_kwargs)

        # Determine checkpoint path for resuming
        ckpt_path = None
        if resume_from_checkpoint:
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.exists():
                # If it's a directory, look for last.ckpt
                if checkpoint_path.is_dir():
                    last_ckpt = checkpoint_path / "last.ckpt"
                    if last_ckpt.exists():
                        ckpt_path = str(last_ckpt)
                    else:
                        # Try to find any .ckpt file
                        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
                        if ckpt_files:
                            # Use the most recent checkpoint
                            ckpt_path = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
                            logger.info(f"Using checkpoint: {ckpt_path}")
                elif checkpoint_path.is_file():
                    ckpt_path = str(checkpoint_path)
            else:
                logger.warning(
                    f"Checkpoint path does not exist: {resume_from_checkpoint}. "
                    "Starting training from scratch."
                )

        # Train with error handling for MPS
        try:
            self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
        except (RuntimeError, SystemError) as e:
            # If MPS fails, try falling back to CPU
            if self.accelerator == "mps" and "MPS" in str(e):
                logger.warning(
                    f"MPS training failed: {e}. Falling back to CPU accelerator."
                )
                # Retry with CPU
                trainer_kwargs["accelerator"] = "cpu"
                trainer_kwargs.pop("precision", None)  # Remove MPS-specific precision
                trainer_kwargs["deterministic"] = True
                self.accelerator = "cpu"
                self.trainer = pl.Trainer(**trainer_kwargs)
                self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
            else:
                raise

        # Set best_iteration for compatibility with training pipeline
        # For PyTorch Lightning, we use max_epochs as best_iteration
        # In practice, the best checkpoint is saved by ModelCheckpoint callback
        self.best_iteration = self.max_epochs

        logger.info("Training completed")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Features (must contain 'SMILES' column)

        Returns:
            For single task: Predicted probabilities of shape (n_samples, 2)
            For multi-task: Predicted probabilities of shape (n_samples, n_tasks)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info(f"Predicting {len(X)} samples")

        # Create dataloader
        smiles = X["SMILES"].tolist()
        test_loader = self._create_dataloader(smiles, labels=None, shuffle=False)

        # For prediction, use single GPU to avoid DDP issues
        # Create a new trainer for prediction (without DDP)
        predict_trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=1,  # Use single device for prediction
            enable_progress_bar=True,
            logger=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        # Predict
        self.model.eval()
        predictions = predict_trainer.predict(self.model, test_loader)

        # Handle predictions (should be simpler without DDP)
        if predictions is None or len(predictions) == 0:
            raise ValueError("No predictions returned from trainer")

        # Convert to numpy array
        # predictions is a list of tensors from each batch
        probs = torch.cat(predictions)

        # For regression/ranking, apply sigmoid to convert logits to probabilities
        if self.objective_type in ["regression", "listmle"]:
            probs = torch.sigmoid(probs)

        probs = probs.cpu().numpy()

        # Ensure correct shape
        n_samples = len(X)

        if self.n_tasks == 1:
            # For single task binary classification, convert to (n_samples, 2) shape
            probs_positive = probs.reshape(-1)
            if len(probs_positive) != n_samples:
                raise ValueError(
                    f"Shape mismatch: expected {n_samples} samples, got {len(probs_positive)}. "
                    f"Model may be duplicating predictions."
                )
            probs_negative = 1 - probs_positive
            probs_binary = np.column_stack([probs_negative, probs_positive])
            return probs_binary
        else:
            # For multi-task, return (n_samples, n_tasks) - probabilities of positive class for each task
            # Ensure shape is correct
            if probs.ndim == 1:
                # If 1D, reshape to (n_samples, n_tasks)
                if len(probs) != n_samples * self.n_tasks:
                    raise ValueError(
                        f"Shape mismatch: expected {n_samples * self.n_tasks} values, "
                        f"got {len(probs)}. Expected shape ({n_samples}, {self.n_tasks})."
                    )
                probs = probs.reshape(n_samples, self.n_tasks)
            elif probs.ndim == 2:
                # If 2D, validate shape
                if probs.shape[0] != n_samples:
                    raise ValueError(
                        f"Shape mismatch: expected {n_samples} samples, got {probs.shape[0]}. "
                        f"Model may be duplicating predictions."
                    )
                if probs.shape[1] != self.n_tasks:
                    # If shape is (n_samples, 1) but n_tasks > 1, try to expand
                    if probs.shape[1] == 1 and self.n_tasks > 1:
                        logger.warning(
                            f"Model output has shape ({probs.shape[0]}, 1) but n_tasks={self.n_tasks}. "
                            f"This may indicate incorrect model configuration."
                        )
                    else:
                        raise ValueError(
                            f"Shape mismatch: expected {self.n_tasks} tasks, got {probs.shape[1]}."
                        )
            else:
                raise ValueError(
                    f"Unexpected prediction shape: {probs.shape}. Expected ({n_samples}, {self.n_tasks})."
                )
            return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features (must contain 'SMILES' column)

        Returns:
            For single task: Predicted labels of shape (n_samples,)
            For multi-task: Predicted labels of shape (n_samples, n_tasks)
        """
        probs = self.predict_proba(X)
        if self.n_tasks == 1:
            return (probs[:, 1] > 0.5).astype(int)
        else:
            # For multi-task, threshold each task independently
            return (probs > 0.5).astype(int)

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

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **override_params,
    ) -> "ChemPropModel":
        """Load ChemPropModel from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory containing checkpoint
            **override_params: Parameters to override from checkpoint

        Returns:
            Loaded ChemPropModel instance
        """
        checkpoint_path = Path(checkpoint_path)

        # If checkpoint_path is a directory, look for checkpoint files
        if checkpoint_path.is_dir():
            # Look for .ckpt files in the directory
            ckpt_files = list(checkpoint_path.glob("*.ckpt"))
            if ckpt_files:
                # Prefer best checkpoint if available, otherwise use last
                best_ckpt = [f for f in ckpt_files if "best" in f.name]
                if best_ckpt:
                    checkpoint_path = sorted(best_ckpt)[-1]
                else:
                    checkpoint_path = sorted(ckpt_files)[-1]
            else:
                raise FileNotFoundError(
                    f"No checkpoint files found in directory {checkpoint_path}"
                )
        elif not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Try to load hyperparameters from checkpoint
        # For newer PyTorch Lightning checkpoints (ZIP format), we need to use Lightning's load_from_checkpoint
        # For older checkpoints, we might be able to read metadata
        hparams = {}
        if checkpoint_path.suffix == ".ckpt" or checkpoint_path.is_file():
            # Try to load checkpoint metadata
            try:
                # For ZIP-based checkpoints, try to extract hyperparameters
                # This might not work for all checkpoint formats
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    # Try different keys where hyperparameters might be stored
                    if "hyper_parameters" in checkpoint:
                        hparams = checkpoint["hyper_parameters"]
                    elif "hparams" in checkpoint:
                        hparams = checkpoint["hparams"]
                    elif "state_dict" in checkpoint and "hyper_parameters" in checkpoint.get("__metadata__", {}):
                        hparams = checkpoint["__metadata__"]["hyper_parameters"]
            except Exception as e:
                # If we can't read metadata, that's okay - we'll use override_params
                logger.debug(f"Could not extract hyperparameters from checkpoint: {e}")

        # Merge checkpoint hyperparameters with override params
        # override_params take precedence
        model_params = {**hparams, **override_params}

        # Create model instance with parameters
        instance = cls(
            max_epochs=model_params.get("max_epochs", 50),
            batch_size=model_params.get("batch_size", 64),
            learning_rate=model_params.get("learning_rate", 1e-3),
            hidden_size=model_params.get("hidden_size", 300),
            depth=model_params.get("depth", 3),
            dropout=model_params.get("dropout", 0.0),
            aggregation=model_params.get("aggregation", "mean"),
            batch_norm=model_params.get("batch_norm", True),
            ffn_num_layers=model_params.get("ffn_num_layers", 2),
            use_foundation=model_params.get("use_foundation", False),
            foundation_name=model_params.get("foundation_name", None),
            use_focal_loss=model_params.get("use_focal_loss", False),
            focal_alpha=model_params.get("focal_alpha", 0.25),
            focal_gamma=model_params.get("focal_gamma", 2.0),
            checkpoint_dir=model_params.get("checkpoint_dir", None),
            random_seed=model_params.get("random_seed", 42),
            accelerator=model_params.get("accelerator", None),
            early_stopping_rounds=model_params.get("early_stopping_rounds", None),
            early_stopping_metric=model_params.get("early_stopping_metric", "roc_auc"),
            n_tasks=model_params.get("n_tasks", 1),
        )

        # Initialize featurizer
        instance.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Load checkpoint into model using Lightning's load_from_checkpoint
        # Always load to CPU first to avoid CUDA errors on CPU-only machines
        # Temporarily patch torch.load to always use map_location="cpu"
        original_load = torch.load

        def cpu_load(*args, **kwargs):
            # torch.load signature: load(f, map_location=None, pickle_module=pickle, ...)
            # Check if map_location is provided as positional argument (2nd arg)
            # or keyword argument
            if len(args) >= 2:
                # map_location is provided as positional argument, replace it
                args_list = list(args)
                args_list[1] = 'cpu'  # Replace map_location (2nd positional arg)
                args = tuple(args_list)
                # Remove map_location from kwargs if present to avoid conflict
                kwargs.pop('map_location', None)
            else:
                # No positional map_location, set/override in kwargs
                kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)

        # Patch torch.load temporarily
        torch.load = cpu_load
        try:
            instance.model = models.MPNN.load_from_checkpoint(str(checkpoint_path))
        finally:
            # Restore original torch.load
            torch.load = original_load

        # Create a minimal trainer for prediction
        instance.trainer = pl.Trainer(
            accelerator=instance.accelerator,
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        return instance

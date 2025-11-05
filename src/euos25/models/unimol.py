"""Uni-Mol-2 model wrapper for binary classification and regression."""

import hashlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from rdkit import Chem
from rdkit.Chem import AllChem
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from tqdm import tqdm

from euos25.models.base import BaseClfModel

logger = logging.getLogger(__name__)

# Set OpenMP environment variables to prevent segmentation faults on macOS
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# Try to import Uni-Mol related packages
UNIMOL_AVAILABLE = False
UNIMOL_USE_TOOLS = False
UNIMOL_USE_UNIMOL = False

try:
    # Try to import from unimol_tools (recommended)
    from unimol_tools.models import UniMolModel as UniMolModelTools
    from unimol_tools.data import DataHub

    UNIMOL_AVAILABLE = True
    UNIMOL_USE_TOOLS = True
    logger.info("Uni-Mol package found (unimol_tools)")
except ImportError:
    try:
        # Fallback to original unimol package
        import unimol
        from unimol import UniMolModel as UniMolModelBase

        UNIMOL_AVAILABLE = True
        UNIMOL_USE_UNIMOL = True
        logger.info("Uni-Mol package found (unimol)")
    except ImportError:
        UNIMOL_AVAILABLE = False
        UNIMOL_USE_UNIMOL = False
        logger.warning(
            "Uni-Mol packages not found. Please install 'unimol_tools' package. "
            "Install with: pip install unimol_tools"
        )


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
        from torchmetrics.functional.classification import binary_auroc

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


# collate_fn is now provided by the model's batch_collate_fn method
# This function is kept for backward compatibility but should not be used
def collate_fn(batch):
    """Deprecated: Use model.batch_collate_fn instead."""
    raise NotImplementedError(
        "collate_fn is deprecated. Use model.batch_collate_fn instead. "
        "This should be handled automatically by the DataLoader."
    )


def _process_single_smiles_worker(args: Tuple[int, str, bool, int, Optional[str]]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Worker function for processing a single SMILES (for multiprocessing).

    Args:
        args: Tuple of (index, smiles, optimize_3d, max_attempts, cache_dir)

    Returns:
        Tuple of (index, coords, atom_types)
    """
    idx, smiles, optimize_3d, max_attempts, cache_dir = args

    # Check cache first
    if cache_dir:
        cache_path = _get_cache_path(smiles, optimize_3d, max_attempts, cache_dir)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    coords, atom_types = pickle.load(f)
                    return idx, coords, atom_types
            except Exception:
                pass

    # Generate 3D structure
    coords, atom_types = _smiles_to_3d_worker(smiles, optimize_3d, max_attempts)

    # Save to cache
    if cache_dir and coords is not None and atom_types is not None:
        cache_path = _get_cache_path(smiles, optimize_3d, max_attempts, cache_dir)
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump((coords, atom_types), f)
            except Exception:
                pass

    return idx, coords, atom_types


def _get_cache_path(smiles: str, optimize_3d: bool, max_attempts: int, cache_dir: str) -> Optional[Path]:
    """Get cache file path for a SMILES string.

    Args:
        smiles: SMILES string
        optimize_3d: Whether 3D optimization was used
        max_attempts: Maximum attempts for embedding
        cache_dir: Cache directory path

    Returns:
        Path to cache file or None
    """
    if not cache_dir:
        return None

    cache_key = hashlib.md5(
        f"{smiles}_{optimize_3d}_{max_attempts}".encode()
    ).hexdigest()
    return Path(cache_dir) / f"{cache_key}.pkl"


def _smiles_to_3d_worker(smiles: str, optimize_3d: bool, max_attempts: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert SMILES to 3D coordinates (worker function for multiprocessing).

    Args:
        smiles: SMILES string
        optimize_3d: Whether to optimize 3D structures
        max_attempts: Maximum attempts for 3D structure generation

    Returns:
        Tuple of (coordinates, atom_types) or (None, None) if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates with improved error handling
        embed_result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=max_attempts)

        if embed_result == -1:
            # If embedding failed, try with random coordinates
            embed_result = AllChem.EmbedMolecule(
                mol,
                randomSeed=42,
                maxAttempts=max_attempts,
                useRandomCoords=True
            )

        if embed_result == -1:
            # If still failed, try ETKDGv3
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.maxAttempts = max_attempts
                embed_result = AllChem.EmbedMolecule(mol, params)
            except Exception:
                pass

        # Optimize if requested and embedding succeeded
        if embed_result != -1 and optimize_3d:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                # If UFF fails, try MMFF
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass

        if embed_result == -1:
            return None, None

        # Get coordinates
        conf = mol.GetConformer()
        coords = conf.GetPositions()  # Shape: (n_atoms, 3)
        atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        return coords.astype(np.float32), atom_types.astype(np.int64)

    except Exception as e:
        logger.debug(f"Error generating 3D structure for {smiles}: {e}")
        return None, None


class UniMolDataset(torch.utils.data.Dataset):
    """Dataset for Uni-Mol that uses unimol_tools DataHub for feature generation."""

    def __init__(
        self,
        smiles: List[str],
        labels: Optional[np.ndarray] = None,
        model_name: str = "unimolv1",
        remove_hs: bool = False,
        max_atoms: int = 128,
        **params,
    ):
        """Initialize Uni-Mol dataset using unimol_tools.

        Args:
            smiles: List of SMILES strings
            labels: Optional labels (shape: (n_samples,) or (n_samples, n_tasks))
            model_name: Model name ("unimolv1" or "unimolv2")
            remove_hs: Whether to remove hydrogens
            max_atoms: Maximum number of atoms
            **params: Additional parameters for DataHub
        """
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "Uni-Mol packages not available. Please install 'unimol_tools' package."
            )

        self.smiles = smiles
        self.labels = labels
        self.model_name = model_name
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms

        # Set default save_path if not provided
        if 'save_path' not in params or params.get('save_path') is None:
            default_save_path = Path('data/processed/unimol_datahub')
            default_save_path.mkdir(parents=True, exist_ok=True)
            params['save_path'] = str(default_save_path)
            logger.info(f"Using default save_path: {params['save_path']}")

        save_path = Path(params['save_path'])
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate cache key from SMILES and parameters
        cache_key_data = {
            'smiles': tuple(smiles),  # Convert to tuple for hashing
            'model_name': model_name,
            'remove_hs': remove_hs,
            'max_atoms': max_atoms,
            'task': 'repr',
            'data_type': params.get('data_type', 'molecule'),
        }
        cache_key_str = str(sorted(cache_key_data.items()))
        cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_file = save_path / f"datahub_cache_{cache_key_hash}.pkl"

        # Try to load datahub from cache
        if cache_file.exists():
            logger.info(f"Loading cached DataHub object from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    datahub = pickle.load(f)
                # Extract processed data from cached datahub
                self.data = datahub.data['unimol_input']
                logger.info(f"Successfully loaded cached DataHub for {len(self.data)} molecules")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}. Regenerating...")

        # Use unimol_tools DataHub to process SMILES
        logger.info(f"Processing {len(smiles)} SMILES strings using unimol_tools...")

        # Create DataHub instance
        datahub_params = {
            'model_name': model_name,
            'remove_hs': remove_hs,
            'max_atoms': max_atoms,
            'task': 'repr',  # Use 'repr' for representation task
            'data_type': 'molecule',
            'is_train': False,  # Don't split data
            **params,
        }

        # Convert SMILES list to numpy array for DataHub
        smiles_array = np.array(smiles)
        datahub = DataHub(data=smiles_array, **datahub_params)

        # Extract processed data
        self.data = datahub.data['unimol_input']

        # Save datahub object to cache
        try:
            logger.info(f"Saving DataHub object to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(datahub, f)
            logger.info(f"Successfully cached DataHub object for {len(self.data)} molecules")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")

        logger.info(f"Successfully processed {len(self.data)} molecules")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the processed data from unimol_tools
        # The data is already in the format expected by the model
        feat_dict = self.data[idx]

        if self.labels is not None:
            if self.labels.ndim == 1:
                label = float(self.labels[idx])
            else:
                label = self.labels[idx].astype(np.float32)[0] if len(self.labels[idx]) == 1 else self.labels[idx].astype(np.float32)
            # Convert to float32 for MPS compatibility
            label = np.float32(label)
            return feat_dict, label
        else:
            return feat_dict, None


class UniMolPredictor(nn.Module):
    """Predictor head for Uni-Mol-2 model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 2,
        dropout: float = 0.0,
        n_tasks: int = 1,
        output_transform: Optional[Any] = None,
    ):
        """Initialize predictor head.

        Args:
            input_dim: Input dimension from backbone
            hidden_dim: Hidden dimension
            n_layers: Number of layers
            dropout: Dropout probability
            n_tasks: Number of tasks
            output_transform: Optional output transformation
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.output_transform = output_transform

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, n_tasks))
        self.predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (shape: (batch_size, input_dim))

        Returns:
            Predictions (shape: (batch_size, n_tasks))
        """
        output = self.predictor(x)
        if self.output_transform is not None:
            output = self.output_transform(output)
        return output


class UniMolLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Uni-Mol-2."""

    def __init__(
        self,
        backbone: nn.Module,
        predictor: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        metrics: Optional[List[Metric]] = None,
    ):
        """Initialize Lightning module.

        Args:
            backbone: Uni-Mol-2 backbone model
            predictor: Predictor head
            loss_fn: Loss function
            learning_rate: Learning rate
            metrics: Optional metrics
        """
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.metrics = metrics or []

        # Initialize metrics
        for metric in self.metrics:
            metric.reset()

    def on_fit_start(self):
        """Called at the beginning of fit."""
        # Move metrics to the same device as the model
        device = next(self.parameters()).device
        for metric in self.metrics:
            if hasattr(metric, 'to'):
                metric.to(device)

    def forward(self, batch_dict: Dict[str, torch.Tensor], return_repr: bool = False) -> torch.Tensor:
        """Forward pass using unimol_tools format.

        Args:
            batch_dict: Dictionary containing model inputs:
                - atom_feat, atom_mask, edge_feat, shortest_path, degree, pair_type, attn_bias, src_tokens, src_coord
            return_repr: Whether to return representations instead of predictions

        Returns:
            Predictions (shape: (batch_size, n_tasks)) or representations if return_repr=True
        """
        # Call backbone forward with all required arguments
        if return_repr:
            # Return CLS token representation
            output = self.backbone(
                atom_feat=batch_dict['atom_feat'],
                atom_mask=batch_dict['atom_mask'],
                edge_feat=batch_dict['edge_feat'],
                shortest_path=batch_dict['shortest_path'],
                degree=batch_dict['degree'],
                pair_type=batch_dict['pair_type'],
                attn_bias=batch_dict['attn_bias'],
                src_tokens=batch_dict['src_tokens'],
                src_coord=batch_dict['src_coord'],
                return_repr=True,
            )
            # Extract CLS representation
            if isinstance(output, dict):
                embeddings = output['cls_repr']
            else:
                embeddings = output
        else:
            # Get logits from backbone (if it has classification_head)
            # Otherwise, get CLS representation and use our predictor
            try:
                # Try to get logits directly from backbone
                logits = self.backbone(
                    atom_feat=batch_dict['atom_feat'],
                    atom_mask=batch_dict['atom_mask'],
                    edge_feat=batch_dict['edge_feat'],
                    shortest_path=batch_dict['shortest_path'],
                    degree=batch_dict['degree'],
                    pair_type=batch_dict['pair_type'],
                    attn_bias=batch_dict['attn_bias'],
                    src_tokens=batch_dict['src_tokens'],
                    src_coord=batch_dict['src_coord'],
                    return_repr=False,
                )
                # If backbone returns logits, use them directly (may need to adjust dimensions)
                if logits.shape[-1] == self.predictor.n_tasks:
                    return logits
                else:
                    # Extract CLS representation and use our predictor
                    repr_output = self.backbone(
                        atom_feat=batch_dict['atom_feat'],
                        atom_mask=batch_dict['atom_mask'],
                        edge_feat=batch_dict['edge_feat'],
                        shortest_path=batch_dict['shortest_path'],
                        degree=batch_dict['degree'],
                        pair_type=batch_dict['pair_type'],
                        attn_bias=batch_dict['attn_bias'],
                        src_tokens=batch_dict['src_tokens'],
                        src_coord=batch_dict['src_coord'],
                        return_repr=True,
                    )
                    if isinstance(repr_output, dict):
                        embeddings = repr_output['cls_repr']
                    else:
                        embeddings = repr_output
            except Exception as e:
                # Fallback: get representation and use our predictor
                logger.debug(f"Getting representation from backbone: {e}")
                repr_output = self.backbone(
                    atom_feat=batch_dict['atom_feat'],
                    atom_mask=batch_dict['atom_mask'],
                    edge_feat=batch_dict['edge_feat'],
                    shortest_path=batch_dict['shortest_path'],
                    degree=batch_dict['degree'],
                    pair_type=batch_dict['pair_type'],
                    attn_bias=batch_dict['attn_bias'],
                    src_tokens=batch_dict['src_tokens'],
                    src_coord=batch_dict['src_coord'],
                    return_repr=True,
                )
                if isinstance(repr_output, dict):
                    embeddings = repr_output['cls_repr']
                else:
                    embeddings = repr_output

        # Use our predictor on CLS representation
        output = self.predictor(embeddings)
        return output

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        """Transfer batch to device with float32 conversion for MPS compatibility."""
        batch_dict, labels = batch

        # Convert batch_dict tensors to float32 if needed (for MPS)
        if isinstance(batch_dict, dict):
            batch_dict = {
                k: v.to(device, dtype=torch.float32) if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch_dict.items()
            }
        else:
            batch_dict = batch_dict.to(device)

        # Convert labels to float32 and reshape to (batch_size, n_tasks)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device, dtype=torch.float32)
            else:
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            # Reshape to (batch_size, n_tasks) if needed
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

        return (batch_dict, labels)

    def training_step(self, batch, batch_idx):
        """Training step."""
        # batch is now (batch_dict, labels) from batch_collate_fn
        batch_dict, labels = batch
        # Ensure labels are float32 and correct shape
        if labels is not None:
            if labels.dtype != torch.float32:
                labels = labels.to(torch.float32)
            # Reshape labels to match preds shape: (batch_size, n_tasks)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        preds = self.forward(batch_dict)
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # batch is now (batch_dict, labels) from batch_collate_fn
        batch_dict, labels = batch
        # Ensure labels are float32 and correct shape
        if labels is not None:
            if labels.dtype != torch.float32:
                labels = labels.to(torch.float32)
            # Reshape labels to match preds shape: (batch_size, n_tasks)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        preds = self.forward(batch_dict)
        loss = self.loss_fn(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics (flatten for metrics that expect 1D)
        for metric in self.metrics:
            metric.update(preds.flatten() if preds.dim() > 1 else preds, labels.flatten() if labels.dim() > 1 else labels)

        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        for metric in self.metrics:
            value = metric.compute()
            self.log(f"val/{metric.__class__.__name__}", value, prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        # Handle both tuple (batch_dict, labels) and dict formats
        if isinstance(batch, tuple):
            batch_dict, _ = batch
        else:
            batch_dict = batch

        # Forward pass
        preds = self.forward(batch_dict)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(preds)
        return probs


class UniMolModel(BaseClfModel):
    """Uni-Mol-2 model wrapper for binary classification and regression.

    This model uses Uni-Mol-2's 3D structure-based molecular representation
    for property prediction.
    """

    def __init__(
        self,
        max_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        hidden_size: int = 300,
        ffn_num_layers: int = 2,
        dropout: float = 0.0,
        pretrained_model: Optional[str] = None,  # "unimol", "unimol2", "unimol2-84m", etc.
        pretrained_model_path: Optional[str] = None,
        freeze_backbone: bool = False,
        freeze_layers: Optional[List[int]] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        objective_type: Optional[str] = None,  # "regression", "listmle", or None (binary)
        regression_loss_type: str = "mse",  # "mse", "mae", "huber", "smooth_l1"
        regression_loss_params: Optional[Dict[str, Any]] = None,
        optimize_3d: bool = True,
        remove_hs: bool = False,  # Whether to remove hydrogens (for unimol_tools)
        checkpoint_dir: Optional[str] = None,
        random_seed: int = 42,
        accelerator: Optional[str] = None,
        devices: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: str = "roc_auc",
        n_tasks: int = 1,
        binary_labels: Optional[np.ndarray] = None,
        cache_dir: Optional[str] = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        """Initialize Uni-Mol-2 model.

        Args:
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training (per GPU when using DDP)
            learning_rate: Learning rate
            hidden_size: Hidden dimension size for predictor
            ffn_num_layers: Number of feed-forward network layers
            dropout: Dropout probability
            pretrained_model: Name of pretrained model ("unimol2", "unimol2-large", "unimol3")
            pretrained_model_path: Path to custom pretrained model
            freeze_backbone: Whether to freeze backbone parameters
            freeze_layers: List of layer indices to freeze (if None, all layers are frozen if freeze_backbone=True)
            use_focal_loss: Whether to use Focal loss for imbalanced data
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            objective_type: Objective type ("regression", "listmle", or None for binary)
            regression_loss_type: Type of regression loss ("mse", "mae", "huber", "smooth_l1")
            regression_loss_params: Parameters for regression loss function
            optimize_3d: Whether to optimize 3D structures
            checkpoint_dir: Directory to save model checkpoints
            random_seed: Random seed for reproducibility
            accelerator: Accelerator to use ('auto', 'cpu', 'cuda', 'mps')
            devices: Number of GPUs to use. If None, automatically detects from CUDA_VISIBLE_DEVICES.
                     When using multiple GPUs, DDP (Distributed Data Parallel) will be enabled.
            early_stopping_rounds: Number of epochs to wait before early stopping
            early_stopping_metric: Metric to monitor for early stopping
            n_tasks: Number of tasks for multi-task learning
            binary_labels: Binary labels (0/1) for ROC AUC calculation in regression/ranking mode
            cache_dir: Directory to cache 3D structures (optional)
            n_jobs: Number of parallel jobs for 3D structure generation (1 = sequential, -1 = auto)
            **kwargs: Additional parameters
        """
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "Uni-Mol packages not available. Please install 'unimol_tools' package. "
                "Install with: pip install unimol_tools"
            )

        super().__init__(name="unimol", **kwargs)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.dropout = dropout
        self.pretrained_model = pretrained_model
        self.pretrained_model_path = pretrained_model_path
        self.freeze_backbone = freeze_backbone
        self.freeze_layers = freeze_layers
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.objective_type = objective_type
        self.regression_loss_type = regression_loss_type
        self.regression_loss_params = regression_loss_params or {}
        self.optimize_3d = optimize_3d
        self.remove_hs = remove_hs
        self.checkpoint_dir = checkpoint_dir or "checkpoints/unimol"
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_metric = early_stopping_metric
        self.n_tasks = n_tasks
        self.binary_labels = binary_labels
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs
        self.devices = devices

        # Determine accelerator
        if accelerator is None or accelerator == "auto":
            if torch.cuda.is_available():
                self.accelerator = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                try:
                    test_tensor = torch.randn(2, 2, device="mps")
                    _ = test_tensor @ test_tensor.T
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    self.accelerator = "mps"
                except Exception:
                    self.accelerator = "cpu"
            else:
                self.accelerator = "cpu"
        else:
            self.accelerator = accelerator

        logger.info(f"Using accelerator: {self.accelerator}")

        self.model = None
        self.trainer = None
        self.backbone = None
        self.best_iteration: int = 0

        # Set random seed
        pl.seed_everything(self.random_seed)

    def _load_pretrained_backbone(self) -> nn.Module:
        """Load pretrained Uni-Mol backbone.

        Returns:
            Backbone model
        """
        if not UNIMOL_AVAILABLE:
            raise ImportError(
                "Uni-Mol packages not available. Please install 'unimol_tools' package. "
                "Install with: pip install unimol_tools"
            )

        # Determine model name and size
        model_name = "unimolv1"  # Default to Uni-Mol v1
        model_size = "84m"  # Default size for v2

        if self.pretrained_model:
            # Map pretrained_model names to unimol_tools model names
            if self.pretrained_model in ["unimol", "unimol1", "unimolv1"]:
                model_name = "unimolv1"
            elif self.pretrained_model in ["unimol2", "unimolv2"]:
                model_name = "unimolv2"
                model_size = "84m"  # Default to 84M model
            elif self.pretrained_model.startswith("unimol2-"):
                # Extract size from name like "unimol2-310m"
                model_name = "unimolv2"
                size_str = self.pretrained_model.replace("unimol2-", "")
                if size_str in ["84m", "164m", "310m", "570m", "1.1B"]:
                    model_size = size_str
                else:
                    logger.warning(f"Unknown model size: {size_str}, using default 84m")
                    model_size = "84m"
            else:
                logger.warning(f"Unknown pretrained_model: {self.pretrained_model}, using default unimolv1")
                model_name = "unimolv1"

        logger.info(f"Loading pretrained Uni-Mol model: {model_name}" + (f" (size: {model_size})" if model_name == "unimolv2" else ""))

        if UNIMOL_USE_TOOLS:
            # Use unimol_tools (recommended approach)
            # Create a UniMolModel instance with output_dim=1 (we'll extract the encoder)
            remove_hs = self.remove_hs

            # Create model instance
            if model_name == "unimolv1":
                backbone = UniMolModelTools(
                    output_dim=1,  # We'll replace this with our own head
                    data_type="molecule",
                    remove_hs=remove_hs,
                )
            elif model_name == "unimolv2":
                from unimol_tools.models import UniMolV2Model
                backbone = UniMolV2Model(
                    output_dim=1,  # We'll replace this with our own head
                    model_size=model_size,
                )
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            # Load custom checkpoint if provided
            if self.pretrained_model_path:
                logger.info(f"Loading custom pretrained model from {self.pretrained_model_path}")
                backbone.load_pretrained_weights(self.pretrained_model_path, strict=False)

            # Freeze backbone if requested
            if self.freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False
                logger.info("Backbone parameters frozen")
            elif self.freeze_layers is not None:
                # Freeze specific layers
                for name, param in backbone.named_parameters():
                    for layer_prefix in self.freeze_layers:
                        if name.startswith(layer_prefix):
                            param.requires_grad = False
                logger.info(f"Frozen layers: {self.freeze_layers}")

            return backbone
        elif UNIMOL_USE_UNIMOL:
            # Use original unimol package (if available)
            if self.pretrained_model_path:
                logger.info(f"Loading pretrained model from {self.pretrained_model_path}")
                backbone = UniMolModelBase.from_pretrained(self.pretrained_model_path)
            else:
                raise ValueError("pretrained_model_path must be specified when using original unimol package")

            if self.freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False
                logger.info("Backbone parameters frozen")

            return backbone
        else:
            raise ImportError("Uni-Mol packages not available. Please install 'unimol_tools' package.")

    def _build_model(self) -> UniMolLightningModule:
        """Build Uni-Mol-2 model.

        Returns:
            Lightning module
        """
        # Load backbone
        backbone = self._load_pretrained_backbone()

        # Get backbone output dimension
        # For unimol_tools, the output dimension is stored in args.encoder_embed_dim
        backbone_output_dim = 512  # Default
        try:
            if UNIMOL_USE_TOOLS:
                # For unimol_tools models, get dimension from args
                if hasattr(backbone, "args"):
                    backbone_output_dim = backbone.args.encoder_embed_dim
                elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "embed_dim"):
                    backbone_output_dim = backbone.encoder.embed_dim
                else:
                    # Try to get from encoder directly
                    if hasattr(backbone, "encoder"):
                        # Check first layer's embedding dimension
                        if hasattr(backbone.encoder, "layers") and len(backbone.encoder.layers) > 0:
                            if hasattr(backbone.encoder.layers[0], "embed_dim"):
                                backbone_output_dim = backbone.encoder.layers[0].embed_dim
                            elif hasattr(backbone.encoder.layers[0], "self_attn"):
                                # Get from attention layer
                                if hasattr(backbone.encoder.layers[0].self_attn, "embed_dim"):
                                    backbone_output_dim = backbone.encoder.layers[0].self_attn.embed_dim
                logger.info(f"Backbone output dimension (from unimol_tools): {backbone_output_dim}")
            else:
                # For original unimol package
                if hasattr(backbone, "config"):
                    if hasattr(backbone.config, "hidden_size"):
                        backbone_output_dim = backbone.config.hidden_size
                    elif hasattr(backbone.config, "d_model"):
                        backbone_output_dim = backbone.config.d_model
                    elif hasattr(backbone.config, "embed_dim"):
                        backbone_output_dim = backbone.config.embed_dim
                elif hasattr(backbone, "hidden_size"):
                    backbone_output_dim = backbone.hidden_size
                elif hasattr(backbone, "embed_dim"):
                    backbone_output_dim = backbone.embed_dim
        except Exception as e:
            logger.warning(f"Could not determine backbone output dimension: {e}. Using default: {backbone_output_dim}")

        logger.info(f"Backbone output dimension: {backbone_output_dim}")

        # Create predictor
        predictor = UniMolPredictor(
            input_dim=backbone_output_dim,
            hidden_dim=self.hidden_size,
            n_layers=self.ffn_num_layers,
            dropout=self.dropout,
            n_tasks=self.n_tasks,
        )

        # Create loss function
        if self.objective_type == "regression":
            from euos25.models.unimol_regression import create_regression_loss

            loss_fn = create_regression_loss(self.regression_loss_type, self.regression_loss_params)
            logger.info(f"Using regression loss: {self.regression_loss_type}")
        elif self.objective_type == "listmle":
            from euos25.models.unimol_listmle import ListMLELoss

            loss_fn = ListMLELoss()
            logger.info("Using ListMLE loss")
        elif self.use_focal_loss:
            from euos25.models.unimol_focal import FocalLoss

            loss_fn = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            logger.info(f"Using Focal Loss with alpha={self.focal_alpha}, gamma={self.focal_gamma}")
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            logger.info("Using BCE loss")

        # Create metrics
        metrics = []
        if self.objective_type in ["regression", "listmle"] and self.binary_labels is not None:
            binary_labels_tensor = torch.from_numpy(self.binary_labels).float()
            roc_metric = BinaryROCAUCMetric(binary_labels=binary_labels_tensor)
            roc_metric.reset()
            metrics.append(roc_metric)
        else:
            from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

            metrics.append(BinaryAUROC())
            metrics.append(BinaryAccuracy())

        # Create Lightning module
        module = UniMolLightningModule(
            backbone=backbone,
            predictor=predictor,
            loss_fn=loss_fn,
            learning_rate=self.learning_rate,
            metrics=metrics,
        )

        # Move metrics to device after module creation
        # This will be handled by Lightning automatically, but we ensure it here
        if hasattr(module, 'metrics'):
            for metric in module.metrics:
                if hasattr(metric, 'to'):
                    # Device will be set by Lightning during training
                    pass

        return module

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
    ) -> "UniMolModel":
        """Train the Uni-Mol-2 model.

        Args:
            X_train: Training features (must contain 'SMILES' column)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            binary_labels_val: Binary labels for validation set ROC AUC calculation
            resume_from_checkpoint: Path to checkpoint file to resume training from
            task_name: Task name for organizing checkpoints (e.g., "transmittance340")
            fold_name: Fold name for organizing checkpoints (e.g., "fold_0", "full")

        Returns:
            Self
        """
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("Training Uni-Mol-2 model")

        # Store validation binary labels for metric
        if binary_labels_val is not None:
            self.binary_labels_val = binary_labels_val
        else:
            self.binary_labels_val = None

        # Build model first to get batch_collate_fn
        if self.binary_labels_val is not None:
            original_binary_labels = self.binary_labels
            self.binary_labels = self.binary_labels_val
        self.model = self._build_model()
        if self.binary_labels_val is not None:
            self.binary_labels = original_binary_labels

        # Determine model name for dataset
        model_name = "unimolv1"
        if self.pretrained_model:
            if self.pretrained_model in ["unimol", "unimol1", "unimolv1"]:
                model_name = "unimolv1"
            elif self.pretrained_model in ["unimol2", "unimolv2"] or self.pretrained_model.startswith("unimol2-"):
                model_name = "unimolv2"

        # Create datasets using unimol_tools format
        train_smiles = X_train["SMILES"].tolist()
        train_dataset = UniMolDataset(
            train_smiles,
            y_train,
            model_name=model_name,
            remove_hs=self.remove_hs,
            max_atoms=128,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.model.backbone.batch_collate_fn,  # Use model's collate function
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_smiles = X_val["SMILES"].tolist()
            val_dataset = UniMolDataset(
                val_smiles,
                y_val,
                model_name=model_name,
                remove_hs=self.remove_hs,
                max_atoms=128,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.model.backbone.batch_collate_fn,  # Use model's collate function
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

        # Setup early stopping
        if (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and val_loader is not None
        ):
            # Map early_stopping_metric to actual logged metric name
            # Metrics are logged as val/{metric.__class__.__name__}
            # Binary classification uses BinaryAUROC, regression/ranking with binary_labels uses BinaryROCAUCMetric
            if self.early_stopping_metric.lower() == "roc_auc":
                if self.objective_type in ["regression", "listmle"] and self.binary_labels is not None:
                    monitor_metric = "val/BinaryROCAUCMetric"
                else:
                    monitor_metric = "val/BinaryAUROC"
            elif self.early_stopping_metric.lower() == "pr_auc":
                monitor_metric = "val/BinaryAveragePrecision"  # Assuming PR-AUC uses BinaryAveragePrecision
            else:
                monitor_metric = "val_loss"

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

        # Determine number of devices
        if self.devices is not None:
            # Use explicitly specified number of devices
            num_devices = self.devices
        else:
            # Auto-detect from CUDA_VISIBLE_DEVICES
            if self.accelerator == "cuda" and torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
            else:
                num_devices = 1

        # Setup strategy for DDP
        strategy = None
        if num_devices > 1 and self.accelerator == "cuda":
            strategy = DDPStrategy(find_unused_parameters=True)
            logger.info(f"Using DDP (Distributed Data Parallel) with {num_devices} GPUs")
            logger.info(f"Effective batch size: {self.batch_size * num_devices} (batch_size={self.batch_size} per GPU)")

        # Setup trainer
        trainer_kwargs = {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "devices": num_devices,
            "callbacks": callbacks,
            "enable_progress_bar": True,
            "logger": False,
            "num_sanity_val_steps": 0,
            "enable_model_summary": False,
        }

        if strategy is not None:
            trainer_kwargs["strategy"] = strategy

        if self.accelerator == "mps":
            trainer_kwargs["precision"] = "32"
            trainer_kwargs["accumulate_grad_batches"] = 1
            trainer_kwargs["deterministic"] = False
            trainer_kwargs["gradient_clip_val"] = None

        self.trainer = pl.Trainer(**trainer_kwargs)

        # Determine checkpoint path for resuming
        ckpt_path = None
        if resume_from_checkpoint:
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.exists():
                if checkpoint_path.is_dir():
                    last_ckpt = checkpoint_path / "last.ckpt"
                    if last_ckpt.exists():
                        ckpt_path = str(last_ckpt)
                    else:
                        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
                        if ckpt_files:
                            ckpt_path = str(sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1])
                elif checkpoint_path.is_file():
                    ckpt_path = str(checkpoint_path)

        # Train
        try:
            self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
        except (RuntimeError, SystemError) as e:
            if self.accelerator == "mps" and "MPS" in str(e):
                logger.warning(f"MPS training failed: {e}. Falling back to CPU.")
                trainer_kwargs["accelerator"] = "cpu"
                trainer_kwargs.pop("precision", None)
                trainer_kwargs["deterministic"] = True
                self.accelerator = "cpu"
                self.trainer = pl.Trainer(**trainer_kwargs)
                self.trainer.fit(self.model, train_loader, val_loader, ckpt_path=ckpt_path)
            else:
                raise

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

        # Determine model name for dataset
        model_name = "unimolv1"
        if self.pretrained_model:
            if self.pretrained_model in ["unimol", "unimol1", "unimolv1"]:
                model_name = "unimolv1"
            elif self.pretrained_model in ["unimol2", "unimolv2"] or self.pretrained_model.startswith("unimol2-"):
                model_name = "unimolv2"

        # Create dataset and dataloader
        smiles = X["SMILES"].tolist()
        dataset = UniMolDataset(
            smiles,
            labels=None,
            model_name=model_name,
            remove_hs=self.remove_hs,
            max_atoms=128,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.model.backbone.batch_collate_fn  # Use model's collate function
        )

        # Predict
        self.model.eval()
        predictions = self.trainer.predict(self.model, test_loader)

        # Convert to numpy array
        probs = torch.cat(predictions)

        # For regression/ranking, apply sigmoid to convert logits to probabilities
        if self.objective_type in ["regression", "listmle"]:
            probs = torch.sigmoid(probs)

        probs = probs.cpu().numpy()

        # Ensure correct shape
        n_samples = len(X)
        if self.n_tasks == 1:
            probs_positive = probs.reshape(-1)
            if len(probs_positive) != n_samples:
                raise ValueError(f"Shape mismatch: expected {n_samples} samples, got {len(probs_positive)}")
            probs_negative = 1 - probs_positive
            probs_binary = np.column_stack([probs_negative, probs_positive])
            return probs_binary
        else:
            if probs.ndim == 1:
                if len(probs) != n_samples * self.n_tasks:
                    raise ValueError(f"Shape mismatch: expected {n_samples * self.n_tasks} values, got {len(probs)}")
                probs = probs.reshape(n_samples, self.n_tasks)
            elif probs.ndim == 2:
                if probs.shape[0] != n_samples:
                    raise ValueError(f"Shape mismatch: expected {n_samples} samples, got {probs.shape[0]}")
                if probs.shape[1] != self.n_tasks:
                    raise ValueError(f"Shape mismatch: expected {self.n_tasks} tasks, got {probs.shape[1]}")
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

    @classmethod
    def load(cls, path: str) -> "UniMolModel":
        """Load model from file.

        This is a convenience method that delegates to load_from_checkpoint.
        For more control over loading parameters, use load_from_checkpoint directly.

        Args:
            path: Path to load model from (checkpoint file or directory)

        Returns:
            Loaded UniMolModel instance
        """
        return cls.load_from_checkpoint(path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **override_params,
    ) -> "UniMolModel":
        """Load UniMolModel from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory
            **override_params: Parameters to override from checkpoint

        Returns:
            Loaded UniMolModel instance
        """
        checkpoint_path = Path(checkpoint_path)

        # If checkpoint_path is a directory, look for checkpoint files
        if checkpoint_path.is_dir():
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
        hparams = {}
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                # Try different keys where hyperparameters might be stored
                if "hyper_parameters" in checkpoint:
                    hparams = checkpoint["hyper_parameters"]
                elif "hparams" in checkpoint:
                    hparams = checkpoint["hparams"]
                elif "callbacks" in checkpoint and "ModelCheckpoint" in str(checkpoint):
                    # Try to extract from callbacks
                    pass
        except Exception as e:
            logger.debug(f"Could not extract hyperparameters from checkpoint: {e}")

        # Merge checkpoint hyperparameters with override params
        model_params = {**hparams, **override_params}

        # Create model instance with parameters
        # Use default values if not in checkpoint
        instance = cls(
            max_epochs=model_params.get("max_epochs", 50),
            batch_size=model_params.get("batch_size", 32),
            learning_rate=model_params.get("learning_rate", 1e-3),
            hidden_size=model_params.get("hidden_size", 300),
            ffn_num_layers=model_params.get("ffn_num_layers", 2),
            dropout=model_params.get("dropout", 0.0),
            pretrained_model=model_params.get("pretrained_model", None),
            pretrained_model_path=model_params.get("pretrained_model_path", None),
            freeze_backbone=model_params.get("freeze_backbone", False),
            freeze_layers=model_params.get("freeze_layers", None),
            use_focal_loss=model_params.get("use_focal_loss", False),
            focal_alpha=model_params.get("focal_alpha", 0.25),
            focal_gamma=model_params.get("focal_gamma", 2.0),
            objective_type=model_params.get("objective_type", None),
            regression_loss_type=model_params.get("regression_loss_type", "mse"),
            regression_loss_params=model_params.get("regression_loss_params", {}),
            optimize_3d=model_params.get("optimize_3d", True),
            checkpoint_dir=model_params.get("checkpoint_dir", None),
            random_seed=model_params.get("random_seed", 42),
            accelerator=model_params.get("accelerator", None),
            devices=model_params.get("devices", None),
            early_stopping_rounds=model_params.get("early_stopping_rounds", None),
            early_stopping_metric=model_params.get("early_stopping_metric", "roc_auc"),
            n_tasks=model_params.get("n_tasks", 1),
            cache_dir=model_params.get("cache_dir", None),
            n_jobs=model_params.get("n_jobs", -1),
        )

        # Load checkpoint into model using Lightning's load_from_checkpoint
        try:
            instance.model = UniMolLightningModule.load_from_checkpoint(
                str(checkpoint_path), strict=False
            )
        except Exception as e:
            logger.warning(f"Could not load checkpoint using Lightning: {e}")
            # Try to load manually
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # Rebuild model and load state dict
                    instance.model = instance._build_model()
                    instance.model.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    raise ValueError("Checkpoint format not recognized")
            except Exception as e2:
                logger.error(f"Failed to load checkpoint: {e2}")
                raise

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

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        params = super().get_params()
        params.update(
            {
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "hidden_size": self.hidden_size,
                "ffn_num_layers": self.ffn_num_layers,
                "dropout": self.dropout,
                "pretrained_model": self.pretrained_model,
                "pretrained_model_path": self.pretrained_model_path,
                "freeze_backbone": self.freeze_backbone,
                "freeze_layers": self.freeze_layers,
                "use_focal_loss": self.use_focal_loss,
                "focal_alpha": self.focal_alpha,
                "focal_gamma": self.focal_gamma,
                "objective_type": self.objective_type,
                "regression_loss_type": self.regression_loss_type,
                "regression_loss_params": self.regression_loss_params,
                "optimize_3d": self.optimize_3d,
                "random_seed": self.random_seed,
                "devices": self.devices,
                "early_stopping_rounds": self.early_stopping_rounds,
                "early_stopping_metric": self.early_stopping_metric,
                "n_tasks": self.n_tasks,
                "cache_dir": self.cache_dir,
                "n_jobs": self.n_jobs,
            }
        )
        return params


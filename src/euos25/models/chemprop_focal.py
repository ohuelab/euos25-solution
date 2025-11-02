"""ChemProp with Focal Loss for imbalanced classification."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from chemprop import nn

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    """Focal Loss for binary classification.

    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (N,) or (N, 1)
            targets: Ground truth labels of shape (N,) or (N, 1)

        Returns:
            Focal loss value
        """
        # Ensure inputs and targets have same shape
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)

        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute modulating factor: (1 - p_t)^gamma
        # where p_t = p if y=1, else 1-p
        p_t = torch.where(targets == 1, probs, 1 - probs)
        modulating_factor = (1 - p_t) ** self.gamma

        # Compute alpha factor: alpha if y=1, else 1-alpha
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # none
            return focal_loss


class BinaryClassificationFFNWithFocalLoss(nn.BinaryClassificationFFN):
    """Binary classification FFN with Focal Loss.

    This extends ChemProp's BinaryClassificationFFN to use Focal Loss
    instead of standard BCE loss.
    """

    def __init__(
        self,
        *args,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        """Initialize FFN with Focal Loss.

        Args:
            *args: Arguments for BinaryClassificationFFN
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            **kwargs: Keyword arguments for BinaryClassificationFFN
        """
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        logger.info(
            f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}"
        )

    def train_step(self, Z: torch.Tensor) -> torch.Tensor:
        """Forward pass during training with Focal Loss.

        Args:
            Z: Input features

        Returns:
            Predictions (logits)
        """
        # Standard forward pass
        return self.forward(Z)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.

        Args:
            preds: Predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        # Remove output transform if present (we want logits)
        if hasattr(self, "output_transform") and self.output_transform is not None:
            # ChemProp applies output transform, but we need raw logits for focal loss
            # The output_transform is applied in eval/predict, not in loss computation
            pass

        return self.focal_loss(preds, targets)


def create_focal_loss_ffn(
    n_tasks: int = 1,
    input_dim: Optional[int] = None,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    output_transform: Optional[nn.UnscaleTransform] = None,
) -> BinaryClassificationFFNWithFocalLoss:
    """Factory function to create FFN with Focal Loss.

    Args:
        n_tasks: Number of tasks (typically 1 for binary classification)
        input_dim: Input dimension (must match message passing output)
        hidden_dim: Hidden layer dimension
        n_layers: Number of FFN layers
        dropout: Dropout probability
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        output_transform: Optional output transformation

    Returns:
        BinaryClassificationFFNWithFocalLoss instance
    """
    return BinaryClassificationFFNWithFocalLoss(
        n_tasks=n_tasks,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        output_transform=output_transform,
    )

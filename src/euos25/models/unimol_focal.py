"""Uni-Mol-2 with Focal Loss for imbalanced classification."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    """
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
            inputs: Predictions (logits) of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)

        Returns:
            Focal loss value
        """
        # Handle multi-task case: process each task independently
        if inputs.dim() == 2 and inputs.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = inputs.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = self._compute_focal_loss(
                    inputs[:, task_idx], targets[:, task_idx]
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if inputs.dim() > 1:
                inputs = inputs.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return self._compute_focal_loss(inputs, targets)

    def _compute_focal_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss for a single task.

        Args:
            inputs: Predictions (logits) of shape (N,)
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value (scalar)
        """
        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

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


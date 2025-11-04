"""Uni-Mol-2 with Regression Loss functions."""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MSELoss(torch.nn.Module):
    """Mean Squared Error Loss for regression."""

    def __init__(self, reduction: str = "mean"):
        """Initialize MSE Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            preds: Predictions of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)

        Returns:
            MSE loss value
        """
        # Handle multi-task case: process each task independently
        if preds.dim() == 2 and preds.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = preds.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = F.mse_loss(
                    preds[:, task_idx], targets[:, task_idx], reduction=self.reduction
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return F.mse_loss(preds, targets, reduction=self.reduction)


class MAELoss(torch.nn.Module):
    """Mean Absolute Error Loss for regression."""

    def __init__(self, reduction: str = "mean"):
        """Initialize MAE Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss.

        Args:
            preds: Predictions of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)

        Returns:
            MAE loss value
        """
        # Handle multi-task case: process each task independently
        if preds.dim() == 2 and preds.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = preds.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = F.l1_loss(
                    preds[:, task_idx], targets[:, task_idx], reduction=self.reduction
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return F.l1_loss(preds, targets, reduction=self.reduction)


class HuberLoss(torch.nn.Module):
    """Huber Loss for regression (smooth transition between MSE and MAE)."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """Initialize Huber Loss.

        Args:
            delta: Threshold for transition between MSE and MAE
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.

        Args:
            preds: Predictions of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)

        Returns:
            Huber loss value
        """
        # Handle multi-task case: process each task independently
        if preds.dim() == 2 and preds.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = preds.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = F.huber_loss(
                    preds[:, task_idx],
                    targets[:, task_idx],
                    delta=self.delta,
                    reduction=self.reduction,
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return F.huber_loss(preds, targets, delta=self.delta, reduction=self.reduction)


class SmoothL1Loss(torch.nn.Module):
    """Smooth L1 Loss for regression (variant of Huber Loss with delta=1.0)."""

    def __init__(self, reduction: str = "mean"):
        """Initialize Smooth L1 Loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Smooth L1 loss.

        Args:
            preds: Predictions of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)

        Returns:
            Smooth L1 loss value
        """
        # Handle multi-task case: process each task independently
        if preds.dim() == 2 and preds.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = preds.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = F.smooth_l1_loss(
                    preds[:, task_idx], targets[:, task_idx], reduction=self.reduction
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return F.smooth_l1_loss(preds, targets, reduction=self.reduction)


def create_regression_loss(
    loss_type: str = "mse", loss_params: Optional[dict] = None
) -> torch.nn.Module:
    """Factory function to create regression loss.

    Args:
        loss_type: Type of loss ('mse', 'mae', 'huber', 'smooth_l1')
        loss_params: Optional parameters for the loss function

    Returns:
        Loss function module
    """
    if loss_params is None:
        loss_params = {}

    if loss_type == "mse":
        return MSELoss(**loss_params)
    elif loss_type == "mae":
        return MAELoss(**loss_params)
    elif loss_type == "huber":
        delta = loss_params.get("delta", 1.0)
        return HuberLoss(delta=delta, **{k: v for k, v in loss_params.items() if k != "delta"})
    elif loss_type == "smooth_l1":
        return SmoothL1Loss(**loss_params)
    else:
        raise ValueError(f"Unknown regression loss type: {loss_type}")


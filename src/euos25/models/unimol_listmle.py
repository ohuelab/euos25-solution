"""Uni-Mol-2 with ListMLE Loss for ranking learning."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ListMLELoss(torch.nn.Module):
    """ListMLE (Listwise Maximum Likelihood Estimation) Loss for ranking.

    ListMLE is a listwise ranking loss that optimizes the likelihood
    of the correct ranking order.

    Reference: https://icml.cc/Conferences/2008/papers/167.pdf
    """

    def __init__(self, eps: float = 1e-12):
        """Initialize ListMLE Loss.

        Args:
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            preds: Predictions (logits) of shape (N,) or (N, 1) or (N, n_tasks)
            targets: Ground truth labels of shape (N,) or (N, 1) or (N, n_tasks)
                     For ranking, these are the quantitative values to rank by.

        Returns:
            ListMLE loss value
        """
        # Handle multi-task case: process each task independently
        if preds.dim() == 2 and preds.shape[1] > 1:
            # Multi-task: shape (N, n_tasks)
            n_tasks = preds.shape[1]
            losses = []
            for task_idx in range(n_tasks):
                task_loss = self._compute_listmle_loss(
                    preds[:, task_idx], targets[:, task_idx]
                )
                losses.append(task_loss)
            return torch.stack(losses).mean()
        else:
            # Single task: shape (N,)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            return self._compute_listmle_loss(preds, targets)

    def _compute_listmle_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute ListMLE loss for a single task.

        Args:
            preds: Predictions of shape (N,)
            targets: Targets of shape (N,)

        Returns:
            ListMLE loss value (scalar)
        """
        # Treat all samples as one group for listwise ranking
        n_samples = preds.shape[0]

        # Get ranking order (descending by target values)
        # pi = argsort(-targets) gives indices sorted by descending target
        _, pi = torch.sort(targets, descending=True, stable=True)

        # Reorder predictions according to ranking
        s_pi = preds[pi]

        # Numerical stability: shift by max before exp
        s_max = s_pi.max()
        s_shift = s_pi - s_max
        exp_s = torch.exp(s_shift)

        # Compute Z_i = sum_{j>=i} exp(s_{pi_j})
        # Use reverse cumsum trick: reverse, cumsum, reverse
        exp_s_rev = exp_s.flip(0)
        Z_rev = torch.cumsum(exp_s_rev, dim=0)
        Z = Z_rev.flip(0) + self.eps

        # Compute log-likelihood: -sum_i log(exp(s_{pi_i}) / Z_i)
        # This is equivalent to: sum_i (log(Z_i) - s_{pi_i})
        log_Z = torch.log(Z)
        log_likelihood = (log_Z - s_pi).sum()

        # ListMLE loss is negative log-likelihood
        loss = log_likelihood / n_samples

        return loss


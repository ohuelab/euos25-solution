"""Seed management for reproducibility."""

import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
        deterministic: If True, ensures deterministic behavior (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: torch if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """Get a numpy RandomState with optional seed.

    Args:
        seed: Random seed value (optional)

    Returns:
        numpy RandomState instance
    """
    return np.random.RandomState(seed)

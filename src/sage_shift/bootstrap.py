"""Paired bootstrap: every condition is scored on the same resamples, given as image weights."""

from __future__ import annotations

import numpy as np


def group_weights(groups: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Resample groups (patients) with replacement; returns image weights of shape (n_resamples, n_images).

    An image's weight is the number of times its patient was drawn. With one group per image this
    is an ordinary image-level bootstrap.
    """
    _, index = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(index.max()) + 1
    draws = np.random.default_rng(seed).integers(0, n_groups, size=(n_resamples, n_groups))
    counts = np.stack([np.bincount(row, minlength=n_groups) for row in draws])
    return counts[:, index.reshape(-1)].astype(float)


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, ignoring nan resamples."""
    tail = 50 * (1 - level)
    low, high = np.nanpercentile(np.asarray(values, dtype=float), [tail, 100 - tail])
    return float(low), float(high)

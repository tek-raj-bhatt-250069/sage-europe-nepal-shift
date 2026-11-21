"""Synthetic endoscopy-like frames for the tests (no dataset image is ever used)."""

from __future__ import annotations

import numpy as np

HEIGHT, WIDTH = 576, 768


def octagon(left: int, top: int, right: int, bottom: int, cut: int) -> np.ndarray:
    """Boolean octagon filling a box, with corners cut by `cut` pixels."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    inside = (x >= left) & (x < right) & (y >= top) & (y < bottom)
    corners = ((x - left) + (y - top) < cut) | ((right - 1 - x) + (y - top) < cut) | \
              ((x - left) + (bottom - 1 - y) < cut) | ((right - 1 - x) + (bottom - 1 - y) < cut)
    return inside & ~corners


def frame(field: np.ndarray, seed: int = 0) -> np.ndarray:
    """A synthetic frame: textured tissue inside `field`, near-black border with JPEG-like noise."""
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 12, size=(HEIGHT, WIDTH, 3)).astype(np.uint8)
    tissue = np.stack([rng.integers(120, 220, size=(HEIGHT, WIDTH)),
                       rng.integers(50, 120, size=(HEIGHT, WIDTH)),
                       rng.integers(40, 100, size=(HEIGHT, WIDTH))], axis=-1).astype(np.uint8)
    out[field] = tissue[field]
    return out

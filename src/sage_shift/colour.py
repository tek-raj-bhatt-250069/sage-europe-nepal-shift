"""Reinhard colour normalisation (Reinhard et al., 2001) in CIELAB, inside the field only.

Each channel's mean and SD inside the field mask are matched to a reference; pixels outside the
field are left as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6 / 29
MIN_PIXELS = 16  # fewer field pixels than this: leave the image unchanged


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELAB (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65
    f = np.where(xyz > _DELTA ** 3, np.cbrt(xyz), xyz / (3 * _DELTA ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB -> sRGB, clipped to [0, 1]."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > _DELTA, f ** 3, 3 * _DELTA ** 2 * (f - 4 / 29)) * _WHITE_D65
    linear = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


@dataclass(frozen=True)
class ColourReference:
    """Target Lab mean and SD."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    images: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std), "images": self.images, "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColourReference:
        mean, std = tuple(float(v) for v in data["mean"]), tuple(float(v) for v in data["std"])
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError(f"invalid colour reference: {data}")
        return cls(mean, std, int(data["images"]), str(data["source"]))  # type: ignore[arg-type]


def masked_lab_stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and SD of the Lab pixels inside `mask`."""
    pixels = lab[mask]
    if len(pixels) < MIN_PIXELS:
        raise ValueError(f"only {len(pixels)} pixels inside the mask")
    return pixels.mean(axis=0), pixels.std(axis=0)


def reference_from_stats(means: np.ndarray, stds: np.ndarray, source: str) -> ColourReference:
    """Average per-image means and SDs into a reference."""
    means, stds = np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3 or means.shape != stds.shape or len(means) == 0:
        raise ValueError("expected two (n_images, 3) arrays of the same shape")
    return ColourReference(tuple(means.mean(axis=0).tolist()), tuple(stds.mean(axis=0).tolist()),  # type: ignore[arg-type]
                           len(means), source)


def transfer(lab_pixels: np.ndarray, reference: ColourReference) -> tuple[np.ndarray, np.ndarray]:
    """Shift and scale Lab pixels to the reference; also returns the per-channel scale."""
    if len(lab_pixels) < MIN_PIXELS:
        raise ValueError(f"only {len(lab_pixels)} pixels to normalise")
    mean, std = lab_pixels.mean(axis=0), lab_pixels.std(axis=0)
    scale = np.asarray(reference.std) / np.maximum(std, 1e-6)
    return (lab_pixels - mean) * scale + np.asarray(reference.mean), scale


def reinhard(rgb: np.ndarray, mask: np.ndarray, reference: ColourReference) -> np.ndarray:
    """Normalise an sRGB image in [0, 1] inside `mask`."""
    out = np.array(rgb, dtype=np.float32)
    if int(mask.sum()) < MIN_PIXELS:
        return out
    pixels, _ = transfer(srgb_to_lab(out[mask]), reference)
    out[mask] = lab_to_srgb(pixels).astype(np.float32)
    return out

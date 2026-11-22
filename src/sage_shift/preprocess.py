"""Model inputs for the four image variants.

    raw         whole frame squashed to size x size (as in the SAGE paper)
    fov         black outside the field, cropped to the field box
    colour      raw + Reinhard colour normalisation inside the field
    fov_colour  fov + Reinhard colour normalisation inside the field

Outputs are float32 arrays in [0, 1]; ImageNet normalisation is applied later.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sage_shift.colour import ColourReference, reinhard
from sage_shift.fov import (FieldChoice, FovLimits, FovParams, Layout, apply_field, choose_field, detect_field,
                            reduce_frame, remove_windows)

VARIANTS: tuple[str, ...] = ("raw", "fov", "colour", "fov_colour")
FIELD_VARIANTS = frozenset({"fov", "colour", "fov_colour"})
COLOUR_VARIANTS = frozenset({"colour", "fov_colour"})


@dataclass(frozen=True)
class PreprocessReference:
    """FOV settings, layouts and colour reference (written by scripts/05_check_fov.py)."""

    params: FovParams
    limits: FovLimits
    size: int
    layouts: Mapping[str, tuple[Layout, ...]] = field(default_factory=dict)  # per dataset
    colour: ColourReference | None = None
    fallback: Mapping[str, bool] = field(default_factory=dict)  # per dataset; missing means on

    def to_dict(self) -> dict[str, Any]:
        return {
            "fov": {"threshold": self.params.threshold, "open_radius_fraction": self.params.open_radius_fraction,
                    "work_size": self.params.work_size},
            "fov_limits": dict(self.limits.__dict__),
            "size": self.size,
            "fallback": dict(self.fallback),
            "colour_reference": None if self.colour is None else self.colour.to_dict(),
            "layouts": {name: [layout.to_dict() for layout in items] for name, items in self.layouts.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessReference:
        colour = data.get("colour_reference")
        return cls(FovParams.from_config(data["fov"]), FovLimits.from_config(data["fov_limits"]), int(data["size"]),
                   {name: tuple(Layout.from_dict(item) for item in items) for name, items in data["layouts"].items()},
                   None if colour is None else ColourReference.from_dict(colour),
                   {name: bool(value) for name, value in data["fallback"].items()})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PreprocessReference:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def check_variants(variants: Iterable[str]) -> tuple[str, ...]:
    """Check variant names and return them in the standard order."""
    chosen = set(variants)
    unknown = chosen - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {VARIANTS}")
    return tuple(name for name in VARIANTS if name in chosen)


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size with bilinear filtering (same as torchvision's Resize on PIL images)."""
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize a boolean mask to size x size."""
    scaled = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(scaled) >= 128


def frame_field(rgb: np.ndarray, reference: PreprocessReference, dataset: str) -> FieldChoice:
    """Detect the field of a frame, check it against the dataset's layouts and take overlay windows out."""
    detected = detect_field(rgb, reference.params)
    choice = choose_field(detected, reference.layouts.get(dataset, ()), reference.limits,
                          fallback=reference.fallback.get(dataset, True))
    return remove_windows(choice, reduce_frame(rgb, detected.factor), reference.limits)


def make_variants(rgb: np.ndarray, choice: FieldChoice | None, size: int,
                  colour: ColourReference | None, variants: Sequence[str]) -> dict[str, np.ndarray]:
    """Build the requested variants of one frame."""
    variants = check_variants(variants)
    if choice is None and FIELD_VARIANTS & set(variants):
        raise ValueError("the fov and colour variants need a field")
    if colour is None and COLOUR_VARIANTS & set(variants):
        raise ValueError("the colour variants need a colour reference")
    out: dict[str, np.ndarray] = {}
    if "raw" in variants or "colour" in variants:
        raw = resize_rgb(rgb, size).astype(np.float32) / 255
        if "raw" in variants:
            out["raw"] = raw
        if "colour" in variants:
            out["colour"] = reinhard(raw, resize_mask(choice.field.full_mask(), size), colour)
    if "fov" in variants or "fov_colour" in variants:
        crop, crop_mask = apply_field(rgb, choice.field)
        fov = resize_rgb(crop, size).astype(np.float32) / 255
        if "fov" in variants:
            out["fov"] = fov
        if "fov_colour" in variants:
            out["fov_colour"] = reinhard(fov, resize_mask(crop_mask, size), colour)
    return {name: out[name] for name in variants}

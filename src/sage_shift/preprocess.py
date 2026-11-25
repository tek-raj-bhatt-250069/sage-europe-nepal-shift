"""Model inputs: the four image variants and the exploratory inputs.

    raw         whole frame squashed to size x size (as in the SAGE paper)
    fov         black outside the field, cropped to the field box
    colour      raw + Reinhard colour normalisation inside the field
    fov_colour  fov + Reinhard colour normalisation inside the field

Exploratory inputs (grey is the ImageNet mean colour, about 0 after ImageNet normalisation):

    masked             black outside the field, whole frame squashed
    masked_grey        grey outside the field, whole frame squashed
    fov_grey           field box crop, grey outside the field
    letterbox          fov crop centred on a black square of side max(box height, box width), then resized
    letterbox_grey     field box crop with grey outside the field, centred on a grey square, then resized
    colour_global      raw + one dataset-level Lab mean/SD transfer inside the field (SAGE only)
    fov_colour_global  fov + the same transfer inside the field (SAGE only)
    sageify            fov crop resized into the box of SAGE's most common field layout, pasted into a
                       black canvas of SAGE's frame size, then squashed (HyperKvasir only)
    sageify_grey       the same with grey outside the field and a grey canvas (HyperKvasir only)

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

from sage_shift.colour import ColourReference, lab_to_srgb, reinhard, srgb_to_lab
from sage_shift.fov import (BBox, FieldChoice, FovLimits, FovParams, FovResult, Layout, apply_field, choose_field,
                            detect_field, reduce_frame, remove_windows)

VARIANTS: tuple[str, ...] = ("raw", "fov", "colour", "fov_colour")
FIELD_VARIANTS = frozenset({"fov", "colour", "fov_colour"})
COLOUR_VARIANTS = frozenset({"colour", "fov_colour"})

# Exploratory inputs, kept apart from the pre-registered variants above.
EXPLORATORY: tuple[str, ...] = ("masked", "masked_grey", "fov_grey", "letterbox", "letterbox_grey", "colour_global",
                                "fov_colour_global", "sageify", "sageify_grey")
INPUTS: tuple[str, ...] = VARIANTS + EXPLORATORY
GEOMETRY: tuple[str, ...] = ("raw", "masked", "masked_grey", "fov", "fov_grey", "letterbox", "letterbox_grey",
                             "sageify", "sageify_grey")  # inputs that change only the frame geometry or fill
GLOBAL_COLOUR = frozenset({"colour_global", "fov_colour_global"})
SAGEIFY = frozenset({"sageify", "sageify_grey"})
ONLY_FOR = {"colour_global": "sage", "fov_colour_global": "sage", "sageify": "hk", "sageify_grey": "hk"}
NEEDS_FIELD = FIELD_VARIANTS | frozenset(EXPLORATORY)
NEEDS_COLOUR = COLOUR_VARIANTS | GLOBAL_COLOUR
GREY_RGB = (0.485, 0.456, 0.406)  # ImageNet mean colour
GREY = np.array([round(255 * value) for value in GREY_RGB], dtype=np.uint8)  # (124, 116, 104)
BLACK = np.zeros(3, dtype=np.uint8)


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


@dataclass(frozen=True)
class Placement:
    """Frame size and field box that sageify copies from SAGE."""

    shape: tuple[int, int]  # (height, width)
    bbox: BBox              # left, top, right, bottom


def sage_placement(reference: PreprocessReference) -> Placement:
    """Frame size and box of the SAGE layout with the most frames."""
    layouts = reference.layouts.get("sage", ())
    if not layouts:
        raise ValueError("sageify needs the SAGE layouts in preprocess_reference.json")
    best = max(layouts, key=lambda layout: layout.frames)
    return Placement(best.shape, best.bbox)


def check_variants(variants: Iterable[str]) -> tuple[str, ...]:
    """Check input names and return them in the standard order (pre-registered variants first)."""
    chosen = set(variants)
    unknown = chosen - set(INPUTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {VARIANTS} or the exploratory inputs "
                         f"{EXPLORATORY}")
    return tuple(name for name in INPUTS if name in chosen)


def check_dataset(dataset: str, variants: Iterable[str]) -> None:
    """Refuse inputs that are defined only for the other dataset."""
    wrong = sorted(name for name in set(variants) if ONLY_FOR.get(name, dataset) != dataset)
    if wrong:
        raise ValueError(f"{', '.join(wrong)} cannot be made for {dataset}: "
                         + "; ".join(f"{name} is for {ONLY_FOR[name]} only" for name in wrong))


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


def resize_box(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to width x height with bilinear filtering."""
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))


def refill(crop: np.ndarray, mask: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Set the pixels outside `mask` to one colour."""
    return np.where(mask[..., None], crop, fill)


def pad_square(rgb: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Centre an image on a square canvas of side max(height, width)."""
    height, width = rgb.shape[:2]
    side = max(height, width)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[:] = fill
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top:top + height, left:left + width] = rgb
    return canvas


def place(rgb: np.ndarray, placement: Placement, fill: np.ndarray) -> np.ndarray:
    """Resize an image to the placement box and paste it into a canvas of the placement's frame size."""
    left, top, right, bottom = placement.bbox
    canvas = np.empty((*placement.shape, 3), dtype=np.uint8)
    canvas[:] = fill
    canvas[top:bottom, left:right] = resize_box(rgb, right - left, bottom - top)
    return canvas


def global_transfer(rgb: np.ndarray, mask: np.ndarray, source: ColourReference,
                    target: ColourReference) -> np.ndarray:
    """One fixed Lab transfer inside `mask`: (lab - source mean) * target SD / source SD + target mean."""
    out = np.array(rgb, dtype=np.float32)
    if not mask.any():
        return out
    scale = np.asarray(target.std) / np.asarray(source.std)
    lab = (srgb_to_lab(out[mask]) - np.asarray(source.mean)) * scale + np.asarray(target.mean)
    out[mask] = lab_to_srgb(lab).astype(np.float32)
    return out


def geometry_images(rgb: np.ndarray, field: FovResult | None, size: int, names: Sequence[str],
                    placement: Placement | None = None) -> dict[str, np.ndarray]:
    """uint8 size x size images of the geometry inputs (before ImageNet normalisation)."""
    unknown = set(names) - set(GEOMETRY)
    if unknown:
        raise ValueError(f"not geometry inputs: {sorted(unknown)}")
    if field is None and set(names) - {"raw"}:
        raise ValueError("these inputs need a field")
    if placement is None and SAGEIFY & set(names):
        raise ValueError("sageify needs a placement (see sage_placement)")
    mask = None if field is None else field.full_mask()
    crop, crop_mask = (None, None) if field is None or not set(names) - {"raw", "masked", "masked_grey"} \
        else apply_field(rgb, field)
    out: dict[str, np.ndarray] = {}
    for name in names:
        fill = GREY if name.endswith("_grey") else BLACK
        if name == "raw":
            image = rgb
        elif name in ("masked", "masked_grey"):
            image = refill(rgb, mask, fill)
        elif name in ("fov", "fov_grey"):
            image = refill(crop, crop_mask, fill)
        elif name in ("letterbox", "letterbox_grey"):
            image = pad_square(refill(crop, crop_mask, fill), fill)
        else:
            image = place(refill(crop, crop_mask, fill), placement, fill)
        out[name] = resize_rgb(image, size)
    return out


def frame_field(rgb: np.ndarray, reference: PreprocessReference, dataset: str) -> FieldChoice:
    """Detect the field of a frame, check it against the dataset's layouts and take overlay windows out."""
    detected = detect_field(rgb, reference.params)
    choice = choose_field(detected, reference.layouts.get(dataset, ()), reference.limits,
                          fallback=reference.fallback.get(dataset, True))
    return remove_windows(choice, reduce_frame(rgb, detected.factor), reference.limits)


def make_variants(rgb: np.ndarray, choice: FieldChoice | None, size: int,
                  colour: ColourReference | None, variants: Sequence[str], sage_colour: ColourReference | None = None,
                  placement: Placement | None = None) -> dict[str, np.ndarray]:
    """Build the requested variants of one frame.

    The colour_global inputs also need `sage_colour` (SAGE train Lab statistics) and the sageify
    inputs a `placement`.
    """
    variants = check_variants(variants)
    if choice is None and FIELD_VARIANTS & set(variants):
        raise ValueError("the fov and colour variants need a field")
    if colour is None and COLOUR_VARIANTS & set(variants):
        raise ValueError("the colour variants need a colour reference")
    if choice is None and NEEDS_FIELD & set(variants):
        raise ValueError("the exploratory inputs need a field")
    if (colour is None or sage_colour is None) and GLOBAL_COLOUR & set(variants):
        raise ValueError("the colour_global inputs need the colour reference and the SAGE train statistics")
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
    shapes = [name for name in variants if name in GEOMETRY and name not in out]
    if shapes:
        for name, image in geometry_images(rgb, choice.field, size, shapes, placement).items():
            out[name] = image.astype(np.float32) / 255
    if "colour_global" in variants:
        raw = resize_rgb(rgb, size).astype(np.float32) / 255
        out["colour_global"] = global_transfer(raw, resize_mask(choice.field.full_mask(), size), sage_colour, colour)
    if "fov_colour_global" in variants:
        crop, crop_mask = apply_field(rgb, choice.field)
        fov = resize_rgb(crop, size).astype(np.float32) / 255
        out["fov_colour_global"] = global_transfer(fov, resize_mask(crop_mask, size), sage_colour, colour)
    return {name: out[name] for name in variants}

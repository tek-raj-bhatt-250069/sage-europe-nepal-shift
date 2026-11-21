"""Per-frame helpers for scripts/05_check_fov.py (kept here so worker processes can import them)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sage_shift.colour import srgb_to_lab
from sage_shift.fov import FovParams, FovResult, apply_field, detect_field, reduce_frame
from sage_shift.preprocess import PreprocessReference, frame_field, load_rgb, resize_mask, resize_rgb

GREEN_MARGIN = 25  # G must exceed R and B by this much
GREEN_MIN = 80
PROBLEM_FLAGS = ("small", "no_border", "not_convex", "beyond_layout", "no_layout")

_REFERENCE: PreprocessReference | None = None


def green_mask(rgb: np.ndarray) -> np.ndarray:
    """Green-dominant pixels (like the ScopeGuide thumbnail in HyperKvasir)."""
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    return (g > r + GREEN_MARGIN) & (g > b + GREEN_MARGIN) & (g > GREEN_MIN)


def detect_task(task: tuple[str, FovParams]) -> dict[str, Any]:
    """Pass 1: detect the field (mask returned bit-packed)."""
    path, params = task
    field = detect_field(load_rgb(path), params)
    return {"packed": np.packbits(field.mask), "mask_shape": field.mask.shape, "factor": field.factor,
            "shape": field.shape, "bbox": field.bbox, "area": field.area_fraction, "solidity": field.solidity}


def unpack_detection(item: dict[str, Any]) -> FovResult:
    """FovResult from detect_task output."""
    height, width = item["mask_shape"]
    mask = np.unpackbits(item["packed"], count=height * width).reshape(height, width).astype(bool)
    return FovResult(mask, item["factor"], item["shape"], item["bbox"], item["area"], item["solidity"])


def set_reference(reference: PreprocessReference) -> None:
    """Pool initializer for pass 2."""
    global _REFERENCE
    _REFERENCE = reference


def measure_task(task: tuple[str, str]) -> dict[str, Any]:
    """Pass 2: choose the field and measure the fov crop."""
    path, dataset = task
    if _REFERENCE is None:
        raise RuntimeError("set_reference was not called in this worker")
    rgb = load_rgb(path)
    choice = frame_field(rgb, _REFERENCE, dataset)
    crop, crop_mask = apply_field(rgb, choice.field)
    fov = resize_rgb(crop, _REFERENCE.size).astype(np.float64) / 255
    mask = resize_mask(crop_mask, _REFERENCE.size)
    lab = srgb_to_lab(fov[mask])
    green = green_mask(reduce_frame(rgb, choice.field.factor))
    return {
        "height": rgb.shape[0], "width": rgb.shape[1],
        "layout": choice.layout, "agreement": choice.agreement, "inside": choice.inside,
        "flags": "|".join(choice.flags), "field_area": choice.field.area_fraction,
        "box_left": choice.field.bbox[0], "box_top": choice.field.bbox[1],
        "box_right": choice.field.bbox[2], "box_bottom": choice.field.bbox[3],
        "L_mean": lab[:, 0].mean(), "a_mean": lab[:, 1].mean(), "b_mean": lab[:, 2].mean(),
        "L_sd": lab[:, 0].std(), "a_sd": lab[:, 1].std(), "b_sd": lab[:, 2].std(),
        "green_total": int(green.sum()), "green_inside": int((green & choice.field.mask).sum()),
    }


def has_problem(flags: str) -> bool:
    """True if the flags need a visual check."""
    return any(flag in PROBLEM_FLAGS for flag in flags.split("|") if flag)

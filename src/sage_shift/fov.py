"""Field-of-view (FOV) detection for endoscopy frames.

The endoscopic image sits on a dark border that can hold text, redaction boxes or a ScopeGuide
thumbnail. We threshold the brightest channel, clean up with an opening, keep the largest region
and fill its holes. Frames where this fails (too small or badly notched) fall back to the field of
their layout, i.e. the typical field of frames with the same size and position. Large green
overlay windows (the ScopeGuide thumbnail) that touch the picture are taken out of the field.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

BBox = tuple[int, int, int, int]  # left, top, right, bottom (full size, right/bottom exclusive)
GREEN_MARGIN = 25  # G must exceed R and B by this much
GREEN_MIN = 80


@dataclass(frozen=True)
class FovParams:
    """Detection settings (values in configs/experiment.json)."""

    threshold: int = 20
    open_radius_fraction: float = 0.01
    work_size: int = 512

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> FovParams:
        return cls(int(cfg["threshold"]), float(cfg["open_radius_fraction"]), int(cfg["work_size"]))


@dataclass(frozen=True)
class FovLimits:
    """Quality limits for detections and layout matching."""

    min_area: float = 0.3           # smaller fields are flagged "small"
    max_area: float = 0.97          # larger means no border was found
    min_solidity: float = 0.9       # less convex fields are flagged "not_convex"
    clean_solidity: float = 0.97    # only these detections are used to build layouts
    layout_tolerance: float = 0.02  # box tolerance for clustering, as a fraction of the frame size
    layout_min_frames: int = 5
    jitter: int = 3                 # reduced pixels
    min_inside: float = 0.97
    window_min_area: float = 0.02   # green windows at least this share of the frame are removed
    window_min_fill: float = 0.5    # ... if they fill at least this share of their box
    window_pad: int = 2             # reduced pixels added around a window

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> FovLimits:
        return cls(float(cfg["min_area"]), float(cfg["max_area"]), float(cfg["min_solidity"]),
                   float(cfg["clean_solidity"]), float(cfg["layout_tolerance"]), int(cfg["layout_min_frames"]),
                   int(cfg["jitter"]), float(cfg["min_inside"]), float(cfg["window_min_area"]),
                   float(cfg["window_min_fill"]), int(cfg["window_pad"]))


@dataclass(frozen=True)
class FovResult:
    """Field mask on the reduced frame, with its full-size box."""

    mask: np.ndarray        # bool mask on the reduced frame
    factor: int             # reduction factor
    shape: tuple[int, int]  # full-size (height, width)
    bbox: BBox
    area_fraction: float
    solidity: float         # area / convex hull area

    def full_mask(self) -> np.ndarray:
        """Mask at full resolution."""
        height, width = self.shape
        full = np.repeat(np.repeat(self.mask, self.factor, axis=0), self.factor, axis=1)
        return full[:height, :width]


@dataclass(frozen=True)
class Layout:
    """Typical field of a group of frames with the same size and box."""

    name: str
    shape: tuple[int, int]
    factor: int
    bbox: BBox
    runs: tuple[tuple[int, int], ...]  # per reduced row: (start, stop) columns
    frames: int

    @cached_property
    def mask(self) -> np.ndarray:
        return runs_mask(self.runs, math.ceil(self.shape[1] / self.factor))

    @cached_property
    def edge(self) -> np.ndarray:
        return boundary(self.mask)

    @cached_property
    def _grown(self) -> dict[int, np.ndarray]:
        return {}

    def grown(self, jitter: int) -> np.ndarray:
        """Layout mask dilated by `jitter` pixels (cached)."""
        if jitter not in self._grown:
            self._grown[jitter] = grow(self.mask, jitter)
        return self._grown[jitter]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "factor": self.factor, "bbox": list(self.bbox),
                "frames": self.frames, "runs": [list(run) for run in self.runs]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Layout:
        return cls(str(data["name"]), (int(data["shape"][0]), int(data["shape"][1])), int(data["factor"]),
                   tuple(int(v) for v in data["bbox"]), tuple((int(a), int(b)) for a, b in data["runs"]),  # type: ignore[arg-type]
                   int(data["frames"]))


@dataclass(frozen=True)
class FieldChoice:
    """Field used for a frame, with its layout and flags."""

    field: FovResult
    layout: str  # "" if no layout holds the detection
    agreement: float
    inside: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class LayoutScore:
    """How well a detection fits a layout."""

    layout: Layout
    agreement: float  # share of the layout boundary close to the detected boundary
    inside: float     # share of the detection inside the (dilated) layout


def reduction_factor(shape: tuple[int, int], work_size: int) -> int:
    """Integer factor that brings the longer side down to `work_size`."""
    return max(1, math.ceil(max(shape) / work_size))


def disk(radius: int) -> np.ndarray:
    """Disk-shaped structuring element."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def reduce_frame(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Downscale by an integer factor (box average)."""
    return np.asarray(Image.fromarray(rgb).reduce(factor)) if factor > 1 else rgb


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a mask by `pixels` steps."""
    if pixels <= 0:
        return mask
    return ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=pixels)


def boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels on the edge of a mask."""
    return mask & ~ndi.binary_erosion(mask)


def solidity(mask: np.ndarray) -> float:
    """Mask area over convex hull area (1.0 for a convex mask)."""
    if not mask.any():
        return 0.0
    points = np.argwhere(boundary(mask))
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):  # degenerate hull
        return 1.0
    inside = hull.volume + hull.area / 2 + 1  # pixels inside the hull (Pick's theorem)
    return float(min(1.0, mask.sum() / inside))


def mask_bbox(mask: np.ndarray, factor: int, shape: tuple[int, int]) -> BBox:
    """Full-size bounding box of a reduced mask."""
    if not mask.any():
        return (0, 0, 0, 0)
    height, width = shape
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]) * factor, int(rows[0]) * factor,
            min(width, (int(cols[-1]) + 1) * factor), min(height, (int(rows[-1]) + 1) * factor))


def make_result(mask: np.ndarray, factor: int, shape: tuple[int, int], bbox: BBox | None = None,
                solidity_value: float | None = None) -> FovResult:
    """Wrap a reduced mask with its box and quality numbers."""
    bbox = mask_bbox(mask, factor, shape) if bbox is None else bbox
    value = solidity(mask) if solidity_value is None else solidity_value
    return FovResult(mask, factor, shape, bbox, float(mask.mean()), value)


def detect_field(rgb: np.ndarray, params: FovParams = FovParams()) -> FovResult:
    """Find the endoscopic field of an RGB uint8 frame."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {rgb.dtype} {rgb.shape}")
    shape = (int(rgb.shape[0]), int(rgb.shape[1]))
    factor = reduction_factor(shape, params.work_size)
    small = reduce_frame(rgb, factor)
    bright = small.max(axis=2) > params.threshold
    radius = max(1, round(params.open_radius_fraction * min(small.shape[:2])))
    bright = ndi.binary_opening(bright, structure=disk(radius))
    labels, count = ndi.label(bright, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return make_result(np.zeros(small.shape[:2], dtype=bool), factor, shape)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return make_result(ndi.binary_fill_holes(labels == int(np.argmax(sizes))), factor, shape)


def is_clean(field: FovResult, limits: FovLimits) -> bool:
    """Clean enough to help define a layout?"""
    return limits.min_area <= field.area_fraction <= limits.max_area and field.solidity >= limits.clean_solidity


def cluster_boxes(boxes: np.ndarray, shape: tuple[int, int], tolerance: float, min_frames: int) -> np.ndarray:
    """Greedy clustering of field boxes (-1 = no cluster).

    Keeps taking the box with the most free neighbours within `tolerance` until the next
    cluster would have fewer than `min_frames` boxes.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    height, width = shape
    unique, inverse, counts = np.unique(np.asarray(boxes), axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    scaled = unique / np.array([width, height, width, height], dtype=float)
    near = np.abs(scaled[:, None, :] - scaled[None, :, :]).max(axis=2) <= tolerance + 1e-12
    free = np.ones(len(unique), dtype=bool)
    cluster_of = np.full(len(unique), -1)
    label = 0
    while free.any():
        weight = (near & free[None, :]).astype(np.int64) @ counts
        weight[~free] = 0
        best = int(np.argmax(weight))
        if weight[best] < min_frames:
            break
        members = near[best] & free
        cluster_of[members] = label
        free[members] = False
        label += 1
    return cluster_of[inverse]


def runs_mask(runs: Sequence[tuple[int, int]], width: int) -> np.ndarray:
    """Mask from per-row (start, stop) spans."""
    out = np.zeros((len(runs), width), dtype=bool)
    for row, (start, stop) in enumerate(runs):
        out[row, start:stop] = True
    return out


def row_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Per-row (start, stop) span of a mask."""
    runs = []
    for row in mask:
        cols = np.flatnonzero(row)
        runs.append((int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, 0))
    return tuple(runs)


def build_layouts(fields: Sequence[FovResult], limits: FovLimits, prefix: str) -> tuple[list[Layout], np.ndarray]:
    """Cluster clean detections into layouts."""
    assignment = np.full(len(fields), -1)
    layouts: list[Layout] = []
    by_shape: dict[tuple[int, int], list[int]] = {}
    for i, field in enumerate(fields):
        if is_clean(field, limits):
            by_shape.setdefault(field.shape, []).append(i)
    for shape in sorted(by_shape, key=lambda s: (-len(by_shape[s]), s)):
        members = np.array(by_shape[shape])
        boxes = np.array([fields[i].bbox for i in members])
        labels = cluster_boxes(boxes, shape, limits.layout_tolerance, limits.layout_min_frames)
        for label in range(labels.max(initial=-1) + 1):
            chosen = members[labels == label]
            vote = np.mean([fields[i].mask for i in chosen], axis=0) >= 0.5
            runs, factor = row_runs(vote), fields[chosen[0]].factor
            bbox = mask_bbox(runs_mask(runs, vote.shape[1]), factor, shape)
            layouts.append(Layout(f"{prefix}{shape[1]}x{shape[0]}-{label + 1}", shape, factor, bbox, runs, len(chosen)))
            assignment[chosen] = len(layouts) - 1
    return layouts, assignment


def score_layouts(detected: FovResult, layouts: Sequence[Layout], jitter: int) -> list[LayoutScore]:
    """Score a detection against each layout of the same frame size."""
    near_edge = grow(boundary(detected.mask), jitter)
    size = int(detected.mask.sum())
    scores = []
    for layout in layouts:
        if layout.shape != detected.shape or layout.factor != detected.factor:
            continue
        inside = int((layout.grown(jitter) & detected.mask).sum()) / size if size else 1.0
        agreement = float((layout.edge & near_edge).sum()) / max(1, int(layout.edge.sum()))
        scores.append(LayoutScore(layout, agreement, inside))
    return scores


def choose_field(detected: FovResult, layouts: Sequence[Layout], limits: FovLimits,
                 fallback: bool = True) -> FieldChoice:
    """Use the frame's own detection, or its layout if the detection failed and `fallback` is on."""
    flags = []
    failed = False
    if detected.area_fraction < limits.min_area:
        flags.append("small")
        failed = True
    if detected.area_fraction > limits.max_area:
        flags.append("no_border")
    if detected.solidity < limits.min_solidity:
        flags.append("not_convex")
        failed = True
    scores = score_layouts(detected, layouts, limits.jitter)
    holding = [score for score in scores if score.inside >= limits.min_inside]
    best = max(holding, key=lambda score: (score.agreement, score.layout.frames), default=None)
    if scores and best is None:
        flags.append("beyond_layout")  # no layout contains this field
    name, agreement, inside = ("", float("nan"), float("nan")) if best is None else (
        best.layout.name, best.agreement, best.inside)
    if failed and best is not None and fallback:
        flags.append("fallback")
        field = make_result(best.layout.mask, best.layout.factor, best.layout.shape, bbox=best.layout.bbox,
                            solidity_value=detected.solidity)
        return FieldChoice(field, name, agreement, inside, tuple(flags))
    if failed:
        flags.append("no_layout" if best is None else "no_fallback")
    if not detected.mask.any():
        raise ValueError(f"no field found and no layout for frame size {detected.shape}")
    return FieldChoice(detected, name, agreement, inside, tuple(flags))


def green_mask(rgb: np.ndarray) -> np.ndarray:
    """Green-dominant pixels (like the ScopeGuide thumbnail in HyperKvasir)."""
    r, g, b = (rgb[..., k].astype(np.int16) for k in range(3))
    return (g > r + GREEN_MARGIN) & (g > b + GREEN_MARGIN) & (g > GREEN_MIN)


def window_mask(small: np.ndarray, limits: FovLimits) -> np.ndarray:
    """Boxes of large, solid green regions on a reduced frame, grown by `window_pad`."""
    green = green_mask(small)
    labels, _ = ndi.label(green, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    out = np.zeros(green.shape, dtype=bool)
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        box_area = (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
        if sizes[index] >= limits.window_min_area * green.size and sizes[index] >= limits.window_min_fill * box_area:
            out[box] = True
    return grow(out, limits.window_pad)


def remove_windows(choice: FieldChoice, small: np.ndarray, limits: FovLimits) -> FieldChoice:
    """Take green overlay windows that touch the picture out of the field (flag "window")."""
    field = choice.field
    windows = window_mask(small, limits) & field.mask
    if not windows.any():
        return choice
    kept = make_result(field.mask & ~windows, field.factor, field.shape, solidity_value=field.solidity)
    return replace(choice, field=kept, flags=(*choice.flags, "window"))


def apply_field(rgb: np.ndarray, field: FovResult) -> tuple[np.ndarray, np.ndarray]:
    """Black out everything outside the field and crop to its box."""
    if (int(rgb.shape[0]), int(rgb.shape[1])) != field.shape:
        raise ValueError(f"frame shape {rgb.shape[:2]} does not match the field shape {field.shape}")
    mask = field.full_mask()
    left, top, right, bottom = field.bbox
    if right <= left or bottom <= top:
        raise ValueError(f"empty field box {field.bbox}")
    out = np.where(mask[..., None], rgb, np.uint8(0))
    return out[top:bottom, left:right], mask[top:bottom, left:right]

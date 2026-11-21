"""Contact sheets for checking the field masks by eye.

Sheets show patient images: keep them in git-ignored local folders only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

OUTLINE_RGB = (0, 255, 255)  # field boundary
BOX_RGB = (255, 255, 0)      # crop box


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Float [0, 1] or uint8 image to uint8."""
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def draw_field(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Draw the mask boundary and an optional box on a copy of the image."""
    out = to_uint8(image).copy()
    edge = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    out[edge] = OUTLINE_RGB
    if box is not None:
        canvas = Image.fromarray(out)
        ImageDraw.Draw(canvas).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline=BOX_RGB)
        out = np.asarray(canvas)
    return out


def compose(cells: Sequence[tuple[str, Sequence[np.ndarray]]], per_row: int, gap: int = 8,
            label_height: int = 14) -> Image.Image:
    """Arrange labelled cells of equal-sized panels in a grid."""
    if not cells:
        raise ValueError("no cells to draw")
    tile_h, tile_w = cells[0][1][0].shape[:2]
    panels = max(len(item[1]) for item in cells)
    cell_w, cell_h = panels * tile_w + (panels - 1) * 2, label_height + tile_h
    rows = -(-len(cells) // per_row)
    sheet = Image.new("RGB", (per_row * cell_w + (per_row + 1) * gap, rows * cell_h + (rows + 1) * gap), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for index, (label, images) in enumerate(cells):
        x0 = gap + (index % per_row) * (cell_w + gap)
        y0 = gap + (index // per_row) * (cell_h + gap)
        draw.text((x0, y0), label, fill=(255, 255, 255))
        for k, image in enumerate(images):
            sheet.paste(Image.fromarray(to_uint8(image)), (x0 + k * (tile_w + 2), y0 + label_height))
    return sheet


def save_sheet(sheet: Image.Image, path: Path) -> None:
    """Save a sheet as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)

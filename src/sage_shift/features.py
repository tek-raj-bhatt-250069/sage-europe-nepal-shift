"""Feature extraction: each frame is decoded once and every variant goes through every backbone."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sage_shift.backbones import IMAGENET_MEAN, IMAGENET_STD, Backbone
from sage_shift.colour import ColourReference
from sage_shift.preprocess import (NEEDS_FIELD, SAGEIFY, PreprocessReference, check_dataset, check_variants,
                                   frame_field, load_rgb, make_variants, sage_placement)


class FrameDataset(Dataset):
    """Returns the variants of a frame as a (V, 3, size, size) normalised tensor."""

    def __init__(self, paths: Sequence[str], dataset: str, variants: Sequence[str],
                 reference: PreprocessReference | None, size: int, sage_colour: ColourReference | None = None) -> None:
        self.paths, self.dataset, self.size = list(paths), dataset, size
        self.variants = check_variants(variants)
        check_dataset(dataset, self.variants)
        self.needs_field = bool(NEEDS_FIELD & set(self.variants))
        if self.needs_field and (reference is None or not reference.layouts.get(dataset)):
            raise ValueError(f"variants {self.variants} need preprocess_reference.json with {dataset} layouts")
        self.reference = reference
        self.sage_colour = sage_colour  # SAGE train Lab statistics for the colour_global inputs
        self.placement = sage_placement(reference) if SAGEIFY & set(self.variants) else None
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rgb = load_rgb(self.paths[index])
        choice = frame_field(rgb, self.reference, self.dataset) if self.needs_field else None
        colour = None if self.reference is None else self.reference.colour
        images = make_variants(rgb, choice, self.size, colour, self.variants, self.sage_colour, self.placement)
        stack = torch.from_numpy(np.stack([images[name] for name in self.variants])).permute(0, 3, 1, 2)
        return (stack - self.mean) / self.std, index


def extract(frames: FrameDataset, backbones: Sequence[Backbone], device: torch.device, batch_size: int,
            workers: int) -> dict[tuple[str, str], np.ndarray]:
    """Features per (backbone, variant), one row per frame in dataset order."""
    loader = DataLoader(frames, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    parts: dict[tuple[str, str], list[np.ndarray]] = {(b.name, v): [] for b in backbones for v in frames.variants}
    seen = 0
    for batch, indices in loader:
        if indices[0].item() != seen:
            raise RuntimeError("the data loader returned frames out of order")
        seen += len(indices)
        batch = batch.to(device, non_blocking=True)
        for k, variant in enumerate(frames.variants):
            for backbone in backbones:
                parts[(backbone.name, variant)].append(backbone(batch[:, k]).float().cpu().numpy())
    return {key: np.concatenate(chunks) for key, chunks in parts.items()}


def save_features(path: Path, features: np.ndarray, ids: Sequence[str], **meta: str) -> None:
    """Save features and ids (write to a temp file, then rename)."""
    if len(features) != len(ids):
        raise ValueError(f"{len(features)} feature rows but {len(ids)} ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, features=features.astype(np.float32), ids=np.asarray(ids, dtype=str),
                 **{key: np.asarray(value) for key, value in meta.items()})
    os.replace(tmp, path)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (features, ids)."""
    with np.load(path, allow_pickle=False) as data:
        return data["features"], data["ids"]

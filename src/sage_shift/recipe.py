"""The SAGE paper's HyperKvasir training recipe: split, transforms, training loop and scoring.

Follows the authors' `hyperkvasir-on-SAGE.py` and `lib/trainer/{transforms,utils}.py`
(commit e24eb0e). The released transforms start with a resize to 224 x 224, so frames are
resized once and cached; the random flip, colour jitter and grayscale then run per batch.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from sage_shift.backbones import IMAGENET_MEAN, IMAGENET_STD
from sage_shift.fov import apply_field
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.metrics import confusion, label_scores
from sage_shift.preprocess import PreprocessReference, frame_field, load_rgb, resize_rgb

K = len(SIX_CLASSES)
_REFERENCE: PreprocessReference | None = None


def seed_everything(seed: int) -> None:
    """Same seeds as the released script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def faithful_split(labels: Sequence[str], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The released random 80/20 split, stratified by class, on rows in image-labels.csv order."""
    train, test = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=seed, stratify=list(labels))
    return np.asarray(train), np.asarray(test)


def train_transform() -> transforms.Compose:
    """Released training transforms, after the resize."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def test_transform() -> transforms.Compose:
    """Released test transforms, after the resize."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class CachedFrames(Dataset):
    """Resized uint8 frames (N x H x W x 3) with integer labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Callable[[Image.Image], torch.Tensor]):
        self.images, self.labels, self.transform = images, np.asarray(labels, dtype=np.int64), transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])


def _resized(task: tuple[str, int]) -> np.ndarray:
    path, size = task
    return resize_rgb(load_rgb(path), size)


def cache_frames(paths: Sequence[str | Path], size: int = 224, workers: int = 2) -> np.ndarray:
    """Decode and resize every frame once (PIL bilinear, as torchvision's Resize on PIL images)."""
    tasks = [(str(path), size) for path in paths]
    if workers <= 1:
        return np.stack([_resized(task) for task in tasks])
    with Pool(workers) as pool:
        return np.stack(pool.map(_resized, tasks, chunksize=16))


def _set_reference(reference: PreprocessReference) -> None:
    global _REFERENCE
    _REFERENCE = reference


def _fov_resized(task: tuple[str, str, int]) -> np.ndarray:
    path, dataset, size = task
    if _REFERENCE is None:
        raise RuntimeError("no preprocessing reference in this worker")
    rgb = load_rgb(path)
    crop, _ = apply_field(rgb, frame_field(rgb, _REFERENCE, dataset).field)
    return resize_rgb(crop, size)


def cache_fov_frames(paths: Sequence[str | Path], dataset: str, reference: PreprocessReference, size: int = 224,
                     workers: int = 2) -> np.ndarray:
    """Field-of-view crops (black outside the field), resized like the fov variant."""
    tasks = [(str(path), dataset, size) for path in paths]
    if workers <= 1:
        _set_reference(reference)
        return np.stack([_fov_resized(task) for task in tasks])
    with Pool(workers, initializer=_set_reference, initargs=(reference,)) as pool:
        return np.stack(pool.map(_fov_resized, tasks, chunksize=16))


def make_model(pretrained: bool = True) -> nn.Module:
    """ResNet-50 (IMAGENET1K_V1) with a new six-way linear head, as released."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, K)
    return model


def seed_worker(worker_id: int) -> None:
    """Give each data-loader worker its own NumPy and random seed (from torch's per-worker seed)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(data: Dataset, batch_size: int, shuffle: bool, workers: int,
                generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=workers > 0)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One epoch of cross-entropy training; returns the mean batch loss (as `train_one_epoch`)."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Softmax probabilities in loader order."""
    model.eval()
    return np.concatenate([torch.softmax(model(images.to(device)), dim=1).cpu().numpy() for images, _ in loader])


def scores(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Weighted F1 (the paper's metric and the checkpoint rule), macro-F1 and accuracy."""
    pred = proba.argmax(axis=1)
    return {"weighted_f1": float(f1_score(y_true, pred, labels=list(range(K)), average="weighted", zero_division=0)),
            "macro_f1": float(label_scores(confusion(y_true, pred, K))["macro_f1"]),
            "accuracy": float(np.mean(pred == y_true))}


def prediction_summary(y_true: np.ndarray, proba: np.ndarray, collapse_share: float) -> dict[str, Any]:
    """Scores, predicted-class counts, the top class's share and the confusion matrix."""
    pred = proba.argmax(axis=1)
    counts = np.bincount(pred, minlength=K)
    top = int(counts.argmax())
    share = float(counts[top] / len(pred))
    return {**{key: round(value, 4) for key, value in scores(y_true, proba).items()},
            "predicted_counts": dict(zip(SIX_CLASSES, counts.tolist())),
            "top_class": SIX_CLASSES[top], "top_class_share": round(share, 4), "collapse": share >= collapse_share,
            "confusion_rows_truth": confusion(y_true, pred, K).astype(int).tolist()}


def save_atomic(obj: Any, path: Path) -> None:
    """torch.save to a temporary file and rename it, so an interrupted save keeps the previous file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "loader": generator.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

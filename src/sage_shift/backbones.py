"""Frozen backbones for feature extraction.

    resnet50_imagenet        torchvision ResNet-50, IMAGENET1K_V1 (as in the SAGE paper)
    resnet50_dino_imagenet   ResNet-50 trained with DINO on ImageNet
    resnet50_gastronet_dino  ResNet-50 trained with DINO on GastroNet-5M (local checkpoint,
                             CC BY-NC-ND 4.0; do not share the file)
    dinov2_vitb14            DINOv2 ViT-B/14 from torch.hub (class token + mean patch token)

ResNets return the 2048-d pooled feature. Loaders record the weight source and SHA-256.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

BACKBONES: tuple[str, ...] = ("resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14")
FEATURE_DIMS = {"resnet50_imagenet": 2048, "resnet50_dino_imagenet": 2048, "resnet50_gastronet_dino": 2048,
                "dinov2_vitb14": 1536}
DINO_RESNET50_URL = "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
DINOV2_REPO = "facebookresearch/dinov2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STRIP_PREFIXES = ("module.", "backbone.", "encoder.", "model.")


@dataclass
class Backbone:
    """Frozen feature extractor plus a record of its weights."""

    name: str
    model: nn.Module
    features: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    dim: int
    record: dict[str, Any] = field(default_factory=dict)

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.features(self.model, batch)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(folder: Path) -> str:
    """SHA-256 over all .py files in a folder (to record the torch.hub code version)."""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hub_checkpoint(url: str) -> Path:
    """torch.hub cache path for a checkpoint URL."""
    return Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]


def resnet_trunk() -> nn.Module:
    """ResNet-50 without its classification layer."""
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    return model


def resnet_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch)


def dinov2_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Class token + mean patch token, as in DINOv2's linear evaluation."""
    out = model.forward_features(batch)
    return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=1)


def clean_state_dict(raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Unwrap a checkpoint (prefer the DINO teacher) and strip key prefixes."""
    state: Any = raw
    for key in ("teacher", "state_dict", "model", "student"):
        if isinstance(state, Mapping) and isinstance(state.get(key), Mapping):
            state = state[key]
            break
    out = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        stripped = True
        while stripped:
            stripped = False
            for prefix in STRIP_PREFIXES:
                if key.startswith(prefix):
                    key, stripped = key[len(prefix):], True
        if key.startswith(("fc.", "head.")):
            continue
        out[key] = value
    return out


def load_trunk_weights(model: nn.Module, state: Mapping[str, torch.Tensor], name: str) -> list[str]:
    """Load weights, failing if any backbone weight is missing. Returns unexpected keys."""
    result = model.load_state_dict(dict(state), strict=False)
    missing = [key for key in result.missing_keys if not key.startswith("fc.")]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} backbone weights are missing from the checkpoint, "
                           f"for example {missing[:3]}")
    return list(result.unexpected_keys)


def load_resnet50_imagenet(random_weights: bool = False) -> Backbone:
    if random_weights:
        return Backbone("resnet50_imagenet", resnet_trunk(), resnet_features, 2048, {"source": "random (smoke test)"})
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    path = hub_checkpoint(weights.url)
    return Backbone("resnet50_imagenet", model, resnet_features, 2048,
                    {"source": weights.url, "sha256": sha256_file(path) if path.is_file() else None})


def load_resnet50_dino(random_weights: bool = False) -> Backbone:
    model = resnet_trunk()
    if random_weights:
        return Backbone("resnet50_dino_imagenet", model, resnet_features, 2048, {"source": "random (smoke test)"})
    state = torch.hub.load_state_dict_from_url(DINO_RESNET50_URL, map_location="cpu", weights_only=True)
    unexpected = load_trunk_weights(model, clean_state_dict(state), "DINO ResNet-50")
    return Backbone("resnet50_dino_imagenet", model, resnet_features, 2048,
                    {"source": DINO_RESNET50_URL, "sha256": sha256_file(hub_checkpoint(DINO_RESNET50_URL)),
                     "unexpected_keys": len(unexpected)})


def load_resnet50_gastronet(path: Path, allow_pickle: bool = False) -> Backbone:
    """GastroNet-5M DINOv1 ResNet-50 from a local checkpoint."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # full pickle, not just tensors
        if not allow_pickle:
            raise RuntimeError(f"{path.name} is not a plain weights file. If you trust it (it came from the "
                               "Theta Vision portal), re-run with --gastronet-allow-pickle") from error
        raw = torch.load(path, map_location="cpu", weights_only=False)
    model = resnet_trunk()
    unexpected = load_trunk_weights(model, clean_state_dict(raw), "GastroNet ResNet-50")
    return Backbone("resnet50_gastronet_dino", model, resnet_features, 2048,
                    {"source": f"local file {path.name} (Theta Vision Cortex portal)", "sha256": sha256_file(path),
                     "unexpected_keys": len(unexpected)})


def load_dinov2(random_weights: bool = False) -> Backbone:
    model = torch.hub.load(DINOV2_REPO, "dinov2_vitb14", pretrained=not random_weights, trust_repo=True)
    code = Path(torch.hub.get_dir()) / (DINOV2_REPO.replace("/", "_") + "_main")
    record: dict[str, Any] = {"source": f"torch.hub {DINOV2_REPO} dinov2_vitb14 (main branch)",
                              "hub_code_sha256": tree_sha256(code) if code.is_dir() else None}
    if random_weights:
        record["source"] += " (random weights, smoke test)"
    else:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_pretrain.pth"
        record["sha256"] = sha256_file(path) if path.is_file() else None
    return Backbone("dinov2_vitb14", model, dinov2_features, 1536, record)


def load_backbone(name: str, device: torch.device, random_weights: bool = False,
                  gastronet_path: Path | None = None, allow_pickle: bool = False) -> Backbone:
    """Load a backbone in eval mode with frozen weights."""
    if name == "resnet50_imagenet":
        backbone = load_resnet50_imagenet(random_weights)
    elif name == "resnet50_dino_imagenet":
        backbone = load_resnet50_dino(random_weights)
    elif name == "resnet50_gastronet_dino":
        if gastronet_path is None:
            raise ValueError("resnet50_gastronet_dino needs --gastronet-weights")
        backbone = load_resnet50_gastronet(gastronet_path, allow_pickle)
    elif name == "dinov2_vitb14":
        backbone = load_dinov2(random_weights)
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {BACKBONES}")
    backbone.model.eval().requires_grad_(False).to(device)
    return backbone

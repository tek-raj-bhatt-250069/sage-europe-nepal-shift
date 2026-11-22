"""Run records: repository commit and software environment."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision


def git_state(root: Path) -> dict[str, Any]:
    """Commit and cleanliness of the repository at `root`."""
    def run(*command: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True, check=True)
            return done.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": None if status is None else bool(status)}


def environment(device: torch.device) -> dict[str, Any]:
    """Library versions and the device used."""
    info: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__,
                            "torchvision": torchvision.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
                            "device": str(device)}
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info

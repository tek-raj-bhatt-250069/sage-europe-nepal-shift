"""Image lists (dataset, id, split, class, path) for the image scripts.

SAGE paths contain patient and visit UUIDs, so the `path` column is only used to open files and
is never printed or saved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sage_shift.data import SagePaths, load_sample_table
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import build_hk_manifest

FRAME_COLUMNS = ["dataset", "id", "split", "class", "in_six", "path"]


def sage_frames(data_dir: Path) -> pd.DataFrame:
    """All SAGE images, sorted by split and sample_uuid."""
    paths = SagePaths(data_dir)
    table = load_sample_table(paths)
    classes = table["class"].fillna("")
    frames = pd.DataFrame({
        "dataset": "sage",
        "id": table["sample_uuid"],
        "split": table["split"],
        "class": classes,
        "in_six": classes.isin(SIX_CLASSES),
        "path": [str(paths.image_path(p, v, s)) for p, v, s in
                 zip(table["patient_uuid"], table["visit_uuid"], table["sample_uuid"])],
    })
    return frames.sort_values(["split", "id"]).reset_index(drop=True)[FRAME_COLUMNS]


def hk_frames(hk_root: Path, split2: Path) -> pd.DataFrame:
    """Six-class HyperKvasir images in split-file order (split = official fold)."""
    labels_csv = hk_root / "image-labels.csv"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"{labels_csv} not found: extract the HyperKvasir labelled images first")
    manifest = build_hk_manifest(split2, None, labels_csv, hk_root)
    mapped = manifest[manifest["sage_class"] != ""].reset_index(drop=True)
    missing = int((mapped["exists"] == 0).sum())
    if missing:
        raise FileNotFoundError(f"{missing} six-class HyperKvasir images are missing below {hk_root}")
    return pd.DataFrame({
        "dataset": "hk",
        "id": mapped["file_name"],
        "split": mapped["fold2"].map(lambda fold: f"fold{fold}"),
        "class": mapped["sage_class"],
        "in_six": True,
        "path": [str(hk_root / rel) for rel in mapped["rel_path"]],
    })[FRAME_COLUMNS]

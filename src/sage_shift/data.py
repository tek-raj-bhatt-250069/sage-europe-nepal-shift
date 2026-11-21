"""Read-only access to the SAGE dataset files (nothing here writes to data/)."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# The 18 multi-label names and the 14 multi-class names, as listed in data/README.md.
SAGE_LABELS: tuple[str, ...] = (
    "blood", "colon", "diverticulum", "duodenum", "esophagitis", "esophagus",
    "foreign-body", "fundus", "gej", "growth", "inflammation", "instrument",
    "polyp", "pylorus", "rectum", "stomach", "ulcer", "z-line",
)
SAGE_CLASSES: tuple[str, ...] = (
    "colon-normal", "diverticulum", "duodenum", "esophagitis", "foreign-body",
    "fundus-retroflexion", "growth", "inflammation", "instrument", "polyp",
    "pylorus-normal", "rectum-retroflexion", "ulcer", "z-line-normal",
)
SPLITS: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True)
class SagePaths:
    """Locations of the SAGE files below a data root directory."""

    root: Path

    @property
    def files_dir(self) -> Path:
        return self.root / "files"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    def image_path(self, patient_uuid: str, visit_uuid: str, sample_uuid: str) -> Path:
        """Expected image location: images/<patient>/<visit>/<sample>/<sample>.jpg."""
        return self.images_dir / patient_uuid / visit_uuid / sample_uuid / f"{sample_uuid}.jpg"


def read_csv_raw(path: Path) -> pd.DataFrame:
    """Read a CSV with every cell as text; only empty cells count as missing."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def read_json(path: Path) -> Any:
    """Load a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_label_list(raw: str) -> list[str]:
    """Parse one `labels` cell, a stringified Python list such as "['polyp', 'colon']"."""
    value = ast.literal_eval(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"expected a list of strings, got {raw!r}")
    return value


def label_indicators(label_lists: pd.Series, labels: Sequence[str]) -> pd.DataFrame:
    """Return one 0/1 column per label, aligned with the index of `label_lists`."""
    columns = {label: [int(label in items) for items in label_lists] for label in labels}
    return pd.DataFrame(columns, index=label_lists.index, dtype="int8")


def load_sample_table(paths: SagePaths) -> pd.DataFrame:
    """samples.csv joined one-to-one with classification.csv, plus a parsed `label_list`."""
    samples = read_csv_raw(paths.files_dir / "samples.csv")
    classification = read_csv_raw(paths.files_dir / "classification.csv")
    table = samples.merge(classification, on="sample_uuid", validate="one_to_one")
    table["label_list"] = table["labels"].map(parse_label_list)
    return table

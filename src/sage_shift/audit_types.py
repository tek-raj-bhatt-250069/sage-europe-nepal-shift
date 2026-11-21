"""Shared containers and small helpers for the SAGE audit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sage_shift.data import SPLITS, SagePaths, read_csv_raw, read_json

ID_COLUMNS = ["patient_uuid", "visit_uuid", "sample_uuid"]
SPLIT_KEYS = (*SPLITS, "total")
BLANK = "(blank)"


@dataclass(frozen=True)
class Check:
    """One PASS/FAIL check. A failed `hard` check makes the audit script exit non-zero."""

    section: str
    name: str
    passed: bool
    detail: str
    hard: bool = False

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class Audit:
    """Collects key numbers, tables, checks and discrepancies while the audit runs."""

    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    examples: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)

    def check(self, section: str, name: str, passed: bool, detail: str, *, hard: bool = False) -> None:
        """Record one check result."""
        self.checks.append(Check(section, name, bool(passed), detail, hard))

    @property
    def hard_failures(self) -> list[Check]:
        return [check for check in self.checks if check.hard and not check.passed]


@dataclass(frozen=True)
class SageData:
    """The five SAGE annotation files: CSVs as raw text tables, JSON files as loaded objects."""

    paths: SagePaths
    samples: pd.DataFrame
    classification: pd.DataFrame
    metadata: pd.DataFrame
    captions: dict[str, dict[str, Any]]
    vqa: dict[str, list[dict[str, Any]]]


def load_sage(paths: SagePaths) -> SageData:
    """Load every annotation file under data/files."""
    files = paths.files_dir
    return SageData(
        paths=paths,
        samples=read_csv_raw(files / "samples.csv"),
        classification=read_csv_raw(files / "classification.csv"),
        metadata=read_csv_raw(files / "metadata.csv"),
        captions=read_json(files / "captions.json"),
        vqa=read_json(files / "vqa.json"),
    )


def pct(part: float, whole: float) -> float:
    """Percentage rounded to one decimal place; 0.0 when the denominator is zero."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def by_split(frame: pd.DataFrame, func: Callable[[pd.DataFrame], Any]) -> dict[str, Any]:
    """Apply `func` to each split's rows and to all rows; keys are train, test, total."""
    result = {split: func(frame[frame["split"] == split]) for split in SPLITS}
    result["total"] = func(frame)
    return result


def count_by_split(values: pd.Series, split: pd.Series) -> pd.DataFrame:
    """Count each distinct value per split; columns are train, test, total."""
    table = pd.crosstab(values, split).reindex(columns=list(SPLITS), fill_value=0)
    table.columns.name = None
    table["total"] = table.sum(axis=1)
    return table

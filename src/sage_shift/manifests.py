"""Per-image manifests for SAGE and HyperKvasir.

Manifests are per-image tables. They are written only to git-ignored folders (local `outputs/` or a
private Google Drive folder on Colab) and are never committed. The SAGE manifest carries no patient
or visit UUIDs: each patient is replaced by an integer `patient_group` for grouping and resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sage_shift.data import SagePaths, load_sample_table, read_json
from sage_shift.labelspace import BASE_LABEL, HK_FINDING_TO_SAGE, MEANING_GROUP, SIX_CLASSES, hk_finding
from sage_shift.strata import VISIBILITY_PREFIX, location_context, nbi_flag, visibility_bin, vqa_answer

HK_SPLIT_COLUMNS = ["file-name", "class-name", "split-index"]
HK_LABEL_COLUMNS = {"Video file", "Finding", "Organ", "Classification"}
HK_ORGAN_DIRS = {"Lower GI": "lower-gi-tract", "Upper GI": "upper-gi-tract"}


def label_column(base_label: str) -> str:
    """Manifest column holding the 0/1 indicator of one SAGE base label, e.g. y_z_line."""
    return "y_" + base_label.replace("-", "_")


LABEL_COLUMNS: dict[str, str] = {cls: label_column(BASE_LABEL[cls]) for cls in SIX_CLASSES}


def counts(series: pd.Series) -> dict[str, int]:
    """Value counts as a JSON-ready dict with string keys in sorted order."""
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


# --- SAGE -----------------------------------------------------------------------------------------

def build_sage_manifest(paths: SagePaths) -> pd.DataFrame:
    """One row per SAGE image with split, class, six base-label indicators and stratification flags."""
    table = load_sample_table(paths)
    vqa = read_json(paths.files_dir / "vqa.json")
    captions = read_json(paths.files_dir / "captions.json")
    groups = {uuid: index for index, uuid in enumerate(sorted(table["patient_uuid"].unique()))}
    classes = table["class"].fillna("")
    manifest = pd.DataFrame({
        "sample_uuid": table["sample_uuid"],
        "patient_group": table["patient_uuid"].map(groups).astype(int),
        "split": table["split"],
        "class": classes,
        "in_six": classes.isin(SIX_CLASSES).astype(int),
        "base_label": classes.map(BASE_LABEL).fillna(""),
        "meaning_group": classes.map(MEANING_GROUP).fillna(""),
        "n_labels": table["label_list"].map(len),
    })
    for cls, column in LABEL_COLUMNS.items():
        manifest[column] = table["label_list"].map(lambda labels, base=BASE_LABEL[cls]: int(base in labels))
    manifest["location_context"] = table["label_list"].map(location_context)
    manifest["visibility_bin"] = [visibility_bin(vqa_answer(vqa.get(s, []), VISIBILITY_PREFIX)) for s in table["sample_uuid"]]
    manifest["nbi"] = [int(nbi_flag(captions.get(s, {}).get("corrected_description", ""))) for s in table["sample_uuid"]]
    return manifest.sort_values(["split", "sample_uuid"]).reset_index(drop=True)


def sage_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    """Aggregate counts of a SAGE manifest (no identifiers)."""
    six = manifest[manifest["in_six"] == 1]
    summary: dict[str, Any] = {
        "images": len(manifest),
        "images_by_split": counts(manifest["split"]),
        "zero_label_images_by_split": counts(manifest.loc[manifest["n_labels"] == 0, "split"]),
        "six_class": {},
    }
    for split, part in six.groupby("split"):
        summary["six_class"][str(split)] = {
            "images": len(part),
            "patients": int(part["patient_group"].nunique()),
            "images_by_class": counts(part["class"]),
            "patients_by_class": {str(k): int(v) for k, v in part.groupby("class")["patient_group"].nunique().items()},
            "visibility_bins": counts(part["visibility_bin"]),
            "nbi_images": int(part["nbi"].sum()),
            "polyp_location_context": counts(part.loc[part["class"] == "polyp", "location_context"]),
        }
    return summary


# --- HyperKvasir ----------------------------------------------------------------------------------

def read_hk_split(path: Path) -> pd.DataFrame:
    """Read an official HyperKvasir split file (semicolon-separated: file-name;class-name;split-index)."""
    frame = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    if list(frame.columns) != HK_SPLIT_COLUMNS:
        raise ValueError(f"{path.name}: expected columns {HK_SPLIT_COLUMNS}, found {list(frame.columns)}")
    if frame["file-name"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate file names")
    return frame


def _attach_labels_csv(manifest: pd.DataFrame, labels_csv: Path) -> pd.DataFrame:
    """Join image-labels.csv (the labelled-images download) and derive each image's relative path."""
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    missing = HK_LABEL_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"{labels_csv.name}: missing columns {sorted(missing)}; found {list(labels.columns)}")
    unknown_organs = set(labels["Organ"]) - set(HK_ORGAN_DIRS)
    if unknown_organs:
        raise ValueError(f"{labels_csv.name}: unexpected Organ values {sorted(unknown_organs)}")
    names = labels["Video file"].map(lambda name: name if name.lower().endswith(".jpg") else f"{name}.jpg")
    labels = labels.assign(file_name=names)[["file_name", "Finding", "Organ", "Classification"]]
    merged = manifest.merge(labels, on="file_name", how="left", validate="one_to_one")
    absent = int(merged["Finding"].isna().sum())
    if absent:
        raise ValueError(f"{absent} images in the split file are missing from {labels_csv.name}")
    disagree = int((merged["Finding"].map(hk_finding) != merged["hk_class"]).sum())
    if disagree:
        raise ValueError(f"{disagree} images have different classes in the split file and {labels_csv.name}")
    merged["rel_path"] = [
        f"{HK_ORGAN_DIRS[organ]}/{group}/{finding}/{name}"
        for organ, group, finding, name in zip(merged["Organ"], merged["Classification"], merged["Finding"], merged["file_name"])
    ]
    return merged.drop(columns=["Finding", "Organ", "Classification"])


def build_hk_manifest(
    split2: Path,
    split5: Path | None = None,
    labels_csv: Path | None = None,
    images_root: Path | None = None,
) -> pd.DataFrame:
    """One row per HyperKvasir labelled image: class, SAGE class (or blank), official folds, path."""
    two = read_hk_split(split2)
    manifest = pd.DataFrame({
        "file_name": two["file-name"],
        "hk_class": two["class-name"].map(hk_finding),
        "fold2": two["split-index"].astype(int),
    })
    manifest["sage_class"] = manifest["hk_class"].map(HK_FINDING_TO_SAGE).fillna("")
    if split5 is not None:
        five = read_hk_split(split5).rename(columns={"file-name": "file_name", "class-name": "class5", "split-index": "fold5"})
        merged = manifest.merge(five, on="file_name", how="left", validate="one_to_one")
        if merged["fold5"].isna().any():
            raise ValueError(f"{split5.name}: some images of {split2.name} are missing")
        if (merged["class5"].map(hk_finding) != merged["hk_class"]).any():
            raise ValueError(f"{split2.name} and {split5.name} disagree on some classes")
        manifest = merged.drop(columns="class5").assign(fold5=lambda frame: frame["fold5"].astype(int))
    if labels_csv is not None:
        manifest = _attach_labels_csv(manifest, labels_csv)
        if images_root is not None:
            manifest["exists"] = [int((images_root / rel).is_file()) for rel in manifest["rel_path"]]
    return manifest


def hk_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    """Aggregate counts of a HyperKvasir manifest."""
    mapped = manifest[manifest["sage_class"] != ""]
    summary: dict[str, Any] = {
        "images": len(manifest),
        "classes": int(manifest["hk_class"].nunique()),
        "mapped_images": len(mapped),
        "mapped_images_by_fold2": counts(mapped["fold2"]),
        "mapped_by_sage_class_and_fold2": {str(cls): counts(part["fold2"]) for cls, part in mapped.groupby("sage_class")},
        "paths_added": "rel_path" in manifest.columns,
    }
    if "exists" in manifest.columns:
        summary["mapped_images_missing_on_disk"] = int((mapped["exists"] == 0).sum())
    return summary

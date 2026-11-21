"""SAGE audit sections 2 and 3: image integrity, then split sizes and train/test leakage."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from sage_shift.audit_types import ID_COLUMNS, Audit, SageData, by_split, pct
from sage_shift.data import SPLITS

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})


def scan_image_tree(images_dir: Path) -> tuple[pd.DataFrame, Counter[str]]:
    """List image files with the IDs implied by their path, and count other file names.

    A file is `well_formed` when it sits at <patient>/<visit>/<sample>/<sample>.jpg.
    """
    records: list[dict[str, Any]] = []
    other_files: Counter[str] = Counter()
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            other_files[path.name] += 1
            continue
        parts = path.relative_to(images_dir).parts
        at_depth = len(parts) == 4
        records.append(
            {
                "path": path,
                "rel_path": "/".join(parts),
                "patient_uuid": parts[0] if at_depth else None,
                "visit_uuid": parts[1] if at_depth else None,
                "sample_uuid": parts[2] if at_depth else None,
                "well_formed": at_depth and path.name == f"{parts[2]}.jpg",
            }
        )
    columns = ["path", "rel_path", *ID_COLUMNS, "well_formed"]
    return pd.DataFrame(records, columns=columns), other_files


def inspect_image(path: Path) -> dict[str, Any]:
    """Size, mode, format and hashes of one image; `error` is set if it cannot be decoded."""
    raw = path.read_bytes()
    info: dict[str, Any] = {
        "file_bytes": len(raw),
        "md5": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "pixel_md5": None,
        "error": None,
    }
    try:
        with Image.open(io.BytesIO(raw)) as image:
            info.update(width=image.width, height=image.height, mode=image.mode, format=image.format)
            image.verify()
        # verify() does not decode pixel data, so decode fully to catch truncated files.
        with Image.open(io.BytesIO(raw)) as image:
            info["pixel_md5"] = hashlib.md5(image.tobytes(), usedforsecurity=False).hexdigest()
    except Exception as error:  # any decode failure is a finding to report, not a crash
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def _duplicate_groups(inventory: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Images sharing a hash, and how many such groups cross patients, visits or splits."""
    duplicated = inventory[inventory[column].notna() & inventory.duplicated(column, keep=False)]
    groups = duplicated.groupby(column)
    stats = {"n_files": len(duplicated), "n_groups": int(groups.ngroups)}
    for name, id_column in (("patient", "patient_uuid"), ("visit", "visit_uuid"), ("split", "split")):
        stats[f"cross_{name}"] = int((groups[id_column].nunique() > 1).sum())
    return duplicated, stats


def audit_integrity(data: SageData, audit: Audit, expected: dict[str, Any]) -> None:
    """Section 2: sample/image correspondence, decodability, image properties and duplicates."""
    section = "2 Integrity"
    found, other_files = scan_image_tree(data.paths.images_dir)
    if found.empty:
        raise FileNotFoundError(f"no image files found under {data.paths.images_dir}")

    samples = data.samples
    sample_keys = set(samples[ID_COLUMNS].itertuples(index=False, name=None))
    found_keys = list(found[ID_COLUMNS].itertuples(index=False, name=None))
    is_match = found["well_formed"] & pd.Series([key in sample_keys for key in found_keys], index=found.index)
    matched = {key for key, ok in zip(found_keys, is_match) if ok}
    missing = sorted(key[2] for key in sample_keys - matched)
    stray = sorted(found.loc[~is_match, "rel_path"])
    audit.tables["per_image_orphans"] = pd.DataFrame(
        [{"kind": "sample_without_image", "identifier": sample_uuid} for sample_uuid in missing]
        + [{"kind": "image_without_sample", "identifier": rel_path} for rel_path in stray],
        columns=["kind", "identifier"],
    )
    audit.check(
        section,
        "every labelled sample has exactly one image file",
        not missing and int(is_match.sum()) == len(samples),
        f"{len(missing)} samples without an image at `<patient>/<visit>/<sample>/<sample>.jpg`",
    )
    audit.check(
        section,
        "every image file belongs to a labelled sample",
        not stray,
        f"{len(stray)} image files without a matching sample row",
    )

    details = pd.DataFrame([inspect_image(path) for path in found["path"]], index=found.index)
    inventory = pd.concat([found.drop(columns="path"), details], axis=1)
    inventory = inventory.merge(samples[["sample_uuid", "split"]], on="sample_uuid", how="left")
    audit.tables["per_image_inventory"] = inventory

    unreadable = int(inventory["error"].notna().sum())
    audit.check(
        section,
        "every image opens (PIL verify and full decode)",
        unreadable == 0,
        f"{unreadable} of {len(inventory)} files failed",
    )

    readable = inventory[inventory["error"].isna()]
    resolution = readable["width"].astype(int).astype(str) + "x" + readable["height"].astype(int).astype(str)
    properties = pd.crosstab([resolution.rename("resolution"), readable["mode"], readable["format"]], readable["split"])
    properties = properties.reindex(columns=list(SPLITS), fill_value=0)
    properties.columns.name = None
    properties["total"] = properties.sum(axis=1)
    audit.tables["image_properties"] = properties.sort_values("total", ascending=False).reset_index()

    size_kb = readable["file_bytes"] / 1024
    file_size = {
        "min_kb": round(float(size_kb.min()), 1),
        "p25_kb": round(float(size_kb.quantile(0.25)), 1),
        "median_kb": round(float(size_kb.median()), 1),
        "mean_kb": round(float(size_kb.mean()), 1),
        "p75_kb": round(float(size_kb.quantile(0.75)), 1),
        "max_kb": round(float(size_kb.max()), 1),
    }
    audit.tables["image_file_size"] = pd.DataFrame([file_size])

    width, height = expected["image_width"], expected["image_height"]
    off_size = readable[(readable["width"] != width) | (readable["height"] != height)]
    off_size_info = {
        "n_images": len(off_size),
        "by_split": {split: int((off_size["split"] == split).sum()) for split in SPLITS},
        "n_patients": int(off_size["patient_uuid"].nunique()),
        "n_visits": int(off_size["visit_uuid"].nunique()),
        "resolutions": sorted(set(resolution[off_size.index])),
    }
    audit.check(
        section,
        f"all images are {width}x{height}",
        off_size.empty,
        f"{len(off_size)} images have another size: {', '.join(off_size_info['resolutions']) or 'none'}",
    )
    if not off_size.empty:
        audit.discrepancies.append(
            f"Image size: {len(off_size)} of {len(readable)} images are {', '.join(off_size_info['resolutions'])} "
            f"instead of {width}x{height} ({off_size_info['by_split']['train']} train, "
            f"{off_size_info['by_split']['test']} test; {off_size_info['n_patients']} patients, "
            f"{off_size_info['n_visits']} visits)."
        )
    formats = sorted(readable["format"].unique())
    modes = sorted(readable["mode"].unique())
    audit.check(
        section,
        f"all images are {expected['image_format']} with one colour mode",
        formats == [expected["image_format"]] and len(modes) == 1,
        f"formats: {', '.join(formats)}; modes: {', '.join(modes)}",
    )

    duplicates, md5_stats = _duplicate_groups(inventory, "md5")
    _, pixel_stats = _duplicate_groups(inventory, "pixel_md5")
    audit.tables["per_image_duplicates"] = duplicates[["md5", *ID_COLUMNS, "split"]].sort_values("md5")
    audit.check(
        section,
        "no exact duplicate image files (MD5)",
        md5_stats["n_groups"] == 0,
        f"{md5_stats['n_groups']} duplicate groups covering {md5_stats['n_files']} files; "
        f"{pixel_stats['n_groups']} groups with identical decoded pixels",
    )
    audit.check(
        section,
        "no duplicate images across patients or splits",
        md5_stats["cross_patient"] == pixel_stats["cross_patient"] == 0
        and md5_stats["cross_split"] == pixel_stats["cross_split"] == 0,
        f"by MD5: {md5_stats['cross_patient']} groups cross patients, {md5_stats['cross_split']} cross splits",
        hard=True,
    )
    audit.summary["images"] = {
        "n_image_files": len(inventory),
        "other_files_in_image_tree": dict(other_files),
        "samples_without_image": len(missing),
        "images_without_sample": len(stray),
        "unreadable": unreadable,
        "file_size": file_size,
        "off_size": off_size_info,
        "duplicates_md5": md5_stats,
        "duplicates_pixels": pixel_stats,
    }


def _size_stats(sizes: pd.Series) -> dict[str, float]:
    """Mean, sample SD (ddof=1), min, median and max of a series of group sizes."""
    return {
        "n_groups": int(len(sizes)),
        "mean": round(float(sizes.mean()), 2),
        "sd": round(float(sizes.std(ddof=1)), 2),
        "min": int(sizes.min()),
        "median": float(sizes.median()),
        "max": int(sizes.max()),
    }


def audit_splits(data: SageData, audit: Audit, expected: dict[str, Any]) -> None:
    """Section 3: split sizes, group sizes and train/test leakage at patient and visit level."""
    section = "3 Splits"
    samples = data.samples
    split_values = {str(key): int(value) for key, value in samples["split"].value_counts(dropna=False).items()}
    audit.check(section, "split is always 'train' or 'test'", set(split_values) == set(SPLITS), str(split_values))

    counts = by_split(
        samples,
        lambda frame: {
            "images": len(frame),
            "patients": int(frame["patient_uuid"].nunique()),
            "visits": int(frame["visit_uuid"].nunique()),
        },
    )
    count_rows = []
    for unit in ("images", "patients", "visits"):
        row: dict[str, Any] = {"unit": unit, **{key: counts[key][unit] for key in (*SPLITS, "total")}}
        row.update({f"{split}_pct": pct(counts[split][unit], counts["total"][unit]) for split in SPLITS})
        count_rows.append(row)
    audit.tables["split_counts"] = pd.DataFrame(count_rows)

    groupings: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "images per patient": lambda frame: frame.groupby("patient_uuid").size(),
        "images per visit": lambda frame: frame.groupby("visit_uuid").size(),
        "visits per patient": lambda frame: frame.groupby("patient_uuid")["visit_uuid"].nunique(),
    }
    group_sizes = {
        name: by_split(samples, lambda frame, sizes=sizes: _size_stats(sizes(frame)))
        for name, sizes in groupings.items()
    }
    audit.tables["split_group_sizes"] = pd.DataFrame(
        [
            {"quantity": name, "split": key, **stats}
            for name, per_split in group_sizes.items()
            for key, stats in per_split.items()
        ]
    )

    def ids(column: str, split: str) -> set[str]:
        return set(samples.loc[samples["split"] == split, column])

    patient_overlap = ids("patient_uuid", "train") & ids("patient_uuid", "test")
    visit_overlap = ids("visit_uuid", "train") & ids("visit_uuid", "test")
    audit.check(
        section,
        "zero patient overlap between train and test",
        not patient_overlap,
        f"{len(patient_overlap)} patients appear in both splits",
        hard=True,
    )
    audit.check(
        section,
        "zero visit overlap between train and test",
        not visit_overlap,
        f"{len(visit_overlap)} visits appear in both splits",
        hard=True,
    )

    visit_key = ["patient_uuid", "visit_uuid"]
    sample_visits = samples.drop_duplicates([*visit_key, "split"])[[*visit_key, "split"]]
    merged = sample_visits.merge(
        data.metadata[[*visit_key, "split"]], on=visit_key, how="outer", suffixes=("_samples", "_metadata")
    )
    disagree = int((merged["split_samples"] != merged["split_metadata"]).sum())
    audit.check(
        section,
        "samples.csv and metadata.csv agree on split for every visit",
        disagree == 0,
        f"{disagree} of {len(merged)} visits differ or are missing from one file",
    )
    spans_samples = int((samples.groupby("patient_uuid")["split"].nunique() > 1).sum())
    spans_metadata = int((data.metadata.groupby("patient_uuid")["split"].nunique() > 1).sum())
    audit.check(
        section,
        "all visits of each patient share one split (in both files)",
        spans_samples == 0 and spans_metadata == 0,
        f"{spans_samples} patients span splits in samples.csv, {spans_metadata} in metadata.csv",
        hard=True,
    )
    audit.summary["splits"] = {
        "counts": counts,
        "train_pct": {unit: pct(counts["train"][unit], counts["total"][unit]) for unit in counts["total"]},
        "nominal_train_pct": expected["nominal_train_pct"],
        "group_sizes": group_sizes,
        "patient_overlap": len(patient_overlap),
        "visit_overlap": len(visit_overlap),
        "patients_with_several_visits": int((groupings["visits per patient"](samples) > 1).sum()),
    }

"""Build per-image manifests for SAGE and HyperKvasir and check their counts.

Manifests go to outputs/manifests/ (or --out-dir) and are never committed. The SAGE manifest uses
integer patient codes instead of UUIDs. HyperKvasir image paths are added once the labelled images
are in --hk-root. Exits with an error if a count differs from configs/experiment.json.

    python scripts/03_build_manifests.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sage_shift.data import SagePaths, read_json
from sage_shift.manifests import build_hk_manifest, build_sage_manifest, hk_summary, sage_summary
from sage_shift.outdir import check_out_dir

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAGE and HyperKvasir manifests (read-only, no network).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--hk-split5", type=Path, default=HK_DIR / "official_splits" / "5_fold_split.csv")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images",
                        help="extracted HyperKvasir labelled images (the folder holding image-labels.csv)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "manifests")
    return parser.parse_args()


def compare(label: str, found: Any, expected: Any, problems: list[str]) -> None:
    status = "ok" if found == expected else "MISMATCH"
    print(f"  [{status}] {label}: {found}" + ("" if found == expected else f" (expected {expected})"))
    if found != expected:
        problems.append(f"{label}: found {found}, expected {expected}")


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    expected = cfg["expected_counts"]
    check_out_dir(args.out_dir, args.data_dir, ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    print("=== SAGE")
    sage = build_sage_manifest(SagePaths(args.data_dir))
    sage.to_csv(args.out_dir / "sage_manifest.csv", index=False)
    sage_counts = sage_summary(sage)
    compare("images", sage_counts["images"], expected["sage_images"], problems)
    for split in ("train", "test"):
        six = sage_counts["six_class"][split]
        for key in ("images", "patients"):
            compare(f"six-class {split} {key}", six[key], expected["sage_six_class"][split][key], problems)
        print(f"    images by class: {six['images_by_class']}")
        print(f"    patients by class: {six['patients_by_class']}")
        print(f"    visibility: {six['visibility_bins']} | NBI images: {six['nbi_images']} | "
              f"polyp location: {six['polyp_location_context']}")

    print("=== HyperKvasir")
    labels_csv = args.hk_root / "image-labels.csv"
    with_images = labels_csv.is_file()
    hk = build_hk_manifest(
        args.hk_split,
        args.hk_split5 if args.hk_split5.is_file() else None,
        labels_csv if with_images else None,
        args.hk_root if with_images else None,
    )
    hk.to_csv(args.out_dir / "hk_manifest.csv", index=False)
    hk_counts = hk_summary(hk)
    compare("labelled images", hk_counts["images"], expected["hk_labelled_images"], problems)
    compare("classes", hk_counts["classes"], expected["hk_classes"], problems)
    compare("images in the six-class space", hk_counts["mapped_images"], expected["hk_mapped_images"], problems)
    compare("six-class images by official fold", hk_counts["mapped_images_by_fold2"], expected["hk_mapped_images_by_fold2"], problems)
    print(f"    by SAGE class and fold: {hk_counts['mapped_by_sage_class_and_fold2']}")
    if with_images:
        missing = hk_counts["mapped_images_missing_on_disk"]
        compare("six-class images missing on disk", missing, 0, problems)
    else:
        print(f"  [note] {labels_csv} not found: image paths not added yet. Re-run after extracting the "
              "labelled images there (or pass --hk-root).")

    summary = {"sage": sage_counts, "hyperkvasir": hk_counts, "problems": problems}
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'sage_manifest.csv'}, {args.out_dir / 'hk_manifest.csv'} and manifest_summary.json")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("All counts match configs/experiment.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

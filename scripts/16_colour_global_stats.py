"""Dataset-level Lab statistics of the SAGE train frames, for the colour_global inputs (laptop, CPU).

Each frame's in-field Lab (D65) mean and SD come from scripts/05_check_fov.py
(sage_fov_quality.csv): they are measured on the fov crop resized to 224 px, over the pixels inside
the field. Their mean over the SAGE train frames without field flags gives the source statistics
of one fixed transfer to the HyperKvasir colour reference,

    lab' = (lab - mean) * (reference SD / SD) + reference mean   (inside the field),

mirroring how that reference was built from HyperKvasir frames. Test frames are never used. Writes
an aggregate JSON (no identifiers); an existing file with other values is kept unless --overwrite.

    python scripts/16_colour_global_stats.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_shift.colour import reference_from_stats
from sage_shift.fieldcheck import has_problem
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import PreprocessReference

ROOT = Path(__file__).resolve().parents[1]
LAB = ["L", "a", "b"]
MEANS, SDS = [f"{c}_mean" for c in LAB], [f"{c}_sd" for c in LAB]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGE train Lab statistics for the colour_global inputs.")
    parser.add_argument("--quality", type=Path, default=ROOT / "outputs" / "preprocess" / "sage_fov_quality.csv",
                        help="per-image SAGE field table from scripts/05_check_fov.py")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs" / "preprocess" / "preprocess_reference.json",
                        help="recorded as the target of the transfer when it holds a colour reference")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "preprocess" / "colour_global_sage.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rounded(values: Any) -> list[float]:
    return [round(float(v), 6) for v in values]


def main() -> int:
    args = parse_args()
    check_out_dir(args.out.parent, args.data_dir, ROOT)
    quality = pd.read_csv(args.quality, dtype={"id": str, "split": str, "flags": str}, keep_default_na=False)
    missing = {"split", "flags", "in_six", *MEANS, *SDS} - set(quality.columns)
    if missing:
        raise SystemExit(f"{args.quality.name}: missing columns {sorted(missing)}")
    flagged = quality["flags"].map(has_problem).to_numpy(dtype=bool)
    train = (quality["split"] == "train").to_numpy()
    six = quality["in_six"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    keep = train & ~flagged
    if not keep.any():
        raise SystemExit(f"{args.quality.name}: no train frames without field flags")
    stats = reference_from_stats(quality.loc[keep, MEANS].to_numpy(), quality.loc[keep, SDS].to_numpy(),
                                 source="SAGE train frames without field flags: mean of per-image in-field Lab "
                                        "means and SDs of the fov crop")
    six_only = reference_from_stats(quality.loc[keep & six, MEANS].to_numpy(), quality.loc[keep & six, SDS].to_numpy(),
                                    source="six-class SAGE train frames only (comparison; not used)")
    out: dict[str, Any] = {
        "mean": rounded(stats.mean), "std": rounded(stats.std), "images": stats.images, "source": stats.source,
        "split": "train", "train_frames": int(train.sum()), "train_frames_with_field_flags": int((train & flagged).sum()),
        "per_frame_statistics": "Lab (D65) mean and SD of the pixels inside the field of the fov crop at 224 px "
                                "(scripts/05_check_fov.py)",
        "transform": "lab' = (lab - mean) * (target std / std) + target mean, inside the field",
        "quality_table_sha256": hashlib.sha256(args.quality.read_bytes()).hexdigest(),
        "six_class_train_only": {"mean": rounded(six_only.mean), "std": rounded(six_only.std), "images": six_only.images},
        "target": None,
    }
    if args.reference.is_file():
        colour = PreprocessReference.load(args.reference).colour
        if colour is not None:
            out["target"] = {"mean": rounded(colour.mean), "std": rounded(colour.std), "images": colour.images,
                             "source": colour.source}
    text = json.dumps(out, indent=2) + "\n"
    if args.out.is_file() and args.out.read_text(encoding="utf-8") != text and not args.overwrite:
        raise SystemExit(f"{args.out} exists with other values; use --overwrite to replace it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"SAGE train frames: {out['train_frames']}, with field flags (left out): {out['train_frames_with_field_flags']}; "
          f"used: {stats.images}")
    print(f"  Lab mean {np.round(stats.mean, 3).tolist()}, SD {np.round(stats.std, 3).tolist()}")
    print(f"  six-class train frames only ({six_only.images}, not used): mean {np.round(six_only.mean, 3).tolist()}, "
          f"SD {np.round(six_only.std, 3).tolist()}")
    if out["target"] is not None:
        scale = np.array(out["target"]["std"]) / np.array(stats.std)
        shift = np.array(out["target"]["mean"]) - np.array(stats.mean)
        print(f"  target (HyperKvasir reference): mean {np.round(out['target']['mean'], 3).tolist()}, "
              f"SD {np.round(out['target']['std'], 3).tolist()}; scale {np.round(scale, 3).tolist()}, "
              f"mean shift {np.round(shift, 3).tolist()}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Field-of-view masks, layouts, colour reference and contact sheets.

Runs on all SAGE images and, once extracted, the six-class HyperKvasir images. Writes
preprocess_reference.json (used by script 06), a summary, per-image quality tables and contact
sheets to outputs/preprocess/ (or --out-dir).

    python scripts/05_check_fov.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_shift.colour import reference_from_stats
from sage_shift.data import read_json
from sage_shift.fieldcheck import (PROBLEM_FLAGS, detect_task, has_problem, measure_task, set_reference,
                                   unpack_detection)
from sage_shift.fov import FovLimits, FovParams, build_layouts, is_clean
from sage_shift.frames import hk_frames, sage_frames
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import PreprocessReference, frame_field, load_rgb, make_variants, resize_mask
from sage_shift.sheets import compose, draw_field, save_sheet

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"
PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
LAB = ["L", "a", "b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Field masks, layouts, colour reference and contact sheets.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--datasets", nargs="+", choices=["sage", "hk"],
                        help="default: sage, plus hk when --hk-root holds image-labels.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "preprocess")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-sheets", action="store_true", help="skip the contact sheets")
    return parser.parse_args()


def pct(values: pd.Series | np.ndarray) -> dict[str, float]:
    """Selected percentiles, rounded, as a JSON-ready dict."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return {str(p): round(float(np.percentile(values, p)), 4) for p in PERCENTILES} if len(values) else {}


def flag_counts(flags: pd.Series) -> dict[str, int]:
    """Frames carrying each flag."""
    return {flag: int(flags.str.split("|").map(lambda items, f=flag: f in items).sum())
            for flag in (*PROBLEM_FLAGS, "fallback", "no_fallback", "window")}


def has_flag(flags: pd.Series, flag: str) -> np.ndarray:
    """Boolean array: frames carrying `flag`."""
    return flags.str.split("|").map(lambda items: flag in items).to_numpy(dtype=bool)


def dataset_summary(frames: pd.DataFrame, quality: pd.DataFrame, layouts: list, limits: FovLimits,
                    clean: int) -> dict[str, Any]:
    """Aggregate statistics of one dataset (no identifiers)."""
    sizes = (quality["width"].astype(str) + "x" + quality["height"].astype(str)).value_counts()
    problem = quality["flags"].map(has_problem)
    six = frames["in_six"].to_numpy()
    green_inside = quality["green_inside"] > 50
    return {
        "frames": len(quality),
        "frame_sizes": {str(k): int(v) for k, v in sizes.items()},
        "detection": {"area_fraction_percentiles": pct(quality["detected_area"]),
                      "solidity_percentiles": pct(quality["detected_solidity"]), "clean_detections": clean},
        "layouts": [{"name": layout.name, "clean_frames": layout.frames,
                     "frames_held": int((quality["layout"] == layout.name).sum()),
                     "bbox": list(layout.bbox), "field_area_fraction": round(float(layout.mask.mean()), 4)}
                    for layout in layouts],
        "flags": flag_counts(quality["flags"]),
        "frames_needing_check": int(problem.sum()),
        "six_class": {str(split): {"frames": int(len(part)), "frames_needing_check": int(problem[part.index].sum()),
                                   "flags": flag_counts(quality.loc[part.index, "flags"])}
                      for split, part in frames[six].groupby("split")},
        "green_check": {"frames_with_green_pixels": int((quality["green_total"] > 50).sum()),
                        "frames_with_green_pixels_inside_field": int(green_inside.sum()),
                        "inside_field_by_class": {str(k): int(v) for k, v in
                                                  frames.loc[green_inside, "class"].value_counts().items()},
                        "windows_removed_by_class": {str(k): int(v) for k, v in frames.loc[
                            has_flag(quality["flags"], "window"), "class"].value_counts().items()}},
        "field_area_percentiles": pct(quality["field_area"]),
        "lab_fov_crop": {"mean_of_image_means": [round(float(quality[f"{c}_mean"].mean()), 3) for c in LAB],
                         "mean_of_image_sds": [round(float(quality[f"{c}_sd"].mean()), 3) for c in LAB],
                         **{f"{c}_mean_percentiles": pct(quality[f"{c}_mean"]) for c in LAB}},
        "limits": dict(limits.__dict__),
    }


def sheet_cells(frames: pd.DataFrame, quality: pd.DataFrame, indices: list[int], dataset: str,
                reference: PreprocessReference) -> tuple[list[tuple[str, list[np.ndarray]]], pd.DataFrame]:
    """Panels for each chosen frame: raw with field outline and crop box, then each variant."""
    variants = ["raw", "fov"] + (["colour", "fov_colour"] if reference.colour is not None else [])
    size, cells, index_rows = reference.size, [], []
    for number, i in enumerate(indices, start=1):
        rgb = load_rgb(frames.at[i, "path"])
        choice = frame_field(rgb, reference, dataset)
        images = make_variants(rgb, choice, size, reference.colour, variants)
        height, width = rgb.shape[:2]
        left, top, right, bottom = choice.field.bbox
        box = (round(left * size / width), round(top * size / height), round(right * size / width),
               round(bottom * size / height))
        panels = [draw_field(images["raw"], resize_mask(choice.field.full_mask(), size), box)]
        panels += [images[name] for name in variants[1:]]
        flags = quality.at[i, "flags"]
        cells.append((f"{number}. {choice.layout or 'own field'}" + (f" [{flags}]" if flags else ""), panels))
        index_rows.append({"tile": number, "id": frames.at[i, "id"], "split": frames.at[i, "split"],
                           "class": frames.at[i, "class"], "layout": choice.layout, "flags": flags})
    return cells, pd.DataFrame(index_rows)


def write_sheets(frames: pd.DataFrame, quality: pd.DataFrame, dataset: str, reference: PreprocessReference,
                 cfg: dict[str, Any], out_dir: Path) -> list[str]:
    """Random, flagged and per-layout contact sheets for one dataset; returns the files written."""
    rng = np.random.default_rng(int(cfg["seed"]))
    six = np.flatnonzero(frames["in_six"].to_numpy())
    chosen = {"random": sorted(rng.choice(six, size=min(int(cfg["random_frames"]), len(six)), replace=False).tolist())}
    problem = np.flatnonzero(quality["flags"].map(has_problem).to_numpy())
    if len(problem) > int(cfg["max_flagged"]):
        problem = np.sort(rng.choice(problem, size=int(cfg["max_flagged"]), replace=False))
    chosen["flagged"] = problem.tolist()
    per_layout = [int(rng.choice(np.flatnonzero((quality["layout"] == name).to_numpy())))
                  for name in sorted(set(quality["layout"]) - {""})]
    chosen["layouts"] = per_layout
    windows = np.flatnonzero(has_flag(quality["flags"], "window"))
    if len(windows) > int(cfg["max_flagged"]):
        windows = np.sort(rng.choice(windows, size=int(cfg["max_flagged"]), replace=False))
    chosen["windows"] = windows.tolist()
    written = []
    panels = 4 if reference.colour is not None else 2
    for kind, indices in chosen.items():
        if not indices:
            continue
        cells, index = sheet_cells(frames, quality, indices, dataset, reference)
        png = out_dir / "sheets" / f"{dataset}_{kind}.png"
        save_sheet(compose(cells, per_row=8 // panels), png)
        index.to_csv(png.with_name(f"{dataset}_{kind}_index.csv"), index=False)
        written.append(f"{png.name} ({len(indices)} frames)")
    return written


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    pre = cfg["preprocessing"]
    params, limits = FovParams.from_config(pre["fov"]), FovLimits.from_config(pre["fov_limits"])
    check_out_dir(args.out_dir, args.data_dir, ROOT)
    datasets = args.datasets or (["sage"] + (["hk"] if (args.hk_root / "image-labels.csv").is_file() else []))
    frames = {"sage": sage_frames(args.data_dir)} if "sage" in datasets else {}
    if "hk" in datasets:
        frames["hk"] = hk_frames(args.hk_root, args.hk_split)
    print("Datasets:", ", ".join(f"{name} ({len(table)} frames)" for name, table in frames.items()))

    with Pool(args.workers) as pool:
        detections = {name: [unpack_detection(item) for item in
                             pool.imap(detect_task, [(path, params) for path in table["path"]], chunksize=16)]
                      for name, table in frames.items()}
    layouts, clean = {}, {}
    for name, fields in detections.items():
        layouts[name], _ = build_layouts(fields, limits, prefix=f"{name}-")
        clean[name] = sum(is_clean(field, limits) for field in fields)
    reference = PreprocessReference(params, limits, int(cfg["image"]["size"]),
                                    {name: tuple(items) for name, items in layouts.items()},
                                    fallback={name: bool(pre["fallback"][name]) for name in frames})

    with Pool(args.workers, initializer=set_reference, initargs=(reference,)) as pool:
        quality = {name: pd.DataFrame(list(pool.imap(measure_task, [(path, name) for path in table["path"]],
                                                     chunksize=16)))
                   for name, table in frames.items()}
    for name, table in quality.items():
        table.insert(0, "detected_area", [field.area_fraction for field in detections[name]])
        table.insert(1, "detected_solidity", [field.solidity for field in detections[name]])

    if "hk" in quality:
        ok = ~quality["hk"]["flags"].map(has_problem)
        part = quality["hk"][ok]
        colour = reference_from_stats(part[[f"{c}_mean" for c in LAB]].to_numpy(), part[[f"{c}_sd" for c in LAB]].to_numpy(),
                                      source="HyperKvasir six-class images without field flags: mean of per-image "
                                             "in-field Lab means and SDs of the fov crop")
        reference = replace(reference, colour=colour)
        print(f"Colour reference from {colour.images} HyperKvasir frames: Lab mean "
              f"{np.round(colour.mean, 2).tolist()}, SD {np.round(colour.std, 2).tolist()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference.save(args.out_dir / "preprocess_reference.json")
    summary: dict[str, Any] = {"datasets": {}, "colour_reference": None if reference.colour is None else reference.colour.to_dict()}
    for name, table in frames.items():
        q = quality[name]
        summary["datasets"][name] = dataset_summary(table, q, layouts[name], limits, clean[name])
        if reference.colour is not None:
            scales = np.array(reference.colour.std) / q[[f"{c}_sd" for c in LAB]].to_numpy()
            summary["datasets"][name]["reinhard_scale"] = {
                **{f"{c}_percentiles": pct(scales[:, k]) for k, c in enumerate(LAB)},
                "frames_with_a_scale_outside_0.25_to_4": int(((scales < 0.25) | (scales > 4)).any(axis=1).sum())}
        out = pd.concat([table[["id", "split", "class", "in_six"]].reset_index(drop=True), q], axis=1)
        out.to_csv(args.out_dir / f"{name}_fov_quality.csv", index=False, float_format="%.5f")
        s = summary["datasets"][name]
        print(f"\n=== {name}: {s['frames']} frames, sizes {s['frame_sizes']}")
        print(f"  detected kept area p1/p50/p99: {s['detection']['area_fraction_percentiles'].get('1')}"
              f" / {s['detection']['area_fraction_percentiles'].get('50')} / {s['detection']['area_fraction_percentiles'].get('99')};"
              f" clean detections {s['detection']['clean_detections']}")
        for item in s["layouts"]:
            print(f"  layout {item['name']}: {item['clean_frames']} clean frames, holds {item['frames_held']}, "
                  f"box {item['bbox']}, field {item['field_area_fraction']:.3f} of the frame")
        print(f"  flags: {s['flags']}; frames needing a visual check: {s['frames_needing_check']}")
        print(f"  six-class frames needing a check, by split: "
              f"{ {k: v['frames_needing_check'] for k, v in s['six_class'].items()} }")
        print(f"  green-pixel check: {s['green_check']}")
        print(f"  fov-crop Lab: mean of means {s['lab_fov_crop']['mean_of_image_means']}, "
              f"mean of SDs {s['lab_fov_crop']['mean_of_image_sds']}")
        if "reinhard_scale" in s:
            print(f"  Reinhard scales outside [0.25, 4]: {s['reinhard_scale']['frames_with_a_scale_outside_0.25_to_4']} frames")
    (args.out_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.no_sheets:
        for name, table in frames.items():
            written = write_sheets(table, quality[name], name, reference, pre["contact_sheet"], args.out_dir)
            print(f"\nContact sheets for {name}: "
                  + ", ".join(written))
    print(f"\nWrote {args.out_dir / 'preprocess_reference.json'}, preprocess_summary.json and the per-image quality tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

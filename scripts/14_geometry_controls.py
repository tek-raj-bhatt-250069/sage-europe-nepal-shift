"""Score kept fine-tuned models on the geometry controls (needs a GPU; --limit runs a CPU smoke test).

Loads <tag>_best.pt, written by scripts/08_paper_recipe.py with --keep-model, and scores

    SAGE six-class test    raw, masked, masked_grey, fov, fov_grey, letterbox, letterbox_grey
    HyperKvasir fold 1     raw, sageify, sageify_grey (the adjusted recipe uses fold 1 only to
                           choose the epoch)

with the re-run's test transform: frames resized to 224 x 224 once, then ImageNet normalisation.
Only runs with a summary file (finished training) are scored. Writes, per run tag,
<tag>_geometry_sage_probs.csv and <tag>_geometry_hk_probs.csv (per image: keep them out of the
repository) and <tag>_geometry_summary.json (aggregate). Runs that already have a summary are
skipped, so an interrupted session can simply be restarted.

    python scripts/14_geometry_controls.py --runs-dir <folder> --reference <preprocess_reference.json>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sage_shift.backbones import sha256_file
from sage_shift.data import read_json
from sage_shift.frames import hk_frames, sage_frames
from sage_shift.geometry import HK_INPUTS, LOWER_GI, POLYP, SAGE_INPUTS, prepare_inputs
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import PreprocessReference, sage_placement
from sage_shift.recipe import CachedFrames, make_loader, make_model, predict, prediction_summary, test_transform
from sage_shift.runinfo import environment, git_state

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"
CLASS_INDEX = {name: i for i, name in enumerate(SIX_CLASSES)}
RUN_FILES = {"raw": "sage_probs", "fov": "sage_fov_probs", "masked": "sage_masked_probs"}  # written by script 08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score kept fine-tuned models on SAGE and HyperKvasir geometry controls.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "paper_recipe_keep",
                        help="scripts/08 output folder of runs made with --keep-model")
    parser.add_argument("--tags", nargs="+", help="run tags such as adjusted_seed0; default: every finished kept run")
    parser.add_argument("--out-dir", type=Path, help="default: --runs-dir")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs" / "preprocess" / "preprocess_reference.json",
                        help="preprocess_reference.json from scripts/05_check_fov.py (with both datasets)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "cache",
                        help="prepared frames, reused when the script is run again (session disk is fine)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--limit", type=int, help="only N images per set (smoke test; outputs are marked _smoke)")
    parser.add_argument("--overwrite", action="store_true", help="score finished runs again")
    return parser.parse_args()


def kept_runs(runs_dir: Path, tags: list[str] | None) -> tuple[list[str], list[str]]:
    """Kept checkpoints with a run summary (finished) and without one (still training)."""
    kept = sorted(path.name.removesuffix("_best.pt") for path in runs_dir.glob("*_best.pt"))
    if tags:
        absent = [tag for tag in tags if tag not in kept]
        if absent:
            raise SystemExit(f"no kept checkpoint for {absent} in {runs_dir} (run scripts/08 with --keep-model)")
        kept = [tag for tag in kept if tag in tags]
    done = [tag for tag in kept if (runs_dir / f"{tag}_summary.json").is_file()]
    return done, [tag for tag in kept if tag not in done]


def cached_inputs(name: str, ids: list[str], inputs: tuple[str, ...], make: Callable[[], dict[str, np.ndarray]],
                  args: argparse.Namespace, size: int) -> dict[str, np.ndarray]:
    """Prepared frames per input, read from the cache when it holds exactly these ids and reference."""
    key = {"ids": ids, "reference_sha256": sha256_file(args.reference)}
    key_file = args.cache_dir / f"geometry_{name}_{size}.json"
    files = {item: args.cache_dir / f"geometry_{name}_{item}_{size}.npy" for item in inputs}
    if not args.limit and key_file.is_file() and all(path.is_file() for path in files.values()) \
            and json.loads(key_file.read_text(encoding="utf-8")) == key:
        return {item: np.load(path) for item, path in files.items()}
    tick = time.time()
    images = make()
    print(f"  prepared {len(ids)} {name} frames x {len(inputs)} inputs in {time.time() - tick:.0f} s")
    if not args.limit:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        for item, path in files.items():
            np.save(path, images[item])
        key_file.write_text(json.dumps(key), encoding="utf-8")
    return images


def input_summary(y: np.ndarray, proba: np.ndarray, collapse: float) -> dict[str, Any]:
    """prediction_summary plus the polyp share and, for HyperKvasir, the upper-GI polyp-or-rectum share."""
    pred = proba.argmax(axis=1)
    upper = ~np.isin(y, LOWER_GI)
    out = prediction_summary(y, proba, collapse)
    out["polyp_share"] = round(float(np.mean(pred == POLYP)), 4)
    out["upper_gi_polyp_or_rectum_share"] = round(float(np.isin(pred[upper], LOWER_GI).mean()), 4) if upper.any() else None
    return out


def long_table(key: str, ids: list[str], probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per image and input: id, input, six probabilities."""
    parts = [pd.DataFrame(proba, columns=list(SIX_CLASSES)).assign(**{key: ids, "input": name})
             for name, proba in probs.items()]
    return pd.concat(parts, ignore_index=True)[[key, "input", *SIX_CLASSES]]


def run_check(runs_dir: Path, tag: str, ids: list[str], probs: dict[str, np.ndarray]) -> dict[str, float | None]:
    """Largest difference to the SAGE probabilities saved by script 08 for the same weights (None if absent)."""
    out: dict[str, float | None] = {}
    for name, stem in RUN_FILES.items():
        path = runs_dir / f"{tag}_{stem}.csv"
        if not path.is_file():
            out[name] = None
            continue
        saved = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid")
        out[name] = None if not set(ids) <= set(saved.index) else round(
            float(np.abs(saved.loc[ids, list(SIX_CLASSES)].to_numpy(dtype=float) - probs[name]).max()), 6)
    return out


def write_json(obj: Any, path: Path) -> None:
    """Write JSON to a temporary file, then rename (the summary marks a finished run)."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    out_dir = args.out_dir or args.runs_dir
    check_out_dir(out_dir, args.data_dir, ROOT)
    check_out_dir(args.cache_dir, args.data_dir, ROOT)
    size = int(cfg["image"]["size"])
    collapse = float(cfg["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type != "cuda" and not args.limit:
        raise SystemExit("this script scores ResNet-50 on about 8,300 frames per run and needs a GPU "
                         "(use --limit for a CPU smoke test)")
    torch.backends.cudnn.benchmark = False
    suffix = "_geometry" + ("_smoke" if args.limit else "")

    done, training = kept_runs(args.runs_dir, args.tags)
    if training:
        print(f"Not finished (no summary yet), skipped: {training}")
    if not done:
        raise SystemExit(f"no finished kept runs in {args.runs_dir}")
    todo = [tag for tag in done if args.overwrite or not (out_dir / f"{tag}{suffix}_summary.json").is_file()]
    for tag in sorted(set(done) - set(todo)):
        print(f"{tag}: already scored ({tag}{suffix}_summary.json exists); skipped. Use --overwrite to score it again.")
    if not todo:
        return 0

    if not args.reference.is_file():
        raise SystemExit(f"{args.reference} not found: run scripts/05_check_fov.py first")
    reference = PreprocessReference.load(args.reference)
    if reference.size != size:
        raise SystemExit(f"reference size {reference.size} differs from the config ({size})")
    placement = sage_placement(reference)
    expected = cfg["exploratory"]["sageify_placement"]
    if list(placement.shape) != expected["frame"] or list(placement.bbox) != expected["bbox"]:
        raise SystemExit(f"the most common SAGE layout ({placement}) differs from the config ({expected})")

    sage = sage_frames(args.data_dir)
    test = sage[sage["in_six"] & (sage["split"] == "test")].reset_index(drop=True)
    hk = hk_frames(args.hk_root, args.hk_split)
    fold1 = hk[hk["split"] == "fold1"].reset_index(drop=True)
    counts = cfg["expected_counts"]
    if args.limit:
        test, fold1 = test.head(args.limit), fold1.head(args.limit)
    elif len(test) != counts["sage_six_class"]["test"]["images"] or len(fold1) != counts["hk_mapped_images_by_fold2"]["1"]:
        raise SystemExit(f"found {len(test)} SAGE six-class test and {len(fold1)} HyperKvasir fold-1 images; "
                         "the config expects other counts")
    y_test = test["class"].map(CLASS_INDEX).to_numpy()
    y_hk = fold1["class"].map(CLASS_INDEX).to_numpy()
    print(f"Device: {device}; runs to score: {todo}; SAGE six-class test {len(test)} images x {len(SAGE_INPUTS)} "
          f"inputs; HyperKvasir fold 1 {len(fold1)} images x {len(HK_INPUTS)} inputs; sageify box {placement.bbox} "
          f"in a {placement.shape[1]} x {placement.shape[0]} canvas")
    sage_ids, hk_ids = test["id"].tolist(), fold1["id"].tolist()
    sage_images = cached_inputs("sage_test", sage_ids, SAGE_INPUTS, lambda: prepare_inputs(
        test["path"].tolist(), "sage", SAGE_INPUTS, reference, size, args.workers), args, size)
    hk_images = cached_inputs("hk_fold1", hk_ids, HK_INPUTS, lambda: prepare_inputs(
        fold1["path"].tolist(), "hk", HK_INPUTS, reference, size, args.workers), args, size)

    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in todo:
        tick = time.time()
        info = read_json(args.runs_dir / f"{tag}_summary.json")
        batch = int(info["settings"]["batch_size"])
        best = args.runs_dir / f"{tag}_best.pt"
        model = make_model(pretrained=False)
        model.load_state_dict(torch.load(best, map_location="cpu", weights_only=True))
        model.to(device)

        def score(images: np.ndarray) -> np.ndarray:
            data = CachedFrames(images, np.zeros(len(images), dtype=np.int64), test_transform())
            return predict(model, make_loader(data, batch, False, args.workers), device)

        sage_probs = {name: score(sage_images[name]) for name in SAGE_INPUTS}
        hk_probs = {name: score(hk_images[name]) for name in HK_INPUTS}
        long_table("sample_uuid", sage_ids, sage_probs).to_csv(out_dir / f"{tag}{suffix}_sage_probs.csv", index=False,
                                                               float_format="%.6f")
        long_table("file_name", hk_ids, hk_probs).to_csv(out_dir / f"{tag}{suffix}_hk_probs.csv", index=False,
                                                         float_format="%.6f")
        if info["recipe"] != "adjusted":
            print(f"  note: the {info['recipe']} recipe trained on part of HyperKvasir fold 1")
        summary = {
            "tag": tag, "recipe": info["recipe"], "seed": info["seed"], "epochs": info["epochs"],
            "best_epoch": info["best_epoch"], "checkpoint_sha256": sha256_file(best),
            "hk_fold1_held_out": info["recipe"] == "adjusted", "smoke_test": bool(args.limit),
            "sage_six_class_test_images": len(test), "hk_fold1_images": len(fold1),
            "transform": "resize to 224 x 224 (PIL bilinear), then ImageNet normalisation",
            "sageify_placement": {"frame": list(placement.shape), "bbox": list(placement.bbox)},
            "sage_six_class_test": {name: input_summary(y_test, proba, collapse) for name, proba in sage_probs.items()},
            "hk_fold1": {name: input_summary(y_hk, proba, collapse) for name, proba in hk_probs.items()},
            "max_abs_difference_to_run_probabilities": run_check(args.runs_dir, tag, sage_ids, sage_probs),
            "environment": environment(device), "git": git_state(ROOT),
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - tick, 1),
        }
        write_json(summary, out_dir / f"{tag}{suffix}_summary.json")
        print(f"\n{tag} (best epoch {info['best_epoch']}), {summary['seconds']} s")
        for dataset, key in (("SAGE test", "sage_six_class_test"), ("HyperKvasir fold 1", "hk_fold1")):
            for name, result in summary[key].items():
                share = result["upper_gi_polyp_or_rectum_share"]
                extra = "" if key == "sage_six_class_test" else \
                    f", upper-GI frames predicted polyp or rectum {'-' if share is None else f'{share:.2f}'}"
                print(f"  {dataset:18s} {name:15s} macro-F1 {result['macro_f1']:.3f}, top {result['top_class']} "
                      f"{result['top_class_share']:.2f} (collapse {result['collapse']}), polyp {result['polyp_share']:.2f}"
                      + extra)
        print(f"  largest difference to the run's own SAGE probabilities: "
              f"{summary['max_abs_difference_to_run_probabilities']}")
    print(f"\nWrote <tag>{suffix}_sage_probs.csv, _hk_probs.csv and _summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

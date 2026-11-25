"""Extract frozen-backbone features for SAGE and HyperKvasir (uses a GPU if available).

Writes <out-dir>/<dataset>/<backbone>/<variant>.npz with `features` and `ids`, and appends a run
record (versions, weights, timings) to features_run.json. Existing files are skipped unless
--overwrite, so an interrupted run can simply be restarted.

Besides the four pre-registered variants, the exploratory inputs of src/sage_shift/preprocess.py
can be requested. colour_global and fov_colour_global are made for SAGE only and need
--colour-global (scripts/16_colour_global_stats.py); sageify and sageify_grey are made for
HyperKvasir only. Choose the dataset with --datasets.

    python scripts/06_extract_features.py --variants raw --out-dir <folder>
    python scripts/06_extract_features.py --random-weights --limit 8    # quick offline test
    python scripts/06_extract_features.py --datasets hk --variants sageify sageify_grey --out-dir <folder>
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sage_shift.backbones import BACKBONES, load_backbone
from sage_shift.colour import ColourReference
from sage_shift.data import read_json
from sage_shift.features import FrameDataset, extract, save_features
from sage_shift.frames import hk_frames, sage_frames
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import (EXPLORATORY, GLOBAL_COLOUR, NEEDS_COLOUR, NEEDS_FIELD, SAGEIFY,
                                   PreprocessReference, check_dataset, check_variants, sage_placement)
from sage_shift.runinfo import environment, git_state

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"
DEFAULT_BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "dinov2_vitb14"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen-backbone features (raw, fov, colour, fov_colour "
                                                 "and the exploratory inputs).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--datasets", nargs="+", choices=["sage", "hk"],
                        help="default: sage, plus hk when --hk-root holds image-labels.csv")
    parser.add_argument("--variants", nargs="+", default=["raw"],
                        help="raw fov colour fov_colour; exploratory: " + " ".join(EXPLORATORY))
    parser.add_argument("--colour-global", type=Path,
                        default=ROOT / "outputs" / "preprocess" / "colour_global_sage.json",
                        help="SAGE train Lab statistics from scripts/16_colour_global_stats.py (colour_global inputs)")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES,
                        help=f"default: {' '.join(DEFAULT_BACKBONES)}, plus resnet50_gastronet_dino with --gastronet-weights")
    parser.add_argument("--gastronet-weights", type=Path, help="local GastroNet-5M DINOv1 ResNet-50 checkpoint")
    parser.add_argument("--gastronet-allow-pickle", action="store_true",
                        help="allow a full pickle checkpoint (only for the file from the Theta Vision portal)")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs" / "preprocess" / "preprocess_reference.json",
                        help="written by scripts/05_check_fov.py; needed for fov and colour variants")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, help="default outputs/features (outputs/features_smoke with --limit)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2, help="data-loader worker processes")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--limit", type=int, help="only the first N images per dataset (smoke test)")
    parser.add_argument("--random-weights", action="store_true", help="no downloads; for smoke tests only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    variants = check_variants(args.variants)
    out_dir = args.out_dir or ROOT / "outputs" / ("features_smoke" if args.limit else "features")
    check_out_dir(out_dir, args.data_dir, ROOT)
    if args.random_weights and args.out_dir is not None and args.limit is None:
        raise SystemExit("--random-weights is for smoke tests: use it with --limit")
    names = args.backbones or DEFAULT_BACKBONES + (["resnet50_gastronet_dino"] if args.gastronet_weights else [])
    datasets = args.datasets or (["sage"] + (["hk"] if (args.hk_root / "image-labels.csv").is_file() else []))
    for dataset in datasets:
        try:
            check_dataset(dataset, variants)
        except ValueError as error:
            raise SystemExit(f"{error}. Choose the dataset with --datasets.") from None
    reference = None
    if NEEDS_FIELD & set(variants):
        if not args.reference.is_file():
            raise SystemExit(f"{args.reference} not found: run scripts/05_check_fov.py first")
        reference = PreprocessReference.load(args.reference)
        if NEEDS_COLOUR & set(variants) and reference.colour is None:
            raise SystemExit("the colour variants need a colour reference: run scripts/05_check_fov.py with HyperKvasir")
        if reference.size != int(cfg["image"]["size"]):
            raise SystemExit(f"reference size {reference.size} differs from the config ({cfg['image']['size']})")
    sage_colour, extra = None, {}
    if GLOBAL_COLOUR & set(variants):
        if not args.colour_global.is_file():
            raise SystemExit(f"{args.colour_global} not found: run scripts/16_colour_global_stats.py first")
        stats = read_json(args.colour_global)
        sage_colour = ColourReference.from_dict(stats)
        target = stats.get("target")
        if target is not None and not (np.allclose(target["mean"], reference.colour.mean)
                                       and np.allclose(target["std"], reference.colour.std)):
            raise SystemExit(f"{args.colour_global.name} was written for another colour reference than {args.reference}")
        extra["colour_global"] = {"file": args.colour_global.name, "mean": stats["mean"], "std": stats["std"],
                                  "images": stats["images"]}
    if SAGEIFY & set(variants):
        placement = sage_placement(reference)
        expected = cfg["exploratory"]["sageify_placement"]
        if list(placement.shape) != expected["frame"] or list(placement.bbox) != expected["bbox"]:
            raise SystemExit(f"the most common SAGE layout ({placement}) differs from the config ({expected})")
        extra["sageify_placement"] = {"frame": list(placement.shape), "bbox": list(placement.bbox)}
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(int(cfg["seeds"]["global"]))
    torch.backends.cudnn.benchmark = False

    started = time.time()
    record: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "args": {key: str(value) for key, value in vars(args).items()},
                              "environment": environment(device), "git": git_state(ROOT), "backbones": {}, "outputs": [],
                              **extra}
    print(f"Device: {device}; datasets: {datasets}; variants: {list(variants)}; backbones: {names}")
    backbones = []
    for name in names:
        backbone = load_backbone(name, device, args.random_weights, args.gastronet_weights, args.gastronet_allow_pickle)
        record["backbones"][name] = {"dim": backbone.dim, **backbone.record}
        backbones.append(backbone)
        print(f"  loaded {name}: {backbone.record}")

    for dataset in datasets:
        frames = sage_frames(args.data_dir) if dataset == "sage" else hk_frames(args.hk_root, args.hk_split)
        if args.limit:
            frames = frames.head(args.limit)
        todo = [b for b in backbones if args.overwrite or not all(
            (out_dir / dataset / b.name / f"{v}.npz").is_file() for v in variants)]
        if not todo:
            print(f"{dataset}: all requested files exist; skipped (use --overwrite to redo)")
            continue
        tick = time.time()
        loader_set = FrameDataset(frames["path"].tolist(), dataset, variants, reference, int(cfg["image"]["size"]),
                                  sage_colour)
        features = extract(loader_set, todo, device, args.batch_size, args.workers)
        seconds = round(time.time() - tick, 1)
        for (name, variant), values in features.items():
            path = out_dir / dataset / name / f"{variant}.npz"
            if path.is_file() and not args.overwrite:
                continue
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite features for {dataset}/{name}/{variant}")
            save_features(path, values, frames["id"].tolist(), dataset=dataset, backbone=name, variant=variant)
            record["outputs"].append({"dataset": dataset, "backbone": name, "variant": variant, "rows": int(values.shape[0]),
                                      "dim": int(values.shape[1]), "file": str(path.relative_to(out_dir))})
            print(f"  {dataset}/{name}/{variant}: {values.shape[0]} x {values.shape[1]}")
        print(f"{dataset}: {len(frames)} frames in {seconds} s")

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["seconds"] = round(time.time() - started, 1)
    log = out_dir / "features_run.json"
    runs = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else []
    log.write_text(json.dumps(runs + [record], indent=2) + "\n", encoding="utf-8")
    print(f"Done in {record['seconds']} s; run record appended to {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

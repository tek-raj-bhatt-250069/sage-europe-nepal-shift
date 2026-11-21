"""Extract frozen-backbone features for SAGE and HyperKvasir (uses a GPU if available).

Writes <out-dir>/<dataset>/<backbone>/<variant>.npz with `features` and `ids`, and appends a run
record (versions, weights, timings) to features_run.json. Existing files are skipped unless
--overwrite, so an interrupted run can simply be restarted.

    python scripts/06_extract_features.py --variants raw --out-dir <folder>
    python scripts/06_extract_features.py --random-weights --limit 8    # quick offline test
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision

from sage_shift.backbones import BACKBONES, load_backbone
from sage_shift.data import read_json
from sage_shift.features import FrameDataset, extract, save_features
from sage_shift.frames import hk_frames, sage_frames
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import FIELD_VARIANTS, COLOUR_VARIANTS, PreprocessReference, check_variants

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"
DEFAULT_BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "dinov2_vitb14"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen-backbone features (raw, fov, colour, fov_colour).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--datasets", nargs="+", choices=["sage", "hk"],
                        help="default: sage, plus hk when --hk-root holds image-labels.csv")
    parser.add_argument("--variants", nargs="+", default=["raw"], help="raw fov colour fov_colour")
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


def git_state() -> dict[str, Any]:
    """Commit and cleanliness of the repository the script runs from."""
    def run(*command: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(ROOT), *command], capture_output=True, text=True, check=True)
            return done.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": None if status is None else bool(status)}


def environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__,
                            "torchvision": torchvision.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
                            "device": str(device)}
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    variants = check_variants(args.variants)
    out_dir = args.out_dir or ROOT / "outputs" / ("features_smoke" if args.limit else "features")
    check_out_dir(out_dir, args.data_dir, ROOT)
    if args.random_weights and args.out_dir is not None and args.limit is None:
        raise SystemExit("--random-weights is for smoke tests: use it with --limit")
    names = args.backbones or DEFAULT_BACKBONES + (["resnet50_gastronet_dino"] if args.gastronet_weights else [])
    reference = None
    if FIELD_VARIANTS & set(variants):
        if not args.reference.is_file():
            raise SystemExit(f"{args.reference} not found: run scripts/05_check_fov.py first")
        reference = PreprocessReference.load(args.reference)
        if COLOUR_VARIANTS & set(variants) and reference.colour is None:
            raise SystemExit("the colour variants need a colour reference: run scripts/05_check_fov.py with HyperKvasir")
        if reference.size != int(cfg["image"]["size"]):
            raise SystemExit(f"reference size {reference.size} differs from the config ({cfg['image']['size']})")
    datasets = args.datasets or (["sage"] + (["hk"] if (args.hk_root / "image-labels.csv").is_file() else []))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(int(cfg["seeds"]["global"]))
    torch.backends.cudnn.benchmark = False

    started = time.time()
    record: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "args": {key: str(value) for key, value in vars(args).items()},
                              "environment": environment(device), "git": git_state(), "backbones": {}, "outputs": []}
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
        loader_set = FrameDataset(frames["path"].tolist(), dataset, variants, reference, int(cfg["image"]["size"]))
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

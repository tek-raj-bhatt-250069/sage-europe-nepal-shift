"""Re-run the SAGE paper's HyperKvasir -> SAGE training recipe (needs a GPU).

    faithful   the released recipe: ResNet-50 (ImageNet), Adam lr 0.01, StepLR(5, 0.5), batch 32,
               100 epochs, random 80/20 split of the six-class HyperKvasir images (seed 0), epoch
               chosen by weighted F1 on that 20%
    adjusted   the same model, transforms and schedule at lr 1e-4 for 30 epochs, trained on official
               fold 0 with the epoch chosen on fold 1

The chosen epoch is evaluated on every SAGE image. The six-class SAGE test images are also scored
after each epoch to show how predictions develop; they are never used to choose anything. With
--reference, the chosen model is also scored on field-of-view crops of the SAGE frames (the fov
variant of scripts/05 and 06). Outputs (history, per-image SAGE probabilities, summary) go to
--out-dir. An interrupted run resumes from its last saved state when restarted with the same
arguments.

    python scripts/08_paper_recipe.py --recipe faithful --out-dir <folder> --reference <preprocess_reference.json>
    python scripts/08_paper_recipe.py --recipe adjusted --out-dir <folder> --reference <preprocess_reference.json>
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet50_Weights

from sage_shift.backbones import hub_checkpoint, sha256_file
from sage_shift.data import read_json
from sage_shift.frames import sage_frames
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import build_hk_manifest
from sage_shift.outdir import check_out_dir
from sage_shift.preprocess import PreprocessReference
from sage_shift.recipe import (CachedFrames, cache_fov_frames, cache_frames, faithful_split, make_loader, make_model,
                               predict, prediction_summary, rng_state, save_atomic, scores, seed_everything,
                               set_rng_state, test_transform, train_epoch, train_transform)
from sage_shift.runinfo import environment, git_state

ROOT = Path(__file__).resolve().parents[1]
HK_DIR = ROOT / "external" / "hyperkvasir"
CLASS_INDEX = {name: i for i, name in enumerate(SIX_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful and adjusted re-runs of the paper's training recipe.")
    parser.add_argument("--recipe", required=True, choices=["faithful", "adjusted"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--hk-root", type=Path, default=HK_DIR / "labeled-images")
    parser.add_argument("--hk-split", type=Path, default=HK_DIR / "official_splits" / "2_fold_split.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--reference", type=Path,
                        help="preprocess_reference.json; also score the chosen model on fov crops of SAGE")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "paper_recipe")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "cache",
                        help="resized frames, reused by the second recipe (session disk is fine)")
    parser.add_argument("--epochs", type=int, help="default: the config value for the recipe")
    parser.add_argument("--seed", type=int, help="default: seeds.global in the config (0, as released)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--save-every", type=int, default=5, help="epochs between resume points")
    parser.add_argument("--restart", action="store_true", help="ignore a saved state and start again")
    parser.add_argument("--keep-model", action="store_true", help="keep the chosen weights (about 95 MB)")
    parser.add_argument("--limit", type=int, help="only N images per set (smoke test)")
    parser.add_argument("--random-weights", action="store_true", help="no download; smoke tests only")
    return parser.parse_args()


def hk_tables(args: argparse.Namespace, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and epoch-selection rows (file_name, rel_path, sage_class) for the recipe."""
    labels_csv = args.hk_root / "image-labels.csv"
    manifest = build_hk_manifest(args.hk_split, None, labels_csv, args.hk_root)
    mapped = manifest[manifest["sage_class"] != ""].set_index("file_name")
    if (mapped["exists"] == 0).any():
        raise SystemExit(f"{int((mapped['exists'] == 0).sum())} six-class HyperKvasir images are missing below {args.hk_root}")
    if args.recipe == "adjusted":
        return mapped[mapped["fold2"] == 0].reset_index(), mapped[mapped["fold2"] == 1].reset_index()
    # The released script splits image-labels.csv rows in file order after dropping unmapped classes.
    names = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)["Video file"]
    names = names.map(lambda name: name if name.lower().endswith(".jpg") else f"{name}.jpg")
    ordered = mapped.loc[[name for name in names if name in mapped.index]].reset_index()
    train, select = faithful_split(ordered["sage_class"].tolist(), seed)
    return ordered.iloc[train].reset_index(drop=True), ordered.iloc[select].reset_index(drop=True)


def cached(name: str, ids: list[str], make: Callable[[], np.ndarray], args: argparse.Namespace,
           size: int) -> np.ndarray:
    """Resized frames, read from the cache when it holds exactly these ids."""
    images_file, ids_file = args.cache_dir / f"{name}_{size}.npy", args.cache_dir / f"{name}_{size}_ids.json"
    if not args.limit and images_file.is_file() and ids_file.is_file() \
            and json.loads(ids_file.read_text(encoding="utf-8")) == ids:
        return np.load(images_file)
    tick = time.time()
    images = make()
    print(f"  prepared {len(ids)} {name} frames in {time.time() - tick:.0f} s")
    if not args.limit:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(images_file, images)
        ids_file.write_text(json.dumps(ids), encoding="utf-8")
    return images


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    spec = cfg["paper_recipe_rerun"][args.recipe]
    seed = int(cfg["seeds"]["global"]) if args.seed is None else args.seed
    epochs = int(spec["epochs"]) if args.epochs is None else args.epochs
    size, collapse = int(cfg["image"]["size"]), float(cfg["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"])
    check_out_dir(args.out_dir, args.data_dir, ROOT)
    if args.random_weights and not args.limit:
        raise SystemExit("--random-weights is for smoke tests: use it with --limit")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type != "cuda" and not args.limit:
        raise SystemExit("this script trains ResNet-50 and needs a GPU (use --limit for a CPU smoke test)")
    tag = f"{args.recipe}_seed{seed}" + ("_smoke" if args.limit else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_file, best_file = args.out_dir / f"{tag}_state.pt", args.out_dir / f"{tag}_best.pt"
    if args.restart:
        state_file.unlink(missing_ok=True)
        best_file.unlink(missing_ok=True)
    seed_everything(seed)

    train_rows, select_rows = hk_tables(args, seed)
    sage = sage_frames(args.data_dir)
    if args.limit:
        train_rows, select_rows = train_rows.head(args.limit), select_rows.head(args.limit)
        sage = pd.concat([sage[sage["in_six"] & (sage["split"] == "test")].head(args.limit),
                          sage[~sage["in_six"]].head(2)]).reset_index(drop=True)
    hk_rows = pd.concat([train_rows, select_rows]).drop_duplicates("file_name").sort_values("file_name")
    hk_paths = [str(args.hk_root / rel) for rel in hk_rows["rel_path"]]
    hk_images = cached("hk_six", hk_rows["file_name"].tolist(), lambda: cache_frames(hk_paths, size, args.workers),
                       args, size)
    position = {name: i for i, name in enumerate(hk_rows["file_name"])}
    sage_images = cached("sage", sage["id"].tolist(), lambda: cache_frames(sage["path"].tolist(), size, args.workers),
                         args, size)
    sage_fov = None
    if args.reference is not None:
        if not args.reference.is_file():
            raise SystemExit(f"{args.reference} not found: run scripts/05_check_fov.py first")
        reference = PreprocessReference.load(args.reference)
        sage_fov = cached("sage_fov", sage["id"].tolist(),
                          lambda: cache_fov_frames(sage["path"].tolist(), "sage", reference, size, args.workers),
                          args, size)
    y_train = train_rows["sage_class"].map(CLASS_INDEX).to_numpy()
    y_select = select_rows["sage_class"].map(CLASS_INDEX).to_numpy()
    test = (sage["in_six"] & (sage["split"] == "test")).to_numpy()
    y_test = sage.loc[test, "class"].map(CLASS_INDEX).to_numpy()
    print(f"{args.recipe}: lr {spec['lr']}, {epochs} epochs, batch {spec['batch_size']}, seed {seed}; "
          f"HyperKvasir train {len(train_rows)}, epoch selection {len(select_rows)}; SAGE frames {len(sage)} "
          f"({int(test.sum())} six-class test)")

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(CachedFrames(hk_images[[position[n] for n in train_rows["file_name"]]], y_train,
                                            train_transform()), int(spec["batch_size"]), True, args.workers, generator)
    select_loader = make_loader(CachedFrames(hk_images[[position[n] for n in select_rows["file_name"]]], y_select,
                                             test_transform()), int(spec["batch_size"]), False, args.workers)
    sage_loader = make_loader(CachedFrames(sage_images, np.zeros(len(sage), dtype=np.int64), test_transform()),
                              int(spec["batch_size"]), False, args.workers)
    test_loader = make_loader(CachedFrames(sage_images[test], y_test, test_transform()), int(spec["batch_size"]),
                              False, args.workers)

    model = make_model(pretrained=not args.random_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(spec["step_size"]), gamma=float(spec["gamma"]))
    history: list[dict[str, Any]] = []
    best_f1, best_epoch, start = 0.0, 0, 0
    if state_file.is_file() and not args.restart:
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_f1, best_epoch, start = state["history"], state["best_f1"], state["best_epoch"], state["epoch"]
        set_rng_state(state["rng"], generator)
        print(f"  resumed after epoch {start} (best weighted F1 {best_f1:.4f} at epoch {best_epoch})")

    started = time.time()
    for epoch in range(start + 1, epochs + 1):
        tick = time.time()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        select = scores(y_select, predict(model, select_loader, device))
        monitor = prediction_summary(y_test, predict(model, test_loader, device), collapse)
        if select["weighted_f1"] > best_f1:  # strict, as released
            best_f1, best_epoch = select["weighted_f1"], epoch
            save_atomic({key: value.cpu() for key, value in model.state_dict().items()}, best_file)
        history.append({"epoch": epoch, "lr": lr, "train_loss": round(loss, 6),
                        **{f"select_{key}": round(value, 4) for key, value in select.items()},
                        "sage_test_macro_f1": monitor["macro_f1"], "sage_test_top_class": monitor["top_class"],
                        "sage_test_top_share": monitor["top_class_share"], "seconds": round(time.time() - tick, 1)})
        print(f"  epoch {epoch:3d}  lr {lr:.2e}  loss {loss:.4f}  selection wF1 {select['weighted_f1']:.4f}  "
              f"SAGE test macro-F1 {monitor['macro_f1']:.3f}, top {monitor['top_class']} {monitor['top_class_share']:.2f}")
        if epoch % args.save_every == 0 or epoch == epochs:
            save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "history": history, "best_f1": best_f1,
                         "best_epoch": best_epoch, "epoch": epoch, "rng": rng_state(generator)}, state_file)

    if not best_file.is_file():
        raise SystemExit("no epoch improved on a weighted F1 of 0; nothing to evaluate")
    model.load_state_dict(torch.load(best_file, map_location=device, weights_only=True))
    proba = predict(model, sage_loader, device)
    probs = pd.DataFrame(proba, columns=list(SIX_CLASSES)).assign(sample_uuid=sage["id"])[["sample_uuid", *SIX_CLASSES]]
    probs.to_csv(args.out_dir / f"{tag}_sage_probs.csv", index=False, float_format="%.6f")
    proba_fov = None
    if sage_fov is not None:
        proba_fov = predict(model, make_loader(CachedFrames(sage_fov, np.zeros(len(sage), dtype=np.int64),
                                                            test_transform()), int(spec["batch_size"]), False,
                                               args.workers), device)
        pd.DataFrame(proba_fov, columns=list(SIX_CLASSES)).assign(sample_uuid=sage["id"])[
            ["sample_uuid", *SIX_CLASSES]].to_csv(args.out_dir / f"{tag}_sage_fov_probs.csv", index=False,
                                                  float_format="%.6f")
    pd.DataFrame(history).to_csv(args.out_dir / f"{tag}_history.csv", index=False)
    weights_url = ResNet50_Weights.IMAGENET1K_V1.url
    weights_path = hub_checkpoint(weights_url)
    summary = {
        "recipe": args.recipe, "seed": seed, "epochs": epochs, "settings": spec,
        "hk_train_images": len(train_rows), "hk_selection_images": len(select_rows),
        "best_epoch": best_epoch, "best_selection_weighted_f1": round(best_f1, 4),
        "selection_at_best_epoch": next((row for row in history if row["epoch"] == best_epoch), None),
        "sage_six_class_test": prediction_summary(y_test, proba[test], collapse),
        "sage_six_class_test_fov": None if proba_fov is None else prediction_summary(y_test, proba_fov[test], collapse),
        "weights": {"source": "random (smoke test)"} if args.random_weights else
                   {"source": weights_url, "sha256": sha256_file(weights_path) if weights_path.is_file() else None},
        "environment": environment(device), "git": git_state(ROOT),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds_this_session": round(time.time() - started, 1),
    }
    (args.out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    state_file.unlink(missing_ok=True)
    if not args.keep_model:
        best_file.unlink(missing_ok=True)
    result = summary["sage_six_class_test"]
    print(f"\n{args.recipe}: best epoch {best_epoch} (selection weighted F1 {best_f1:.4f})")
    print(f"SAGE six-class test: weighted F1 {result['weighted_f1']:.4f}, macro-F1 {result['macro_f1']:.4f}, "
          f"accuracy {result['accuracy']:.4f}; top class {result['top_class']} gets {result['top_class_share']:.0%} "
          f"of predictions (collapse: {result['collapse']})")
    print(f"Predicted counts: {result['predicted_counts']}")
    if summary["sage_six_class_test_fov"] is not None:
        fov = summary["sage_six_class_test_fov"]
        print(f"Same model on fov crops: weighted F1 {fov['weighted_f1']:.4f}, macro-F1 {fov['macro_f1']:.4f}; "
              f"top class {fov['top_class']} gets {fov['top_class_share']:.0%} (collapse: {fov['collapse']})")
    print(f"Wrote {tag}_summary.json, {tag}_history.csv and {tag}_sage_probs.csv to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

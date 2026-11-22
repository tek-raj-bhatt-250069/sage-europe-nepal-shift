"""RQ1: transfer gap on the SAGE six-class test set.

For each backbone, a probe trained on HyperKvasir (transfer) and a probe trained on SAGE train
(ceiling) are scored on the SAGE test images. C is chosen on training data only. Intervals come
from a paired patient-level bootstrap; the gap is ceiling minus transfer. Per-image predictions
are written to outputs/ (git-ignored).

    python scripts/07_rq1_probes.py --variant raw
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_shift.baselines import constant_predictor_table
from sage_shift.bootstrap import group_weights, percentile_ci
from sage_shift.data import read_json
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.metrics import evaluate
from sage_shift.outdir import check_out_dir
from sage_shift.probes import fit_predict, grouped_folds, official_folds, select_c

ROOT = Path(__file__).resolve().parents[1]
BACKBONE_ORDER = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
METRICS = ["macro_f1", "macro_auroc", "weighted_f1", "balanced_accuracy"]
CLASS_METRICS = ["f1", "auroc", "ap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1 probes, SAGE ceiling and paired patient bootstrap.")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "rq1")
    parser.add_argument("--variant", default="raw", choices=["raw", "fov", "colour", "fov_colour"])
    parser.add_argument("--backbones", nargs="+", help="default: every backbone with SAGE and HyperKvasir features")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def aligned_features(path: Path, ids: pd.Series) -> np.ndarray:
    """Feature rows in the order of `ids`."""
    with np.load(path, allow_pickle=False) as data:
        features, file_ids = data["features"], data["ids"]
    index = {key: row for row, key in enumerate(file_ids.tolist())}
    missing = [key for key in ids if key not in index]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} images have no features")
    return features[[index[key] for key in ids]].astype(np.float64)


def boot_scores(y: np.ndarray, proba: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics on every resample: scalars as (B,), per-class scores as (B, classes)."""
    runs = [evaluate(y, proba, truth, w) for w in weights]
    out = {m: np.array([r[m] for r in runs]) for m in METRICS}
    out.update({m: np.stack([r[m] for r in runs]) for m in CLASS_METRICS})
    return out


def summarise(point: float, values: np.ndarray, level: float) -> dict[str, float]:
    low, high = percentile_ci(values, level)
    return {"estimate": round(float(point), 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y]


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    grid, seed, max_iter = cfg["probe"]["C_grid"], int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"])
    n_boot, level = int(cfg["bootstrap"]["n_resamples"]), float(cfg["bootstrap"]["ci"])
    k = len(SIX_CLASSES)
    class_index = {name: i for i, name in enumerate(SIX_CLASSES)}

    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    six = sage[sage["in_six"] == 1]
    train = six[six["split"] == "train"].reset_index(drop=True)
    test = six[six["split"] == "test"].reset_index(drop=True)
    y_train, y_test = train["class"].map(class_index).to_numpy(), test["class"].map(class_index).to_numpy()
    truth = test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float)
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    y_hk, hk_fold = hk["sage_class"].map(class_index).to_numpy(), hk["fold2"].astype(int).to_numpy()

    names = args.backbones or [b for b in BACKBONE_ORDER if all(
        (args.features / d / b / f"{args.variant}.npz").is_file() for d in ("sage", "hk"))]
    if not names:
        raise SystemExit(f"no {args.variant} features found below {args.features}")
    weights = group_weights(test["patient_group"].to_numpy(), n_boot, int(cfg["seeds"]["bootstrap"]))
    hk_weights = group_weights(np.arange(len(hk)), n_boot, int(cfg["seeds"]["bootstrap"]) + 1)
    best_constant = float(constant_predictor_table(test["class"].tolist(), SIX_CLASSES)["macro_f1"].max())
    print(f"SAGE test: {len(test)} images, {test['patient_group'].nunique()} patients; SAGE train: {len(train)} images, "
          f"{train['patient_group'].nunique()} patients; HyperKvasir: {len(hk)} images. Variant: {args.variant}; "
          f"{n_boot} paired patient resamples.")

    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"variant": args.variant, "test_images": len(test),
                               "test_patients": int(test["patient_group"].nunique()), "bootstrap": cfg["bootstrap"],
                               "best_constant_macro_f1": round(best_constant, 4), "classes": list(SIX_CLASSES),
                               "backbones": {}}
    for name in names:
        x_train = aligned_features(args.features / "sage" / name / f"{args.variant}.npz", train["sample_uuid"])
        x_test = aligned_features(args.features / "sage" / name / f"{args.variant}.npz", test["sample_uuid"])
        x_hk = aligned_features(args.features / "hk" / name / f"{args.variant}.npz", hk["file_name"])

        c_hk, cv_hk, oof_hk = select_c(x_hk, y_hk, official_folds(hk_fold), grid, seed, k, args.jobs, max_iter)
        transfer = fit_predict(x_hk, y_hk, x_test, c_hk, seed, k, max_iter)
        folds = grouped_folds(y_train, train["patient_group"].to_numpy(), 5, seed)
        c_sage, cv_sage, _ = select_c(x_train, y_train, folds, grid, seed, k, args.jobs, max_iter)
        ceiling = fit_predict(x_train, y_train, x_test, c_sage, seed, k, max_iter)
        for label, proba in (("transfer", transfer), ("ceiling", ceiling)):
            frame = pd.DataFrame(proba, columns=list(SIX_CLASSES)).assign(sample_uuid=test["sample_uuid"])
            frame[["sample_uuid", *SIX_CLASSES]].to_csv(pred_dir / f"{name}__{args.variant}__{label}.csv",
                                                        index=False, float_format="%.6f")

        point = {"transfer": evaluate(y_test, transfer, truth), "ceiling": evaluate(y_test, ceiling, truth)}
        boot = {"transfer": boot_scores(y_test, transfer, truth, weights),
                "ceiling": boot_scores(y_test, ceiling, truth, weights)}
        in_domain = evaluate(y_hk, oof_hk, one_hot(y_hk, k))
        in_boot = boot_scores(y_hk, oof_hk, one_hot(y_hk, k), hk_weights)

        result: dict[str, Any] = {"C_hyperkvasir": c_hk, "cv_macro_f1_hyperkvasir": cv_hk,
                                  "C_sage": c_sage, "cv_macro_f1_sage": cv_sage, "metrics": {}, "per_class": {},
                                  "confusion": {c: point[c]["confusion"].astype(int).tolist() for c in point}}
        for metric in METRICS:
            entries = {
                "transfer": summarise(point["transfer"][metric], boot["transfer"][metric], level),
                "ceiling": summarise(point["ceiling"][metric], boot["ceiling"][metric], level),
                "gap": summarise(point["ceiling"][metric] - point["transfer"][metric],
                                 boot["ceiling"][metric] - boot["transfer"][metric], level),
                "hk_in_domain": summarise(in_domain[metric], in_boot[metric], level),
                "paper_style_drop": summarise(in_domain[metric] - point["transfer"][metric],
                                              in_boot[metric] - boot["transfer"][metric], level),
            }
            result["metrics"][metric] = entries
            for condition, values in entries.items():
                rows.append({"backbone": name, "variant": args.variant, "metric": metric, "condition": condition, **values})
        for metric in CLASS_METRICS:
            result["per_class"][metric] = {
                cls: {cond: summarise(point[cond][metric][j], boot[cond][metric][:, j], level) for cond in point}
                | {"gap": summarise(point["ceiling"][metric][j] - point["transfer"][metric][j],
                                    boot["ceiling"][metric][:, j] - boot["transfer"][metric][:, j], level)}
                for j, cls in enumerate(SIX_CLASSES)}
        macro = result["metrics"]["macro_f1"]
        result["h2"] = {"transfer_ci_above_best_constant": macro["transfer"]["ci_low"] > best_constant,
                        "gap_ci_excludes_zero": macro["gap"]["ci_low"] > 0 or macro["gap"]["ci_high"] < 0}
        summary["backbones"][name] = result
        print(f"\n{name}: C (HyperKvasir) = {c_hk}, C (SAGE) = {c_sage}")
        for metric in ("macro_f1", "macro_auroc", "weighted_f1"):
            parts = [f"{cond} {v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
                     for cond, v in result["metrics"][metric].items()]
            print(f"  {metric:12s} " + "; ".join(parts))
        print(f"  H2: transfer CI above best constant ({best_constant:.4f}): {result['h2']['transfer_ci_above_best_constant']}; "
              f"gap CI excludes 0: {result['h2']['gap_ci_excludes_zero']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / f"rq1_{args.variant}_table.csv", index=False)
    (args.out_dir / f"rq1_{args.variant}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote rq1_{args.variant}_table.csv, rq1_{args.variant}_summary.json and per-image predictions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

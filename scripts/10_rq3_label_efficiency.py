"""RQ3: how many Nepali patients does a probe need?

For k SAGE training patients (seeded random draws of whole patients), two regimes are trained on
frozen features and scored on the SAGE six-class test set:

    hk_plus_k   HyperKvasir plus the k patients' images; classes balanced within each source and
                the two sources weighted equally (k = 0 is the HyperKvasir-only transfer probe)
    sage_only   the k patients' images only, classes balanced (k = all patients is the ceiling)

C values are reused from scripts/07 (HyperKvasir's C for hk_plus_k, the ceiling's C for
sage_only), so nothing is tuned on the test set. The target is the smallest k whose mean macro-F1
over draws reaches 90% of the SAGE-only ceiling. Intervals for the mean over draws come from the
same paired patient-level bootstrap as RQ1.

    python scripts/10_rq3_label_efficiency.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sage_shift.bootstrap import group_weights, percentile_ci
from sage_shift.data import read_json
from sage_shift.features import load_features
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.metrics import evaluate, macro_f1_many
from sage_shift.outdir import check_out_dir
from sage_shift.probes import balanced_weights, fit_predict_weighted

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
K = len(SIX_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 label-efficiency curves.")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--rq1", type=Path, default=ROOT / "outputs" / "rq1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "rq3")
    parser.add_argument("--variants", nargs="+", default=["raw", "fov_colour"])
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def features_for(path: Path, ids: pd.Series) -> np.ndarray:
    values, file_ids = load_features(path)
    index = {key: row for row, key in enumerate(file_ids.tolist())}
    return values[[index[key] for key in ids]].astype(np.float64)


def patient_draws(patients: np.ndarray, ks: list[int], draws: int, seed: int) -> list[tuple[int, int, np.ndarray]]:
    """(k, draw, patients) for every k; one draw when k is 0 or covers every patient."""
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        if k == 0 or k >= len(patients):
            out.append((min(k, len(patients)), 0, patients if k else patients[:0]))
            continue
        for draw in range(draws):
            out.append((k, draw, np.sort(rng.choice(patients, size=k, replace=False))))
    return out


def run_one(regime: str, x_hk: np.ndarray, y_hk: np.ndarray, x_sage: np.ndarray, y_sage: np.ndarray,
            x_test: np.ndarray, C: float, seed: int, max_iter: int) -> np.ndarray:
    """Test probabilities for one regime and one draw."""
    if regime == "hk_plus_k":
        x = np.vstack([x_hk, x_sage])
        y = np.r_[y_hk, y_sage]
        source = np.r_[np.zeros(len(y_hk)), np.ones(len(y_sage))]
    else:
        x, y, source = x_sage, y_sage, np.zeros(len(y_sage))
    if len(np.unique(y)) < 2:  # a small draw can hold one class only: predict it for every image
        return np.tile(np.eye(K)[y[0]], (len(x_test), 1))
    return fit_predict_weighted(x, y, balanced_weights(y, source), x_test, C, seed, K, max_iter)


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    rq3, seed, max_iter = cfg["rq3"], int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"])
    n_boot, level = int(cfg["bootstrap"]["n_resamples"]), float(cfg["bootstrap"]["ci"])
    class_index = {name: i for i, name in enumerate(SIX_CLASSES)}

    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    six = sage[sage["in_six"] == 1]
    train = six[six["split"] == "train"].reset_index(drop=True)
    test = six[six["split"] == "test"].reset_index(drop=True)
    y_train, y_test = train["class"].map(class_index).to_numpy(), test["class"].map(class_index).to_numpy()
    truth = test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float)
    weights = group_weights(test["patient_group"].to_numpy(), n_boot, int(cfg["seeds"]["bootstrap"]))
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    y_hk = hk["sage_class"].map(class_index).to_numpy()
    patients = np.sort(train["patient_group"].unique())
    draws = patient_draws(patients, [int(k) for k in rq3["k_patients"]], int(rq3["draws_per_k"]),
                          int(cfg["seeds"]["rq3_patient_draws"]))
    print(f"SAGE train: {len(train)} images, {len(patients)} patients; test: {len(test)} images. "
          f"{len(draws)} patient draws per regime; variants {args.variants}.")

    runs: list[dict[str, Any]] = []
    boot_rows: dict[tuple[str, str, str, int], list[np.ndarray]] = {}
    summary: dict[str, Any] = {"target_fraction": rq3["target_fraction_of_ceiling"], "results": {}}
    for variant in args.variants:
        rq1 = read_json(args.rq1 / f"rq1_{variant}_summary.json")["backbones"]
        for backbone in [b for b in BACKBONES if b in rq1 and (args.features / "hk" / b / f"{variant}.npz").is_file()]:
            x_hk = features_for(args.features / "hk" / backbone / f"{variant}.npz", hk["file_name"])
            x_train = features_for(args.features / "sage" / backbone / f"{variant}.npz", train["sample_uuid"])
            x_test = features_for(args.features / "sage" / backbone / f"{variant}.npz", test["sample_uuid"])
            c_values = {"hk_plus_k": float(rq1[backbone]["C_hyperkvasir"]), "sage_only": float(rq1[backbone]["C_sage"])}
            tasks = [(regime, k, draw, members) for regime in ("hk_plus_k", "sage_only")
                     for k, draw, members in draws if not (regime == "sage_only" and k == 0)]
            results = Parallel(n_jobs=args.jobs)(
                delayed(run_one)(regime, x_hk, y_hk, x_train[np.isin(train["patient_group"], members)],
                                 y_train[np.isin(train["patient_group"], members)], x_test, c_values[regime], seed, max_iter)
                for regime, k, draw, members in tasks)
            for (regime, k, draw, members), proba in zip(tasks, results):
                chosen = np.isin(train["patient_group"], members)
                point = evaluate(y_test, proba, truth)
                runs.append({"backbone": backbone, "variant": variant, "regime": regime, "k": k, "draw": draw,
                             "sage_images": int(chosen.sum()), "classes_covered": int(len(np.unique(y_train[chosen]))),
                             "C": c_values[regime], "macro_f1": round(point["macro_f1"], 4),
                             "macro_auroc": round(point["macro_auroc"], 4), "weighted_f1": round(point["weighted_f1"], 4)})
                boot_rows.setdefault((backbone, variant, regime, k), []).append(
                    macro_f1_many(y_test, proba.argmax(axis=1), K, weights))
            print(f"  {backbone}/{variant}: {len(tasks)} probes fitted")

    runs_frame = pd.DataFrame(runs)
    curves = []
    all_patients = len(patients)
    for (backbone, variant, regime, k), part in runs_frame.groupby(["backbone", "variant", "regime", "k"], sort=False):
        mean_boot = np.mean(boot_rows[(backbone, variant, regime, k)], axis=0)
        ceiling_boot = boot_rows[(backbone, variant, "sage_only", all_patients)][0]
        ceiling_point = runs_frame[(runs_frame["backbone"] == backbone) & (runs_frame["variant"] == variant)
                                   & (runs_frame["regime"] == "sage_only") & (runs_frame["k"] == all_patients)]["macro_f1"].iloc[0]
        low, high = percentile_ci(mean_boot, level)
        diff_low, diff_high = percentile_ci(mean_boot - ceiling_boot, level)
        curves.append({"backbone": backbone, "variant": variant, "regime": regime, "k": int(k), "draws": len(part),
                       "macro_f1_mean": round(part["macro_f1"].mean(), 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
                       "minus_ceiling": round(part["macro_f1"].mean() - ceiling_point, 4),
                       "minus_ceiling_ci_low": round(diff_low, 4), "minus_ceiling_ci_high": round(diff_high, 4),
                       "macro_f1_sd_over_draws": round(part["macro_f1"].std(ddof=1), 4) if len(part) > 1 else 0.0,
                       "macro_auroc_mean": round(part["macro_auroc"].mean(), 4),
                       "classes_covered_mean": round(part["classes_covered"].mean(), 2),
                       "sage_images_mean": round(part["sage_images"].mean(), 1)})
    curves_frame = pd.DataFrame(curves)
    for (backbone, variant), part in curves_frame.groupby(["backbone", "variant"], sort=False):
        ceiling = float(read_json(args.rq1 / f"rq1_{variant}_summary.json")["backbones"][backbone]["metrics"]["macro_f1"]["ceiling"]["estimate"])
        target = float(rq3["target_fraction_of_ceiling"]) * ceiling
        entry: dict[str, Any] = {"ceiling_macro_f1": ceiling, "target_macro_f1": round(target, 4)}
        for regime in ("hk_plus_k", "sage_only"):
            reached = part[(part["regime"] == regime) & (part["macro_f1_mean"] >= target)].sort_values("k")
            entry[f"smallest_k_{regime}"] = int(reached["k"].iloc[0]) if len(reached) else None
        above = part[(part["regime"] == "hk_plus_k") & (part["macro_f1_mean"] > ceiling)]
        entry["hk_plus_k_above_ceiling_at_k"] = above["k"].tolist()
        entry["hk_plus_k_minus_ceiling"] = {int(r.k): [r.minus_ceiling, r.minus_ceiling_ci_low, r.minus_ceiling_ci_high]
                                            for r in above.itertuples()}
        summary["results"][f"{backbone}/{variant}"] = entry
    summary["h5_supported"] = any(e["smallest_k_hk_plus_k"] is not None and e["smallest_k_hk_plus_k"] <= 20
                                  for e in summary["results"].values())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_frame.to_csv(args.out_dir / "rq3_runs.csv", index=False)
    curves_frame.to_csv(args.out_dir / "rq3_curves.csv", index=False)
    (args.out_dir / "rq3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nMean macro-F1 over draws [95% CI] by k:")
    for (backbone, variant, regime), part in curves_frame.groupby(["backbone", "variant", "regime"], sort=False):
        cells = "  ".join(f"k={r.k}: {r.macro_f1_mean:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in part.itertuples())
        print(f"  {backbone:24s} {variant:11s} {regime:10s} {cells}")
    print("\nTarget (90% of the SAGE-only ceiling) and the smallest k that reaches it:")
    for key, entry in summary["results"].items():
        print(f"  {key:36s} ceiling {entry['ceiling_macro_f1']:.3f}, target {entry['target_macro_f1']:.3f}; "
              f"HyperKvasir + k: {entry['smallest_k_hk_plus_k']}; SAGE only: {entry['smallest_k_sage_only']}; "
              f"HyperKvasir + k above the ceiling at k = {entry['hk_plus_k_above_ceiling_at_k']} "
              f"(difference [95% CI]: {entry['hk_plus_k_minus_ceiling']})")
    print(f"H5 (k <= 20 with HyperKvasir for at least one backbone): {summary['h5_supported']}")
    print(f"\nWrote rq3_runs.csv, rq3_curves.csv and rq3_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

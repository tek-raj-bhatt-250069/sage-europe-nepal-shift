"""RQ2: what explains the transfer gap, plus the paper-recipe table.

Reads the per-image predictions of scripts/07 (outputs/rq1/predictions) and scripts/08
(outputs/paper_recipe), and refits two probes from the features (the open-set check and the
format-matched source). All intervals come from the same paired patient-level bootstrap as RQ1.

    decomposition   transfer, ceiling and gap for each variant, and EM prior correction on raw
                    and fov_colour, each with its change from raw
    meaning groups  per-class F1 gaps; H3 compares the shifted (polyp, pylorus) and matched classes
    polyp context   transfer recall on SAGE polyps by location label (descriptive)
    sensitivity     without the two heaviest test patients, without NBI images, and where test
                    images outside the label space go (open-set check)
    format          exploratory: a probe trained only on HyperKvasir's older-format frames
    paper recipe    every re-run on raw frames, fov crops and masked frames

Writes aggregate tables and a summary to outputs/rq2/.

    python scripts/09_rq2_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sage_shift.baselines import constant_predictor_table
from sage_shift.bootstrap import group_weights
from sage_shift.data import read_json
from sage_shift.features import load_features
from sage_shift.labelspace import MEANING_GROUP, SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.outdir import check_out_dir
from sage_shift.probes import fit_predict, official_folds, select_c
from sage_shift.rq2 import boot_scores, ci, meaning_gap_difference, saerens_em

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
VARIANTS = ["raw", "fov", "colour", "fov_colour"]
EM_VARIANTS = ["raw", "fov_colour"]
CHECK_VARIANTS = ["raw", "fov_colour"]
K = len(SIX_CLASSES)
POLYP = SIX_CLASSES.index("polyp")
SHIFTED = np.array([MEANING_GROUP[name] == "shifted" for name in SIX_CLASSES])
UNIFORM = np.full(K, 1 / K)  # the prior implied by balanced class weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ2 decomposition, meaning groups, checks and the paper-recipe table.")
    parser.add_argument("--rq1", type=Path, default=ROOT / "outputs" / "rq1")
    parser.add_argument("--paper-recipe", type=Path, default=ROOT / "outputs" / "paper_recipe")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--hk-quality", type=Path, default=ROOT / "outputs" / "preprocess" / "hk_fov_quality.csv",
                        help="per-image HyperKvasir field table from scripts/05 (frame widths)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "rq2")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


class Table:
    """Long-format results: one row per (section, backbone or run, row, metric)."""

    def __init__(self, level: float) -> None:
        self.rows: list[dict[str, Any]] = []
        self.level = level

    def add(self, section: str, model: str, row: str, metric: str, point: float, values: np.ndarray | None = None,
            **extra: Any) -> dict[str, Any]:
        entry = ci(point, values, self.level) if values is not None else {"estimate": round(float(point), 4),
                                                                         "ci_low": None, "ci_high": None}
        self.rows.append({"section": section, "model": model, "row": row, "metric": metric, **entry, **extra})
        return entry


def aligned(path: Path, ids: pd.Series) -> np.ndarray:
    """Probabilities (six columns) in the order of `ids`."""
    frame = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid")
    missing = set(ids) - set(frame.index)
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} test images have no prediction")
    return frame.loc[list(ids), list(SIX_CLASSES)].to_numpy(dtype=float)


def feature_rows(path: Path, ids: pd.Series) -> np.ndarray:
    features, file_ids = load_features(path)
    index = {key: row for row, key in enumerate(file_ids.tolist())}
    return features[[index[key] for key in ids]].astype(np.float64)


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    n_boot, level, boot_seed = int(cfg["bootstrap"]["n_resamples"]), float(cfg["bootstrap"]["ci"]), int(cfg["seeds"]["bootstrap"])
    seed, max_iter, grid = int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"]), cfg["probe"]["C_grid"]
    table = Table(level)
    summary: dict[str, Any] = {"bootstrap": cfg["bootstrap"], "classes": list(SIX_CLASSES)}

    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    test = sage[(sage["in_six"] == 1) & (sage["split"] == "test")].reset_index(drop=True)
    class_index = {name: i for i, name in enumerate(SIX_CLASSES)}
    y = test["class"].map(class_index).to_numpy()
    truth = test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float)
    weights = group_weights(test["patient_group"].to_numpy(), n_boot, boot_seed)
    ids = test["sample_uuid"]
    pred = args.rq1 / "predictions"
    backbones = [b for b in BACKBONES if all((pred / f"{b}__{v}__{c}.csv").is_file()
                                             for v in VARIANTS for c in ("transfer", "ceiling"))]
    print(f"SAGE six-class test: {len(test)} images, {test['patient_group'].nunique()} patients; "
          f"{n_boot} paired patient resamples; backbones: {backbones}")

    # Decomposition (T5) and meaning groups (T4, H3).
    summary["decomposition"], summary["h3"], summary["em_prior"] = {}, {}, {}
    for backbone in backbones:
        runs: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            transfer = aligned(pred / f"{backbone}__{variant}__transfer.csv", ids)
            runs[variant] = {"proba": transfer, "transfer": boot_scores(y, transfer, truth, weights, POLYP),
                             "ceiling": boot_scores(y, aligned(pred / f"{backbone}__{variant}__ceiling.csv", ids),
                                                    truth, weights, POLYP)}
        for variant in EM_VARIANTS:
            proba = runs[variant]["proba"]
            runs[f"{variant}+EM"] = {"transfer": boot_scores(y, proba, truth, weights, POLYP, em_prior=UNIFORM),
                                     "ceiling": runs[variant]["ceiling"]}
            prior = saerens_em(proba, UNIFORM)[1]
            summary["em_prior"][f"{backbone}/{variant}"] = dict(zip(SIX_CLASSES, np.round(prior, 3).tolist()))
        (raw_tp, raw_tb), (raw_cp, raw_cb) = runs["raw"]["transfer"], runs["raw"]["ceiling"]
        out: dict[str, Any] = {}
        for row, run in runs.items():
            (tp, tb), (cp, cb) = run["transfer"], run["ceiling"]
            out[row] = {}
            for metric in ("macro_f1", "macro_auroc"):
                gap_p, gap_b = cp[metric] - tp[metric], cb[metric] - tb[metric]
                raw_gap_p, raw_gap_b = raw_cp[metric] - raw_tp[metric], raw_cb[metric] - raw_tb[metric]
                out[row][metric] = {
                    "transfer": table.add("decomposition", backbone, row, f"transfer_{metric}", tp[metric], tb[metric]),
                    "ceiling": table.add("decomposition", backbone, row, f"ceiling_{metric}", cp[metric], cb[metric]),
                    "gap": table.add("decomposition", backbone, row, f"gap_{metric}", gap_p, gap_b),
                    "transfer_change": table.add("decomposition", backbone, row, f"transfer_change_{metric}",
                                                 tp[metric] - raw_tp[metric], tb[metric] - raw_tb[metric]),
                    "gap_reduction": table.add("decomposition", backbone, row, f"gap_reduction_{metric}",
                                               raw_gap_p - gap_p, raw_gap_b - gap_b)}
            out[row]["polyp_share"] = table.add("decomposition", backbone, row, "transfer_polyp_share",
                                                tp["polyp_share"], tb["polyp_share"])
        summary["decomposition"][backbone] = out
        for variant in VARIANTS:
            (tp, tb), (cp, cb) = runs[variant]["transfer"], runs[variant]["ceiling"]
            for j, name in enumerate(SIX_CLASSES):
                table.add("meaning_groups", backbone, variant, f"f1_gap_{name}", cp["f1"][j] - tp["f1"][j],
                          cb["f1"][:, j] - tb["f1"][:, j], group=MEANING_GROUP[name])
            difference = table.add("meaning_groups", backbone, variant, "shifted_minus_matched_f1_gap",
                                   meaning_gap_difference(cp["f1"], tp["f1"], SHIFTED),
                                   meaning_gap_difference(cb["f1"], tb["f1"], SHIFTED))
            summary["h3"][f"{backbone}/{variant}"] = {**difference, "ci_excludes_zero":
                                                      difference["ci_low"] > 0 or difference["ci_high"] < 0}
    imagenet = summary["decomposition"].get("resnet50_imagenet", {})
    summary["h4_imagenet"] = {row: imagenet[row]["macro_f1"]["gap_reduction"] for row in ("fov", "colour") if row in imagenet}
    summary["h4_imagenet"]["supported"] = bool(imagenet) and all(
        imagenet[row]["macro_f1"]["gap_reduction"]["ci_low"] > 0 for row in ("fov", "colour"))
    summary["h4_imagenet"]["gastronet_available"] = "resnet50_gastronet_dino" in backbones

    # Polyp context: recall on test polyps by location label (descriptive).
    polyp_rows = []
    polyps = test["class"] == "polyp"
    for backbone in backbones:
        for variant in CHECK_VARIANTS:
            for condition in ("transfer", "ceiling"):
                hit = aligned(pred / f"{backbone}__{variant}__{condition}.csv", ids).argmax(axis=1) == POLYP
                for context, part in test[polyps].groupby("location_context"):
                    polyp_rows.append({"model": backbone, "variant": variant, "condition": condition,
                                       "location": context, "images": len(part),
                                       "patients": int(part["patient_group"].nunique()),
                                       "recalled": int(hit[part.index].sum())})
    polyp_context = pd.DataFrame(polyp_rows)

    # Sensitivity: drop the two heaviest test patients, drop NBI images.
    heaviest = test["patient_group"].value_counts().index[:2]
    checks = {"without_two_heaviest_patients": ~test["patient_group"].isin(heaviest).to_numpy(),
              "without_nbi": test["nbi"].astype(int).to_numpy() == 0}
    summary["sensitivity_sizes"] = {name: {"images": int(keep.sum()), "patients": int(test.loc[keep, "patient_group"].nunique())}
                                    for name, keep in checks.items()}
    for name, keep in checks.items():
        part = test[keep].reset_index(drop=True)
        w = group_weights(part["patient_group"].to_numpy(), n_boot, boot_seed)
        for backbone in backbones:
            for variant in CHECK_VARIANTS:
                scored = {c: boot_scores(y[keep], aligned(pred / f"{backbone}__{variant}__{c}.csv", ids)[keep],
                                         truth[keep], w, POLYP) for c in ("transfer", "ceiling")}
                (tp, tb), (cp, cb) = scored["transfer"], scored["ceiling"]
                table.add("sensitivity", backbone, f"{variant}/{name}", "transfer_macro_f1", tp["macro_f1"], tb["macro_f1"])
                table.add("sensitivity", backbone, f"{variant}/{name}", "gap_macro_f1",
                          cp["macro_f1"] - tp["macro_f1"], cb["macro_f1"] - tb["macro_f1"])

    # Open-set check: refit each transfer probe and score every SAGE test image.
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    y_hk, hk_fold = hk["sage_class"].map(class_index).to_numpy(), hk["fold2"].astype(int).to_numpy()
    all_test = sage[sage["split"] == "test"].reset_index(drop=True)
    groups = {"six_class": all_test["in_six"].to_numpy() == 1,
              "outside_label_space_with_labels": (all_test["in_six"].to_numpy() == 0) & (all_test["n_labels"].astype(int) > 0).to_numpy(),
              "zero_label": all_test["n_labels"].astype(int).to_numpy() == 0}
    open_rows, summary["open_set_check"] = [], {}
    for backbone in backbones:
        for variant in CHECK_VARIANTS:
            rq1 = read_json(args.rq1 / f"rq1_{variant}_summary.json")["backbones"][backbone]
            x_hk = feature_rows(args.features / "hk" / backbone / f"{variant}.npz", hk["file_name"])
            x_all = feature_rows(args.features / "sage" / backbone / f"{variant}.npz", all_test["sample_uuid"])
            proba = fit_predict(x_hk, y_hk, x_all, float(rq1["C_hyperkvasir"]), seed, K, max_iter)
            six = groups["six_class"]
            saved = aligned(pred / f"{backbone}__{variant}__transfer.csv", all_test.loc[six, "sample_uuid"])
            summary["open_set_check"][f"{backbone}/{variant}"] = {"max_diff_to_rq1_predictions": float(np.abs(proba[six] - saved).max())}
            for group, mask in groups.items():
                counts = np.bincount(proba[mask].argmax(axis=1), minlength=K)
                open_rows.append({"model": backbone, "variant": variant, "group": group, "images": int(mask.sum()),
                                  **{f"share_{name}": round(float(n / mask.sum()), 4) for name, n in zip(SIX_CLASSES, counts)}})
    open_set = pd.DataFrame(open_rows)

    # Exploratory: transfer probe trained on older-format HyperKvasir frames only.
    widths = pd.read_csv(args.hk_quality, usecols=["id", "width"]).set_index("id")["width"]
    older = (hk["file_name"].map(widths) < 700).to_numpy()
    summary["format_matched"] = {"older_format_hk_images": int(older.sum()),
                                 "by_class": hk.loc[older, "sage_class"].value_counts().to_dict()}
    for backbone in backbones:
        for variant in ("raw", "fov"):
            x_hk = feature_rows(args.features / "hk" / backbone / f"{variant}.npz", hk["file_name"])
            x_test = feature_rows(args.features / "sage" / backbone / f"{variant}.npz", ids)
            c_older = select_c(x_hk[older], y_hk[older], official_folds(hk_fold[older]), grid, seed, K, args.jobs, max_iter)[0]
            proba = fit_predict(x_hk[older], y_hk[older], x_test, c_older, seed, K, max_iter)
            tp, tb = boot_scores(y, proba, truth, weights, POLYP)
            fp, fb = boot_scores(y, aligned(pred / f"{backbone}__{variant}__transfer.csv", ids), truth, weights, POLYP)
            for metric in ("macro_f1", "polyp_share"):
                table.add("format_matched", backbone, variant, f"older_format_{metric}", tp[metric], tb[metric], C=c_older)
                table.add("format_matched", backbone, variant, f"change_vs_all_hk_{metric}", tp[metric] - fp[metric],
                          tb[metric] - fb[metric])

    # Paper-recipe re-runs (T2): every run and input, paired changes, means over seeds.
    collapse = float(cfg["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"])
    table.add("paper_recipe", "paper", "table6", "weighted_f1", cfg["paper_table6"]["hyperkvasir"]["resnet50"]["sage"])
    constants = constant_predictor_table(test["class"].tolist(), SIX_CLASSES)
    for _, row in constants.iterrows():
        for metric in ("weighted_f1", "macro_f1"):
            table.add("paper_recipe", "constant", f"always_{row['predict_always']}", metric, float(row[metric]))
    runs_found = sorted(path.name.removesuffix("_summary.json") for path in args.paper_recipe.glob("*_seed*_summary.json")
                        if re.fullmatch(r"(faithful|adjusted)_seed\d+", path.name.removesuffix("_summary.json")))
    inputs = {"raw": "sage_probs", "fov": "sage_fov_probs", "masked": "sage_masked_probs"}
    boot_by_run: dict[str, dict[str, Any]] = {}
    summary["paper_recipe"] = {}
    for tag in runs_found:
        info = read_json(args.paper_recipe / f"{tag}_summary.json")
        boot_by_run[tag] = {}
        for name, stem in inputs.items():
            path = args.paper_recipe / f"{tag}_{stem}.csv"
            if not path.is_file():
                continue
            proba = aligned(path, ids)
            point, boot = boot_scores(y, proba, truth, weights, POLYP)
            boot_by_run[tag][name] = (point, boot)
            top_share = float(np.bincount(proba.argmax(axis=1), minlength=K).max() / len(proba))
            for metric in ("weighted_f1", "macro_f1", "macro_auroc", "polyp_share"):
                table.add("paper_recipe", tag, name, metric, point[metric], boot[metric],
                          epochs=info["epochs"], best_epoch=info["best_epoch"])
            summary["paper_recipe"][f"{tag}/{name}"] = {"top_class_share": round(top_share, 4),
                                                        "collapse": top_share >= collapse,
                                                        "macro_f1": round(point["macro_f1"], 4)}
        raw_point, raw_boot = boot_by_run[tag]["raw"]
        for name in ("fov", "masked"):
            if name in boot_by_run[tag]:
                point, boot = boot_by_run[tag][name]
                for metric in ("macro_f1", "polyp_share"):
                    table.add("paper_recipe", tag, f"{name}_minus_raw", metric, point[metric] - raw_point[metric],
                              boot[metric] - raw_boot[metric])
    for recipe in ("faithful", "adjusted"):
        for name in ("raw", "fov", "masked"):
            tags = [t for t in runs_found if t.startswith(recipe) and name in boot_by_run[t]]
            if not tags:
                continue
            points = np.array([boot_by_run[t][name][0]["macro_f1"] for t in tags])
            mean_boot = np.mean([boot_by_run[t][name][1]["macro_f1"] for t in tags], axis=0)
            table.add("paper_recipe", f"{recipe}_mean_over_seeds", name, "macro_f1", points.mean(), mean_boot,
                      seeds=len(tags), seed_sd=round(float(points.std(ddof=1)) if len(tags) > 1 else 0.0, 4))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table.rows).to_csv(args.out_dir / "rq2_table.csv", index=False)
    polyp_context.to_csv(args.out_dir / "rq2_polyp_context.csv", index=False)
    open_set.to_csv(args.out_dir / "rq2_open_set.csv", index=False)
    (args.out_dir / "rq2_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    report(pd.DataFrame(table.rows), summary, polyp_context, open_set)
    print(f"\nWrote rq2_table.csv, rq2_polyp_context.csv, rq2_open_set.csv and rq2_summary.json to {args.out_dir}")
    return 0


def fmt(entry: pd.Series) -> str:
    if pd.isna(entry["ci_low"]):
        return f"{entry['estimate']:.3f}"
    return f"{entry['estimate']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def report(rows: pd.DataFrame, summary: dict[str, Any], polyp_context: pd.DataFrame, open_set: pd.DataFrame) -> None:
    """Compact printout of the main tables."""
    def pick(section: str, model: str, row: str, metric: str) -> str:
        hit = rows[(rows["section"] == section) & (rows["model"] == model) & (rows["row"] == row) & (rows["metric"] == metric)]
        return fmt(hit.iloc[0]) if len(hit) else "-"
    print("\nDecomposition (macro-F1): transfer | gap | gap reduction vs raw | polyp share")
    for model in summary["decomposition"]:
        print(f"  {model}")
        for row in summary["decomposition"][model]:
            print(f"    {row:16s} {pick('decomposition', model, row, 'transfer_macro_f1'):24s} "
                  f"{pick('decomposition', model, row, 'gap_macro_f1'):24s} "
                  f"{pick('decomposition', model, row, 'gap_reduction_macro_f1'):26s} "
                  f"{pick('decomposition', model, row, 'transfer_polyp_share')}")
    print("\nH3 (shifted minus matched mean per-class F1 gap):")
    for key, value in summary["h3"].items():
        print(f"  {key:30s} {value['estimate']:+.3f} [{value['ci_low']:+.3f}, {value['ci_high']:+.3f}]"
              f"  CI excludes 0: {value['ci_excludes_zero']}")
    print(f"\nH4, ImageNet ResNet-50 gap reduction: fov {summary['h4_imagenet'].get('fov')}, "
          f"colour {summary['h4_imagenet'].get('colour')}; supported: {summary['h4_imagenet']['supported']}")
    print("\nEM-estimated SAGE class priors (true test shares: "
          + ", ".join(f"{c[:5]} {s}" for c, s in zip(SIX_CLASSES, [15, 16, 43, 57, 10, 17])) + " of 158):")
    for key, prior in summary["em_prior"].items():
        print(f"  {key:30s} " + " ".join(f"{c[:5]} {v:.2f}" for c, v in prior.items()))
    print("\nSensitivity (transfer macro-F1 | gap):")
    for model in summary["decomposition"]:
        for row in sorted(set(rows[(rows["section"] == "sensitivity") & (rows["model"] == model)]["row"])):
            print(f"  {model:24s} {row:44s} {pick('sensitivity', model, row, 'transfer_macro_f1'):24s} "
                  f"{pick('sensitivity', model, row, 'gap_macro_f1')}")
    print("\nPolyp context (transfer probe; recalled / images):")
    sub = polyp_context[polyp_context["condition"] == "transfer"]
    for (model, variant), part in sub.groupby(["model", "variant"], sort=False):
        print(f"  {model:24s} {variant:11s} " + "; ".join(f"{r.location}: {r.recalled}/{r.images}" for r in part.itertuples()))
    print("\nOpen-set check: share predicted polyp by group")
    for (model, variant), part in open_set.groupby(["model", "variant"], sort=False):
        print(f"  {model:24s} {variant:11s} " + "; ".join(f"{r.group} ({r.images}): {r.share_polyp:.2f}" for r in part.itertuples()))
    print("\nFormat-matched source (older-format HyperKvasir only), change vs all HyperKvasir:")
    for model in summary["decomposition"]:
        for variant in ("raw", "fov"):
            print(f"  {model:24s} {variant:4s} macro-F1 {pick('format_matched', model, variant, 'older_format_macro_f1'):24s} "
                  f"change {pick('format_matched', model, variant, 'change_vs_all_hk_macro_f1'):24s} "
                  f"polyp share change {pick('format_matched', model, variant, 'change_vs_all_hk_polyp_share')}")
    print("\nPaper recipe (macro-F1; polyp share):")
    for key, value in summary["paper_recipe"].items():
        tag, name = key.split("/")
        print(f"  {key:26s} {pick('paper_recipe', tag, name, 'macro_f1'):24s} "
              f"{pick('paper_recipe', tag, name, 'polyp_share'):24s} collapse {value['collapse']}")
    for recipe in ("faithful", "adjusted"):
        for name in ("raw", "fov", "masked"):
            hit = rows[(rows["model"] == f"{recipe}_mean_over_seeds") & (rows["row"] == name)]
            if len(hit):
                r = hit.iloc[0]
                print(f"  mean of {int(r['seeds'])} seeds, {recipe:8s} {name:6s} {fmt(r)}  (seed SD {r['seed_sd']})")


if __name__ == "__main__":
    raise SystemExit(main())

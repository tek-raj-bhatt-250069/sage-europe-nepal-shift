"""Exploratory geometry and colour analyses of the GPU outputs (laptop, CPU).

    probes       HyperKvasir raw-trained transfer probes (at the RQ1 raw C) scored on SAGE test under
                 raw, fov and the geometry inputs, with paired differences vs raw and vs masked
    hk probe     a probe trained on HyperKvasir fold-0 raw features, scored on fold-1 raw, sageify
                 and sageify_grey features
    colour       HyperKvasir raw -> SAGE colour_global and HyperKvasir fov -> SAGE fov_colour_global,
                 with SAGE-only references refitted on those inputs (5-fold patient-grouped CV over
                 the C grid): gap, transfer change and gap reduction vs raw / fov
    fine-tuned   the kept re-runs scored by scripts/14_geometry_controls.py, per seed and input
    verdicts     the reading rules in sage_shift.geometry.RULES, fixed before the runs

SAGE intervals come from the paired patient-level bootstrap of RQ1. HyperKvasir has no patient
IDs, so its intervals come from an image-level bootstrap (bootstrap seed + 1). Missing inputs are
skipped and listed. Writes aggregate tables and a summary to outputs/exploratory_gpu/.

    python scripts/15_geometry_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from sage_shift.bootstrap import group_weights
from sage_shift.data import read_json
from sage_shift.geometry import (HK_INPUTS, LOWER_GI, PAIRS, RULES, SAGE_INPUTS, hk_scores, pair_verdict, sage_scores,
                                 shortcut_shown)
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.outdir import check_out_dir
from sage_shift.probes import grouped_folds, make_probe, select_c
from sage_shift.rq2 import ci

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
K = len(SIX_CLASSES)
CLASS_INDEX = {name: i for i, name in enumerate(SIX_CLASSES)}
SAGE_METRICS = ("macro_f1", "macro_auroc", "polyp_share", "top_class_share")
HK_METRICS = (*(f"accuracy_{name}" for name in SIX_CLASSES), "macro_f1", "upper_gi_polyp_or_rectum_share",
              "hd_upper_gi_polyp_or_rectum_share")
COLOUR_ROWS = (("raw", "colour_global"), ("fov", "fov_colour_global"))  # (baseline, new input) for SAGE
FINETUNED_DIFFERENCES = (("masked_grey", "masked"), ("letterbox_grey", "letterbox"), ("fov_grey", "fov"),
                         ("fov", "masked"))
HD_WIDTH = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory geometry and colour analyses (probes and kept re-runs).")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--rq1", type=Path, default=ROOT / "outputs" / "rq1", help="RQ1 summaries (chosen C values)")
    parser.add_argument("--geometry", type=Path, default=ROOT / "outputs" / "paper_recipe_keep",
                        help="outputs of scripts/14_geometry_controls.py")
    parser.add_argument("--hk-quality", type=Path, default=ROOT / "outputs" / "preprocess" / "hk_fov_quality.csv",
                        help="per-image HyperKvasir field table from scripts/05 (frame widths)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "exploratory_gpu")
    parser.add_argument("--backbones", nargs="+", help="default: every backbone with raw features for both datasets")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


class Table:
    """Long-format results: one row per (section, model, row, metric) with a percentile interval."""

    def __init__(self, level: float) -> None:
        self.rows: list[dict[str, Any]] = []
        self.level = level

    def add(self, section: str, model: str, row: str, metric: str, point: float, values: np.ndarray,
            **extra: Any) -> dict[str, Any]:
        entry = ci(point, values, self.level) if np.isfinite(point) else {"estimate": None, "ci_low": None,
                                                                          "ci_high": None}
        self.rows.append({"section": section, "model": model, "row": row, "metric": metric, **entry, **extra})
        return entry


class Probes:
    """Feature lookup and fitted probes for one backbone (fits are reused)."""

    def __init__(self, args: argparse.Namespace, backbone: str, seed: int, max_iter: int) -> None:
        self.args, self.backbone, self.seed, self.max_iter = args, backbone, seed, max_iter
        self.fitted: dict[tuple[str, ...], Any] = {}

    def features(self, dataset: str, name: str, ids: pd.Series) -> np.ndarray | None:
        """Feature rows in the order of `ids`, or None if the file is missing."""
        path = self.args.features / dataset / self.backbone / f"{name}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            values, file_ids = data["features"], data["ids"]
        index = {key: row for row, key in enumerate(file_ids.tolist())}
        missing = [key for key in ids if key not in index]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} images have no features")
        return values[[index[key] for key in ids]].astype(np.float64)

    def fit(self, key: tuple[str, ...], x: np.ndarray, y: np.ndarray, C: float) -> Any:
        if key not in self.fitted:
            self.fitted[key] = make_probe(C, self.seed, self.max_iter).fit(x, y)
        return self.fitted[key]


def predict(model: Any, x: np.ndarray) -> np.ndarray:
    """Probabilities for all six classes (zero for classes unseen in training)."""
    proba = np.zeros((len(x), K))
    proba[:, model.classes_] = model.predict_proba(x)
    return proba


def aligned(path: Path, key: str, ids: pd.Series) -> dict[str, np.ndarray]:
    """Per-input probabilities from a long table of scripts/14, in the order of `ids`."""
    table = pd.read_csv(path, dtype={key: str})
    out = {}
    for name, part in table.groupby("input", sort=False):
        part = part.set_index(key)
        missing = set(ids) - set(part.index)
        if missing:
            raise SystemExit(f"{path.name}: {len(missing)} images have no {name} prediction")
        out[str(name)] = part.loc[list(ids), list(SIX_CLASSES)].to_numpy(dtype=float)
    return out


def sage_block(table: Table, section: str, model: str, probs: dict[str, np.ndarray], y: np.ndarray,
               truth: np.ndarray, weights: np.ndarray, metrics: tuple[str, ...] = SAGE_METRICS,
               **extra: Any) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]]:
    """Scores per input with intervals; returns the summary entries and the raw resamples."""
    scored = {name: sage_scores(y, proba, truth, weights) for name, proba in probs.items()}
    out = {name: {"top_class": point["top_class"],
                  **{metric: table.add(section, model, name, metric, point[metric], boot[metric], **extra)
                     for metric in metrics}}
           for name, (point, boot) in scored.items()}
    return out, scored


def difference(table: Table, section: str, model: str, scored: dict[str, Any], first: str, second: str,
               metrics: tuple[str, ...], **extra: Any) -> dict[str, Any]:
    """Paired difference first - second on the same resamples."""
    (p1, b1), (p2, b2) = scored[first], scored[second]
    return {metric: table.add(section, model, f"{first}_minus_{second}", metric, p1[metric] - p2[metric],
                              b1[metric] - b2[metric], **extra) for metric in metrics}


def hk_block(table: Table, section: str, model: str, preds: dict[str, np.ndarray], y: np.ndarray, upper: np.ndarray,
             hd: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """HyperKvasir fold-1 counterfactual: scores per input, paired differences and the source-side rule."""
    scored = {name: hk_scores(y, pred, upper, hd, weights) for name, pred in preds.items()}
    out: dict[str, Any] = {"inputs": {name: {metric: table.add(section, model, name, metric, point[metric], boot[metric])
                                             for metric in HK_METRICS} for name, (point, boot) in scored.items()},
                           "differences": {}}
    for name in ("sageify", "sageify_grey"):
        if name in scored and "raw" in scored:
            out["differences"][f"{name}_minus_raw"] = difference(table, section, model, scored, name, "raw", HK_METRICS)
    if not {"raw", "sageify", "sageify_grey"} <= set(scored):
        return out
    # (sageify - raw) - (sageify_grey - raw), on the same resamples
    (ps, bs), (pg, bg) = scored["sageify"], scored["sageify_grey"]
    out["differences"]["increase_difference"] = {
        metric: table.add(section, model, "increase_difference", metric, ps[metric] - pg[metric], bs[metric] - bg[metric])
        for metric in HK_METRICS}
    rule = {}
    for share in ("upper_gi_polyp_or_rectum_share", "hd_upper_gi_polyp_or_rectum_share"):
        increase = out["differences"]["sageify_minus_raw"][share]
        diff = out["differences"]["increase_difference"][share]
        rule[share] = {"sageify_minus_raw": increase,
                       "sageify_grey_minus_raw": out["differences"]["sageify_grey_minus_raw"][share],
                       "increase_difference": diff,
                       "shown": None if increase["estimate"] is None or diff["ci_low"] is None
                       else shortcut_shown(increase, diff)}
    out["source_shortcut"] = {"rule": RULES["source_shortcut"], "shown": rule["upper_gi_polyp_or_rectum_share"]["shown"],
                              "all_upper_gi": rule["upper_gi_polyp_or_rectum_share"],
                              "hd_only_descriptive": rule["hd_upper_gi_polyp_or_rectum_share"]}
    return out


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    n_boot, level, boot_seed = int(cfg["bootstrap"]["n_resamples"]), float(cfg["bootstrap"]["ci"]), int(cfg["seeds"]["bootstrap"])
    seed, max_iter, grid = int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"]), cfg["probe"]["C_grid"]
    collapse = float(cfg["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"])
    table, skipped = Table(level), []

    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    six = sage[sage["in_six"].astype(int) == 1]
    train = six[six["split"] == "train"].reset_index(drop=True)
    test = six[six["split"] == "test"].reset_index(drop=True)
    y_train, y_test = train["class"].map(CLASS_INDEX).to_numpy(), test["class"].map(CLASS_INDEX).to_numpy()
    truth = test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float)
    weights = group_weights(test["patient_group"].to_numpy(), n_boot, boot_seed)
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    y_hk, fold = hk["sage_class"].map(CLASS_INDEX).to_numpy(), hk["fold2"].astype(int).to_numpy()
    fold1 = hk[fold == 1].reset_index(drop=True)
    y_fold1 = y_hk[fold == 1]
    widths = pd.read_csv(args.hk_quality, usecols=["id", "width"], dtype={"id": str}).set_index("id")["width"]
    width = fold1["file_name"].map(widths)
    if width.isna().any():
        raise SystemExit(f"{args.hk_quality.name}: {int(width.isna().sum())} fold-1 images have no width")
    upper, hd = ~np.isin(y_fold1, LOWER_GI), (width >= HD_WIDTH).to_numpy()
    hk_weights = group_weights(np.arange(len(fold1)), n_boot, boot_seed + 1)
    rq1 = {variant: read_json(args.rq1 / f"rq1_{variant}_summary.json")["backbones"] for variant in ("raw", "fov")
           if (args.rq1 / f"rq1_{variant}_summary.json").is_file()}
    print(f"SAGE six-class test: {len(test)} images, {test['patient_group'].nunique()} patients; train {len(train)} "
          f"images; HyperKvasir fold 1: {len(fold1)} images ({int(upper.sum())} upper GI, {int((upper & hd).sum())} of "
          f"them HD); {n_boot} resamples")
    summary: dict[str, Any] = {"bootstrap": {**cfg["bootstrap"], "hk": "image-level, seed + 1"},
                               "classes": list(SIX_CLASSES), "rules": RULES, "probes": {}, "fine_tuned": {}}

    # (i) Probe level.
    names = args.backbones or [b for b in BACKBONES if all(
        (args.features / d / b / "raw.npz").is_file() for d in ("sage", "hk"))]
    for backbone in names:
        if backbone not in rq1.get("raw", {}):
            skipped.append(f"probes/{backbone}: no RQ1 raw summary")
            continue
        probes = Probes(args, backbone, seed, max_iter)
        x_hk = {"raw": probes.features("hk", "raw", hk["file_name"])}
        if x_hk["raw"] is None:
            skipped.append(f"probes/{backbone}: no HyperKvasir raw features")
            continue
        c_raw = float(rq1["raw"][backbone]["C_hyperkvasir"])
        transfer = probes.fit(("hk", "raw"), x_hk["raw"], y_hk, c_raw)
        result: dict[str, Any] = {"C_hyperkvasir_raw": c_raw}
        x_test = {name: probes.features("sage", name, test["sample_uuid"]) for name in SAGE_INPUTS}
        probs = {name: predict(transfer, x) for name, x in x_test.items() if x is not None}
        skipped += [f"probes/{backbone}/sage/{name}: no features" for name, x in x_test.items() if x is None]
        saved = args.rq1 / "predictions" / f"{backbone}__raw__transfer.csv"
        if "raw" in probs and saved.is_file():
            frame = pd.read_csv(saved, dtype={"sample_uuid": str}).set_index("sample_uuid")
            result["max_diff_to_rq1_raw_predictions"] = round(float(np.abs(
                frame.loc[list(test["sample_uuid"]), list(SIX_CLASSES)].to_numpy() - probs["raw"]).max()), 6)
        result["sage"], scored = sage_block(table, "probe_sage", backbone, probs, y_test, truth, weights)
        for base in ("raw", "masked"):
            if base in scored:
                result[f"sage_minus_{base}"] = {name: difference(table, "probe_sage", backbone, scored, name, base,
                                                                 SAGE_METRICS) for name in scored if name != base}

        x_fold1 = {name: probes.features("hk", name, fold1["file_name"]) for name in HK_INPUTS}
        if x_fold1["sageify"] is not None or x_fold1["sageify_grey"] is not None:
            fold0 = probes.fit(("hk", "raw", "fold0"), x_hk["raw"][fold == 0], y_hk[fold == 0], c_raw)
            preds = {name: predict(fold0, x).argmax(axis=1) for name, x in x_fold1.items() if x is not None}
            result["hk_fold1"] = hk_block(table, "probe_hk", backbone, preds, y_fold1, upper, hd, hk_weights)
        else:
            skipped.append(f"probes/{backbone}/hk: no sageify features")

        result["colour_global"] = {}
        for base, new in COLOUR_ROWS:
            x_new_train = probes.features("sage", new, train["sample_uuid"])
            x_new_test = probes.features("sage", new, test["sample_uuid"])
            if x_new_train is None or base not in rq1 or backbone not in rq1[base]:
                skipped.append(f"probes/{backbone}/{new}: no features or no RQ1 {base} summary")
                continue
            x_hk_base = x_hk.get(base)
            if x_hk_base is None:
                x_hk_base = x_hk[base] = probes.features("hk", base, hk["file_name"])
            x_base_train = probes.features("sage", base, train["sample_uuid"])
            x_base_test = probes.features("sage", base, test["sample_uuid"])
            if x_hk_base is None or x_base_train is None:
                skipped.append(f"probes/{backbone}/{new}: no {base} features")
                continue
            hk_probe = probes.fit(("hk", base), x_hk_base, y_hk, float(rq1[base][backbone]["C_hyperkvasir"]))
            reference_base = probes.fit(("sage", base), x_base_train, y_train, float(rq1[base][backbone]["C_sage"]))
            folds = grouped_folds(y_train, train["patient_group"].to_numpy(), 5, seed)
            c_new, cv_new, _ = select_c(x_new_train, y_train, folds, grid, seed, K, args.jobs, max_iter)
            reference_new = probes.fit(("sage", new), x_new_train, y_train, c_new)
            colour_probs = {f"transfer_{base}": predict(hk_probe, x_base_test), f"transfer_{new}": predict(hk_probe, x_new_test),
                            f"reference_{base}": predict(reference_base, x_base_test),
                            f"reference_{new}": predict(reference_new, x_new_test)}
            entries, cs = sage_block(table, "colour_global", backbone, colour_probs, y_test, truth, weights)
            row: dict[str, Any] = {"C_sage_reference_new": c_new, "cv_macro_f1_new": cv_new, "scores": entries}
            for metric in ("macro_f1", "macro_auroc"):
                gaps = {name: (cs[f"reference_{name}"][0][metric] - cs[f"transfer_{name}"][0][metric],
                               cs[f"reference_{name}"][1][metric] - cs[f"transfer_{name}"][1][metric]) for name in (base, new)}
                row[metric] = {
                    f"gap_{base}": table.add("colour_global", backbone, f"gap_{base}", metric, *gaps[base]),
                    f"gap_{new}": table.add("colour_global", backbone, f"gap_{new}", metric, *gaps[new]),
                    "transfer_change": table.add("colour_global", backbone, f"transfer_change_{new}_vs_{base}", metric,
                                                 cs[f"transfer_{new}"][0][metric] - cs[f"transfer_{base}"][0][metric],
                                                 cs[f"transfer_{new}"][1][metric] - cs[f"transfer_{base}"][1][metric]),
                    "gap_reduction": table.add("colour_global", backbone, f"gap_reduction_{new}_vs_{base}", metric,
                                               gaps[base][0] - gaps[new][0], gaps[base][1] - gaps[new][1])}
            result["colour_global"][new] = row
        summary["probes"][backbone] = result
        print(f"  probes done: {backbone}")

    # (ii) Fine-tuned re-runs scored by scripts/14.
    tags = sorted(path.name.removesuffix("_geometry_summary.json") for path in args.geometry.glob("*_geometry_summary.json"))
    runs_rows, collapse_flags, shortcuts = [], {}, {}
    for tag in tags:
        info = read_json(args.geometry / f"{tag}_geometry_summary.json")
        recipe = info.get("recipe") or re.sub(r"_seed\d+$", "", tag)
        entry: dict[str, Any] = {"recipe": recipe, "seed": info.get("seed"), "best_epoch": info.get("best_epoch")}
        sage_file, hk_file = args.geometry / f"{tag}_geometry_sage_probs.csv", args.geometry / f"{tag}_geometry_hk_probs.csv"
        if sage_file.is_file():
            probs = aligned(sage_file, "sample_uuid", test["sample_uuid"])
            entry["sage"], scored = sage_block(table, "finetuned_sage", tag, probs, y_test, truth, weights, recipe=recipe)
            for name, values in entry["sage"].items():
                values["collapse"] = bool(scored[name][0]["top_class_share"] >= collapse)
                runs_rows.append({"tag": tag, "recipe": recipe, "seed": entry["seed"], "input": name,
                                  "macro_f1": values["macro_f1"]["estimate"], "top_class": values["top_class"],
                                  "top_class_share": values["top_class_share"]["estimate"],
                                  "polyp_share": values["polyp_share"]["estimate"], "collapse": values["collapse"]})
            collapse_flags.setdefault(recipe, {})[tag] = {name: values["collapse"] for name, values in entry["sage"].items()}
            entry["differences"] = {f"{a}_minus_{b}": difference(table, "finetuned_sage", tag, scored, a, b, SAGE_METRICS,
                                                                 recipe=recipe)
                                    for a, b in FINETUNED_DIFFERENCES if a in scored and b in scored}
        else:
            skipped.append(f"fine_tuned/{tag}: no SAGE probabilities")
        if hk_file.is_file():
            preds = {name: proba.argmax(axis=1) for name, proba in aligned(hk_file, "file_name", fold1["file_name"]).items()}
            entry["hk_fold1"] = hk_block(table, "finetuned_hk", tag, preds, y_fold1, upper, hd, hk_weights)
            if "source_shortcut" in entry["hk_fold1"]:
                shortcuts.setdefault(recipe, {})[tag] = entry["hk_fold1"]["source_shortcut"]["shown"]
        else:
            skipped.append(f"fine_tuned/{tag}: no HyperKvasir probabilities")
        summary["fine_tuned"][tag] = entry
    if not tags:
        skipped.append(f"fine_tuned: no *_geometry_summary.json in {args.geometry}")

    # (iii) Verdicts by the reading rules.
    verdicts: dict[str, Any] = {}
    for recipe, flags in collapse_flags.items():
        verdicts[recipe] = {}
        for black, grey in PAIRS:
            seeds = [tag for tag, by_input in flags.items() if black in by_input and grey in by_input]
            verdicts[recipe][f"{black}_vs_{grey}"] = {
                **pair_verdict([flags[t][black] for t in seeds], [flags[t][grey] for t in seeds]),
                "per_seed": {t: {black: flags[t][black], grey: flags[t][grey]} for t in seeds}}
        shown = shortcuts.get(recipe, {})
        verdicts[recipe]["source_shortcut"] = {"per_seed": shown, "seeds": len(shown),
                                               "seeds_shown": sum(bool(v) for v in shown.values())}
    verdicts["probes_source_shortcut"] = {backbone: result["hk_fold1"]["source_shortcut"]["shown"]
                                          for backbone, result in summary["probes"].items()
                                          if "source_shortcut" in result.get("hk_fold1", {})}
    summary["verdicts"] = verdicts
    summary["skipped"] = skipped
    summary["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__, "cpu_count": os.cpu_count(),
                              "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "jobs": args.jobs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table.rows).to_csv(args.out_dir / "geometry_table.csv", index=False)
    pd.DataFrame(runs_rows, columns=["tag", "recipe", "seed", "input", "macro_f1", "top_class", "top_class_share",
                                     "polyp_share", "collapse"]).to_csv(args.out_dir / "geometry_runs.csv", index=False)
    (args.out_dir / "geometry_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                         encoding="utf-8")
    report(summary)
    print(f"\nWrote geometry_table.csv, geometry_runs.csv and geometry_summary.json to {args.out_dir}")
    return 0


def fmt(entry: dict[str, Any] | None, signed: bool = False) -> str:
    if not entry or entry.get("estimate") is None:
        return "-"
    form = "+.3f" if signed else ".3f"
    if entry.get("ci_low") is None:
        return f"{entry['estimate']:{form}}"
    return f"{entry['estimate']:{form}} [{entry['ci_low']:{form}}, {entry['ci_high']:{form}}]"


def report(summary: dict[str, Any]) -> None:
    """Compact printout of the main results and the verdicts."""
    for backbone, result in summary["probes"].items():
        print(f"\nProbe {backbone} (HyperKvasir raw, C = {result['C_hyperkvasir_raw']}) on SAGE test: "
              "macro-F1 | top class share | polyp share")
        for name, values in result["sage"].items():
            print(f"  {name:15s} {fmt(values['macro_f1']):28s} {values['top_class']:20s} "
                  f"{fmt(values['top_class_share']):28s} {fmt(values['polyp_share'])}")
        for name, values in result.get("sage_minus_masked", {}).items():
            print(f"  {name} - masked: macro-F1 {fmt(values['macro_f1'], True)}")
        rule = result.get("hk_fold1", {}).get("source_shortcut")
        if rule:
            print(f"  HyperKvasir fold 1, upper-GI polyp-or-rectum share: sageify - raw "
                  f"{fmt(rule['all_upper_gi']['sageify_minus_raw'], True)}, difference of increases "
                  f"{fmt(rule['all_upper_gi']['increase_difference'], True)}; shortcut shown: {rule['shown']}")
        for new, row in result.get("colour_global", {}).items():
            print(f"  {new}: transfer change {fmt(row['macro_f1']['transfer_change'], True)}, gap reduction "
                  f"{fmt(row['macro_f1']['gap_reduction'], True)} (reference C {row['C_sage_reference_new']})")
    for tag, entry in summary["fine_tuned"].items():
        print(f"\nRe-run {tag}: macro-F1 | top class share (collapse)")
        for name, values in entry.get("sage", {}).items():
            print(f"  {name:15s} {fmt(values['macro_f1']):28s} {values['top_class']:20s} "
                  f"{fmt(values['top_class_share'])} ({values['collapse']})")
        for name, values in entry.get("differences", {}).items():
            print(f"  {name}: macro-F1 {fmt(values['macro_f1'], True)}, top class share "
                  f"{fmt(values['top_class_share'], True)}")
        rule = entry.get("hk_fold1", {}).get("source_shortcut")
        if rule:
            print(f"  HyperKvasir fold 1: sageify - raw {fmt(rule['all_upper_gi']['sageify_minus_raw'], True)}, "
                  f"difference of increases {fmt(rule['all_upper_gi']['increase_difference'], True)}; "
                  f"shown: {rule['shown']}")
    print("\nVerdicts:")
    for recipe, items in summary["verdicts"].items():
        if recipe == "probes_source_shortcut":
            print(f"  probes, source-side shortcut shown: {items}")
            continue
        for name, value in items.items():
            text = value.get("verdict", f"shown in {value.get('seeds_shown')} of {value.get('seeds')} seeds")
            print(f"  {recipe} {name}: {text}")
    if summary["skipped"]:
        print(f"\nSkipped ({len(summary['skipped'])}): " + "; ".join(summary["skipped"]))


if __name__ == "__main__":
    raise SystemExit(main())

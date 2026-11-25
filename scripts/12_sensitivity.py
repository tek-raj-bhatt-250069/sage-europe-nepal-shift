"""Post hoc sensitivity analyses of the RQ1 and RQ2 results; none replaces a pre-registered result.

Reads the per-image predictions of scripts/07 (outputs/rq1/predictions) and scripts/08
(outputs/paper_recipe), and the features for the refits. Unless stated otherwise, intervals use
the same 2,000 paired patient-level resamples as RQ1 (percentile 95%), the estimand is the
pre-registered image-weighted one, and the gap is the SAGE-only reference (`ceiling`) minus the
transfer probe.

    patient weights   every test patient's images weighted 1/n_i, beside the image-weighted values
    bca               bias-corrected and accelerated intervals (patient jackknife)
    few clusters      bootstrap-t intervals (jackknife SE within each resample) and patient-level
                      paired swap tests for the gap and the per-class F1 gaps
    h3                Holm, Bonferroni and max-|z| adjustments, the pooled contrast, the AUROC version
    clustering        Kish effective numbers of patients, images per patient, resamples missing a class
    power             minimal detectable effects at 80% power and power at the observed effects
    h2 subset         without the two heaviest test patients, best constant recomputed per resample
    mcc               multi-class MCC and balanced accuracy for RQ1 and the re-runs
    influence         leave-one-patient-out changes of the headline quantities
    polyp context     polyp recall by location label with patient-bootstrap and Wilson intervals
    training          refits on resampled SAGE train patients and HyperKvasir images
    numerics          lbfgs tolerance 1e-6, one BLAS thread, and C selection at tolerance 1e-6

Writes aggregate tables and sensitivity_summary.json to outputs/sensitivity/. The one per-patient
table, loo_influence.csv, holds integer patient codes only.

With --converged, only the converged-fit check runs: every probe of RQ1 and RQ3 is refitted at
lbfgs tolerance 1e-6 with C re-selected by the pre-registered folds, grid and tie rule, and the RQ1,
H3, RQ2 and RQ3 results and the H1-H5 verdicts are recomputed beside the saved ones (converged_*
tables and converged_summary.json; the RQ1-RQ3 outputs are not changed).

    python scripts/12_sensitivity.py
    python scripts/12_sensitivity.py --converged
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import matthews_corrcoef
from threadpoolctl import threadpool_limits

from sage_shift.bootstrap import group_weights, percentile_ci
from sage_shift.data import read_json
from sage_shift.labelspace import MEANING_GROUP, SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.metrics import evaluate, macro_f1_many
from sage_shift.outdir import check_out_dir
from sage_shift.probes import grouped_folds, official_folds
from sage_shift.rq2 import meaning_gap_difference
from sage_shift.sensitivity import (PatientScores, Replicates, bca_interval, bootstrap_p, bootstrap_t_interval,
                                    class_f1, cv_choice, draw_counts, fit_predict_with, fit_tight, holm,
                                    inverse_size_weights, jackknife_se, kish_effective, macro_f1, max_abs_z_intervals,
                                    mde, metric_replicates, min_swap_p, nanmean_last, patient_bootstrap_rows,
                                    patient_index, patient_weighted, power_at, resample_deletions, rq3_draws, rq3_fit,
                                    select_c_with, stratified_bootstrap_rows, swap_test, wilson_interval)

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
VARIANTS = ["raw", "fov", "colour", "fov_colour"]
CHECK_VARIANTS = ["raw", "fov_colour"]
INPUTS = {"raw": "sage_probs", "fov": "sage_fov_probs", "masked": "sage_masked_probs"}
K = len(SIX_CLASSES)
POLYP = SIX_CLASSES.index("polyp")
SHIFTED = np.array([MEANING_GROUP[name] == "shifted" for name in SIX_CLASSES])
TIGHT_TOL = 1e-6
HEADLINE = ["transfer_macro_f1", "ceiling_macro_f1", "gap_macro_f1", "transfer_macro_auroc", "ceiling_macro_auroc",
            "gap_macro_auroc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post hoc sensitivity analyses (outputs/sensitivity/).")
    parser.add_argument("--rq1", type=Path, default=ROOT / "outputs" / "rq1")
    parser.add_argument("--rq2", type=Path, default=ROOT / "outputs" / "rq2")
    parser.add_argument("--paper-recipe", type=Path, default=ROOT / "outputs" / "paper_recipe")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sensitivity")
    parser.add_argument("--jobs", type=int, default=3, help="parallel refit processes (one BLAS thread each)")
    parser.add_argument("--swaps", type=int, default=10000, help="random swap vectors per swap test")
    parser.add_argument("--train-resamples", type=int, default=100, help="refits on resampled SAGE train patients")
    parser.add_argument("--hk-resamples", type=int, default=30, help="refits on resampled HyperKvasir images")
    parser.add_argument("--training-backbones", nargs="+", help="default: every backbone (raw frames)")
    parser.add_argument("--skip-refits", action="store_true", help="skip the training and numerical refits")
    parser.add_argument("--rq3", type=Path, default=ROOT / "outputs" / "rq3")
    parser.add_argument("--converged", action="store_true",
                        help="run only the converged-fit check (tolerance 1e-6 everywhere; converged_* outputs)")
    return parser.parse_args()


# --- Small helpers --------------------------------------------------------------------------------

def rnd(value: Any, digits: int = 4) -> Any:
    """Round a number for tables; nan becomes None, infinities are kept."""
    if value is None:
        return None
    v = float(value)
    return None if np.isnan(v) else (v if np.isinf(v) else round(v, digits))


def sig(value: float) -> float:
    """Four significant digits (p-values)."""
    return float(f"{float(value):.4g}")


def json_ready(obj: Any) -> Any:
    """Plain JSON types; nan and infinities become None."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj):
            return None
        return sig(obj) if 0 < abs(obj) < 1e-3 else round(float(obj), 4)
    return obj


def ci_entry(rep: Replicates, level: float) -> dict[str, float]:
    """Estimate with its percentile interval."""
    low, high = percentile_ci(rep.boot, level)
    return {"estimate": float(rep.point), "ci_low": low, "ci_high": high}


def fmt(entry: dict[str, Any], low: str = "ci_low", high: str = "ci_high", est: str = "estimate") -> str:
    def one(v: Any) -> str:
        return "unbounded" if v is None or not np.isfinite(v) else f"{v:.3f}"
    return f"{entry[est]:.3f} [{one(entry[low])}, {one(entry[high])}]"


def excludes_zero(low: float, high: float) -> bool:
    return bool(low > 0 or high < 0)


def aligned(path: Path, ids: pd.Series) -> np.ndarray:
    """Probabilities (six columns) in the order of `ids`."""
    frame = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid")
    missing = set(ids) - set(frame.index)
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} test images have no prediction")
    return frame.loc[list(ids), list(SIX_CLASSES)].to_numpy(dtype=float)


def feature_rows(path: Path, ids: pd.Series) -> np.ndarray:
    """Feature rows in the order of `ids` (as scripts/07 reads them)."""
    with np.load(path, allow_pickle=False) as data:
        features, file_ids = data["features"], data["ids"]
    index = {key: row for row, key in enumerate(file_ids.tolist())}
    return features[[index[key] for key in ids]].astype(np.float64)


def h3_stat(f1_ceiling: np.ndarray, f1_transfer: np.ndarray) -> np.ndarray:
    return meaning_gap_difference(f1_ceiling, f1_transfer, SHIFTED)


def cell_quantities(rt: dict[str, Replicates], rc: dict[str, Replicates]) -> dict[str, Replicates]:
    """Named quantities of one backbone x variant cell from the transfer and ceiling replicates."""
    q: dict[str, Replicates] = {}
    for metric in ("macro_f1", "macro_auroc", "mcc", "balanced_accuracy"):
        q[f"transfer_{metric}"], q[f"ceiling_{metric}"] = rt[metric], rc[metric]
        q[f"gap_{metric}"] = rc[metric] - rt[metric]
    q["best_constant_macro_f1"] = rt["best_constant"]
    q["transfer_lead_over_best_constant"] = rt["macro_f1"] - rt["best_constant"]
    q["h3_shifted_minus_matched"] = rc["f1"].apply(h3_stat, rt["f1"])
    q["h3_auroc_shifted_minus_matched"] = rc["auroc"].apply(h3_stat, rt["auroc"])
    for j, name in enumerate(SIX_CLASSES):
        q[f"f1_gap_{name}"] = (rc["f1"] - rt["f1"]).pick(j)
    return q


def add_variant_changes(quantities: dict[tuple[str, str], dict[str, Replicates]], backbones: list[str]) -> None:
    """RQ2 transfer change and gap reduction against raw frames (macro-F1)."""
    for b in backbones:
        raw = quantities[(b, "raw")]
        for v in VARIANTS[1:]:
            q = quantities[(b, v)]
            q["transfer_change_macro_f1"] = q["transfer_macro_f1"] - raw["transfer_macro_f1"]
            q["gap_reduction_macro_f1"] = raw["gap_macro_f1"] - q["gap_macro_f1"]


# --- Reproduction check ---------------------------------------------------------------------------

def reproduction_check(q_img: dict, rq1: dict, rq2: dict, rq2_table: pd.DataFrame, level: float,
                       check: dict[str, Any]) -> dict[str, Any]:
    """The image-weighted estimates and intervals must equal the saved RQ1 and RQ2 values.

    RQ1 computed AUROC from unrounded probabilities; the saved predictions (read here and by RQ2)
    are rounded to 6 decimals, which creates ties. AUROC is therefore checked against the RQ2 table
    and its difference from the RQ1 summary is only reported.
    """
    def diff(mine: dict[str, float], saved: Any) -> float:
        return max(abs(round(mine[k], 4) - float(saved[k])) for k in mine)
    worst, auroc_rq1 = 0.0, 0.0
    decomposition = rq2_table[rq2_table["section"] == "decomposition"].set_index(["model", "row", "metric"])
    for (b, v), q in q_img.items():
        saved = rq1[v]["backbones"][b]["metrics"]
        for cond in ("transfer", "ceiling", "gap"):
            for metric in ("macro_f1", "balanced_accuracy"):
                worst = max(worst, diff(ci_entry(q[f"{cond}_{metric}"], level), saved[metric][cond]))
            mine = ci_entry(q[f"{cond}_macro_auroc"], level)
            worst = max(worst, diff(mine, decomposition.loc[(b, v, f"{cond}_macro_auroc")]))
            auroc_rq1 = max(auroc_rq1, diff(mine, saved["macro_auroc"][cond]))
        worst = max(worst, diff(ci_entry(q["h3_shifted_minus_matched"], level), rq2["h3"][f"{b}/{v}"]))
    if worst > 1.01e-4 or max(check.values()) > 1e-9:
        raise SystemExit(f"reproduction check failed: tables differ by {worst}, direct scores by {check}")
    return {"max_abs_diff_to_rq1_rq2_tables": worst, "macro_auroc_diff_to_rq1_unrounded": round(auroc_rq1, 4), **check,
            "note": "AUROC uses the saved 6-decimal probabilities, as RQ2 does; RQ1's AUROC used unrounded ones."}


def direct_check(y: np.ndarray, proba: np.ndarray, truth: np.ndarray, groups: np.ndarray, weights: np.ndarray,
                 img: dict[str, Replicates], pat: dict[str, Replicates]) -> dict[str, float]:
    """Vectorised scores against metrics.evaluate and scikit-learn's MCC on the first resamples."""
    diffs = {"image_weighted": 0.0, "patient_weighted": 0.0, "mcc_vs_sklearn": 0.0}
    pw = patient_weighted(weights, groups)
    pred = proba.argmax(axis=1)
    for b in range(25):
        for name, reps, w in (("image_weighted", img, weights[b]), ("patient_weighted", pat, pw[b])):
            direct = evaluate(y, proba, truth, w)
            for metric in ("macro_f1", "macro_auroc", "balanced_accuracy"):
                diffs[name] = max(diffs[name], abs(direct[metric] - reps[metric].boot[b]))
        diffs["mcc_vs_sklearn"] = max(diffs["mcc_vs_sklearn"], abs(
            matthews_corrcoef(y, pred, sample_weight=weights[b]) - img["mcc"].boot[b]))
    diffs["mcc_vs_sklearn"] = max(diffs["mcc_vs_sklearn"], abs(matthews_corrcoef(y, pred) - float(img["mcc"].point)))
    return diffs


# --- 1. Patient-weighted estimand -----------------------------------------------------------------

PW_QUANTITIES = ["transfer_macro_f1", "ceiling_macro_f1", "gap_macro_f1", "transfer_macro_auroc", "ceiling_macro_auroc",
                 "gap_macro_auroc", "transfer_mcc", "ceiling_mcc", "gap_mcc", "transfer_balanced_accuracy",
                 "ceiling_balanced_accuracy", "gap_balanced_accuracy", "best_constant_macro_f1",
                 "transfer_lead_over_best_constant", "transfer_change_macro_f1", "gap_reduction_macro_f1",
                 "h3_shifted_minus_matched"]


def h2_checks(q: dict[str, Replicates], level: float, bar: float) -> dict[str, Any]:
    transfer, lead, gap = (ci_entry(q[n], level) for n in ("transfer_macro_f1", "transfer_lead_over_best_constant",
                                                           "gap_macro_f1"))
    return {"transfer_ci_low": transfer["ci_low"], "transfer_ci_above_fixed_bar": transfer["ci_low"] > bar,
            "best_constant": float(q["best_constant_macro_f1"].point), "lead": lead, "lead_ci_above_zero": lead["ci_low"] > 0,
            "gap": gap, "gap_ci_excludes_zero": excludes_zero(gap["ci_low"], gap["ci_high"])}


def patient_weighted_item(q_img: dict, q_pat: dict, level: float, bar: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for (b, v), qi in q_img.items():
        for name in PW_QUANTITIES:
            if name not in qi:
                continue
            ei, ep = ci_entry(qi[name], level), ci_entry(q_pat[(b, v)][name], level)
            rows.append({"backbone": b, "variant": v, "quantity": name,
                         **{f"image_{k}": rnd(x) for k, x in ei.items()}, **{f"patient_{k}": rnd(x) for k, x in ep.items()}})
    summary: dict[str, Any] = {"h2": {}, "h3": {}, "h4_imagenet_gap_reduction": {}}
    for (b, v), qi in q_img.items():
        summary["h2"][f"{b}/{v}"] = {"image_weighted": h2_checks(qi, level, bar),
                                     "patient_weighted": h2_checks(q_pat[(b, v)], level, bar)}
        summary["h3"][f"{b}/{v}"] = {"image_weighted": ci_entry(qi["h3_shifted_minus_matched"], level),
                                     "patient_weighted": ci_entry(q_pat[(b, v)]["h3_shifted_minus_matched"], level)}
    for v in ("fov", "colour", "fov_colour"):
        key = ("resnet50_imagenet", v)
        if key in q_img:
            summary["h4_imagenet_gap_reduction"][v] = {
                "image_weighted": ci_entry(q_img[key]["gap_reduction_macro_f1"], level),
                "patient_weighted": ci_entry(q_pat[key]["gap_reduction_macro_f1"], level)}
    p = np.array([bootstrap_p(q_pat[key]["h3_shifted_minus_matched"].boot) for key in q_img])
    summary["h3_patient_weighted_tests"] = {
        "cells_ci_excluding_zero": [k for k, e in summary["h3"].items()
                                    if excludes_zero(e["patient_weighted"]["ci_low"], e["patient_weighted"]["ci_high"])],
        "p_boot": {f"{b}/{v}": sig(x) for (b, v), x in zip(q_img, p)},
        "holm12_rejections": [f"{b}/{v}" for (b, v), x in zip(q_img, holm(p)) if x <= 1 - level]}
    failing = {est: [k for k, e in summary["h2"].items() if not (e[est]["lead_ci_above_zero"] and e[est]["gap_ci_excludes_zero"])]
               for est in ("image_weighted", "patient_weighted")}
    summary["h2_cells_failing_paired_lead_or_gap"] = failing
    summary["note"] = ("H2 is checked here as the paired lead over the best constant predictor, both recomputed "
                       "in every resample; the pre-registered rule compares the transfer interval with the fixed bar.")
    return pd.DataFrame(rows), summary


# --- 2. BCa ---------------------------------------------------------------------------------------

def bca_item(q_img: dict, runs: dict, level: float, bar: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = []
    for (b, v), q in q_img.items():
        for name in HEADLINE + ["h3_shifted_minus_matched", "transfer_change_macro_f1", "gap_reduction_macro_f1"]:
            if name in q:
                targets.append((b, v, name, q[name]))
    for tag, reps in runs.items():
        if "fov" in reps:
            targets.append((tag, "fov_minus_raw", "macro_f1_difference", reps["fov"]["macro_f1"] - reps["raw"]["macro_f1"]))
    rows = []
    for model, variant, name, rep in targets:
        pct, bca = ci_entry(rep, level), bca_interval(float(rep.point), rep.boot, rep.jack, level)
        rows.append({"model": model, "variant": variant, "quantity": name, "estimate": rnd(rep.point),
                     "pct_low": rnd(pct["ci_low"]), "pct_high": rnd(pct["ci_high"]), "bca_low": rnd(bca["low"]),
                     "bca_high": rnd(bca["high"]), "z0": rnd(bca["z0"]), "acceleration": rnd(bca["acceleration"]),
                     "bound_shift": rnd(max(abs(bca["low"] - pct["ci_low"]), abs(bca["high"] - pct["ci_high"])))})
    frame = pd.DataFrame(rows)

    def get(model: str, variant: str, name: str) -> dict[str, Any]:
        row = frame[(frame["model"] == model) & (frame["variant"] == variant) & (frame["quantity"] == name)].iloc[0]
        return {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in row.items()}
    summary: dict[str, Any] = {"max_bound_shift": float(frame["bound_shift"].max()), "h2": {}, "h3": {}, "h4": {}}
    for (b, v) in q_img:
        t, g = get(b, v, "transfer_macro_f1"), get(b, v, "gap_macro_f1")
        summary["h2"][f"{b}/{v}"] = {"transfer_bca_low": t["bca_low"], "transfer_above_bar": bool(t["bca_low"] > bar),
                                     "gap_bca": [g["bca_low"], g["bca_high"]],
                                     "gap_excludes_zero": excludes_zero(g["bca_low"], g["bca_high"])}
        h = get(b, v, "h3_shifted_minus_matched")
        summary["h3"][f"{b}/{v}"] = {"bca": [h["bca_low"], h["bca_high"]],
                                     "excludes_zero": excludes_zero(h["bca_low"], h["bca_high"])}
    for v in ("fov", "colour"):
        if ("resnet50_imagenet", v) in q_img:
            r = get("resnet50_imagenet", v, "gap_reduction_macro_f1")
            summary["h4"][v] = {"bca": [r["bca_low"], r["bca_high"]], "excludes_zero": excludes_zero(r["bca_low"], r["bca_high"])}
    recipe = frame[frame["variant"] == "fov_minus_raw"]
    summary["paper_recipe_fov_minus_raw"] = {r.model: [r.estimate, r.bca_low, r.bca_high] for r in recipe.itertuples()}
    summary["interval_verdict_changes"] = [
        f"{r.model}/{r.variant}/{r.quantity}: percentile [{r.pct_low}, {r.pct_high}], BCa [{r.bca_low}, {r.bca_high}]"
        for r in frame.itertuples() if excludes_zero(r.pct_low, r.pct_high) != excludes_zero(r.bca_low, r.bca_high)]
    summary["h2_cells_failing"] = [k for k, e in summary["h2"].items() if not (e["transfer_above_bar"] and e["gap_excludes_zero"])]
    summary["note"] = ("z0 from the share of resamples below the estimate (ties one half); acceleration from the "
                       "leave-one-patient-out jackknife of the 62 test patients.")
    return frame, summary


# --- 3. Few-cluster checks ------------------------------------------------------------------------

def few_cluster_item(scores: dict, q_img: dict, counts: np.ndarray, class_patients: dict[str, int], level: float,
                     n_swaps: int, swap_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    drawn = counts > 0
    for (b, v), q in q_img.items():
        st, sc = scores[(b, v)]["transfer"], scores[(b, v)]["ceiling"]
        f1_t, f1_c = class_f1(resample_deletions(st.parts, counts)), class_f1(resample_deletions(sc.parts, counts))
        deleted = {"gap_macro_f1": nanmean_last(f1_c) - nanmean_last(f1_t), "h3_shifted_minus_matched": h3_stat(f1_c, f1_t)}
        for j, name in enumerate(SIX_CLASSES):
            deleted[f"f1_gap_{name}"] = f1_c[..., j] - f1_t[..., j]
        _, p_gap = swap_test(sc.parts, st.parts, lambda cr, ct: macro_f1(cr) - macro_f1(ct), n_swaps, swap_seed)
        _, p_class = swap_test(sc.parts, st.parts, lambda cr, ct: class_f1(cr) - class_f1(ct), n_swaps, swap_seed)
        swap = {"gap_macro_f1": (float(p_gap), len(drawn[0]))}
        swap.update({f"f1_gap_{name}": (float(p_class[j]), class_patients[name]) for j, name in enumerate(SIX_CLASSES)})
        for name, values in deleted.items():
            rep = q[name]
            boot_se = jackknife_se(np.where(drawn, values, np.nan), counts)
            se = float(jackknife_se(rep.jack))
            pct, bt = ci_entry(rep, level), bootstrap_t_interval(float(rep.point), se, rep.boot, boot_se, level)
            p, g = swap.get(name, (None, None))
            rows.append({"backbone": b, "variant": v, "quantity": name, "estimate": rnd(rep.point),
                         "pct_low": rnd(pct["ci_low"]), "pct_high": rnd(pct["ci_high"]), "jackknife_se": rnd(se),
                         "boot_t_low": rnd(bt["low"]), "boot_t_high": rnd(bt["high"]), "boot_t_resamples": bt["resamples"],
                         "boot_t_zero_se": bt["zero_se"], "swap_p": None if p is None else sig(p),
                         "swap_patients": g, "swap_min_p": None if g is None else sig(min_swap_p(g))})
    frame = pd.DataFrame(rows)
    changed = []
    for r in frame.itertuples():
        pct_ex = excludes_zero(r.pct_low, r.pct_high)
        bt_ex = excludes_zero(r.boot_t_low, r.boot_t_high)
        if pct_ex != bt_ex:
            changed.append(f"{r.backbone}/{r.variant}/{r.quantity}: percentile {'excludes' if pct_ex else 'includes'} 0, "
                           f"bootstrap-t {'excludes' if bt_ex else 'includes'} 0")
    gaps = frame[frame["quantity"] == "gap_macro_f1"]
    h3 = frame[frame["quantity"] == "h3_shifted_minus_matched"]
    summary = {
        "interval_verdict_changes": changed,
        "gap_bootstrap_t": {f"{r.backbone}/{r.variant}": [r.estimate, r.boot_t_low, r.boot_t_high] for r in gaps.itertuples()},
        "gap_swap_p": {f"{r.backbone}/{r.variant}": r.swap_p for r in gaps.itertuples()},
        "h3_bootstrap_t": {f"{r.backbone}/{r.variant}": [r.estimate, r.boot_t_low, r.boot_t_high] for r in h3.itertuples()},
        "per_class_swap_p_below_0.05": int((frame["quantity"].str.startswith("f1_gap_") & (frame["swap_p"] < 0.05)).sum()),
        "per_class_tests": int(frame["quantity"].str.startswith("f1_gap_").sum()),
        "swap_patients_by_class": class_patients,
        "min_swap_p_by_class": {c: sig(min_swap_p(g)) for c, g in class_patients.items()},
        "max_boot_t_zero_se": int(frame["boot_t_zero_se"].max()),
        "per_class_boot_t_bounds_outside_range": int(((frame["boot_t_low"] < -1) | (frame["boot_t_high"] > 1))[
            frame["quantity"].str.startswith("f1_gap_")].sum()),
        "note": ("Bootstrap-t: SE in each resample from a jackknife that deletes one drawn patient copy at a time "
                 "(vectorised; no nested bootstrap needed); the SE of the estimate from the leave-one-patient-out "
                 "jackknife; a resample whose SE is 0 counts as t = +-inf. Bounds are not truncated to the range "
                 "of the statistic (a per-class F1 gap lies in [-1, 1]). Swap test: 10,000 random patient-level "
                 "swaps of the two probes' predictions; two-sided Monte Carlo p = (1 + #{|T*| >= |T|}) / "
                 "(1 + swaps). The minimum p of a sign-flip over the G patients with the class is 2^(1-G); false "
                 "positives from other patients also enter the per-class F1, so this floor is nominal. The swap "
                 "test is not applied to H3: its null (equal gaps in the two meaning groups) is not "
                 "exchangeability of the two probes, whose null is a zero gap."),
    }
    return frame, summary


# --- 4. H3 multiplicity ---------------------------------------------------------------------------

def h3_item(q_img: dict, level: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(q_img)
    reps = [q_img[k]["h3_shifted_minus_matched"] for k in keys]
    auc = [q_img[k]["h3_auroc_shifted_minus_matched"] for k in keys]
    p = np.array([bootstrap_p(r.boot) for r in reps])
    p_auc = np.array([bootstrap_p(r.boot) for r in auc])
    alpha = 1 - level
    holm12 = holm(p)
    raw = [i for i, k in enumerate(keys) if k[1] == "raw"]
    holm3 = dict(zip(raw, holm(p[raw])))
    points = np.array([float(r.point) for r in reps])
    boot = np.stack([r.boot for r in reps], axis=1)
    maxz_low, maxz_high, q = max_abs_z_intervals(points, boot, level)
    rows = []
    for i, (b, v) in enumerate(keys):
        pct = ci_entry(reps[i], level)
        bon12 = percentile_ci(reps[i].boot, 1 - alpha / len(keys))
        bon3 = percentile_ci(reps[i].boot, 1 - alpha / len(raw)) if i in holm3 else (None, None)
        a = ci_entry(auc[i], level)
        rows.append({"backbone": b, "variant": v, "estimate": rnd(pct["estimate"]), "ci_low": rnd(pct["ci_low"]),
                     "ci_high": rnd(pct["ci_high"]), "p_boot": sig(p[i]), "holm12_p": sig(holm12[i]),
                     "holm12_reject": bool(holm12[i] <= alpha), "holm3_raw_p": sig(holm3[i]) if i in holm3 else None,
                     "holm3_raw_reject": bool(holm3[i] <= alpha) if i in holm3 else None,
                     "bonferroni12_low": rnd(bon12[0]), "bonferroni12_high": rnd(bon12[1]),
                     "bonferroni3_low": rnd(bon3[0]), "bonferroni3_high": rnd(bon3[1]),
                     "maxz_low": rnd(maxz_low[i]), "maxz_high": rnd(maxz_high[i]),
                     "auroc_estimate": rnd(a["estimate"]), "auroc_ci_low": rnd(a["ci_low"]),
                     "auroc_ci_high": rnd(a["ci_high"]), "auroc_p_boot": sig(p_auc[i])})
    pooled = Replicates(points.mean(), boot.mean(axis=1), np.mean([r.jack for r in reps], axis=0))
    pooled_auc = Replicates(np.mean([float(r.point) for r in auc]), np.mean([r.boot for r in auc], axis=0),
                            np.mean([r.jack for r in auc], axis=0))
    pe, pa = ci_entry(pooled, level), ci_entry(pooled_auc, level)
    rows.append({"backbone": "pooled_mean_of_12_cells", "variant": "all", "estimate": rnd(pe["estimate"]),
                 "ci_low": rnd(pe["ci_low"]), "ci_high": rnd(pe["ci_high"]), "p_boot": sig(bootstrap_p(pooled.boot)),
                 "auroc_estimate": rnd(pa["estimate"]), "auroc_ci_low": rnd(pa["ci_low"]), "auroc_ci_high": rnd(pa["ci_high"]),
                 "auroc_p_boot": sig(bootstrap_p(pooled_auc.boot))})
    frame = pd.DataFrame(rows)
    cells = frame.iloc[:-1]
    summary = {
        "holm12_rejections": [f"{r.backbone}/{r.variant}" for r in cells.itertuples() if r.holm12_reject],
        "holm3_raw_rejections": [f"{r.backbone}/{r.variant}" for r in cells.itertuples() if r.holm3_raw_reject is True],
        "smallest_p": {"cell": "/".join(keys[int(np.argmin(p))]), "p_boot": sig(p.min()),
                       "holm12_threshold": sig(alpha / len(keys))},
        "positive_estimates": int((points > 0).sum()), "cells": len(keys), "maxz_quantile": rnd(q, 3),
        "pooled": pe, "pooled_auroc": pa,
        "auroc_estimate_range": [rnd(cells["auroc_estimate"].min()), rnd(cells["auroc_estimate"].max())],
        "auroc_cells_excluding_zero": int(sum(excludes_zero(r.auroc_ci_low, r.auroc_ci_high) for r in cells.itertuples())),
        "note": ("p_boot = twice the smaller share of resamples on either side of 0. The primary family is all 12 "
                 "backbone x variant cells (Holm); the three raw cells are a post hoc alternative family. The max-|z| "
                 "intervals use each cell's bootstrap SD and the 95% quantile of the largest |z| over the 12 cells."),
    }
    return frame, summary


# --- 5. Clustering --------------------------------------------------------------------------------

def clustering_item(test: pd.DataFrame, counts: np.ndarray, groups: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = patient_index(groups)
    sizes = np.bincount(index)
    per_class = pd.crosstab(index, test["class"]).reindex(columns=list(SIX_CLASSES), fill_value=0).to_numpy()
    support = counts @ per_class  # (B, K) images of each class in each resample
    rows = [{"group": "all", "images": int(sizes.sum()), "patients": len(sizes),
             "kish_effective_patients": rnd(kish_effective(sizes), 2),
             "largest_patient_images": int(sizes.max()), "largest_patient_share": rnd(sizes.max() / sizes.sum(), 3),
             "resamples_missing": int((support == 0).any(axis=1).sum())}]
    for j, name in enumerate(SIX_CLASSES):
        n = per_class[:, j]
        rows.append({"group": name, "images": int(n.sum()), "patients": int((n > 0).sum()),
                     "kish_effective_patients": rnd(kish_effective(n[n > 0]), 2), "largest_patient_images": int(n.max()),
                     "largest_patient_share": rnd(n.max() / n.sum(), 3), "resamples_missing": int((support[:, j] == 0).sum())})
    frame = pd.DataFrame(rows)
    distribution = {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
    summary = {"images_per_patient": {"patients_by_image_count": distribution, "median": float(np.median(sizes)),
                                      "mean": rnd(sizes.mean(), 2), "max": int(sizes.max()),
                                      "five_largest": sorted(sizes.tolist(), reverse=True)[:5],
                                      "share_of_images_in_two_largest": rnd(np.sort(sizes)[-2:].sum() / sizes.sum(), 3)},
               "kish": {r.group: r.kish_effective_patients for r in frame.itertuples()},
               "resamples_missing_a_class": int(frame.loc[0, "resamples_missing"]),
               "resamples_missing_by_class": {r.group: r.resamples_missing for r in frame.iloc[1:].itertuples()}}
    return frame, summary


# --- 6. Power -------------------------------------------------------------------------------------

def power_item(q_img: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    families = ["gap_macro_f1", "transfer_lead_over_best_constant", "transfer_change_macro_f1", "gap_reduction_macro_f1",
                "h3_shifted_minus_matched", "gap_macro_auroc"]
    rows = []
    for family in families:
        for (b, v), q in q_img.items():
            if family in q:
                se = q[family].se()
                rows.append({"family": family, "backbone": b, "variant": v, "estimate": rnd(q[family].point),
                             "bootstrap_se": rnd(se), "mde80": rnd(mde(se)),
                             "power_at_estimate": rnd(power_at(float(q[family].point), se), 3)})
    frame = pd.DataFrame(rows)
    summary = {f: {"quantities": int(len(part)), "mde80_range": [rnd(part["mde80"].min()), rnd(part["mde80"].max())],
                   "power_range": [rnd(part["power_at_estimate"].min(), 3), rnd(part["power_at_estimate"].max(), 3)]}
               for f, part in frame.groupby("family", sort=False)}
    summary["note"] = "MDE80 = (z_0.975 + z_0.80) x bootstrap SE = 2.80 x SE; power is two-sided, normal approximation."
    return frame, summary


# --- 7. H2 without the two heaviest patients ------------------------------------------------------

def h2_subset_item(test: pd.DataFrame, cells: dict, y: np.ndarray, truth: np.ndarray, n_boot: int, boot_seed: int,
                   level: float, bar: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    heaviest = test["patient_group"].value_counts()
    if heaviest.iloc[1] == heaviest.iloc[2]:
        raise SystemExit("the two heaviest test patients are not unique")
    keep = ~test["patient_group"].isin(heaviest.index[:2]).to_numpy()
    groups = test.loc[keep, "patient_group"].to_numpy()
    counts = draw_counts(group_weights(groups, n_boot, boot_seed), groups)
    rows = []
    for (b, v), probs in cells.items():
        rt, rc = (metric_replicates(PatientScores(y[keep], probs[c][keep], truth[keep], groups), counts)
                  for c in ("transfer", "ceiling"))
        t, lead = ci_entry(rt["macro_f1"], level), ci_entry(rt["macro_f1"] - rt["best_constant"], level)
        const, gap = ci_entry(rt["best_constant"], level), ci_entry(rc["macro_f1"] - rt["macro_f1"], level)
        rows.append({"backbone": b, "variant": v, "images": int(keep.sum()), "patients": len(np.unique(groups)),
                     "transfer_estimate": rnd(t["estimate"]), "transfer_ci_low": rnd(t["ci_low"]),
                     "transfer_ci_high": rnd(t["ci_high"]),
                     "best_constant": rnd(const["estimate"]), "best_constant_ci_low": rnd(const["ci_low"]),
                     "best_constant_ci_high": rnd(const["ci_high"]), "lead_estimate": rnd(lead["estimate"]),
                     "lead_ci_low": rnd(lead["ci_low"]), "lead_ci_high": rnd(lead["ci_high"]),
                     "gap_estimate": rnd(gap["estimate"]), "gap_ci_low": rnd(gap["ci_low"]), "gap_ci_high": rnd(gap["ci_high"]),
                     "transfer_ci_above_fixed_bar": bool(t["ci_low"] > bar),
                     "transfer_ci_above_subset_best_constant": bool(t["ci_low"] > const["estimate"]),
                     "lead_ci_above_zero": bool(lead["ci_low"] > 0),
                     "gap_ci_excludes_zero": excludes_zero(gap["ci_low"], gap["ci_high"])})
    frame = pd.DataFrame(rows)
    summary = {"images": int(keep.sum()), "patients": len(np.unique(groups)),
               "removed_patients_images": heaviest.iloc[:2].tolist(),
               "subset_best_constant": rnd(frame["best_constant"].iloc[0]),
               "cells_failing_paired_lead": [f"{r.backbone}/{r.variant}" for r in frame.itertuples() if not r.lead_ci_above_zero],
               "cells_failing_fixed_bar": [f"{r.backbone}/{r.variant}" for r in frame.itertuples()
                                           if not r.transfer_ci_above_fixed_bar],
               "cells_failing_gap": [f"{r.backbone}/{r.variant}" for r in frame.itertuples() if not r.gap_ci_excludes_zero],
               "note": ("New patient resamples of the reduced set (2,000, seed as RQ1); the best constant is "
                        "recomputed in each resample.")}
    return frame, summary


# --- 8. MCC and balanced accuracy -----------------------------------------------------------------

def mcc_item(q_img: dict, runs: dict, level: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for (b, v), q in q_img.items():
        for cond in ("transfer", "ceiling", "gap"):
            for metric in ("mcc", "balanced_accuracy"):
                e = ci_entry(q[f"{cond}_{metric}"], level)
                rows.append({"section": "rq1", "model": b, "row": v, "condition": cond, "metric": metric,
                             **{k: rnd(x) for k, x in e.items()}})
    for tag, reps in runs.items():
        for name, rep in reps.items():
            for metric in ("macro_f1", "mcc", "balanced_accuracy"):
                e = ci_entry(rep[metric], level)
                rows.append({"section": "paper_recipe", "model": tag, "row": name, "condition": "fine_tuned", "metric": metric,
                             **{k: rnd(x) for k, x in e.items()}})
        for name in ("fov", "masked"):
            if name in reps:
                for metric in ("macro_f1", "mcc", "balanced_accuracy"):
                    e = ci_entry(reps[name][metric] - reps["raw"][metric], level)
                    rows.append({"section": "paper_recipe", "model": tag, "row": f"{name}_minus_raw", "condition": "fine_tuned",
                                 "metric": metric, **{k: rnd(x) for k, x in e.items()}})
    frame = pd.DataFrame(rows)
    rq1 = frame[(frame["section"] == "rq1") & (frame["condition"] == "transfer")]
    summary = {"rq1_transfer_mcc_range": [rnd(rq1.loc[rq1["metric"] == "mcc", "estimate"].min()),
                                          rnd(rq1.loc[rq1["metric"] == "mcc", "estimate"].max())],
               "rq1_gap_mcc_cells_excluding_zero": int(sum(excludes_zero(r.ci_low, r.ci_high) for r in frame.itertuples()
                                                           if r.section == "rq1" and r.condition == "gap" and r.metric == "mcc")),
               "paper_recipe_mcc": {f"{r.model}/{r.row}": [r.estimate, r.ci_low, r.ci_high] for r in frame.itertuples()
                                    if r.section == "paper_recipe" and r.metric == "mcc"},
               "note": "Multi-class MCC (Gorodkin) from the weighted confusion matrix; 0 when undefined (a constant predictor)."}
    return frame, summary


# --- 9. Leave-one-patient-out influence -----------------------------------------------------------

def influence_item(q_img: dict, runs: dict, test: pd.DataFrame, groups: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = patient_index(groups)
    codes, sizes = np.unique(groups), np.bincount(index)
    composition = pd.crosstab(index, test["class"]).reindex(columns=list(SIX_CLASSES), fill_value=0)
    targets = []
    for (b, v), q in q_img.items():
        if v in CHECK_VARIANTS:
            for name in ("transfer_macro_f1", "ceiling_macro_f1", "gap_macro_f1"):
                targets.append((b, v, name, q[name]))
    for tag, reps in runs.items():
        if tag.startswith("adjusted") and "fov" in reps:
            targets.append((tag, "fov_minus_raw", "macro_f1_difference", reps["fov"]["macro_f1"] - reps["raw"]["macro_f1"]))
    rows, summary = [], {}
    for model, variant, name, rep in targets:
        delta = rep.jack - rep.point
        rows += [{"backbone_or_run": model, "variant": variant, "quantity": name, "patient_group": int(codes[p]),
                  "n_images": int(sizes[p]), "delta": round(float(delta[p]), 4)} for p in range(len(codes))]
        top = np.argsort(-np.abs(delta), kind="mergesort")[:3]
        summary[f"{model}/{variant}/{name}"] = {
            "estimate": rnd(rep.point), "jackknife_se": rnd(jackknife_se(rep.jack)),
            "largest": [{"delta": rnd(delta[p]), "n_images": int(sizes[p]),
                         "classes": {c: int(n) for c, n in composition.iloc[p].items() if n > 0}} for p in top]}
    return pd.DataFrame(rows), summary


# --- 10. Polyp-context recall ---------------------------------------------------------------------

def polyp_context_item(test: pd.DataFrame, cells: dict, counts: np.ndarray, groups: np.ndarray,
                       level: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = patient_index(groups)
    polyps = (test["class"] == "polyp").to_numpy()
    strata = [("all polyps", polyps)] + [(ctx, polyps & (test["location_context"] == ctx).to_numpy())
                                         for ctx in ("upper GI", "lower GI", "no location label")]
    rows = []
    for (b, v), probs in cells.items():
        if v not in CHECK_VARIANTS:
            continue
        for cond in ("transfer", "ceiling"):
            hit = probs[cond].argmax(axis=1) == POLYP
            for name, mask in strata:
                n_p = np.bincount(index, weights=mask.astype(float), minlength=counts.shape[1])
                hit_p = np.bincount(index, weights=(mask & hit).astype(float), minlength=counts.shape[1])
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot = (counts @ hit_p) / (counts @ n_p)
                low, high = percentile_ci(boot, level)
                w_low, w_high = wilson_interval(int(hit[mask].sum()), int(mask.sum()), level)
                rows.append({"backbone": b, "variant": v, "condition": cond, "location": name, "images": int(mask.sum()),
                             "patients": int(len(np.unique(groups[mask]))), "recalled": int(hit[mask].sum()),
                             "recall": rnd(hit[mask].mean()), "patient_ci_low": rnd(low), "patient_ci_high": rnd(high),
                             "resamples_without_stratum": int(np.isnan(boot).sum()),
                             "wilson_low": rnd(w_low), "wilson_high": rnd(w_high)})
    frame = pd.DataFrame(rows)
    summary = {f"{r.backbone}/{r.variant}/{r.condition}/{r.location}": f"{r.recalled}/{r.images} = {r.recall:.2f} "
               f"[patients {r.patient_ci_low:.2f}, {r.patient_ci_high:.2f}; Wilson {r.wilson_low:.2f}, {r.wilson_high:.2f}]"
               for r in frame.itertuples()}
    return frame, {"recall": summary, "note": "Descriptive only. Patient intervals reuse the RQ1 resamples restricted to the "
                                              "stratum's images; Wilson intervals treat images as independent."}


# --- 11. Training variability ---------------------------------------------------------------------

def training_item(args: argparse.Namespace, backbones: list[str], train: pd.DataFrame, y_train: np.ndarray,
                  hk: pd.DataFrame, y_hk: np.ndarray, test: pd.DataFrame, y: np.ndarray, truth: np.ndarray,
                  q_img: dict, rq1: dict, seed: int, max_iter: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, summary = [], {}
    patients = train["patient_group"].to_numpy()
    for b in backbones:
        info = rq1["raw"]["backbones"][b]
        x_train = feature_rows(args.features / "sage" / b / "raw.npz", train["sample_uuid"])
        x_test = feature_rows(args.features / "sage" / b / "raw.npz", test["sample_uuid"])
        x_hk = feature_rows(args.features / "hk" / b / "raw.npz", hk["file_name"])
        sage_rng, hk_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        tasks = [("ceiling", patient_bootstrap_rows(patients, sage_rng)) for _ in range(args.train_resamples)]
        tasks += [("transfer", stratified_bootstrap_rows(y_hk, hk_rng)) for _ in range(args.hk_resamples)]
        data = {"ceiling": (x_train, y_train, float(info["C_sage"])), "transfer": (x_hk, y_hk, float(info["C_hyperkvasir"]))}
        results = Parallel(n_jobs=args.jobs)(
            delayed(fit_predict_with)(data[c][0], data[c][1], x_test, data[c][2], seed, K, train_rows=rows_, threads=1,
                                      max_iter=max_iter) for c, rows_ in tasks)
        scores = {"ceiling": [], "transfer": []}
        for (cond, _), proba in zip(tasks, results):
            out = evaluate(y, proba, truth)
            scores[cond].append((out["macro_f1"], out["macro_auroc"]))
        q = q_img[(b, "raw")]
        entry: dict[str, Any] = {}
        for metric, col in (("macro_f1", 0), ("macro_auroc", 1)):
            values = {c: np.array([s[col] for s in scores[c]]) for c in scores}
            sd = {c: float(values[c].std(ddof=1)) for c in values}
            gap = q[f"gap_{metric}"]
            se_test = gap.se()
            se_total = float(np.sqrt(se_test ** 2 + sd["ceiling"] ** 2 + sd["transfer"] ** 2))
            low, high = percentile_ci(gap.boot, args.level)
            t_rep = q[f"transfer_{metric}"]
            t_total = float(np.sqrt(t_rep.se() ** 2 + sd["transfer"] ** 2))
            entry[metric] = {
                "ceiling_refits": int(len(values["ceiling"])), "transfer_refits": int(len(values["transfer"])),
                "ceiling_training_sd": rnd(sd["ceiling"]), "transfer_training_sd": rnd(sd["transfer"]),
                "ceiling_refit_mean": rnd(values["ceiling"].mean()), "transfer_refit_mean": rnd(values["transfer"].mean()),
                "ceiling_test_se": rnd(q[f"ceiling_{metric}"].se()), "transfer_test_se": rnd(t_rep.se()),
                "gap_estimate": rnd(gap.point), "gap_test_se": rnd(se_test), "gap_total_se": rnd(se_total),
                "gap_se_inflation": rnd(se_total / se_test - 1, 3), "gap_percentile_ci": [rnd(low), rnd(high)],
                "gap_widened_ci": [rnd(float(gap.point) - 1.96 * se_total), rnd(float(gap.point) + 1.96 * se_total)],
                "transfer_widened_ci": [rnd(float(t_rep.point) - 1.96 * t_total), rnd(float(t_rep.point) + 1.96 * t_total)]}
            rows.append({"backbone": b, "variant": "raw", "metric": metric, **{k: v for k, v in entry[metric].items()
                                                                               if not isinstance(v, list)},
                         "gap_ci_low": entry[metric]["gap_percentile_ci"][0], "gap_ci_high": entry[metric]["gap_percentile_ci"][1],
                         "gap_widened_low": entry[metric]["gap_widened_ci"][0],
                         "gap_widened_high": entry[metric]["gap_widened_ci"][1],
                         "transfer_widened_low": entry[metric]["transfer_widened_ci"][0],
                         "transfer_widened_high": entry[metric]["transfer_widened_ci"][1]})
        summary[f"{b}/raw"] = entry
        print(f"  training variability {b}: {len(tasks)} refits", flush=True)
    summary["note"] = ("The SAGE-only reference is refitted on patient-level bootstrap resamples of the 145 train patients "
                       "and the transfer probe on class-stratified bootstrap resamples of the HyperKvasir images (no "
                       "patient IDs, so a lower bound), C fixed at the RQ1 values, one BLAS thread. Total SE = "
                       "sqrt(test SE^2 + reference SD^2 + transfer SD^2), assuming independent sources; widened "
                       "interval = estimate +- 1.96 total SE (normal approximation).")
    return pd.DataFrame(rows), summary


# --- 12. Numerical robustness ---------------------------------------------------------------------

def numerics_item(args: argparse.Namespace, backbones: list[str], train: pd.DataFrame, y_train: np.ndarray,
                  hk: pd.DataFrame, y_hk: np.ndarray, test: pd.DataFrame, y: np.ndarray, truth: np.ndarray,
                  cells: dict, counts: np.ndarray, groups: np.ndarray, rq1: dict, cfg: dict,
                  bar: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed, max_iter, grid = int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"]), cfg["probe"]["C_grid"]
    hk_folds = official_folds(hk["fold2"].astype(int).to_numpy())
    sage_folds = grouped_folds(y_train, train["patient_group"].to_numpy(), 5, seed)
    rows, c_changes = [], []

    def score(proba: np.ndarray) -> float:
        return float(evaluate(y, proba, truth)["macro_f1"])

    def h2(transfer: np.ndarray, ceiling: np.ndarray) -> tuple[bool, bool]:
        rt, rc = (metric_replicates(PatientScores(y, p, None, groups), counts)["macro_f1"] for p in (transfer, ceiling))
        t_low = percentile_ci(rt.boot, args.level)[0]
        g_low, g_high = percentile_ci((rc - rt).boot, args.level)
        return bool(t_low > bar), excludes_zero(g_low, g_high)

    for v in VARIANTS:
        for b in backbones:
            info = rq1[v]["backbones"][b]
            x_train = feature_rows(args.features / "sage" / b / f"{v}.npz", train["sample_uuid"])
            x_test = feature_rows(args.features / "sage" / b / f"{v}.npz", test["sample_uuid"])
            x_hk = feature_rows(args.features / "hk" / b / f"{v}.npz", hk["file_name"])
            data = {"transfer": (x_hk, y_hk, float(info["C_hyperkvasir"]), hk_folds),
                    "ceiling": (x_train, y_train, float(info["C_sage"]), sage_folds)}
            settings = {"tol_1e-6": (TIGHT_TOL, 1), "one_thread": (None, 1)}
            tasks = [(c, s) for c in data for s in settings]
            results = Parallel(n_jobs=args.jobs)(
                delayed(fit_predict_with)(data[c][0], data[c][1], x_test, data[c][2], seed, K, tol=settings[s][0],
                                          threads=settings[s][1], max_iter=max_iter) for c, s in tasks)
            refit = dict(zip(tasks, results))
            for s in settings:
                h2_t, h2_g = h2(refit[("transfer", s)], refit[("ceiling", s)])
                for c in data:
                    saved = cells[(b, v)][c]
                    rows.append({"backbone": b, "variant": v, "probe": c, "setting": s, "C": data[c][2],
                                 "saved_macro_f1": rnd(score(saved)), "refit_macro_f1": rnd(score(refit[(c, s)])),
                                 "delta_macro_f1": rnd(score(refit[(c, s)]) - score(saved), 6),
                                 "max_abs_delta_proba": rnd(np.abs(refit[(c, s)] - saved).max(), 6),
                                 "changed_predictions": int((refit[(c, s)].argmax(1) != saved.argmax(1)).sum()),
                                 "h2_transfer_above_bar": h2_t, "h2_gap_excludes_zero": h2_g})
            for c, (x, labels, c_rq1, folds) in data.items():
                chosen, cv = select_c_with(x, labels, folds, grid, seed, K, tol=TIGHT_TOL, threads=1, jobs=args.jobs,
                                           max_iter=max_iter)
                saved_cv = info["cv_macro_f1_hyperkvasir" if c == "transfer" else "cv_macro_f1_sage"]
                entry = {"backbone": b, "variant": v, "probe": c, "setting": "c_selection_tol_1e-6", "C": chosen,
                         "C_rq1": c_rq1, "C_changed": bool(chosen != c_rq1),
                         "max_abs_delta_cv_macro_f1": rnd(max(abs(cv[g] - float(saved_cv[str(g)])) for g in grid), 6)}
                if chosen != c_rq1:
                    proba = fit_predict_with(x, labels, x_test, chosen, seed, K, tol=TIGHT_TOL, threads=1,
                                             max_iter=max_iter)
                    entry.update(refit_macro_f1=rnd(score(proba)),
                                 delta_macro_f1=rnd(score(proba) - score(cells[(b, v)][c]), 6))
                    c_changes.append({"probe": f"{b}/{v}/{c}", "C_rq1": c_rq1, "C_tol_1e-6": chosen,
                                      "cv_macro_f1_rq1": {str(g): rnd(saved_cv[str(g)]) for g in grid},
                                      "cv_macro_f1_tol_1e-6": {str(g): rnd(cv[g]) for g in grid},
                                      "test_macro_f1_saved": rnd(score(cells[(b, v)][c])),
                                      "test_macro_f1_new_C_tol_1e-6": rnd(score(proba))})
                rows.append(entry)
            print(f"  numerics {b}/{v} done", flush=True)
    frame = pd.DataFrame(rows)
    fits = frame[frame["setting"] != "c_selection_tol_1e-6"]
    summary: dict[str, Any] = {}
    for s, part in fits.groupby("setting"):
        worst = part.loc[part["delta_macro_f1"].abs().idxmax()]
        summary[s] = {"max_abs_delta_macro_f1": rnd(part["delta_macro_f1"].abs().max(), 4),
                      "worst": f"{worst['backbone']}/{worst['variant']}/{worst['probe']}: {worst['saved_macro_f1']:.4f} -> "
                               f"{worst['refit_macro_f1']:.4f}",
                      "max_abs_delta_proba": rnd(part["max_abs_delta_proba"].max(), 6),
                      "probes_with_changed_predictions": int((part["changed_predictions"] > 0).sum()),
                      "h2_cells_failing": sorted({f"{r.backbone}/{r.variant}" for r in part.itertuples()
                                                  if not (r.h2_transfer_above_bar and r.h2_gap_excludes_zero)})}
    summary["c_selection_tol_1e-6"] = {"C_changes": c_changes,
                                       "probes_checked": int((frame["setting"] == "c_selection_tol_1e-6").sum())}
    summary["note"] = ("Saved RQ1 predictions were fitted at the default lbfgs tolerance (1e-4) with all BLAS threads. "
                       "Refits use one BLAS thread; rq1 outputs are not changed.")
    return frame, summary


# --- Converged-fit check (--converged) ------------------------------------------------------------

def count_warnings(warn: dict[str, int], fits: list[tuple[np.ndarray, int, int]]) -> None:
    for _, first, left in fits:
        warn["fits"] += 1
        warn["convergence_warnings"] += first
        warn["left_after_retry"] += left


def converged_rq1(args: argparse.Namespace, cfg: dict, ctx: dict) -> tuple[dict, pd.DataFrame, dict[str, int]]:
    """Re-select C (pre-registered folds, grid and tie rule) and refit both RQ1 probes at tolerance 1e-6."""
    seed, max_iter, grid = int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"]), cfg["probe"]["C_grid"]
    train, test, hk = ctx["train"], ctx["test"], ctx["hk"]
    folds = {"transfer": official_folds(hk["fold2"].astype(int).to_numpy()),
             "ceiling": grouped_folds(ctx["y_train"], train["patient_group"].to_numpy(), 5, seed)}
    probs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    rows, warn = [], {"fits": 0, "convergence_warnings": 0, "left_after_retry": 0}
    for v in VARIANTS:
        data = {}
        for b in ctx["backbones"]:
            sage_path = args.features / "sage" / b / f"{v}.npz"
            x_test = feature_rows(sage_path, test["sample_uuid"])
            data[(b, "transfer")] = (feature_rows(args.features / "hk" / b / f"{v}.npz", hk["file_name"]), ctx["y_hk"], x_test)
            data[(b, "ceiling")] = (feature_rows(sage_path, train["sample_uuid"]), ctx["y_train"], x_test)
        tasks = [(key, C, k) for key in data for C in grid for k in range(len(folds[key[1]]))]
        fold_fits = Parallel(n_jobs=args.jobs)(
            delayed(fit_tight)(data[key][0], data[key][1], data[key][0], C, seed, K, folds[key[1]][k][0],
                               folds[key[1]][k][1], tol=TIGHT_TOL, max_iter=max_iter) for key, C, k in tasks)
        chosen = {key: cv_choice(data[key][1], folds[key[1]], grid,
                                 {(C, k): fit[0] for (kk, C, k), fit in zip(tasks, fold_fits) if kk == key}, K)
                  for key in data}
        finals = Parallel(n_jobs=args.jobs)(
            delayed(fit_tight)(data[key][0], data[key][1], data[key][2], chosen[key][0], seed, K, tol=TIGHT_TOL,
                               max_iter=max_iter) for key in data)
        count_warnings(warn, [*fold_fits, *finals])
        for (b, probe), fit in zip(data, finals):
            probs.setdefault((b, v), {})[probe] = fit[0]
            saved = ctx["rq1"][v]["backbones"][b]
            saved_cv = saved["cv_macro_f1_hyperkvasir" if probe == "transfer" else "cv_macro_f1_sage"]
            rows.append({"backbone": b, "variant": v, "probe": probe,
                         "C_default": float(saved["C_hyperkvasir" if probe == "transfer" else "C_sage"]),
                         "C_converged": chosen[(b, probe)][0],
                         **{f"cv_default_{g}": rnd(saved_cv[str(g)]) for g in grid},
                         **{f"cv_converged_{g}": rnd(chosen[(b, probe)][1][g]) for g in grid}})
        print(f"  converged RQ1 {v}: {len(tasks) + len(finals)} fits", flush=True)
    return {key: probs[key] for key in ctx["q_default"]}, pd.DataFrame(rows), warn


def converged_rq3(args: argparse.Namespace, cfg: dict, ctx: dict, c_conv: dict[tuple[str, str, str], float],
                  reference: dict[tuple[str, str], float]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, int]]:
    """The RQ3 design of scripts/10 (same draws and seeds) with converged fits at the converged C values."""
    rq3, seed, max_iter = cfg["rq3"], int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"])
    train, test, hk, y, truth = ctx["train"], ctx["test"], ctx["hk"], ctx["y"], ctx["truth"]
    patients = np.sort(train["patient_group"].unique())
    draws = rq3_draws(patients, [int(k) for k in rq3["k_patients"]], int(rq3["draws_per_k"]),
                      int(cfg["seeds"]["rq3_patient_draws"]))
    train_groups = train["patient_group"].to_numpy()
    saved_runs = pd.read_csv(args.rq3 / "rq3_runs.csv").set_index(["backbone", "variant", "regime", "k", "draw"])
    run_rows, boots, warn = [], {}, {"fits": 0, "convergence_warnings": 0, "left_after_retry": 0}
    for v in CHECK_VARIANTS:
        data = {b: (feature_rows(args.features / "hk" / b / f"{v}.npz", hk["file_name"]),
                    feature_rows(args.features / "sage" / b / f"{v}.npz", train["sample_uuid"]),
                    feature_rows(args.features / "sage" / b / f"{v}.npz", test["sample_uuid"])) for b in ctx["backbones"]}
        tasks = [(b, regime, k, draw, np.flatnonzero(np.isin(train_groups, members)))
                 for b in ctx["backbones"] for regime in ("hk_plus_k", "sage_only")
                 for k, draw, members in draws if not (regime == "sage_only" and k == 0)]
        probe = {"hk_plus_k": "transfer", "sage_only": "ceiling"}
        fits = Parallel(n_jobs=args.jobs)(
            delayed(rq3_fit)(data[b][0], ctx["y_hk"], data[b][1], ctx["y_train"], rows, data[b][2], regime,
                             c_conv[(b, v, probe[regime])], seed, K, TIGHT_TOL, max_iter)
            for b, regime, k, draw, rows in tasks)
        count_warnings(warn, fits)
        for (b, regime, k, draw, rows), (proba, _, _) in zip(tasks, fits):
            saved = saved_runs.loc[(b, v, regime, k, draw)]
            classes = int(len(np.unique(ctx["y_train"][rows])))
            if int(saved["sage_images"]) != len(rows) or int(saved["classes_covered"]) != classes:
                raise SystemExit(f"RQ3 draw differs from scripts/10: {b}/{v}/{regime}/k={k}/draw {draw}")
            point = evaluate(y, proba, truth)
            run_rows.append({"backbone": b, "variant": v, "regime": regime, "k": k, "draw": draw,
                             "sage_images": len(rows), "classes_covered": classes, "C": c_conv[(b, v, probe[regime])],
                             "macro_f1": round(point["macro_f1"], 4), "macro_auroc": round(point["macro_auroc"], 4),
                             "default_C": float(saved["C"]), "default_macro_f1": float(saved["macro_f1"])})
            boots.setdefault((b, v, regime, k), []).append(macro_f1_many(y, proba.argmax(axis=1), K, ctx["weights"]))
        print(f"  converged RQ3 {v}: {len(tasks)} fits", flush=True)

    runs = pd.DataFrame(run_rows)
    saved_curves = pd.read_csv(args.rq3 / "rq3_curves.csv").set_index(["backbone", "variant", "regime", "k"])
    n_all, curves = len(patients), []
    for (b, v, regime, k), part in runs.groupby(["backbone", "variant", "regime", "k"], sort=False):
        mean_boot = np.mean(boots[(b, v, regime, k)], axis=0)
        ref_boot = boots[(b, v, "sage_only", n_all)][0]
        ref_point = runs[(runs["backbone"] == b) & (runs["variant"] == v) & (runs["regime"] == "sage_only")
                         & (runs["k"] == n_all)]["macro_f1"].iloc[0]
        low, high = percentile_ci(mean_boot, ctx["level"])
        d_low, d_high = percentile_ci(mean_boot - ref_boot, ctx["level"])
        saved = saved_curves.loc[(b, v, regime, k)]
        curves.append({"backbone": b, "variant": v, "regime": regime, "k": int(k), "draws": len(part),
                       "macro_f1_mean": round(part["macro_f1"].mean(), 4), "ci_low": rnd(low), "ci_high": rnd(high),
                       "minus_reference": round(part["macro_f1"].mean() - ref_point, 4),
                       "minus_reference_ci_low": rnd(d_low), "minus_reference_ci_high": rnd(d_high),
                       "default_macro_f1_mean": float(saved["macro_f1_mean"]),
                       "default_minus_reference": float(saved["minus_ceiling"]),
                       "default_minus_reference_ci_low": float(saved["minus_ceiling_ci_low"]),
                       "default_minus_reference_ci_high": float(saved["minus_ceiling_ci_high"])})
    curves_frame = pd.DataFrame(curves)
    saved_summary = read_json(args.rq3 / "rq3_summary.json")["results"]
    results = {}
    for (b, v), part in curves_frame.groupby(["backbone", "variant"], sort=False):
        ref = round(reference[(b, v)], 4)
        target = float(rq3["target_fraction_of_ceiling"]) * ref
        entry: dict[str, Any] = {"reference_macro_f1": ref, "target_macro_f1": round(target, 4)}
        for regime in ("hk_plus_k", "sage_only"):
            reached = part[(part["regime"] == regime) & (part["macro_f1_mean"] >= target)].sort_values("k")
            entry[f"smallest_k_{regime}"] = int(reached["k"].iloc[0]) if len(reached) else None
        entry["hk_plus_k_above_reference_at_k"] = part[(part["regime"] == "hk_plus_k")
                                                       & (part["macro_f1_mean"] > ref)]["k"].tolist()
        last = part[(part["regime"] == "hk_plus_k") & (part["k"] == n_all)].iloc[0]
        entry["hk_plus_145_minus_reference"] = [last["minus_reference"], last["minus_reference_ci_low"],
                                                last["minus_reference_ci_high"]]
        entry["sage_only_145_minus_rq1_reference"] = round(float(
            part[(part["regime"] == "sage_only") & (part["k"] == n_all)]["macro_f1_mean"].iloc[0]) - ref, 4)
        default = saved_summary[f"{b}/{v}"]
        entry["default"] = {"reference_macro_f1": default["ceiling_macro_f1"],
                            "smallest_k_hk_plus_k": default["smallest_k_hk_plus_k"],
                            "smallest_k_sage_only": default["smallest_k_sage_only"],
                            "hk_plus_k_above_reference_at_k": default["hk_plus_k_above_ceiling_at_k"],
                            "hk_plus_145_minus_reference": [last["default_minus_reference"],
                                                            last["default_minus_reference_ci_low"],
                                                            last["default_minus_reference_ci_high"]]}
        results[f"{b}/{v}"] = entry
    return runs, curves_frame, results, warn


def h3_family(q: dict, level: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """H3 per cell with bootstrap p-values, Holm over the 12 cells and over the 3 raw cells."""
    keys = list(q)
    reps = [q[key]["h3_shifted_minus_matched"] for key in keys]
    p = np.array([bootstrap_p(r.boot) for r in reps])
    raw = [i for i, key in enumerate(keys) if key[1] == "raw"]
    holm12, holm3 = holm(p), dict(zip(raw, holm(p[raw])))
    rows = []
    for i, (b, v) in enumerate(keys):
        e = ci_entry(reps[i], level)
        rows.append({"backbone": b, "variant": v, "estimate": rnd(e["estimate"]), "ci_low": rnd(e["ci_low"]),
                     "ci_high": rnd(e["ci_high"]), "p_boot": sig(p[i]), "holm12_p": sig(holm12[i]),
                     "holm3_raw_p": sig(holm3[i]) if i in holm3 else None})
    frame = pd.DataFrame(rows)
    alpha = 1 - level
    summary = {"cells": len(keys),
               "nominal": [f"{r.backbone}/{r.variant}" for r in frame.itertuples() if excludes_zero(r.ci_low, r.ci_high)],
               "holm12": [f"{r.backbone}/{r.variant}" for r in frame.itertuples() if r.holm12_p <= alpha],
               "holm3_raw": [f"{r.backbone}/{r.variant}" for r in frame.itertuples()
                             if r.holm3_raw_p is not None and r.holm3_raw_p <= alpha]}
    return frame, summary


def h3_verdict(s: dict[str, Any]) -> str:
    if s["holm12"]:
        return f"Supported after Holm over {s['cells']} cells in {s['holm12']}"
    if s["nominal"]:
        return (f"Not supported: CI excludes 0 in {len(s['nominal'])} of {s['cells']} cells ({', '.join(s['nominal'])}), "
                f"none after Holm")
    return f"Not supported: no CI excludes 0 ({s['cells']} cells)"


def converged_main(args: argparse.Namespace, cfg: dict, ctx: dict, reproduction: dict[str, Any]) -> int:
    """Every pre-registered probe analysis refitted at tolerance 1e-6 with C re-selected; converged_* outputs."""
    level, bar, timings, started = ctx["level"], ctx["bar"], {}, time.time()
    probs, c_table, warn_rq1 = converged_rq1(args, cfg, ctx)
    timings["rq1_refits"] = time.time() - started
    q_conv = {key: cell_quantities(*(metric_replicates(PatientScores(ctx["y"], p[c], ctx["truth"], ctx["groups"]),
                                                       ctx["counts"]) for c in ("transfer", "ceiling")))
              for key, p in probs.items()}
    add_variant_changes(q_conv, ctx["backbones"])
    q_def = ctx["q_default"]

    rq1_rows, h2 = [], {}
    for (b, v), q in q_conv.items():
        saved = ctx["rq1"][v]["backbones"][b]["metrics"]
        for metric in ("macro_f1", "macro_auroc"):
            for cond in ("transfer", "ceiling", "gap"):
                d, c = saved[metric][cond], ci_entry(q[f"{cond}_{metric}"], level)
                rq1_rows.append({"backbone": b, "variant": v, "metric": metric, "condition": cond,
                                 "default_estimate": d["estimate"], "default_ci_low": d["ci_low"], "default_ci_high": d["ci_high"],
                                 "converged_estimate": rnd(c["estimate"]), "converged_ci_low": rnd(c["ci_low"]),
                                 "converged_ci_high": rnd(c["ci_high"]), "change": rnd(c["estimate"] - d["estimate"])})
        t, g = ci_entry(q["transfer_macro_f1"], level), ci_entry(q["gap_macro_f1"], level)
        h2[f"{b}/{v}"] = {"default": ctx["rq1"][v]["backbones"][b]["h2"],
                          "converged": {"transfer_ci_above_best_constant": bool(t["ci_low"] > bar),
                                        "gap_ci_excludes_zero": excludes_zero(g["ci_low"], g["ci_high"])}}
    rq1_frame = pd.DataFrame(rq1_rows)

    h3_def, h3_def_s = h3_family(q_def, level)
    h3_conv, h3_conv_s = h3_family(q_conv, level)
    h3_frame = h3_def.merge(h3_conv, on=["backbone", "variant"], suffixes=("_default", "_converged"))
    rq2_rows = []
    for (b, v), q in q_conv.items():
        for name in ("transfer_change_macro_f1", "gap_reduction_macro_f1"):
            if name in q:
                d, c = ci_entry(q_def[(b, v)][name], level), ci_entry(q[name], level)
                rq2_rows.append({"backbone": b, "variant": v, "quantity": name, **{f"default_{k}": rnd(x) for k, x in d.items()},
                                 **{f"converged_{k}": rnd(x) for k, x in c.items()}})
    rq2_frame = pd.DataFrame(rq2_rows)

    tick = time.time()
    c_conv = {(r.backbone, r.variant, r.probe): float(r.C_converged) for r in c_table.itertuples()}
    reference = {key: float(q["ceiling_macro_f1"].point) for key, q in q_conv.items()}
    runs, curves, rq3_results, warn_rq3 = converged_rq3(args, cfg, ctx, c_conv, reference)
    timings["rq3_refits"] = time.time() - tick

    table = verdict_table(args, cfg, ctx, h2, h3_def_s, h3_conv_s, rq2_frame, rq3_results)
    changes = largest_changes(rq1_frame, h3_frame, rq2_frame, curves, rq3_results)
    summary = {"settings": {"tol": TIGHT_TOL, "max_iter": int(cfg["probe"]["max_iter"]), "retry_max_iter": 50000,
                            "c_grid": cfg["probe"]["C_grid"], "c_selection": cfg["probe"]["c_selection"],
                            "resamples": int(cfg["bootstrap"]["n_resamples"]), "bootstrap_seed": int(cfg["seeds"]["bootstrap"]),
                            "rq3": cfg["rq3"], "one_blas_thread_per_fit": True},
               "reproduction_check": reproduction,
               "convergence_warnings": {"rq1": warn_rq1, "rq3": warn_rq3},
               "c_changes": [f"{r.backbone}/{r.variant}/{r.probe}: {r.C_default} -> {r.C_converged}"
                             for r in c_table.itertuples() if r.C_default != r.C_converged],
               "h2": h2, "h3": {"default": h3_def_s, "converged": h3_conv_s},
               "h4": {v: {est: ci_entry(q[("resnet50_imagenet", v)]["gap_reduction_macro_f1"], level)
                          for est, q in (("default", q_def), ("converged", q_conv))} for v in ("fov", "colour")},
               "rq3": rq3_results, "verdicts": table, "largest_changes": changes,
               "note": ("Converged refits: lbfgs tolerance 1e-6 for every probe, C re-selected with the pre-registered "
                        "folds, grid and tie rule, one BLAS thread per fit. Default columns are the saved RQ1-RQ3 "
                        "results (RQ1 AUROC from unrounded probabilities; H3 and RQ2 from the saved predictions). "
                        "AUROC here uses unrounded converged probabilities.")}
    timings["total"] = time.time() - ctx["started"]
    summary["runtime_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("c_choice", c_table), ("rq1", rq1_frame), ("h3", h3_frame), ("rq2", rq2_frame),
                        ("rq3_runs", runs), ("rq3_curves", curves), ("verdicts", pd.DataFrame(table))):
        frame.to_csv(args.out_dir / f"converged_{name}.csv", index=False)
    (args.out_dir / "converged_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print("\nConvergenceWarnings:", summary["convergence_warnings"])
    print("C changes:", summary["c_changes"] or "none")
    print("\nVerdicts (default tolerance | converged):")
    for row in table:
        print(f"  {row['hypothesis']}: {row['default_verdict']} | {row['converged_verdict']}")
        print(f"      default:   {row['default_evidence']}\n      converged: {row['converged_evidence']}")
    print("\nLargest absolute changes:")
    for c in changes:
        print(f"  {c['quantity']:34s} {c['largest_abs_change']} ({c['where']}: {c['default']} -> {c['converged']})")
    print(f"\nWrote converged_* tables and converged_summary.json to {args.out_dir} ({timings['total']:.0f} s)")
    return 0


def verdict_table(args: argparse.Namespace, cfg: dict, ctx: dict, h2: dict, h3_def: dict, h3_conv: dict,
                  rq2_frame: pd.DataFrame, rq3: dict) -> list[dict[str, str]]:
    """H1-H5 by the pre-registered rules, for the default-tolerance and the converged results."""
    proba = aligned(args.paper_recipe / "faithful_seed0_sage_probs.csv", ctx["test"]["sample_uuid"])
    share = float(np.bincount(proba.argmax(axis=1), minlength=K).max() / len(proba))
    h1 = "Supported" if share >= float(cfg["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"]) \
        else "Not supported"
    h1_evidence = f"faithful re-run, seed 0: top class {share:.1%} of predictions (no probe refitted)"
    rows = [{"hypothesis": "H1", "rule": "top predicted class >= 80% of the 158 predictions (faithful re-run)",
             "default_verdict": h1, "default_evidence": h1_evidence, "converged_verdict": h1, "converged_evidence": h1_evidence}]

    def h2_text(est: str) -> tuple[str, str]:
        failing = [cell for cell, e in h2.items() if not all(e[est].values())]
        verdict = f"Supported ({len(h2) - len(failing)}/{len(h2)} cells)" if not failing else f"Not supported: fails for {failing}"
        return verdict, f"cells failing a condition: {failing or 'none'}"
    rows.append({"hypothesis": "H2", "rule": "every transfer macro-F1 CI above 0.0884 and every gap CI excludes 0",
                 **dict(zip(("default_verdict", "default_evidence"), h2_text("default"))),
                 **dict(zip(("converged_verdict", "converged_evidence"), h2_text("converged")))})
    rows.append({"hypothesis": "H3", "rule": "CI of shifted - matched mean per-class F1 gap excludes 0 (Holm over 12 cells)",
                 "default_verdict": h3_verdict(h3_def), "default_evidence": f"Holm over the 3 raw cells: {h3_def['holm3_raw'] or 'none'}",
                 "converged_verdict": h3_verdict(h3_conv),
                 "converged_evidence": f"Holm over the 3 raw cells: {h3_conv['holm3_raw'] or 'none'}"})

    def h4_text(est: str) -> tuple[str, str]:
        part = rq2_frame[(rq2_frame["backbone"] == "resnet50_imagenet") & (rq2_frame["quantity"] == "gap_reduction_macro_f1")
                         & rq2_frame["variant"].isin(["fov", "colour"])]
        ok = bool(len(part) == 2 and (part[f"{est}_ci_low"] > 0).all())
        evidence = "; ".join(f"{r.variant} {getattr(r, est + '_estimate'):+.3f} [{getattr(r, est + '_ci_low'):.3f}, "
                             f"{getattr(r, est + '_ci_high'):.3f}]" for r in part.itertuples())
        return ("Supported" if ok else "Not supported"), f"ImageNet gap reduction: {evidence}; in-domain condition not tested"
    rows.append({"hypothesis": "H4", "rule": "ImageNet ResNet-50 gap-reduction CI excludes 0 for fov and for colour",
                 **dict(zip(("default_verdict", "default_evidence"), h4_text("default"))),
                 **dict(zip(("converged_verdict", "converged_evidence"), h4_text("converged")))})

    def h5_text(est: str) -> tuple[str, str]:
        def pick(entry: dict) -> dict:
            return entry["default"] if est == "default" else entry
        ks = {cell: pick(e)["smallest_k_hk_plus_k"] for cell, e in rq3.items()}
        ok = any(k is not None and k <= 20 for k in ks.values())
        above = {cell: pick(e)["hk_plus_145_minus_reference"] for cell, e in rq3.items()
                 if pick(e)["hk_plus_145_minus_reference"][1] > 0}
        evidence = (f"smallest k: {ks}; HyperKvasir + 145 above the reference with CI above 0: "
                    f"{ {c: [round(float(x), 3) for x in v] for c, v in above.items()} or 'none'}")
        return ("Supported" if ok else "Not supported"), evidence
    rows.append({"hypothesis": "H5", "rule": "smallest k reaching 90% of the SAGE-only reference <= 20 for some backbone",
                 **dict(zip(("default_verdict", "default_evidence"), h5_text("default"))),
                 **dict(zip(("converged_verdict", "converged_evidence"), h5_text("converged")))})
    return rows


def largest_changes(rq1: pd.DataFrame, h3: pd.DataFrame, rq2: pd.DataFrame, curves: pd.DataFrame,
                    rq3: dict) -> list[dict[str, Any]]:
    """The largest absolute change (converged minus default) in each headline number."""
    out = []

    def add(name: str, frame: pd.DataFrame, default: str, converged: str, where: list[str]) -> None:
        diff = (frame[converged] - frame[default]).abs()
        row = frame.loc[diff.idxmax()]
        out.append({"quantity": name, "largest_abs_change": rnd(diff.max()), "where": "/".join(str(row[w]) for w in where),
                    "default": row[default], "converged": row[converged]})
    for metric in ("macro_f1", "macro_auroc"):
        for cond in ("transfer", "ceiling", "gap"):
            part = rq1[(rq1["metric"] == metric) & (rq1["condition"] == cond)]
            add(f"rq1_{cond}_{metric}", part, "default_estimate", "converged_estimate", ["backbone", "variant"])
    add("h3_shifted_minus_matched", h3, "estimate_default", "estimate_converged", ["backbone", "variant"])
    for name in ("transfer_change_macro_f1", "gap_reduction_macro_f1"):
        add(f"rq2_{name}", rq2[rq2["quantity"] == name], "default_estimate", "converged_estimate", ["backbone", "variant"])
    for regime in ("hk_plus_k", "sage_only"):
        add(f"rq3_{regime}_mean_macro_f1", curves[curves["regime"] == regime], "default_macro_f1_mean", "macro_f1_mean",
            ["backbone", "variant", "k"])
    last = curves[(curves["regime"] == "hk_plus_k") & (curves["k"] == curves["k"].max())]
    add("rq3_hk_plus_145_minus_reference", last, "default_minus_reference", "minus_reference", ["backbone", "variant"])
    out.append({"quantity": "rq3_smallest_k_hk_plus_k", "largest_abs_change": None, "where": "all",
                "default": {c: e["default"]["smallest_k_hk_plus_k"] for c, e in rq3.items()},
                "converged": {c: e["smallest_k_hk_plus_k"] for c, e in rq3.items()}})
    return out


# --- Main -----------------------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    threadpool_limits(limits=args.jobs)  # this process's BLAS threads; refit workers use one each
    started, timings = time.time(), {}
    n_boot, level, boot_seed = int(cfg["bootstrap"]["n_resamples"]), float(cfg["bootstrap"]["ci"]), int(cfg["seeds"]["bootstrap"])
    seed, max_iter = int(cfg["seeds"]["global"]), int(cfg["probe"]["max_iter"])
    args.level = level
    class_index = {name: i for i, name in enumerate(SIX_CLASSES)}

    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    six = sage[sage["in_six"] == 1]
    train = six[six["split"] == "train"].reset_index(drop=True)
    test = six[six["split"] == "test"].reset_index(drop=True)
    y, y_train = test["class"].map(class_index).to_numpy(), train["class"].map(class_index).to_numpy()
    truth = test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float)
    groups = test["patient_group"].to_numpy()
    weights = group_weights(groups, n_boot, boot_seed)
    counts = draw_counts(weights, groups)
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    y_hk = hk["sage_class"].map(class_index).to_numpy()
    rq1 = {v: read_json(args.rq1 / f"rq1_{v}_summary.json") for v in VARIANTS}
    bar = float(rq1["raw"]["best_constant_macro_f1"])

    pred = args.rq1 / "predictions"
    backbones = [b for b in BACKBONES if all((pred / f"{b}__{v}__{c}.csv").is_file() for v in VARIANTS
                                             for c in ("transfer", "ceiling"))]
    cells = {(b, v): {c: aligned(pred / f"{b}__{v}__{c}.csv", test["sample_uuid"]) for c in ("transfer", "ceiling")}
             for v in VARIANTS for b in backbones}
    cells = {key: cells[key] for key in sorted(cells, key=lambda k: (backbones.index(k[0]), VARIANTS.index(k[1])))}
    runs: dict[str, dict[str, dict[str, Replicates]]] = {}
    for path in sorted(args.paper_recipe.glob("*_seed*_summary.json")):
        tag = path.name.removesuffix("_summary.json")
        if re.fullmatch(r"(faithful|adjusted)_seed\d+", tag):
            runs[tag] = {name: metric_replicates(PatientScores(y, aligned(args.paper_recipe / f"{tag}_{stem}.csv",
                                                                          test["sample_uuid"]), None, groups), counts)
                         for name, stem in INPUTS.items() if (args.paper_recipe / f"{tag}_{stem}.csv").is_file()}
    print(f"SAGE six-class test: {len(test)} images, {len(np.unique(groups))} patients; {n_boot} paired patient "
          f"resamples (seed {boot_seed}); backbones {backbones}; paper-recipe runs {list(runs)}.", flush=True)

    inverse = inverse_size_weights(groups)
    scores = {key: {c: PatientScores(y, p, truth, groups) for c, p in probs.items()} for key, probs in cells.items()}
    img = {key: {c: metric_replicates(s, counts) for c, s in pair.items()} for key, pair in scores.items()}
    pat = {key: {c: metric_replicates(PatientScores(y, p, truth, groups, inverse), counts) for c, p in probs.items()}
           for key, probs in cells.items()}
    q_img = {key: cell_quantities(r["transfer"], r["ceiling"]) for key, r in img.items()}
    q_pat = {key: cell_quantities(r["transfer"], r["ceiling"]) for key, r in pat.items()}
    add_variant_changes(q_img, backbones)
    add_variant_changes(q_pat, backbones)
    first = next(iter(cells))
    check = direct_check(y, cells[first]["transfer"], truth, groups, weights, img[first]["transfer"], pat[first]["transfer"])
    summary: dict[str, Any] = {
        "settings": {"test_images": len(test), "test_patients": int(len(np.unique(groups))), "resamples": n_boot,
                     "bootstrap_seed": boot_seed, "level": level, "swaps": args.swaps, "swap_seed": boot_seed,
                     "training_seeds": {"sage_patients": seed, "hyperkvasir_images": seed + 1},
                     "fixed_h2_bar_best_constant": bar, "backbones": backbones, "paper_recipe_runs": list(runs),
                     "gap": "SAGE-only reference (ceiling) minus transfer"},
        "reproduction_check": reproduction_check(q_img, rq1, read_json(args.rq2 / "rq2_summary.json"),
                                                 pd.read_csv(args.rq2 / "rq2_table.csv"), level, check)}
    timings["setup"] = time.time() - started
    if args.converged:
        context = {"backbones": backbones, "train": train, "y_train": y_train, "hk": hk, "y_hk": y_hk, "test": test,
                   "y": y, "truth": truth, "groups": groups, "weights": weights, "counts": counts, "rq1": rq1,
                   "q_default": q_img, "bar": bar, "level": level, "started": started}
        return converged_main(args, cfg, context, summary["reproduction_check"])

    tables: dict[str, pd.DataFrame] = {}
    steps = [("patient_weighted", lambda: patient_weighted_item(q_img, q_pat, level, bar)),
             ("bca", lambda: bca_item(q_img, runs, level, bar)),
             ("few_cluster", lambda: few_cluster_item(
                 scores, q_img, counts, {c: int(test.loc[test["class"] == c, "patient_group"].nunique()) for c in SIX_CLASSES},
                 level, args.swaps, boot_seed)),
             ("h3_multiplicity", lambda: h3_item(q_img, level)),
             ("clustering", lambda: clustering_item(test, counts, groups)),
             ("power", lambda: power_item(q_img)),
             ("h2_without_heaviest", lambda: h2_subset_item(test, cells, y, truth, n_boot, boot_seed, level, bar)),
             ("mcc_balanced_accuracy", lambda: mcc_item(q_img, runs, level)),
             ("loo_influence", lambda: influence_item(q_img, runs, test, groups)),
             ("polyp_context_recall", lambda: polyp_context_item(test, cells, counts, groups, level))]
    if not args.skip_refits:
        train_backbones = args.training_backbones or backbones
        steps += [("training_variability", lambda: training_item(args, train_backbones, train, y_train, hk, y_hk, test, y,
                                                                 truth, q_img, rq1, seed, max_iter)),
                  ("numerical_robustness", lambda: numerics_item(args, backbones, train, y_train, hk, y_hk, test, y, truth,
                                                                 cells, counts, groups, rq1, cfg, bar))]
    for name, step in steps:
        tick = time.time()
        tables[name], summary[name] = step()
        timings[name] = time.time() - tick
        print(f"{name}: {timings[name]:.1f} s", flush=True)

    summary["verdict_flags"] = verdict_flags(summary)
    summary["runtime_seconds"] = {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.time() - started, 1)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    (args.out_dir / "sensitivity_summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    report(summary, tables)
    print(f"\nWrote {len(tables)} tables and sensitivity_summary.json to {args.out_dir} "
          f"({summary['runtime_seconds']['total']:.0f} s)")
    return 0


def verdict_flags(summary: dict[str, Any]) -> list[str]:
    """Sensitivity results that bear on a pre-registered verdict (H2, H3, H4)."""
    flags = []
    pw = summary["patient_weighted"]
    for cell in pw["h2_cells_failing_paired_lead_or_gap"]["patient_weighted"]:
        e = pw["h2"][cell]["patient_weighted"]
        flags.append(f"H2, patient-weighted: {cell} lead {fmt(e['lead'])}, gap {fmt(e['gap'])}")
    for cell in pw["h2_cells_failing_paired_lead_or_gap"]["image_weighted"]:
        flags.append(f"H2, image-weighted paired lead or gap fails: {cell}")
    for cell in summary["bca"]["h2_cells_failing"]:
        flags.append(f"H2, BCa: {cell} fails ({summary['bca']['h2'][cell]})")
    sub = summary["h2_without_heaviest"]
    for cell in sub["cells_failing_paired_lead"]:
        flags.append(f"H2 without the two heaviest patients: paired lead interval includes 0 for {cell}")
    for cell in sub["cells_failing_fixed_bar"]:
        bar = summary["settings"]["fixed_h2_bar_best_constant"]
        flags.append(f"H2 without the two heaviest patients: transfer interval not above {bar} for {cell}")
    for change in summary["few_cluster"]["interval_verdict_changes"]:
        if "gap_macro_f1" in change or "h3_" in change:
            flags.append(f"Bootstrap-t: {change}")
    h3 = summary["h3_multiplicity"]
    flags.append(f"H3: Holm over 12 cells rejects {h3['holm12_rejections'] or 'none'}; over the 3 raw cells (post hoc) "
                 f"{h3['holm3_raw_rejections'] or 'none'}")
    for cell, e in summary["bca"]["h3"].items():
        if e["excludes_zero"]:
            flags.append(f"H3, BCa interval excludes 0: {cell} {e['bca']}")
    h3_pw = summary["patient_weighted"]["h3_patient_weighted_tests"]
    flags.append(f"H3, patient-weighted: intervals exclude 0 for {len(h3_pw['cells_ci_excluding_zero'])} of 12 cells; "
                 f"Holm over 12 rejects {h3_pw['holm12_rejections'] or 'none'}")
    for v, e in summary["patient_weighted"]["h4_imagenet_gap_reduction"].items():
        pw_e = e["patient_weighted"]
        if v in ("fov", "colour") and excludes_zero(pw_e["ci_low"], pw_e["ci_high"]):
            flags.append(f"H4, patient-weighted ImageNet {v} gap reduction excludes 0: {fmt(pw_e)} (H4 needs both fov and colour)")
    for v, e in summary["bca"]["h4"].items():
        if e["excludes_zero"]:
            flags.append(f"H4, BCa ImageNet {v} gap reduction excludes 0: {e['bca']}")
    for key in ("tol_1e-6", "one_thread"):
        if key in summary.get("numerical_robustness", {}) and summary["numerical_robustness"][key]["h2_cells_failing"]:
            flags.append(f"Numerics ({key}): H2 fails for {summary['numerical_robustness'][key]['h2_cells_failing']}")
    for change in summary.get("numerical_robustness", {}).get("c_selection_tol_1e-6", {}).get("C_changes", []):
        flags.append(f"Numerics: C selection at tol 1e-6 changes {change['probe']}: C {change['C_rq1']} -> "
                     f"{change['C_tol_1e-6']}, test macro-F1 {change['test_macro_f1_saved']} -> "
                     f"{change['test_macro_f1_new_C_tol_1e-6']}")
    for cell, entry in summary.get("training_variability", {}).items():
        if cell != "note" and not excludes_zero(*entry["macro_f1"]["gap_widened_ci"]):
            flags.append(f"Training variability: widened gap interval includes 0 for {cell}")
    return flags


def report(summary: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    """Compact printout."""
    print("\nReproduction check:", summary["reproduction_check"])
    pw = tables["patient_weighted"]
    print("\n1. Patient-weighted (image-weighted) macro-F1: transfer | gap | lead over best constant")
    for (b, v), part in pw.groupby(["backbone", "variant"], sort=False):
        def pick(q: str) -> str:
            r = part[part["quantity"] == q].iloc[0]
            return (f"{r.patient_estimate:.3f} [{r.patient_ci_low:.3f}, {r.patient_ci_high:.3f}] "
                    f"({r.image_estimate:.3f})")
        print(f"  {b:23s} {v:10s} {pick('transfer_macro_f1'):30s} {pick('gap_macro_f1'):30s} "
              f"{pick('transfer_lead_over_best_constant')}")
    h4 = summary["patient_weighted"]["h4_imagenet_gap_reduction"]
    print("  H4 (ImageNet gap reduction):", {v: fmt(e["patient_weighted"]) for v, e in h4.items()})
    print(f"\n2. BCa: largest bound shift {summary['bca']['max_bound_shift']:.3f}; "
          f"H2 cells failing: {summary['bca']['h2_cells_failing']}")
    print("   H3 BCa:", {k: e["bca"] for k, e in summary["bca"]["h3"].items() if "dinov2" in k})
    print("   paper recipe fov - raw [BCa]:", summary["bca"]["paper_recipe_fov_minus_raw"])
    print("   BCa and percentile disagree on excluding 0:", summary["bca"]["interval_verdict_changes"] or "never")
    fc = summary["few_cluster"]
    print(f"\n3. Bootstrap-t gap: {fc['gap_bootstrap_t']}\n   swap p (gap): {fc['gap_swap_p']}")
    print(f"   per-class swap tests with p < 0.05: {fc['per_class_swap_p_below_0.05']} of {fc['per_class_tests']}; "
          f"interval verdict changes: {fc['interval_verdict_changes']}")
    h3 = summary["h3_multiplicity"]
    print(f"\n4. H3: smallest p {h3['smallest_p']}; Holm-12 rejects {h3['holm12_rejections']}; Holm-3 (raw) rejects "
          f"{h3['holm3_raw_rejections']}; pooled {fmt(h3['pooled'])}; AUROC H3 range {h3['auroc_estimate_range']}")
    row = tables["h3_multiplicity"].iloc[int(np.argmin(tables["h3_multiplicity"]["p_boot"].iloc[:-1]))]
    print(f"   {row.backbone}/{row.variant}: Bonferroni-12 [{row.bonferroni12_low}, {row.bonferroni12_high}], "
          f"Bonferroni-3 [{row.bonferroni3_low}, {row.bonferroni3_high}], max-|z| [{row.maxz_low}, {row.maxz_high}]")
    cl = summary["clustering"]
    print(f"\n5. Kish: {cl['kish']}; resamples missing a class: {cl['resamples_missing_by_class']}")
    print("\n6. Power:", {k: v for k, v in summary["power"].items() if k != "note"})
    sub = tables["h2_without_heaviest"]
    print(f"\n7. Without the two heaviest patients ({summary['h2_without_heaviest']['images']} images): transfer | lead")
    for r in sub.itertuples():
        print(f"  {r.backbone:23s} {r.variant:10s} {r.transfer_estimate:.3f} [{r.transfer_ci_low:.3f}, {r.transfer_ci_high:.3f}] "
              f"best constant {r.best_constant:.4f}; lead {r.lead_estimate:.3f} [{r.lead_ci_low:.3f}, {r.lead_ci_high:.3f}]")
    print(f"\n8. MCC: RQ1 transfer range {summary['mcc_balanced_accuracy']['rq1_transfer_mcc_range']}; paper recipe "
          f"{summary['mcc_balanced_accuracy']['paper_recipe_mcc']}")
    print("\n9. Largest leave-one-patient-out influences:")
    for key, e in summary["loo_influence"].items():
        top = e["largest"][0]
        print(f"  {key:52s} {top['delta']:+.3f} (patient with {top['n_images']} images: {top['classes']})")
    print("\n10. Polyp recall (transfer, raw):")
    for key, text in summary["polyp_context_recall"]["recall"].items():
        if key.endswith(("transfer/upper GI", "transfer/lower GI", "transfer/no location label")) and "/raw/" in key:
            print(f"  {key:60s} {text}")
    if "training_variability" in summary:
        print("\n11. Training variability (macro-F1):")
        for key, e in summary["training_variability"].items():
            if key != "note":
                m = e["macro_f1"]
                print(f"  {key:28s} reference SD {m['ceiling_training_sd']}, transfer SD {m['transfer_training_sd']}, gap SE "
                      f"{m['gap_test_se']} -> {m['gap_total_se']} (+{100 * m['gap_se_inflation']:.0f}%), "
                      f"widened {m['gap_widened_ci']}")
    if "numerical_robustness" in summary:
        num = summary["numerical_robustness"]
        print("\n12. Numerics (refit vs saved predictions):")
        for setting in ("tol_1e-6", "one_thread"):
            e = num[setting]
            print(f"  {setting:10s} max |delta macro-F1| {e['max_abs_delta_macro_f1']} ({e['worst']}); probes with changed "
                  f"predictions {e['probes_with_changed_predictions']}; H2 cells failing {e['h2_cells_failing']}")
        changes = [f"{c['probe']}: {c['C_rq1']} -> {c['C_tol_1e-6']}" for c in num["c_selection_tol_1e-6"]["C_changes"]]
        print(f"  C selection at tol 1e-6: {changes or 'no change'}")
    print("\nVerdict-relevant flags:")
    for flag in summary["verdict_flags"]:
        print(f"  - {flag}")


if __name__ == "__main__":
    raise SystemExit(main())

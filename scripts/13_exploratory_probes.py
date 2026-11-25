"""Exploratory probe analyses on the SAGE six-class test set (not pre-registered).

Every interval comes from the paired patient-level bootstrap of RQ1 (the same resamples).

    a  like-for-like row: the HyperKvasir raw-trained transfer probe scored on SAGE fov, colour and
       fov_colour features, next to the probes trained and scored on the same variant
    b  prior and calibration ladder on the transfer probabilities: none, EM, oracle prior (the true
       class shares of each resample), bias-corrected temperature scaling (BCTS, fitted on
       HyperKvasir out-of-fold probabilities) and BCTS then EM
    c  regrouped meaning groups: strict (the two retroflexions), boundary (z-line, esophagitis) and
       meaning (polyp, pylorus); group contrasts of per-class F1 and AUROC gaps
    d  a transfer probe trained without HyperKvasir esophagitis grade A
    e  RQ3 additions: AUROC curves, ratios to the SAGE-only reference, HyperKvasir + k minus SAGE
       only, regularisation sensitivity with all patients, and draws reaching the target
    f  intercept-only recalibration of the transfer probe on the RQ3 patient draws
    g  SAGE field layout by anatomical location (descriptive)

Probes are fitted in worker processes. Results depend slightly on the BLAS thread count, so the
defaults follow the saved results: cross-validation uses --cv-threads per worker (scripts/07 with
its default --jobs), every other probe --worker-threads (one thread reproduces the saved RQ3 runs).
When the worker refit of the raw transfer probe does not reproduce the saved RQ1 predictions, it is
refitted in this process with the default thread count, as scripts/07 fits it. Fits are cached in
<out-dir>/fits (test probabilities without identifiers). Writes aggregate tables and
exploratory_summary.json.

    python scripts/13_exploratory_probes.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.special import log_softmax, softmax

from sage_shift import exploratory as ex
from sage_shift.bootstrap import group_weights
from sage_shift.data import read_json
from sage_shift.features import load_features
from sage_shift.labelspace import SIX_CLASSES, hk_finding
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.outdir import check_out_dir
from sage_shift.probes import fit_predict, grouped_folds, official_folds, select_c
from sage_shift.rq2 import ci, saerens_em

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ["resnet50_imagenet", "resnet50_dino_imagenet", "resnet50_gastronet_dino", "dinov2_vitb14"]
VARIANTS = ["raw", "fov", "colour", "fov_colour"]
FIT_VARIANTS = ["raw", "fov_colour"]
REGIMES = ["hk_plus_k", "sage_only"]
K = len(SIX_CLASSES)
CLASS = {name: i for i, name in enumerate(SIX_CLASSES)}
POLYP, ZLINE, ESO = CLASS["polyp"], CLASS["z-line-normal"], CLASS["esophagitis"]
UNIFORM = np.full(K, 1 / K)  # the prior implied by balanced class weights
GROUPS = {"strict": [CLASS["fundus-retroflexion"], CLASS["rectum-retroflexion"]],
          "boundary": [CLASS["z-line-normal"], CLASS["esophagitis"]],
          "meaning": [CLASS["polyp"], CLASS["pylorus-normal"]]}
CONTRASTS = [("meaning", "strict"), ("boundary", "strict"), ("meaning", "boundary")]
GRADE_A = "esophagitis-a"
METRICS = ["macro_f1", "macro_auroc", "mcc", "balanced_accuracy", "polyp_share"]
LADDER_METRICS = ["macro_f1", "balanced_accuracy", "mcc", "polyp_share"]
RQ3_METRICS = ["macro_f1", "macro_auroc", "mcc"]
COMMON_C = 0.01


def parse_args() -> argparse.Namespace:
    cpus = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="Exploratory probe analyses (items a-g).")
    parser.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--manifests", type=Path, default=ROOT / "outputs" / "manifests")
    parser.add_argument("--rq1", type=Path, default=ROOT / "outputs" / "rq1")
    parser.add_argument("--rq2", type=Path, default=ROOT / "outputs" / "rq2")
    parser.add_argument("--rq3", type=Path, default=ROOT / "outputs" / "rq3")
    parser.add_argument("--quality", type=Path, default=ROOT / "outputs" / "preprocess" / "sage_fov_quality.csv",
                        help="per-image SAGE field table from scripts/05 (layouts)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "exploratory")
    parser.add_argument("--backbones", nargs="+", help="default: every backbone with RQ1 results")
    parser.add_argument("--jobs", type=int, default=min(3, cpus))
    parser.add_argument("--cv-threads", type=int, default=max(cpus // min(8, cpus), 1),
                        help="BLAS threads per worker for cross-validation")
    parser.add_argument("--worker-threads", type=int, default=1, help="BLAS threads per worker for other probes")
    parser.add_argument("--penalty", type=float, default=1e-3, help="L2 pull of refitted intercepts to their values")
    parser.add_argument("--refit", action="store_true", help="ignore cached fits")
    return parser.parse_args()


@dataclass
class Setup:
    """Data shared by every analysis."""

    test: pd.DataFrame
    train: pd.DataFrame
    y_test: np.ndarray
    y_train: np.ndarray
    truth: np.ndarray
    weights: np.ndarray
    hk: pd.DataFrame
    y_hk: np.ndarray
    hk_fold: np.ndarray
    hk_class: np.ndarray
    draws: list[tuple[int, int, np.ndarray]]
    folds: list[tuple[np.ndarray, np.ndarray]]
    grid: list[float]
    seed: int
    max_iter: int
    level: float
    target_fraction: float

    @property
    def unit(self) -> np.ndarray:
        return np.ones((1, len(self.y_test)))


def load_setup(args: argparse.Namespace, cfg: dict[str, Any]) -> Setup:
    sage = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    six = sage[sage["in_six"] == 1]
    train = six[six["split"] == "train"].reset_index(drop=True)
    test = six[six["split"] == "test"].reset_index(drop=True)
    hk = pd.read_csv(args.manifests / "hk_manifest.csv", dtype=str, keep_default_na=False)
    hk = hk[hk["sage_class"] != ""].reset_index(drop=True)
    seed, rq3 = int(cfg["seeds"]["global"]), cfg["rq3"]
    y_train = train["class"].map(CLASS).to_numpy()
    patients = np.sort(train["patient_group"].unique())
    return Setup(
        test=test, train=train, y_test=test["class"].map(CLASS).to_numpy(), y_train=y_train,
        truth=test[[LABEL_COLUMNS[name] for name in SIX_CLASSES]].to_numpy(dtype=float),
        weights=group_weights(test["patient_group"].to_numpy(), int(cfg["bootstrap"]["n_resamples"]),
                              int(cfg["seeds"]["bootstrap"])),
        hk=hk, y_hk=hk["sage_class"].map(CLASS).to_numpy(), hk_fold=hk["fold2"].astype(int).to_numpy(),
        hk_class=hk["hk_class"].map(hk_finding).to_numpy(),
        draws=ex.patient_draws(patients, [int(k) for k in rq3["k_patients"]], int(rq3["draws_per_k"]),
                               int(cfg["seeds"]["rq3_patient_draws"])),
        folds=grouped_folds(y_train, train["patient_group"].to_numpy(), 5, seed),
        grid=[float(c) for c in cfg["probe"]["C_grid"]], seed=seed, max_iter=int(cfg["probe"]["max_iter"]),
        level=float(cfg["bootstrap"]["ci"]), target_fraction=float(rq3["target_fraction_of_ceiling"]))


def feature_rows(path: Path, ids: pd.Series) -> np.ndarray:
    """Feature rows in the order of `ids`, as float64 (as scripts/07 and 10)."""
    values, file_ids = load_features(path)
    index = {key: row for row, key in enumerate(file_ids.tolist())}
    missing = sum(key not in index for key in ids)
    if missing:
        raise SystemExit(f"{path}: {missing} images have no features")
    return values[[index[key] for key in ids]].astype(np.float64)


def saved_proba(path: Path, ids: pd.Series) -> np.ndarray:
    """Saved test probabilities (six columns) in the order of `ids`."""
    frame = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid")
    missing = len(set(ids) - set(frame.index))
    if missing:
        raise SystemExit(f"{path.name}: {missing} test images have no prediction")
    return frame.loc[list(ids), list(SIX_CLASSES)].to_numpy(dtype=float)


def c_key(C: float) -> str:
    return f"{C:g}"


# --- Fitting (cached) ----------------------------------------------------------------------------

def fit_combo(s: Setup, args: argparse.Namespace, backbone: str, variant: str) -> dict[str, Any]:
    """Every probe one backbone and variant needs, loaded from the cache when its settings match."""
    rq1 = read_json(args.rq1 / f"rq1_{variant}_summary.json")["backbones"][backbone]
    c_hk, c_sage = float(rq1["C_hyperkvasir"]), float(rq1["C_sage"])
    settings = {"C_hk": c_hk, "C_sage": c_sage, "grid": s.grid, "seed": s.seed, "max_iter": s.max_iter,
                "draws": len(s.draws), "cv_threads": args.cv_threads, "worker_threads": args.worker_threads,
                "transfer_refit": "worker, then main"}
    cache = args.out_dir / "fits" / f"{backbone}__{variant}.joblib"
    if cache.is_file() and not args.refit:
        fits = joblib.load(cache)
        if fits["settings"] == settings:
            return fits
    started = time.time()
    sage_dir = args.features / "sage" / backbone
    x_hk = feature_rows(args.features / "hk" / backbone / f"{variant}.npz", s.hk["file_name"])
    x_train = feature_rows(sage_dir / f"{variant}.npz", s.train["sample_uuid"])
    x_test = feature_rows(sage_dir / f"{variant}.npz", s.test["sample_uuid"])
    fits: dict[str, Any] = {"settings": settings}
    keep = s.hk_class != GRADE_A
    patients = s.train["patient_group"].to_numpy()
    c_regime = {"hk_plus_k": c_hk, "sage_only": c_sage}
    with parallel_config(backend="loky", inner_max_num_threads=args.cv_threads):
        _, cv_oof, oof = select_c(x_hk, s.y_hk, official_folds(s.hk_fold), [c_hk], s.seed, K, args.jobs, s.max_iter)
        c_ref, cv_ref, _ = select_c(x_train, s.y_train, s.folds, s.grid, s.seed, K, args.jobs, s.max_iter)
        c_nga, cv_nga, _ = select_c(x_hk[keep], s.y_hk[keep], official_folds(s.hk_fold[keep]), s.grid, s.seed, K,
                                    args.jobs, s.max_iter)
    with parallel_config(backend="loky", inner_max_num_threads=args.worker_threads):
        stacked, y_stacked = np.vstack([x_hk, x_train]), np.r_[s.y_hk, s.y_train]
        source = np.r_[np.zeros(len(s.y_hk)), np.ones(len(s.y_train))]
        tasks: list[tuple[tuple, Any, tuple]] = []
        for regime in REGIMES:
            hk_part = (x_hk, s.y_hk) if regime == "hk_plus_k" else (None, None)
            for k, draw, members in s.draws:
                if regime == "sage_only" and k == 0:
                    continue
                chosen = np.isin(patients, members)
                tasks.append((("rq3", regime, k, draw), ex.fit_regime,
                              (regime, *hk_part, x_train[chosen], s.y_train[chosen], x_test, c_regime[regime], s.seed, K,
                               s.max_iter)))
        for C in s.grid:
            for regime in REGIMES:
                if C != c_regime[regime]:  # the RQ3 run with every patient covers the RQ1 C
                    hk_part = (x_hk, s.y_hk) if regime == "hk_plus_k" else (None, None)
                    tasks.append((("all", regime, C), ex.fit_regime,
                                  (regime, *hk_part, x_train, s.y_train, x_test, C, s.seed, K, s.max_iter)))
            for f, fold in enumerate(ex.with_source_block(len(s.y_hk), s.folds)):
                tasks.append((("cv", C, f), ex.weighted_fold_score, (stacked, y_stacked, source, fold, C, s.seed, K,
                                                                     s.max_iter)))
        for C in sorted({c_hk, c_nga}):
            tasks.append((("no_grade_a", C), fit_predict, (x_hk[keep], s.y_hk[keep], x_test, C, s.seed, K, s.max_iter)))
        tasks.append((("transfer_worker",), ex.fit_linear, (x_hk, s.y_hk, c_hk, s.seed, s.max_iter)))
        results = Parallel(n_jobs=args.jobs)(delayed(fn)(*fn_args) for _, fn, fn_args in tasks)
    done = {key: value for (key, _, _), value in zip(tasks, results)}

    # The transfer probe is the worker refit. On raw frames, if that refit does not reproduce the saved
    # RQ1 predictions, it is refitted in this process with the default BLAS threads, as scripts/07 does.
    saved = saved_proba(args.rq1 / "predictions" / f"{backbone}__{variant}__transfer.csv", s.test["sample_uuid"])
    worker_parts = done[("transfer_worker",)]
    worker_proba = softmax(ex.logits(worker_parts, x_test), axis=1)
    parts, fitted_in = worker_parts, "worker process"
    if variant == "raw" and np.abs(worker_proba - saved).max() > 1e-6:
        refit_started = time.time()
        parts, fitted_in = ex.fit_linear(x_hk, s.y_hk, c_hk, s.seed, s.max_iter), "main process"
        fits["seconds_main_process_refit"] = round(time.time() - refit_started, 1)
    inputs = VARIANTS if variant == "raw" else [variant]
    fits["transfer"] = {
        "parts": parts, "fitted_in": fitted_in, "proba": softmax(ex.logits(parts, x_test), axis=1),
        "test_logits": {v: ex.logits(parts, x_test if v == variant else
                                     feature_rows(sage_dir / f"{v}.npz", s.test["sample_uuid"])) for v in inputs},
        "train_z": ex.logits(parts, x_train, intercept=False), "test_z": ex.logits(parts, x_test, intercept=False)}
    fits["transfer_worker"] = worker_proba

    runs = []
    for regime in REGIMES:
        for k, draw, members in s.draws:
            if regime == "sage_only" and k == 0:
                continue
            chosen = np.isin(patients, members)
            runs.append({"regime": regime, "k": k, "draw": draw, "sage_images": int(chosen.sum()),
                         "classes_covered": int(len(np.unique(s.y_train[chosen]))), "C": c_regime[regime],
                         "proba": done[("rq3", regime, k, draw)]})
    every = len(np.unique(patients))
    all_patients = {regime: {C: done[("all", regime, C)] for C in s.grid if C != c_regime[regime]} for regime in REGIMES}
    for regime in REGIMES:
        all_patients[regime][c_regime[regime]] = next(r["proba"] for r in runs if r["regime"] == regime and r["k"] == every)
    fits.update({
        "oof_hk": oof, "oof_cv_macro_f1": cv_oof[c_hk],
        "reference_cv": {"C": c_ref, "scores": cv_ref},
        "no_grade_a": {"C": c_nga, "cv": cv_nga, "images": int(keep.sum()),
                       "proba": {C: done[("no_grade_a", C)] for C in {c_hk, c_nga}}},
        "rq3": runs, "all_patients": all_patients,
        "hk_plus_all_cv": {C: [done[("cv", C, f)] for f in range(len(s.folds))] for C in s.grid},
        "seconds": round(time.time() - started, 1)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fits, cache)
    return fits


# --- Scoring helpers -----------------------------------------------------------------------------

class Rows:
    """Long-format results: estimate and percentile interval per section, model, row and metric."""

    def __init__(self, level: float) -> None:
        self.rows: list[dict[str, Any]] = []
        self.level = level

    def add(self, section: str, backbone: str, variant: str, row: str, quantity: str, metric: str, point: float,
            values: np.ndarray | None = None, **extra: Any) -> dict[str, Any]:
        entry = ci(point, values, self.level) if values is not None else {"estimate": round(float(point), 4),
                                                                          "ci_low": None, "ci_high": None}
        self.rows.append({"section": section, "backbone": backbone, "variant": variant, "row": row,
                          "quantity": quantity, "metric": metric, **entry, **extra})
        return entry

    def add_given(self, section: str, backbone: str, variant: str, row: str, quantity: str, metric: str,
                  entry: dict[str, float]) -> None:
        """A row whose estimate and interval were computed elsewhere."""
        self.rows.append({"section": section, "backbone": backbone, "variant": variant, "row": row,
                          "quantity": quantity, "metric": metric, **entry})

    def frame(self, section: str) -> pd.DataFrame:
        frame = pd.DataFrame([r for r in self.rows if r["section"] == section])
        return frame.drop(columns="section").dropna(axis=1, how="all")


def scored(s: Setup, proba: np.ndarray, with_auroc: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Point scores and their values on every resample."""
    point = {key: value[0] for key, value in ex.resample_scores(s.y_test, proba, s.truth, s.unit, POLYP,
                                                                with_auroc).items()}
    return point, ex.resample_scores(s.y_test, proba, s.truth, s.weights, POLYP, with_auroc)


def from_predictions(s: Setup, point_pred: np.ndarray, boot_pred: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scores of argmax predictions that may differ between resamples."""
    point = {key: value[0] for key, value in
             ex.confusion_scores(ex.confusion_many(s.y_test, point_pred, K, s.unit), POLYP).items()}
    return point, ex.confusion_scores(ex.confusion_many(s.y_test, boot_pred, K, s.weights), POLYP)


def to_json(value: Any) -> Any:
    """JSON form of NumPy scalars and arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def interval(entry: dict[str, Any]) -> str:
    return f"{entry['estimate']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"


def excludes_zero(entry: dict[str, Any]) -> bool:
    return entry["ci_low"] > 0 or entry["ci_high"] < 0


# --- (a) like-for-like row -----------------------------------------------------------------------

def like_for_like(s: Setup, args: argparse.Namespace, rows: Rows, backbone: str, fits: dict[str, Any],
                  rq2_table: pd.DataFrame) -> dict[str, Any]:
    pred_dir, ids = args.rq1 / "predictions", s.test["sample_uuid"]
    saved_raw = saved_proba(pred_dir / f"{backbone}__raw__transfer.csv", ids)
    refit = fits["transfer"]["proba"]
    out: dict[str, Any] = {"max_abs_diff_vs_rq1_raw_predictions": float(np.abs(refit - saved_raw).max()),
                           "argmax_agreement_vs_rq1_raw": f"{int((refit.argmax(1) == saved_raw.argmax(1)).sum())}/{len(ids)}"}
    lfl = {v: scored(s, softmax(fits["transfer"]["test_logits"][v], axis=1)) for v in VARIANTS}
    both = {v: scored(s, saved_proba(pred_dir / f"{backbone}__{v}__transfer.csv", ids)) for v in VARIANTS}
    for v in VARIANTS:
        (lp, lb), (bp, bb) = lfl[v], both[v]
        for m in METRICS:
            entry = rows.add("a", backbone, "raw", v, "hk_raw_probe", m, lp[m], lb[m])
            if v == "raw":
                out.setdefault("raw", {})[m] = entry
                continue
            change = rows.add("a", backbone, "raw", v, "change_vs_raw", m, lp[m] - lfl["raw"][0][m],
                              lb[m] - lfl["raw"][1][m])
            rows.add("a", backbone, "raw", v, "both_processed", m, bp[m], bb[m])
            both_change = rows.add("a", backbone, "raw", v, "both_processed_change", m, bp[m] - both["raw"][0][m],
                                   bb[m] - both["raw"][1][m])
            versus = rows.add("a", backbone, "raw", v, "like_for_like_minus_both_processed", m, lp[m] - bp[m],
                              lb[m] - bb[m])
            out.setdefault(v, {})[m] = {"score": entry, "change_vs_raw": change,
                                        "both_processed_change": both_change, "like_for_like_minus_both": versus}
        if v != "raw":
            hit = rq2_table[(rq2_table["section"] == "decomposition") & (rq2_table["model"] == backbone)
                            & (rq2_table["row"] == v) & (rq2_table["metric"] == "transfer_change_macro_f1")]
            if len(hit):
                ref = {key: float(hit.iloc[0][key]) for key in ("estimate", "ci_low", "ci_high")}
                rows.add_given("a", backbone, "raw", v, "both_processed_change_rq2_table", "macro_f1", ref)
                mine = out[v]["macro_f1"]["both_processed_change"]
                out[v]["macro_f1"]["rq2_table_both_processed_change"] = ref
                out[v]["macro_f1"]["max_diff_vs_rq2_table"] = max(abs(mine[key] - ref[key]) for key in ref)
    return out


# --- (b) prior and calibration ladder ------------------------------------------------------------

def ladder_inputs(s: Setup, args: argparse.Namespace, backbone: str, variant: str, fits: dict[str, Any]) -> dict[str, Any]:
    """Transfer probabilities, BCTS fits on HyperKvasir out-of-fold probabilities and the calibrated outputs."""
    saved = saved_proba(args.rq1 / "predictions" / f"{backbone}__{variant}__transfer.csv", s.test["sample_uuid"])
    logit = fits["transfer"]["test_logits"][variant]
    diff = float(np.abs(softmax(logit, axis=1) - saved).max())
    if diff <= 1e-6:  # the refit equals the saved predictions: use its unrounded values
        proba, log_proba, origin = softmax(logit, axis=1), log_softmax(logit, axis=1), "refit, equal to the saved predictions"
    else:
        proba, log_proba = saved, np.log(np.maximum(saved, 5e-7))
        origin = "saved predictions; zeros floored at 5e-7 for logarithms"
    oof_log = np.log(np.maximum(fits["oof_hk"], np.finfo(float).tiny))
    counts = np.bincount(s.y_hk, minlength=K)
    balanced = len(s.y_hk) / (K * counts[s.y_hk])
    t_bal, b_bal = ex.fit_bcts(oof_log, s.y_hk, balanced)
    t_plain, b_plain = ex.fit_bcts(oof_log, s.y_hk)
    q_bal, q_plain = ex.apply_bcts(log_proba, t_bal, b_bal), ex.apply_bcts(log_proba, t_plain, b_plain)
    hk_prior = counts / counts.sum()
    calibration = {
        "probabilities": origin, "max_abs_diff_refit_vs_saved": diff,
        "hk_oof_cv_macro_f1": fits["oof_cv_macro_f1"],
        "bcts_balanced": {"temperature": round(t_bal, 4), "bias_centred": dict(zip(SIX_CLASSES, np.round(b_bal - b_bal.mean(), 4).tolist())),
                          "hk_oof_nll_balanced_before": round(ex.nll(oof_log, s.y_hk, balanced), 4),
                          "hk_oof_nll_balanced_after": round(ex.nll(np.log(ex.apply_bcts(oof_log, t_bal, b_bal)), s.y_hk, balanced), 4),
                          "sage_test_nll_before": round(ex.nll(log_proba, s.y_test), 4),
                          "sage_test_nll_after": round(ex.nll(np.log(q_bal), s.y_test), 4)},
        "bcts_plain": {"temperature": round(t_plain, 4), "bias_centred": dict(zip(SIX_CLASSES, np.round(b_plain - b_plain.mean(), 4).tolist())),
                       "hk_oof_nll_before": round(ex.nll(oof_log, s.y_hk), 4),
                       "hk_oof_nll_after": round(ex.nll(np.log(ex.apply_bcts(oof_log, t_plain, b_plain)), s.y_hk), 4)},
        "hk_class_prior": dict(zip(SIX_CLASSES, np.round(hk_prior, 4).tolist()))}
    return {"proba": proba, "q_bal": q_bal, "q_plain": q_plain, "hk_prior": hk_prior, "calibration": calibration}


def em_ladder(s: Setup, rows: Rows, backbone: str, variant: str, inputs: dict[str, Any],
              em_boot: dict[str, tuple[np.ndarray, np.ndarray]], rq2_table: pd.DataFrame) -> dict[str, Any]:
    p, q_bal, q_plain, hk_prior = inputs["proba"], inputs["q_bal"], inputs["q_plain"], inputs["hk_prior"]
    true_shares = ex.class_shares(s.y_test, K, s.unit)[0]
    boot_shares = ex.class_shares(s.y_test, K, s.weights)
    rungs: dict[str, dict[str, Any]] = {}
    for name, posterior, source in (("em", p, UNIFORM), ("bcts_em", q_bal, UNIFORM), ("bcts_plain_em", q_plain, hk_prior)):
        adjusted, prior, iterations = saerens_em(posterior, source)
        rungs[name] = {"point": adjusted.argmax(1), "boot": em_boot[name][0], "prior": prior,
                       "boot_prior": em_boot[name][1], "iterations": iterations}
    rungs["none"] = {"point": p.argmax(1), "boot": p.argmax(1), "prior": p.mean(axis=0), "prior_kind": "mean posterior"}
    rungs["oracle"] = {"point": ex.reweight_prior(p, true_shares, UNIFORM).argmax(1),
                       "boot": ex.reweight_prior(p, boot_shares, UNIFORM).argmax(-1), "prior": true_shares,
                       "prior_kind": "true test class shares"}
    rungs["bcts"] = {"point": q_bal.argmax(1), "boot": q_bal.argmax(1), "prior": q_bal.mean(axis=0),
                     "prior_kind": "mean posterior"}
    order = ["none", "em", "oracle", "bcts", "bcts_em", "bcts_plain_em"]
    scores = {name: from_predictions(s, rungs[name]["point"], rungs[name]["boot"]) for name in order}
    out: dict[str, Any] = {"calibration": inputs["calibration"], "rungs": {}}
    for name in order:
        (pt, bt), (p0, b0) = scores[name], scores["none"]
        entry: dict[str, Any] = {}
        for m in LADDER_METRICS:
            entry[m] = rows.add("b", backbone, variant, name, "score", m, pt[m], bt[m])
            if name != "none":
                entry[f"{m}_change_vs_none"] = rows.add("b", backbone, variant, name, "change_vs_none", m,
                                                        pt[m] - p0[m], bt[m] - b0[m])
        rung = rungs[name]
        entry["prior"] = dict(zip(SIX_CLASSES, np.round(rung["prior"], 4).tolist()))
        entry["prior_kind"] = rung.get("prior_kind", "EM estimate")
        if "boot_prior" in rung:
            entry["polyp_prior"] = rows.add("b", backbone, variant, name, "estimated_prior", "polyp", rung["prior"][POLYP],
                                            rung["boot_prior"][:, POLYP])
            entry["em_iterations"] = int(rung["iterations"])
        out["rungs"][name] = entry
    oracle, em = out["rungs"]["oracle"]["macro_f1_change_vs_none"], out["rungs"]["em"]["macro_f1_change_vs_none"]
    if oracle["ci_low"] > 0:
        out["reading"] = ("the oracle prior helps but EM does not: the failure lies in estimating the prior or in calibration"
                          if em["ci_low"] <= 0 else "both the oracle prior and EM help")
    else:
        out["reading"] = ("even the oracle prior does not help (interval includes 0)" if oracle["ci_high"] >= 0
                          else "even the oracle prior lowers macro-F1") + ": prevalence shift is not a material cause"
    checks = {}
    for row, name in ((variant, "none"), (f"{variant}+EM", "em")):
        hit = rq2_table[(rq2_table["section"] == "decomposition") & (rq2_table["model"] == backbone)
                        & (rq2_table["row"] == row) & (rq2_table["metric"] == "transfer_macro_f1")]
        if len(hit):
            mine = out["rungs"][name]["macro_f1"]
            checks[name] = max(abs(mine[key] - float(hit.iloc[0][key])) for key in ("estimate", "ci_low", "ci_high"))
    out["max_diff_vs_rq2_table"] = checks
    return out


# --- (c) regrouped meaning groups ----------------------------------------------------------------

def meaning_groups(s: Setup, args: argparse.Namespace, rows: Rows, backbone: str, variant: str,
                   rq2_table: pd.DataFrame) -> dict[str, Any]:
    pred_dir, ids = args.rq1 / "predictions", s.test["sample_uuid"]
    (tp, tb) = scored(s, saved_proba(pred_dir / f"{backbone}__{variant}__transfer.csv", ids))
    (rp, rb) = scored(s, saved_proba(pred_dir / f"{backbone}__{variant}__ceiling.csv", ids))
    out: dict[str, Any] = {}
    for key in ("f1", "auroc"):
        gap_p, gap_b = rp[key] - tp[key], rb[key] - tb[key]
        means_p, means_b = ex.group_means(gap_p, GROUPS), ex.group_means(gap_b, GROUPS)
        for group in GROUPS:
            out[f"{key}/{group}"] = rows.add("c", backbone, variant, group, "mean_gap", f"{key}_gap", means_p[group],
                                             means_b[group], classes=" + ".join(SIX_CLASSES[c] for c in GROUPS[group]))
        contrasts_p, contrasts_b = ex.group_contrasts(gap_p, GROUPS, CONTRASTS), ex.group_contrasts(gap_b, GROUPS, CONTRASTS)
        for name in contrasts_p:
            out[f"{key}/{name}"] = rows.add("c", backbone, variant, name, "contrast", f"{key}_gap", contrasts_p[name],
                                            contrasts_b[name])
    part = rq2_table[(rq2_table["section"] == "meaning_groups") & (rq2_table["model"] == backbone)
                     & (rq2_table["row"] == variant)].set_index("metric")["estimate"]
    gaps = rp["f1"] - tp["f1"]
    out["max_diff_per_class_f1_gap_vs_rq2_table"] = float(max(abs(round(gaps[j], 4) - part[f"f1_gap_{name}"])
                                                              for j, name in enumerate(SIX_CLASSES)))
    return out


# --- (d) source without esophagitis grade A ------------------------------------------------------

def no_grade_a(s: Setup, args: argparse.Namespace, rows: Rows, backbone: str, variant: str,
               fits: dict[str, Any]) -> dict[str, Any]:
    nga = fits["no_grade_a"]
    c_hk, c_nga = fits["settings"]["C_hk"], nga["C"]
    models = {"rq1_transfer": saved_proba(args.rq1 / "predictions" / f"{backbone}__{variant}__transfer.csv",
                                          s.test["sample_uuid"]),
              "transfer_refit_worker": fits["transfer_worker"],
              "no_grade_a_selected_C": nga["proba"][c_nga], "no_grade_a_rq1_C": nga["proba"][c_hk]}
    values: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, proba in models.items():
        pred = proba.argmax(axis=1)
        cms = (ex.confusion_many(s.y_test, pred, K, s.unit), ex.confusion_many(s.y_test, pred, K, s.weights))
        both = []
        for cm in cms:
            sc = ex.confusion_scores(cm, POLYP)
            both.append({"macro_f1": sc["macro_f1"], "f1_esophagitis": sc["f1"][:, ESO], "f1_z_line": sc["f1"][:, ZLINE],
                         "z_line_predicted_esophagitis": cm[:, ZLINE, ESO], "esophagitis_predicted_z_line": cm[:, ESO, ZLINE],
                         "polyp_share": sc["polyp_share"]})
        values[name] = ({k: v[0] for k, v in both[0].items()}, both[1])
    out: dict[str, Any] = {"C_selected": c_nga, "C_rq1": c_hk, "cv_macro_f1": {c_key(c): round(v, 4) for c, v in nga["cv"].items()},
                           "hk_images": nga["images"], "models": {}}
    for name, (point, boot) in values.items():
        entry: dict[str, Any] = {}
        for m in point:
            entry[m] = rows.add("d", backbone, variant, name, "score", m, point[m], boot[m],
                                C=c_nga if name == "no_grade_a_selected_C" else c_hk)
            if name.startswith("no_grade_a"):
                for base in ("rq1_transfer", "transfer_refit_worker"):
                    entry[f"{m}_change_vs_{base}"] = rows.add("d", backbone, variant, name, f"change_vs_{base}", m,
                                                              point[m] - values[base][0][m], boot[m] - values[base][1][m])
        out["models"][name] = entry
    return out


# --- (e) RQ3 additions ---------------------------------------------------------------------------

def score_runs(s: Setup, runs: list[dict[str, Any]]) -> None:
    """Add point scores and resample values (macro-F1, macro AUROC, MCC) to every run."""
    for run in runs:
        point, boot = scored(s, run["proba"])
        run["point"] = {m: float(point[m]) for m in RQ3_METRICS}
        run["boot"] = {m: boot[m] for m in RQ3_METRICS}


def mean_over_draws(runs: list[dict[str, Any]], metric: str) -> tuple[float, np.ndarray]:
    return float(np.mean([r["point"][metric] for r in runs])), np.mean([r["boot"][metric] for r in runs], axis=0)


def rq3_additions(s: Setup, args: argparse.Namespace, backbone: str, variant: str, fits: dict[str, Any],
                  saved_runs: pd.DataFrame, tables: dict[str, list[dict[str, Any]]], ceiling_rq1: float,
                  rq1_cv_sage: dict[str, float]) -> dict[str, Any]:
    runs = fits["rq3"]
    every = max(r["k"] for r in runs)
    key = {"backbone": backbone, "variant": variant}
    out: dict[str, Any] = {}

    # Reproduction of scripts/10.
    saved = saved_runs[(saved_runs["backbone"] == backbone) & (saved_runs["variant"] == variant)].set_index(["regime", "k", "draw"])
    diffs, auroc_diffs, mismatched = [], [], 0
    for run in runs:
        row = saved.loc[(run["regime"], run["k"], run["draw"])]
        diffs.append(abs(run["point"]["macro_f1"] - row["macro_f1"]))
        auroc_diffs.append(abs(run["point"]["macro_auroc"] - row["macro_auroc"]))
        mismatched += int(row["sage_images"] != run["sage_images"] or row["classes_covered"] != run["classes_covered"])
    out["reproduction"] = {"runs": len(runs), "saved_runs": len(saved), "max_abs_diff_macro_f1": float(max(diffs)),
                           "max_abs_diff_macro_auroc": float(max(auroc_diffs)), "runs_with_other_images": mismatched}

    by_k: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        by_k.setdefault((run["regime"], run["k"]), []).append(run)
    ref_point, ref_boot = mean_over_draws(by_k[("sage_only", every)], "macro_f1")
    const_point, const_boot = float(ex.best_constant_many(s.y_test, K, s.unit)[0]), ex.best_constant_many(s.y_test, K, s.weights)
    means = {(regime, k): {m: mean_over_draws(part, m) for m in RQ3_METRICS} for (regime, k), part in by_k.items()}
    out["reference"] = {"macro_f1": round(ref_point, 4), "rq1_ceiling": ceiling_rq1, "best_constant": round(const_point, 4)}

    # AUROC curves, ratios to the reference and draws reaching the target, per regime and k.
    ratios: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {regime: [] for regime in REGIMES}
    all_point: dict[tuple[str, float], float] = {}
    all_boot: dict[tuple[str, float], np.ndarray] = {}
    for regime in REGIMES:
        for C, proba in fits["all_patients"][regime].items():
            point, boot = scored(s, proba, with_auroc=False)
            all_point[(regime, C)], all_boot[(regime, C)] = float(point["macro_f1"]), boot["macro_f1"]
    ref_common = all_point[("sage_only", COMMON_C)]
    for (regime, k), part in sorted(by_k.items(), key=lambda item: (REGIMES.index(item[0][0]), item[0][1])):
        (f1_point, f1_boot), (auc_point, auc_boot) = means[(regime, k)]["macro_f1"], means[(regime, k)]["macro_auroc"]
        auc_ci = ci(auc_point, auc_boot, s.level)
        tables["auroc_curves"].append({**key, "regime": regime, "k": k, "auroc_mean": auc_ci["estimate"], "ci_low": auc_ci["ci_low"],
                             "ci_high": auc_ci["ci_high"], "draws": len(part)})
        ratio = ci(f1_point / ref_point, f1_boot / ref_boot, s.level)
        corrected = ci((f1_point - const_point) / (ref_point - const_point), (f1_boot - const_boot) / (ref_boot - const_boot), s.level)
        ratios[regime].append((k, ratio, corrected))
        tables["ratios"].append({**key, "regime": regime, "k": k, "draws": len(part), "macro_f1_mean": round(f1_point, 4),
                             "ratio": ratio["estimate"], "ratio_ci_low": ratio["ci_low"], "ratio_ci_high": ratio["ci_high"],
                             "chance_corrected": corrected["estimate"], "chance_corrected_ci_low": corrected["ci_low"],
                             "chance_corrected_ci_high": corrected["ci_high"]})
        target = s.target_fraction * ceiling_rq1
        tables["draws_reaching"].append({**key, "regime": regime, "k": k, "draws": len(part),
                             "draws_reaching_target": int(sum(r["point"]["macro_f1"] >= target for r in part)),
                             "target": round(target, 4),
                             "draws_reaching_target_reference_C_0.01": int(sum(r["point"]["macro_f1"] >= s.target_fraction * ref_common for r in part)),
                             "target_reference_C_0.01": round(s.target_fraction * ref_common, 4)})
    out["smallest_k"] = {}
    for regime in REGIMES:
        ks = [k for k, _, _ in ratios[regime]]
        out["smallest_k"][regime] = {
            "ratio_point": ex.smallest_k(ks, [r["estimate"] for _, r, _ in ratios[regime]], s.target_fraction),
            "ratio_ci_low": ex.smallest_k(ks, [r["ci_low"] for _, r, _ in ratios[regime]], s.target_fraction),
            "chance_corrected_point": ex.smallest_k(ks, [c["estimate"] for _, _, c in ratios[regime]], s.target_fraction),
            "chance_corrected_ci_low": ex.smallest_k(ks, [c["ci_low"] for _, _, c in ratios[regime]], s.target_fraction),
            "mean_reaches_90pct_of_rq1_ceiling": ex.smallest_k(ks, [means[(regime, k)]["macro_f1"][0] for k in ks],
                                                               s.target_fraction * ceiling_rq1),
            "mean_reaches_90pct_of_reference_at_C_0.01": ex.smallest_k(ks, [means[(regime, k)]["macro_f1"][0] for k in ks],
                                                                        s.target_fraction * ref_common)}
    out["ratios"] = {regime: {str(k): {"ratio": r, "chance_corrected": c} for k, r, c in ratios[regime]} for regime in REGIMES}

    # HyperKvasir + k minus SAGE only.
    out["hk_minus_sage_only"] = {}
    for k in sorted({k for regime, k in by_k if regime == "sage_only"}):
        (hp, hb), (sp, sb) = means[("hk_plus_k", k)]["macro_f1"], means[("sage_only", k)]["macro_f1"]
        entry = ci(hp - sp, hb - sb, s.level)
        tables["hk_minus_sage_only"].append({**key, "k": k, "hk_plus_k": round(hp, 4), "sage_only": round(sp, 4), "difference": entry["estimate"],
                             "ci_low": entry["ci_low"], "ci_high": entry["ci_high"]})
        out["hk_minus_sage_only"][str(k)] = entry

    # Regularisation sensitivity with every patient.
    cv = {C: float(np.mean(scores)) for C, scores in fits["hk_plus_all_cv"].items()}
    c_cv = ex.choose_c(cv)
    c_hk, c_sage = fits["settings"]["C_hk"], fits["settings"]["C_sage"]
    c_ref_cv = fits["reference_cv"]["C"]
    out["regularisation"] = {
        "reference_by_C": {c_key(C): round(all_point[("sage_only", C)], 4) for C in s.grid},
        "hk_plus_all_by_C": {c_key(C): round(all_point[("hk_plus_k", C)], 4) for C in s.grid},
        "hk_plus_all_cv_macro_f1": {c_key(C): round(v, 4) for C, v in cv.items()}, "hk_plus_all_cv_C": c_cv,
        "reference_cv_C_recomputed": c_ref_cv, "reference_C_rq1": c_sage,
        "reference_cv_max_abs_diff_vs_rq1": float(max(abs(fits["reference_cv"]["scores"][C] - rq1_cv_sage[c_key(C)]) for C in s.grid)),
        "comparisons": {}}
    for regime, name in (("sage_only", "reference"), ("hk_plus_k", "hk_plus_all")):
        for C in s.grid:
            entry = ci(all_point[(regime, C)], all_boot[(regime, C)], s.level)
            tables["scores_by_C"].append({**key, "model": name, "C": C, "macro_f1": entry["estimate"], "ci_low": entry["ci_low"],
                                        "ci_high": entry["ci_high"], "cv_macro_f1": round(cv[C], 4) if regime == "hk_plus_k"
                                        else round(fits["reference_cv"]["scores"][C], 4)})
    comparisons = [("pre_registered", c_hk, c_sage), ("common_C_0.01", COMMON_C, COMMON_C), ("both_cv_tuned", c_cv, c_ref_cv)]
    comparisons += [(f"common_C_{c_key(C)}", C, C) for C in s.grid]
    bonferroni = 1 - (1 - s.level) / 6
    for name, c_plus, c_ref in comparisons:
        point = all_point[("hk_plus_k", c_plus)] - all_point[("sage_only", c_ref)]
        boot = all_boot[("hk_plus_k", c_plus)] - all_boot[("sage_only", c_ref)]
        entry, wide = ci(point, boot, s.level), ci(point, boot, bonferroni)
        tables["regularisation"].append({**key, "comparison": name, "C_hk_plus_all": c_plus, "C_reference": c_ref,
                             "hk_plus_all": round(all_point[("hk_plus_k", c_plus)], 4),
                             "reference": round(all_point[("sage_only", c_ref)], 4), "difference": entry["estimate"],
                             "ci_low": entry["ci_low"], "ci_high": entry["ci_high"], "bonferroni6_ci_low": wide["ci_low"],
                             "bonferroni6_ci_high": wide["ci_high"]})
        out["regularisation"]["comparisons"][name] = {**entry, "bonferroni6": [wide["ci_low"], wide["ci_high"]],
                                                      "C_hk_plus_all": c_plus, "C_reference": c_ref}
    out["means"] = means
    return out


# --- (f) intercept-only recalibration ------------------------------------------------------------

def intercept_only(s: Setup, args: argparse.Namespace, rows: Rows, backbone: str, variant: str, fits: dict[str, Any],
                   rq3_means: dict[tuple[str, int], dict[str, tuple[float, np.ndarray]]]) -> dict[str, Any]:
    transfer = fits["transfer"]
    z_train, z_test, b0 = transfer["train_z"], transfer["test_z"], transfer["parts"]["intercept"]
    patients = s.train["patient_group"].to_numpy()
    by_k: dict[int, list[tuple[dict[str, Any], dict[str, Any], np.ndarray]]] = {}
    point, boot = scored(s, softmax(z_test + b0, axis=1))
    by_k[0] = [(point, boot, np.zeros(K))]
    for k, _, members in s.draws:
        if k == 0:
            continue
        chosen = np.isin(patients, members)
        b = ex.fit_intercepts(z_train[chosen], s.y_train[chosen], b0, args.penalty)
        point, boot = scored(s, softmax(z_test + b, axis=1))
        by_k.setdefault(k, []).append((point, boot, b - b0))
    out: dict[str, Any] = {"penalty": args.penalty, "by_k": {}}
    for k, part in sorted(by_k.items()):
        entry: dict[str, Any] = {"draws": len(part), "mean_abs_intercept_change": round(float(np.mean([np.abs(d).mean() for _, _, d in part])), 4)}
        for m in RQ3_METRICS:
            mp, mb = float(np.mean([p[m] for p, _, _ in part])), np.mean([b[m] for _, b, _ in part], axis=0)
            entry[m] = rows.add("f", backbone, variant, f"k={k}", "intercept_only", m, mp, mb, draws=len(part))
            for regime in REGIMES:
                if (regime, k) not in rq3_means:
                    continue
                rp, rb = rq3_means[(regime, k)][m]
                rows.add("f", backbone, variant, f"k={k}", regime, m, rp, rb, draws=len(part))
                entry[f"{m}_minus_{regime}"] = rows.add("f", backbone, variant, f"k={k}", f"intercept_only_minus_{regime}",
                                                        m, mp - rp, mb - rb)
        out["by_k"][str(k)] = entry
    return out


# --- (g) field layout by location ----------------------------------------------------------------

def layout_location(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(args.manifests / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str}, keep_default_na=False)
    quality = pd.read_csv(args.quality, usecols=["id", "width", "height", "layout"], dtype={"id": str, "layout": str},
                          keep_default_na=False)
    merged = manifest.merge(quality, left_on="sample_uuid", right_on="id", how="left", validate="one_to_one")
    if merged["layout"].isna().any():
        raise SystemExit(f"{args.quality.name}: some SAGE images have no field row")
    merged = merged.drop(columns=["sample_uuid", "id"])  # identifiers are not needed below
    merged["frame_size"] = merged["width"].astype(str) + "x" + merged["height"].astype(str)
    merged["layout"] = merged["layout"].where(merged["layout"] != "", merged["frame_size"] + " (no layout)")
    subsets = {"all_frames": merged, "six_class_train": merged[(merged["in_six"] == 1) & (merged["split"] == "train")],
               "six_class_test": merged[(merged["in_six"] == 1) & (merged["split"] == "test")]}
    rows = []
    for subset, part in subsets.items():
        for layout, group in part.groupby("layout"):
            counts = group["location_context"].value_counts()
            lower, upper = int(counts.get("lower GI", 0)), int(counts.get("upper GI", 0))
            rows.append({"subset": subset, "layout": layout, "frame_size": group["frame_size"].iloc[0], "frames": len(group),
                         "lower_gi": lower, "upper_gi": upper, "no_location_label": int(counts.get("no location label", 0)),
                         "lower_gi_share": round(lower / len(group), 4),
                         "lower_gi_share_of_located": round(lower / (lower + upper), 4) if lower + upper else None})
    table = pd.DataFrame(rows)
    odd = merged[merged["frame_size"] != "768x576"]
    other = (odd.assign(class_name=odd["class"].where(odd["class"] != "", "(blank)"))
             .groupby(["frame_size", "split", "class_name", "in_six"]).size().rename("frames").reset_index())
    summary = {"frames": len(merged), "layouts": int(merged["layout"].nunique()),
               "lower_gi_share_all_frames": dict(zip(table.loc[table["subset"] == "all_frames", "layout"],
                                                     table.loc[table["subset"] == "all_frames", "lower_gi_share"])),
               "frames_not_768x576": other.to_dict(orient="records")}
    return table, other, summary


# --- Main ----------------------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    started, cpu_started = time.time(), time.process_time()
    s = load_setup(args, cfg)
    rq1 = {v: read_json(args.rq1 / f"rq1_{v}_summary.json")["backbones"] for v in VARIANTS}
    backbones = args.backbones or [b for b in BACKBONES if all(b in rq1[v] for v in VARIANTS)]
    rq2_table = pd.read_csv(args.rq2 / "rq2_table.csv")
    saved_runs = pd.read_csv(args.rq3 / "rq3_runs.csv")
    print(f"SAGE six-class test: {len(s.test)} images, {s.test['patient_group'].nunique()} patients; train: {len(s.train)} "
          f"images; HyperKvasir: {len(s.hk)} images. {len(s.weights)} paired patient resamples; backbones {backbones}; "
          f"{args.jobs} workers; BLAS threads per worker: {args.cv_threads} for CV, {args.worker_threads} otherwise.")

    rows = Rows(s.level)
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in ("auroc_curves", "ratios", "hk_minus_sage_only",
                                                                     "regularisation", "scores_by_C", "draws_reaching")}
    summary: dict[str, Any] = {
        "settings": {"bootstrap": cfg["bootstrap"], "seeds": cfg["seeds"], "C_grid": s.grid, "rq3": cfg["rq3"],
                     "jobs": args.jobs, "cv_blas_threads": args.cv_threads, "worker_blas_threads": args.worker_threads,
                     "cpu_count": os.cpu_count(),
                     "intercept_penalty": args.penalty, "classes": list(SIX_CLASSES),
                     "groups": {g: [SIX_CLASSES[c] for c in cols] for g, cols in GROUPS.items()},
                     "hk_classes_mapped": sorted(set(s.hk_class)), "hk_grade_a_class": GRADE_A,
                     "bcts_objective": "class-balanced NLL on HyperKvasir out-of-fold probabilities (official folds, RQ1 C), "
                                       "so the calibrated posteriors keep the uniform prior of the balanced probe; "
                                       "bcts_plain_em is the plain-NLL fit with EM from the HyperKvasir class prior",
                     "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                  "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}},
        "like_for_like": {}, "em_ladder": {}, "meaning_groups": {}, "no_grade_a": {}, "rq3": {},
        "intercept_only": {}, "fit_seconds": {}}

    # (g) and (c) need no refits.
    layout_table, other_frames, summary["layout"] = layout_location(args)
    for backbone in backbones:
        for variant in VARIANTS:
            summary["meaning_groups"][f"{backbone}/{variant}"] = meaning_groups(s, args, rows, backbone, variant, rq2_table)
    print(f"(c), (g) done after {time.time() - started:.0f} s")

    ladder_jobs = []
    for backbone in backbones:
        for variant in FIT_VARIANTS:
            t0 = time.time()
            fits = fit_combo(s, args, backbone, variant)
            name = f"{backbone}/{variant}"
            summary["fit_seconds"][name] = {"this_run": round(time.time() - t0, 1), "when_fitted": fits.get("seconds"),
                                            "main_process_refit": fits.get("seconds_main_process_refit")}
            print(f"  fits {name}: {time.time() - t0:.0f} s")
            if variant == "raw":
                summary["like_for_like"][backbone] = like_for_like(s, args, rows, backbone, fits, rq2_table)
            check = saved_proba(args.rq1 / "predictions" / f"{backbone}__{variant}__transfer.csv", s.test["sample_uuid"])
            summary.setdefault("transfer_refit_checks", {})[name] = {
                "transfer_probe_fitted_in": fits["transfer"]["fitted_in"],
                "max_abs_diff_vs_rq1_predictions": float(np.abs(fits["transfer"]["proba"] - check).max()),
                "argmax_agreement": int((fits["transfer"]["proba"].argmax(1) == check.argmax(1)).sum()),
                "max_abs_diff_worker_refit_vs_rq1_predictions": float(np.abs(fits["transfer_worker"] - check).max())}
            summary["no_grade_a"][name] = no_grade_a(s, args, rows, backbone, variant, fits)
            score_runs(s, fits["rq3"])
            rq3 = rq3_additions(s, args, backbone, variant, fits, saved_runs, tables,
                                float(rq1[variant][backbone]["metrics"]["macro_f1"]["ceiling"]["estimate"]),
                                rq1[variant][backbone]["cv_macro_f1_sage"])
            summary["intercept_only"][name] = intercept_only(s, args, rows, backbone, variant, fits, rq3.pop("means"))
            summary["rq3"][name] = rq3
            ladder_jobs.append((backbone, variant, ladder_inputs(s, args, backbone, variant, fits)))

    # (b): EM on every resample, in parallel over the backbone x variant x rung cells.
    cells = [(i, rung, posterior, prior) for i, (_, _, inputs) in enumerate(ladder_jobs)
             for rung, posterior, prior in (("em", inputs["proba"], UNIFORM), ("bcts_em", inputs["q_bal"], UNIFORM),
                                            ("bcts_plain_em", inputs["q_plain"], inputs["hk_prior"]))]
    with parallel_config(backend="loky", inner_max_num_threads=args.worker_threads):
        em_results = Parallel(n_jobs=args.jobs)(delayed(ex.em_many)(posterior, prior, s.weights)
                                                for _, _, posterior, prior in cells)
    for i, (backbone, variant, inputs) in enumerate(ladder_jobs):
        em_boot = {rung: result for (j, rung, _, _), result in zip(cells, em_results) if j == i}
        summary["em_ladder"][f"{backbone}/{variant}"] = em_ladder(s, rows, backbone, variant, inputs, em_boot, rq2_table)

    # Outputs (aggregates only).
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows.frame("a").to_csv(out / "like_for_like.csv", index=False)
    rows.frame("b").to_csv(out / "em_ladder.csv", index=False)
    rows.frame("c").to_csv(out / "meaning_groups.csv", index=False)
    rows.frame("d").to_csv(out / "no_grade_a.csv", index=False)
    pd.DataFrame(tables["auroc_curves"]).to_csv(out / "rq3_auroc_curves.csv", index=False)
    pd.DataFrame(tables["ratios"]).to_csv(out / "rq3_ratios.csv", index=False)
    pd.DataFrame(tables["hk_minus_sage_only"]).to_csv(out / "rq3_hk_minus_sage_only.csv", index=False)
    pd.DataFrame(tables["regularisation"]).to_csv(out / "rq3_regularisation.csv", index=False)
    pd.DataFrame(tables["scores_by_C"]).to_csv(out / "rq3_scores_by_C.csv", index=False)
    pd.DataFrame(tables["draws_reaching"]).to_csv(out / "rq3_draws_reaching_target.csv", index=False)
    rows.frame("f").to_csv(out / "intercept_only.csv", index=False)
    layout_table.to_csv(out / "layout_location.csv", index=False)
    other_frames.to_csv(out / "frames_not_768x576.csv", index=False)
    summary["runtime"] = {"wall_seconds": round(time.time() - started, 1),
                          "main_process_cpu_seconds": round(time.process_time() - cpu_started, 1)}
    (out / "exploratory_summary.json").write_text(json.dumps(summary, indent=2, default=to_json) + "\n", encoding="utf-8")
    report(summary, tables)
    print(f"\nWrote aggregate tables and exploratory_summary.json to {out}")
    return 0


def report(summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    """Compact printout of the main numbers."""
    print("\n(a) Like-for-like: HyperKvasir raw probe on SAGE inputs, macro-F1 change vs raw | both processed (rq2) | difference")
    for backbone, entry in summary["like_for_like"].items():
        print(f"  {backbone:24s} refit vs RQ1 raw: max |diff| {entry['max_abs_diff_vs_rq1_raw_predictions']:.1e}, "
              f"argmax agreement {entry['argmax_agreement_vs_rq1_raw']}")
        for v in ("fov", "colour", "fov_colour"):
            m = entry[v]["macro_f1"]
            rq2 = m.get("rq2_table_both_processed_change", {})
            print(f"    {v:11s} {interval(m['change_vs_raw'])} | {rq2.get('estimate', float('nan')):+.3f} "
                  f"[{rq2.get('ci_low', float('nan')):+.3f}, {rq2.get('ci_high', float('nan')):+.3f}] | "
                  f"{interval(m['like_for_like_minus_both'])}")
    print("\n(b) Prior and calibration ladder: macro-F1 change vs none [95% CI]; estimated polyp prior")
    for name, entry in summary["em_ladder"].items():
        rungs = entry["rungs"]
        cal = entry["calibration"]["bcts_balanced"]
        print(f"  {name:30s} none {rungs['none']['macro_f1']['estimate']:.3f}; T = {cal['temperature']:.2f}; {entry['reading']}")
        for rung in ("em", "oracle", "bcts", "bcts_em", "bcts_plain_em"):
            prior = rungs[rung]["prior"]["polyp"]
            print(f"    {rung:14s} {interval(rungs[rung]['macro_f1_change_vs_none'])}  polyp share "
                  f"{rungs[rung]['polyp_share']['estimate']:.2f}  prior(polyp) {prior:.3f}")
    print("\n(c) Regrouped meaning groups, F1-gap contrasts (cells whose 95% CI excludes 0):")
    for contrast in ("meaning_minus_strict", "boundary_minus_strict", "meaning_minus_boundary"):
        for key in ("f1", "auroc"):
            entries = {cell: v[f"{key}/{contrast}"] for cell, v in summary["meaning_groups"].items()}
            hits = [cell for cell, e in entries.items() if excludes_zero(e)]
            values = [e["estimate"] for e in entries.values()]
            print(f"  {contrast:24s} {key:5s} range {min(values):+.3f} to {max(values):+.3f}; excluding 0: {hits}")
    print("\n(d) Without HyperKvasir esophagitis grade A (selected C), change vs RQ1 transfer:")
    for name, entry in summary["no_grade_a"].items():
        model = entry["models"]["no_grade_a_selected_C"]
        base = entry["models"]["rq1_transfer"]
        print(f"  {name:30s} C {entry['C_selected']:g}: macro-F1 {interval(model['macro_f1_change_vs_rq1_transfer'])}; "
              f"F1 eso {interval(model['f1_esophagitis_change_vs_rq1_transfer'])}; F1 z-line "
              f"{interval(model['f1_z_line_change_vs_rq1_transfer'])}; z-line->eso {base['z_line_predicted_esophagitis']['estimate']:.0f}"
              f"->{model['z_line_predicted_esophagitis']['estimate']:.0f}; eso->z-line "
              f"{base['esophagitis_predicted_z_line']['estimate']:.0f}->{model['esophagitis_predicted_z_line']['estimate']:.0f}")
    print("\n(e) RQ3 additions")
    for name, entry in summary["rq3"].items():
        rep, reg = entry["reproduction"], entry["regularisation"]
        print(f"  {name:30s} reproduction: max |diff| macro-F1 {rep['max_abs_diff_macro_f1']:.1e}, AUROC "
              f"{rep['max_abs_diff_macro_auroc']:.1e}, other images {rep['runs_with_other_images']}; smallest k "
              f"{entry['smallest_k']['hk_plus_k']}")
        hk5 = entry["hk_minus_sage_only"].get("5")
        print(f"    HK + 5 minus SAGE only {interval(hk5) if hk5 else '-'}; reference by C {reg['reference_by_C']}; HK + all by C "
              f"{reg['hk_plus_all_by_C']}; CV C {reg['hk_plus_all_cv_C']:g} (reference CV C {reg['reference_cv_C_recomputed']:g})")
        for comp in ("pre_registered", "common_C_0.01", "both_cv_tuned"):
            c = reg["comparisons"][comp]
            print(f"    {comp:16s} {interval(c)}  Bonferroni-6 [{c['bonferroni6'][0]:+.3f}, {c['bonferroni6'][1]:+.3f}]")
    reaching = pd.DataFrame(tables["draws_reaching"])
    part = reaching[reaching["regime"] == "hk_plus_k"].pivot_table(index=["backbone", "variant"], columns="k",
                                                                   values="draws_reaching_target")
    print("  draws reaching 90% of the SAGE-only reference (HyperKvasir + k):")
    print("    " + part.to_string().replace("\n", "\n    "))
    print("\n(f) Intercept-only recalibration, mean macro-F1 [95% CI] and difference vs HyperKvasir + k / SAGE only:")
    for name, entry in summary["intercept_only"].items():
        for k in [k for k in ("5", "10") if k in entry["by_k"]]:
            e = entry["by_k"][k]
            print(f"  {name:30s} k={k:3s} {interval(e['macro_f1'])}; vs HK + k {interval(e['macro_f1_minus_hk_plus_k'])}; "
                  f"vs SAGE only {interval(e['macro_f1_minus_sage_only'])}")
    print("\n(g) Lower-GI share by field layout (all frames):", summary["layout"]["lower_gi_share_all_frames"])
    print("    frames not 768x576:", summary["layout"]["frames_not_768x576"])
    print(f"\nRuntime: {summary['runtime']}")


if __name__ == "__main__":
    raise SystemExit(main())

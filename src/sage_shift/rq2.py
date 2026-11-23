"""RQ2 helpers: prior correction, bootstrap scoring and meaning-group gaps."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from sage_shift.bootstrap import percentile_ci
from sage_shift.metrics import evaluate


def saerens_em(proba: np.ndarray, train_prior: Sequence[float], weights: np.ndarray | None = None,
               max_iter: int = 1000, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, int]:
    """Re-weight posteriors to class priors estimated on the target set by EM (Saerens et al., 2002).

    Returns the adjusted probabilities, the estimated target prior and the number of iterations.
    Rows with zero weight do not influence the estimate.
    """
    p = np.asarray(proba, dtype=float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    source = np.asarray(train_prior, dtype=float)
    target = source.copy()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        adjusted = p * (target / source)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        updated = (w[:, None] * adjusted).sum(axis=0) / w.sum()
        done = np.abs(updated - target).max() < tol
        target = updated
        if done:
            break
    adjusted = p * (target / source)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, target, iterations


def scores(y: np.ndarray, proba: np.ndarray, truth: np.ndarray, weights: np.ndarray | None,
           polyp: int) -> dict[str, Any]:
    """Macro-F1, macro AUROC, weighted F1, per-class F1 and the share of predictions that are polyp."""
    out = evaluate(y, proba, truth, weights)
    cm = out["confusion"]
    return {"macro_f1": out["macro_f1"], "macro_auroc": out["macro_auroc"], "weighted_f1": out["weighted_f1"],
            "f1": out["f1"], "polyp_share": float(cm[:, polyp].sum() / cm.sum())}


def boot_scores(y: np.ndarray, proba: np.ndarray, truth: np.ndarray, weights: np.ndarray,
                polyp: int, em_prior: Sequence[float] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Point scores and their values on every resample.

    With `em_prior`, the EM prior correction is re-estimated on each resample, so its intervals
    include the uncertainty of the estimated prior.
    """
    def one(w: np.ndarray | None) -> dict[str, Any]:
        p = proba if em_prior is None else saerens_em(proba, em_prior, w)[0]
        return scores(y, p, truth, w, polyp)
    point = one(None)
    runs = [one(w) for w in weights]
    boot = {key: np.array([run[key] for run in runs]) for key in ("macro_f1", "macro_auroc", "weighted_f1",
                                                                     "polyp_share")}
    boot["f1"] = np.stack([run["f1"] for run in runs])
    return point, boot


def ci(point: float, values: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Estimate with a percentile interval, rounded for tables."""
    low, high = percentile_ci(values, level)
    return {"estimate": round(float(point), 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}


def meaning_gap_difference(f1_ceiling: np.ndarray, f1_transfer: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    """Mean per-class F1 gap of the meaning-shifted classes minus that of the matched ones.

    Works on one row of per-class scores or on a matrix of resamples (classes in the last axis);
    classes absent from a resample (nan) are left out of their group's mean.
    """
    gap = np.asarray(f1_ceiling, dtype=float) - np.asarray(f1_transfer, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a group absent from a resample gives nan
        return np.nanmean(gap[..., shifted], axis=-1) - np.nanmean(gap[..., ~shifted], axis=-1)

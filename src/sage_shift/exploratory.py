"""Exploratory probe analyses: scores on many bootstrap resamples at once, prior and calibration
adjustments, intercept-only recalibration, cross-validation with a source block that always trains,
and contrasts between groups of per-class gaps.

Resamples are rows of a weight matrix (sage_shift.bootstrap.group_weights). Every score here equals
the one sage_shift.metrics computes for the same weights.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.pipeline import Pipeline

from sage_shift.metrics import confusion, label_scores
from sage_shift.probes import Fold, balanced_weights, fit_predict_weighted, make_probe
from sage_shift.rq2 import saerens_em

LOG_FLOOR = float(np.log(np.finfo(float).tiny))  # keeps log-probabilities of zero finite


# --- Scores on many resamples --------------------------------------------------------------------

def confusion_many(y: np.ndarray, pred: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    """Weighted confusion matrices (B, K, K), one per row of `weights` (B, n).

    `pred` holds one prediction per image (n,) or one per resample and image (B, n).
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    y, pred = np.asarray(y), np.asarray(pred)
    cells = n_classes * n_classes
    if pred.ndim == 1:
        onehot = np.zeros((len(y), cells))
        onehot[np.arange(len(y)), y * n_classes + pred] = 1.0
        return (w @ onehot).reshape(len(w), n_classes, n_classes)
    index = np.arange(len(w))[:, None] * cells + y[None, :] * n_classes + pred
    flat = np.bincount(index.ravel(), weights=w.ravel(), minlength=len(w) * cells)
    return flat.reshape(len(w), n_classes, n_classes)


def mcc(cm: np.ndarray) -> np.ndarray:
    """Multi-class Matthews correlation (as scikit-learn) of confusion matrices (..., K, K); 0 if undefined."""
    cm = np.asarray(cm, dtype=float)
    truth, pred = cm.sum(axis=-1), cm.sum(axis=-2)
    correct, total = np.trace(cm, axis1=-2, axis2=-1), truth.sum(axis=-1)
    cov_tp = correct * total - (truth * pred).sum(axis=-1)
    cov_pp = total ** 2 - (pred * pred).sum(axis=-1)
    cov_tt = total ** 2 - (truth * truth).sum(axis=-1)
    denom = np.sqrt(cov_tt * cov_pp)
    return np.divide(cov_tp, denom, out=np.zeros_like(cov_tp), where=denom > 0)


def confusion_scores(cm: np.ndarray, polyp: int | None = None) -> dict[str, np.ndarray]:
    """Macro-F1, balanced accuracy, accuracy, MCC and per-class F1 (nan for absent classes) of
    confusion matrices (..., K, K), as sage_shift.metrics.label_scores; with `polyp`, also the share
    of predictions in that class."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    total, present = support.sum(axis=-1), support > 0
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=present)
    n_present = present.sum(axis=-1)
    out = {"macro_f1": (f1 * present).sum(axis=-1) / n_present,
           "balanced_accuracy": (recall * present).sum(axis=-1) / n_present,
           "accuracy": tp.sum(axis=-1) / total,
           "mcc": mcc(cm),
           "f1": np.where(present, f1, np.nan)}
    if polyp is not None:
        out["polyp_share"] = predicted[..., polyp] / total
    return out


def auroc_many(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sage_shift.metrics.auroc for every row of `weights` (ties count one half); nan without both classes."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    truth, scores = np.asarray(truth, dtype=float), np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    pos = np.add.reduceat(w[:, order] * truth[order], starts, axis=1)
    neg = np.add.reduceat(w[:, order] * (1 - truth[order]), starts, axis=1)
    n_pos, n_neg = pos.sum(axis=1), neg.sum(axis=1)
    below = np.cumsum(neg, axis=1) - neg
    area = (pos * (below + 0.5 * neg)).sum(axis=1)
    valid = (n_pos > 0) & (n_neg > 0)
    return np.where(valid, area / np.where(valid, n_pos * n_neg, 1.0), np.nan)


def resample_scores(y: np.ndarray, proba: np.ndarray, truth: np.ndarray, weights: np.ndarray, polyp: int,
                    with_auroc: bool = True) -> dict[str, np.ndarray]:
    """confusion_scores of the argmax predictions on every resample, plus per-class AUROC with the
    multi-label `truth` and their mean over the classes that can be scored (macro AUROC)."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    out = confusion_scores(confusion_many(y, proba.argmax(axis=1), proba.shape[1], w), polyp)
    if with_auroc:
        out["auroc"] = np.stack([auroc_many(truth[:, c], proba[:, c], w) for c in range(proba.shape[1])], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # a resample may score no class
            out["macro_auroc"] = np.nanmean(out["auroc"], axis=1)
    return out


def best_constant_many(y: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    """Macro-F1 of the best constant predictor (always one class) on every resample."""
    y = np.asarray(y)
    return np.max([confusion_scores(confusion_many(y, np.full(len(y), c), n_classes, weights))["macro_f1"]
                   for c in range(n_classes)], axis=0)


# --- Prior shift and calibration -----------------------------------------------------------------

def class_shares(y: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    """Weighted class shares of `y`, one row per row of `weights`."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    return (w @ np.eye(n_classes)[np.asarray(y)]) / w.sum(axis=1, keepdims=True)


def reweight_prior(proba: np.ndarray, target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Move posteriors from the `source` class prior to `target` (Saerens et al., 2002).

    A target of shape (B, K) gives one adjusted matrix per row, shape (B, n, K).
    """
    ratio = np.asarray(target, dtype=float) / np.asarray(source, dtype=float)
    adjusted = proba[None] * ratio[:, None, :] if ratio.ndim == 2 else proba * ratio
    return adjusted / adjusted.sum(axis=-1, keepdims=True)


def em_many(proba: np.ndarray, source_prior: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EM prior correction re-estimated on every resample: argmax predictions (B, n) and priors (B, K)."""
    preds, priors = [], []
    for w in np.atleast_2d(np.asarray(weights, dtype=float)):
        adjusted, prior, _ = saerens_em(proba, source_prior, w)
        preds.append(adjusted.argmax(axis=1))
        priors.append(prior)
    return np.array(preds), np.array(priors)


def nll(log_proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Weighted mean negative log-likelihood of the true classes."""
    lp = np.maximum(np.asarray(log_proba, dtype=float), LOG_FLOOR)
    w = np.ones(len(lp)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(-(w * lp[np.arange(len(lp)), np.asarray(y)]).sum() / w.sum())


def fit_bcts(log_proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Bias-corrected temperature scaling (Alexandari et al., 2020): softmax(log_proba / T + b).

    T and b minimise the weighted negative log-likelihood. Returns T and b with b[0] = 0.
    """
    lp = np.maximum(np.asarray(log_proba, dtype=float), LOG_FLOOR)
    n, k = lp.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w = w / w.sum()
    onehot = np.eye(k)[np.asarray(y)]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scale, bias = np.exp(theta[0]), np.r_[0.0, theta[1:]]
        logq = log_softmax(scale * lp + bias, axis=1)
        residual = (np.exp(logq) - onehot) * w[:, None]
        grad_scale = (residual * lp).sum() * scale  # chain rule for log(1 / T)
        return float(-(w * (logq * onehot).sum(axis=1)).sum()), np.r_[grad_scale, residual.sum(axis=0)[1:]]

    result = minimize(objective, np.zeros(k), jac=True, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10})
    return float(np.exp(-result.x[0])), np.r_[0.0, result.x[1:]]


def apply_bcts(log_proba: np.ndarray, temperature: float, bias: np.ndarray) -> np.ndarray:
    """Probabilities after bias-corrected temperature scaling."""
    lp = np.maximum(np.asarray(log_proba, dtype=float), LOG_FLOOR)
    return softmax(lp / temperature + np.asarray(bias, dtype=float), axis=1)


# --- Linear probes: fixed weights, refitted intercepts -------------------------------------------

def linear_parts(model: Pipeline) -> dict[str, np.ndarray]:
    """Scaler mean and scale, coefficients and intercepts of a sage_shift.probes.make_probe pipeline."""
    scaler, logistic = model[0], model[-1]
    return {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy(), "coef": logistic.coef_.copy(),
            "intercept": logistic.intercept_.copy()}


def fit_linear(x: np.ndarray, y: np.ndarray, C: float, seed: int, max_iter: int) -> dict[str, np.ndarray]:
    """Fit a sage_shift.probes.make_probe pipeline and return its linear parts."""
    return linear_parts(make_probe(C, seed, max_iter).fit(x, y))


def logits(parts: Mapping[str, np.ndarray], x: np.ndarray, intercept: bool = True) -> np.ndarray:
    """Decision values of a linear probe, with or without its intercepts."""
    z = ((np.asarray(x, dtype=float) - parts["mean"]) / parts["scale"]) @ parts["coef"].T
    return z + parts["intercept"] if intercept else z


def fit_intercepts(z: np.ndarray, y: np.ndarray, b0: np.ndarray, penalty: float = 1e-3,
                   sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Class intercepts for fixed decision values `z` (without intercepts); all other weights stay fixed.

    Minimises the class-balanced cross-entropy of softmax(z + b), with the softmax and the
    cross-entropy taken over the classes present in `y`, plus penalty * ||b - b0||^2. Intercepts of
    absent classes keep their values in `b0`. `sample_weight` replaces the balanced weights.
    """
    z, y, b0 = np.asarray(z, dtype=float), np.asarray(y), np.asarray(b0, dtype=float)
    present = np.unique(y)
    if sample_weight is None:
        counts = np.bincount(y, minlength=z.shape[1])
        w = 1.0 / (len(present) * counts[y])
    else:
        w = np.asarray(sample_weight, dtype=float)
    w = w / w.sum()
    column = np.searchsorted(present, y)
    onehot = np.eye(len(present))[column]
    zp, start = z[:, present], b0[present]

    def objective(bp: np.ndarray) -> tuple[float, np.ndarray]:
        logq = log_softmax(zp + bp, axis=1)
        loss = -(w * logq[np.arange(len(y)), column]).sum() + penalty * ((bp - start) ** 2).sum()
        grad = ((np.exp(logq) - onehot) * w[:, None]).sum(axis=0) + 2 * penalty * (bp - start)
        return float(loss), grad

    result = minimize(objective, start.copy(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10})
    b = b0.copy()
    b[present] = result.x
    return b


# --- Training regimes and cross-validation -------------------------------------------------------

def patient_draws(patients: np.ndarray, ks: Sequence[int], draws: int, seed: int) -> list[tuple[int, int, np.ndarray]]:
    """(k, draw, patients) for every k; one draw when k is 0 or covers every patient (as scripts/10)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        if k == 0 or k >= len(patients):
            out.append((min(k, len(patients)), 0, patients if k else patients[:0]))
            continue
        for draw in range(draws):
            out.append((k, draw, np.sort(rng.choice(patients, size=k, replace=False))))
    return out


def fit_regime(regime: str, x_hk: np.ndarray | None, y_hk: np.ndarray | None, x_sage: np.ndarray, y_sage: np.ndarray,
               x_test: np.ndarray, C: float, seed: int, n_classes: int, max_iter: int) -> np.ndarray:
    """Test probabilities of an RQ3 probe: "hk_plus_k" (two sources weighted equally, classes balanced
    within each) or "sage_only"; a single training class is predicted for every image (as scripts/10)."""
    if regime == "hk_plus_k":
        x, y = np.vstack([x_hk, x_sage]), np.r_[y_hk, y_sage]
        source = np.r_[np.zeros(len(y_hk)), np.ones(len(y_sage))]
    else:
        x, y, source = x_sage, y_sage, np.zeros(len(y_sage))
    if len(np.unique(y)) < 2:
        return np.tile(np.eye(n_classes)[y[0]], (len(x_test), 1))
    return fit_predict_weighted(x, y, balanced_weights(y, source), x_test, C, seed, n_classes, max_iter)


def with_source_block(n_source: int, target_folds: Sequence[Fold]) -> list[Fold]:
    """Folds over stacked [source; target] rows: every source row trains in every fold and the target
    rows follow `target_folds` (indices into the target rows)."""
    block = np.arange(n_source)
    return [(np.r_[block, n_source + np.asarray(train)], n_source + np.asarray(val)) for train, val in target_folds]


def weighted_fold_score(x: np.ndarray, y: np.ndarray, source: np.ndarray, fold: Fold, C: float, seed: int,
                        n_classes: int, max_iter: int) -> float:
    """Validation macro-F1 of a probe trained on one fold, weighted with balanced_weights over its sources."""
    train, val = fold
    proba = fit_predict_weighted(x[train], y[train], balanced_weights(y[train], source[train]), x[val], C, seed,
                                 n_classes, max_iter)
    return label_scores(confusion(y[val], proba.argmax(axis=1), n_classes))["macro_f1"]


def choose_c(mean_scores: Mapping[float, float]) -> float:
    """Best mean validation score; ties go to the smaller C (the rule of sage_shift.probes.select_c)."""
    return min(mean_scores, key=lambda C: (-round(mean_scores[C], 10), C))


# --- Groups of classes and thresholds ------------------------------------------------------------

def group_means(values: np.ndarray, groups: Mapping[str, Sequence[int]]) -> dict[str, np.ndarray]:
    """Mean over each group's classes (last axis); classes that are nan are left out."""
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a group absent from a resample gives nan
        return {name: np.nanmean(values[..., list(columns)], axis=-1) for name, columns in groups.items()}


def group_contrasts(values: np.ndarray, groups: Mapping[str, Sequence[int]],
                    pairs: Sequence[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Differences of group means, keyed "<a>_minus_<b>"."""
    means = group_means(values, groups)
    return {f"{a}_minus_{b}": means[a] - means[b] for a, b in pairs}


def smallest_k(ks: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    """Smallest k whose value reaches `threshold`, or None."""
    reached = [k for k, value in zip(ks, values) if np.isfinite(value) and value >= threshold]
    return int(min(reached)) if reached else None

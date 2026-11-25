"""Sensitivity analyses for the paired patient-level bootstrap on the SAGE six-class test set.

Scores are built from per-patient pieces (confusion matrices and ranked positive-negative pairs),
so bootstrap resamples, leave-one-patient-out deletions and patient swaps all reduce to weighted
sums over patients. Image weights inside a patient are 1 (the image-weighted estimand) or 1/n_i,
n_i = the patient's test images (the patient-weighted estimand: every patient counts once).
The refit helpers at the end repeat the RQ1 and RQ3 probes with a chosen lbfgs tolerance.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sage_shift.metrics import confusion, label_scores
from sage_shift.probes import Fold, balanced_weights, make_probe

# --- Patients and weights -------------------------------------------------------------------------

def patient_index(groups: np.ndarray) -> np.ndarray:
    """Each image's patient as 0..P-1 in sorted order, the order bootstrap.group_weights draws them in."""
    return np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)


def draw_counts(weights: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-patient draw counts (resamples x patients) behind image weights from group_weights."""
    index = patient_index(groups)
    first = np.unique(index, return_index=True)[1]
    counts = np.asarray(weights, dtype=float)[:, first]
    if not np.array_equal(counts[:, index], np.asarray(weights, dtype=float)):
        raise ValueError("image weights differ within a patient")
    return counts


def inverse_size_weights(groups: np.ndarray) -> np.ndarray:
    """Image weights 1/n_i (n_i = images of the image's patient), so that every patient weighs 1."""
    index = patient_index(groups)
    return 1.0 / np.bincount(index)[index]


def patient_weighted(weights: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Patient-weighted bootstrap image weights: the patient's draw count times 1/n_i."""
    return np.asarray(weights, dtype=float) * inverse_size_weights(groups)


def kish_effective(sizes: Sequence[float]) -> float:
    """Kish effective number of clusters, (sum n)^2 / sum n^2, from per-cluster sizes."""
    n = np.asarray(sizes, dtype=float)
    return float(n.sum() ** 2 / (n ** 2).sum()) if (n ** 2).sum() > 0 else 0.0


# --- Per-patient pieces of the scores -------------------------------------------------------------

class PatientScores:
    """Per-patient confusion matrices and AUROC pair counts of one set of predictions.

    Patient weights `u` of shape (..., P) (ones, bootstrap draw counts or a deletion design)
    give confusion matrices (..., K, K) and per-class AUROCs (..., K) for that weighting.
    """

    def __init__(self, y: np.ndarray, proba: np.ndarray, truth: np.ndarray | None, groups: np.ndarray,
                 image_weights: np.ndarray | None = None) -> None:
        index = patient_index(groups)
        n_patients, k = int(index.max()) + 1, proba.shape[1]
        a = np.ones(len(index)) if image_weights is None else np.asarray(image_weights, dtype=float)
        members = np.eye(n_patients)[index]  # image-to-patient indicator (N, P)
        cells = np.eye(k * k)[np.asarray(y) * k + np.asarray(proba).argmax(axis=1)]
        self.parts = (members.T @ (a[:, None] * cells)).reshape(n_patients, k, k)
        self.pairs: np.ndarray | None = None
        if truth is not None:
            t = np.asarray(truth, dtype=float)
            pos, neg = a[:, None] * t, a[:, None] * (1 - t)
            self.pos, self.neg = (members.T @ pos).T, (members.T @ neg).T  # (K, P)
            self.pairs = np.empty((k, n_patients, n_patients))
            for c in range(k):
                s = np.asarray(proba)[:, c]
                ranked = (s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])  # ties count one half
                self.pairs[c] = members.T @ (pos[:, c, None] * ranked * neg[None, :, c]) @ members

    @property
    def n_patients(self) -> int:
        return self.parts.shape[0]

    def confusion(self, u: np.ndarray) -> np.ndarray:
        """Weighted confusion matrices (rows: truth) for patient weights `u`."""
        return np.tensordot(np.asarray(u, dtype=float), self.parts, axes=(-1, 0))

    def auroc(self, u: np.ndarray) -> np.ndarray:
        """One-vs-rest AUROC per class for patient weights `u`; nan without both outcomes."""
        if self.pairs is None:
            raise ValueError("built without truth: no AUROC")
        u = np.asarray(u, dtype=float)
        num = np.stack([((u @ pairs) * u).sum(axis=-1) for pairs in self.pairs], axis=-1)
        den = (u @ self.pos.T) * (u @ self.neg.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def nanmean_last(values: np.ndarray) -> np.ndarray:
    """Mean over the last axis, ignoring nan (all-nan rows give nan without a warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=-1)


def class_f1(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 of confusion matrices (..., K, K); nan for classes absent from the truth."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, 2 * tp / (support + predicted), np.nan)


def macro_f1(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 over the classes present, as metrics.label_scores computes it."""
    return nanmean_last(class_f1(cm))


def balanced_accuracy(cm: np.ndarray) -> np.ndarray:
    """Mean recall over the classes present."""
    tp, support = np.diagonal(cm, axis1=-2, axis2=-1), cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return nanmean_last(np.where(support > 0, tp / support, np.nan))


def mcc(cm: np.ndarray) -> np.ndarray:
    """Multi-class Matthews correlation (Gorodkin, 2004); 0 when undefined, as in scikit-learn."""
    t, p = cm.sum(axis=-1), cm.sum(axis=-2)
    s, correct = t.sum(axis=-1), np.trace(cm, axis1=-2, axis2=-1)
    cov_tp = correct * s - (t * p).sum(axis=-1)
    denom = (s ** 2 - (p * p).sum(axis=-1)) * (s ** 2 - (t * t).sum(axis=-1))
    ok = denom > 1e-12 * s ** 4  # zero up to rounding: one predicted or one true class
    return np.where(ok, cov_tp / np.sqrt(np.where(ok, denom, 1.0)), 0.0)


def best_constant_macro_f1(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 (over classes present) of the best always-the-same-class predictor for the truth in `cm`.

    Always predicting class c gives F1_c = 2 t_c / (t_c + total) and 0 for every other class.
    """
    support = cm.sum(axis=-1)
    total = support.sum(axis=-1, keepdims=True)
    return (2 * support / (support + total)).max(axis=-1) / (support > 0).sum(axis=-1)


# --- A statistic on the full set, the resamples and the deletions --------------------------------

@dataclass(frozen=True)
class Replicates:
    """A statistic on the full test set (`point`), on each bootstrap resample (`boot`, first axis)
    and with each patient left out (`jack`, first axis). Per-class statistics keep a last axis."""

    point: np.ndarray
    boot: np.ndarray
    jack: np.ndarray

    def __sub__(self, other: Replicates) -> Replicates:
        return Replicates(self.point - other.point, self.boot - other.boot, self.jack - other.jack)

    def pick(self, j: int) -> Replicates:
        """One column of a per-class statistic."""
        return Replicates(self.point[..., j], self.boot[..., j], self.jack[..., j])

    def apply(self, fn: Callable[[np.ndarray], np.ndarray], *others: Replicates) -> Replicates:
        """fn(self, *others) on each of point, boot and jack."""
        parts = [(r.point, r.boot, r.jack) for r in (self, *others)]
        return Replicates(*(fn(*[p[i] for p in parts]) for i in range(3)))

    def se(self) -> float:
        """Bootstrap standard error (SD of the resamples, nan ignored)."""
        return float(np.nanstd(self.boot, ddof=1))


def scores_for(scores: PatientScores, u: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric of one set of predictions for patient weights `u`."""
    cm = scores.confusion(u)
    out = {"macro_f1": macro_f1(cm), "f1": class_f1(cm), "mcc": mcc(cm), "balanced_accuracy": balanced_accuracy(cm),
           "best_constant": best_constant_macro_f1(cm)}
    if scores.pairs is not None:
        out["auroc"] = scores.auroc(u)
        out["macro_auroc"] = nanmean_last(out["auroc"])
    return out


def metric_replicates(scores: PatientScores, counts: np.ndarray) -> dict[str, Replicates]:
    """Every metric on the full set (ones), the resamples (draw counts) and each patient deleted."""
    n = scores.n_patients
    values = [scores_for(scores, u) for u in (np.ones(n), counts, 1.0 - np.eye(n))]
    return {name: Replicates(*(v[name] for v in values)) for name in values[0]}


# --- Intervals and tests --------------------------------------------------------------------------

def jackknife_se(values: np.ndarray, multiplicity: np.ndarray | None = None) -> np.ndarray:
    """Jackknife standard error from leave-one-out values along the last axis.

    `multiplicity` (broadcast to `values`) counts how often a unit occurs, so a patient drawn twice
    in a resample is deleted twice; nan values (statistic undefined after a deletion) are skipped.
    """
    v = np.asarray(values, dtype=float)
    m = np.ones_like(v) if multiplicity is None else np.broadcast_to(np.asarray(multiplicity, dtype=float), v.shape)
    m = np.where(np.isnan(v), 0.0, m)
    v = np.where(m > 0, v, 0.0)
    n = m.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (m * v).sum(axis=-1) / n
        ss = (m * (v - mean[..., None]) ** 2).sum(axis=-1)
        return np.where(n > 1, np.sqrt((n - 1) / n * ss), np.nan)


def jackknife_acceleration(values: np.ndarray) -> float:
    """BCa acceleration from leave-one-out values: sum(d^3) / (6 sum(d^2)^1.5), d = mean - value."""
    v = np.asarray(values, dtype=float)
    d = np.nanmean(v) - v[~np.isnan(v)]
    ss = float((d ** 2).sum())
    return float((d ** 3).sum() / (6 * ss ** 1.5)) if ss > 0 else 0.0


def bca_interval(point: float, boot: np.ndarray, jack: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Bias-corrected and accelerated percentile interval (Efron, 1987).

    The bias correction z0 comes from the share of resamples below the estimate (ties count one
    half, as in scipy.stats.bootstrap); the acceleration from the leave-one-out values `jack`.
    """
    b = np.asarray(boot, dtype=float)
    b = b[~np.isnan(b)]
    share = ((b < point).sum() + 0.5 * (b == point).sum()) / len(b)
    z0 = float(norm.ppf(np.clip(share, 0.5 / len(b), 1 - 0.5 / len(b))))
    a = jackknife_acceleration(jack)
    z = norm.ppf([(1 - level) / 2, (1 + level) / 2])
    tails = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    low, high = np.percentile(b, 100 * tails)
    return {"low": float(low), "high": float(high), "z0": z0, "acceleration": a,
            "tail_low": float(tails[0]), "tail_high": float(tails[1])}


def bootstrap_t_interval(point: float, se: float, boot: np.ndarray, boot_se: np.ndarray,
                         level: float = 0.95) -> dict[str, float]:
    """Studentized (bootstrap-t) interval: [point - q_high * se, point - q_low * se].

    t* = (boot - point) / boot_se. A resample with SE 0 gives t* = +-inf (0 if it equals the
    estimate); resamples where the statistic or its SE is undefined are dropped. A bound whose
    t quantile is infinite is returned as -inf or +inf (unbounded).
    """
    b, s = np.asarray(boot, dtype=float), np.asarray(boot_se, dtype=float)
    keep = ~np.isnan(b) & ~np.isnan(s)
    diff, s = b[keep] - point, s[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(s > 0, diff / s, np.where(diff == 0, 0.0, np.sign(diff) * np.inf))
    big = 1e300  # stands in for infinity so that interpolation stays defined
    q_low, q_high = np.quantile(np.clip(t, -big, big), [(1 - level) / 2, (1 + level) / 2])
    q_low, q_high = (np.sign(q) * np.inf if abs(q) > 1e290 else q for q in (q_low, q_high))
    return {"low": float(point - q_high * se), "high": float(point - q_low * se), "t_low": float(q_low),
            "t_high": float(q_high), "resamples": int(keep.sum()), "zero_se": int((s == 0).sum())}


def bootstrap_p(boot: np.ndarray, null: float = 0.0) -> float:
    """Two-sided percentile-bootstrap p-value: twice the smaller share of resamples on either side of `null`."""
    b = np.asarray(boot, dtype=float)
    b = b[~np.isnan(b)]
    return float(min(1.0, 2 * min((b <= null).mean(), (b >= null).mean())))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def max_abs_z_intervals(points: np.ndarray, boot: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous intervals point +- q * SE for the columns of `boot` (resamples x quantities).

    SE is each column's bootstrap SD and q the `level` quantile over resamples of max_j |z_j|,
    z_j = (boot_j - point_j) / SE_j.
    """
    points = np.asarray(points, dtype=float)
    se = np.nanstd(boot, axis=0, ddof=1)
    q = float(np.nanquantile(np.nanmax(np.abs(boot - points) / se, axis=1), level))
    return points - q * se, points + q * se, q


def mde(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimal detectable effect of a two-sided normal test: (z_{1-alpha/2} + z_power) * SE."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def power_at(effect: float, se: float, alpha: float = 0.05) -> float:
    """Power of a two-sided normal test at the given true effect."""
    z, shift = norm.ppf(1 - alpha / 2), abs(effect) / se
    return float(norm.cdf(shift - z) + norm.cdf(-shift - z))


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (treats images as independent)."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf((1 + level) / 2)
    p = successes / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def resample_deletions(parts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Confusion matrices of each resample with one copy of each patient deleted, (B, P, K, K).

    Entries for patients not drawn in a resample are meaningless; give them multiplicity 0.
    """
    total = np.tensordot(counts, parts, axes=(-1, 0))
    return total[:, None] - parts[None]


def swap_test(parts_reference: np.ndarray, parts_transfer: np.ndarray,
              statistic: Callable[[np.ndarray, np.ndarray], np.ndarray], n_swaps: int,
              seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level paired swap (randomisation) test of a reference-minus-transfer statistic.

    Under H0 the two probes' predictions are exchangeable within each patient, so each random swap
    vector exchanges them for a random subset of patients. `statistic(cm_reference, cm_transfer)`
    works on stacks of confusion matrices. Returns the observed statistic and the two-sided Monte
    Carlo p-values (1 + #{|T*| >= |T|}) / (1 + n_swaps).
    """
    n, k = parts_reference.shape[0], parts_reference.shape[-1]
    flips = np.random.default_rng(seed).integers(0, 2, size=(n_swaps, n)).astype(float)
    moved = (flips @ (parts_transfer - parts_reference).reshape(n, -1)).reshape(-1, k, k)
    total_r, total_t = parts_reference.sum(axis=0), parts_transfer.sum(axis=0)
    observed = statistic(total_r, total_t)
    null = statistic(total_r + moved, total_t - moved)
    extreme = (np.abs(null) >= np.abs(observed) - 1e-12).sum(axis=0)
    return observed, (1 + extreme) / (1 + n_swaps)


def min_swap_p(n_patients: int) -> float:
    """Smallest two-sided p-value a sign-flip test over `n_patients` patients can reach: 2^(1-G)."""
    return float(2.0 ** (1 - n_patients))


# --- Training resamples and refits ----------------------------------------------------------------

def patient_bootstrap_rows(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one patient-level bootstrap resample (a drawn patient brings all its rows)."""
    groups = np.asarray(groups)
    patients = np.unique(groups)
    rows = {p: np.flatnonzero(groups == p) for p in patients}
    return np.concatenate([rows[p] for p in rng.choice(patients, size=len(patients), replace=True)])


def stratified_bootstrap_rows(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one class-stratified image bootstrap (every class keeps its size)."""
    y = np.asarray(y)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True)
                           for c in np.unique(y)])


def fit_predict_with(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, C: float, seed: int,
                     n_classes: int, train_rows: np.ndarray | None = None, test_rows: np.ndarray | None = None,
                     tol: float | None = None, threads: int | None = None, max_iter: int = 5000) -> np.ndarray:
    """The RQ1 probe (probes.make_probe) with an optional lbfgs tolerance and BLAS thread limit.

    Fits on X_train[train_rows] and returns probabilities for X_test[test_rows] (all rows by
    default), with zeros for classes unseen in training.
    """
    x, y = (X_train, y_train) if train_rows is None else (X_train[train_rows], y_train[train_rows])
    x_test = X_test if test_rows is None else X_test[test_rows]
    model = make_probe(C, seed, max_iter)
    if tol is not None:
        model.set_params(logisticregression__tol=tol)
    with threadpool_limits(limits=threads):
        model.fit(x, y)
        proba = np.zeros((len(x_test), n_classes))
        proba[:, model.classes_] = model.predict_proba(x_test)
    return proba


def cv_choice(y: np.ndarray, folds: Sequence[Fold], grid: Sequence[float], probas: dict[tuple[float, int], np.ndarray],
              n_classes: int) -> tuple[float, dict[float, float]]:
    """Mean validation macro-F1 per C from out-of-fold probabilities keyed (C, fold), and the chosen C
    (ties go to the smaller C), as probes.select_c computes them."""
    scores: dict[float, list[float]] = {C: [] for C in grid}
    for (C, k), proba in probas.items():
        val = folds[k][1]
        scores[C].append(label_scores(confusion(y[val], proba.argmax(axis=1), n_classes))["macro_f1"])
    mean = {C: float(np.mean(values)) for C, values in scores.items()}
    return min(grid, key=lambda C: (-round(mean[C], 10), C)), mean


def select_c_with(X: np.ndarray, y: np.ndarray, folds: Sequence[Fold], grid: Sequence[float], seed: int,
                  n_classes: int, tol: float | None = None, threads: int | None = None, jobs: int = 1,
                  max_iter: int = 5000) -> tuple[float, dict[float, float]]:
    """probes.select_c (mean validation macro-F1, ties to the smaller C) with an lbfgs tolerance."""
    tasks = [(C, k) for C in grid for k in range(len(folds))]
    results = Parallel(n_jobs=jobs)(
        delayed(fit_predict_with)(X, y, X, C, seed, n_classes, folds[k][0], folds[k][1], tol, threads, max_iter)
        for C, k in tasks)
    return cv_choice(y, folds, grid, dict(zip(tasks, results)), n_classes)


def fit_tight(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, C: float, seed: int, n_classes: int,
              train_rows: np.ndarray | None = None, test_rows: np.ndarray | None = None,
              sample_weight: np.ndarray | None = None, tol: float = 1e-6, max_iter: int = 5000,
              retry_iter: int = 50000, threads: int | None = 1) -> tuple[np.ndarray, int, int]:
    """A probe fitted at a tight lbfgs tolerance, refitted with `retry_iter` iterations if lbfgs stops at `max_iter`.

    Without `sample_weight` it is the RQ1 probe (probes.make_probe, balanced class weights); with it,
    the RQ3 probe (explicit sample weights, as probes.fit_predict_weighted). Returns the probabilities
    for X_test[test_rows] (zeros for classes unseen in training) and the ConvergenceWarnings of the
    first fit and of the retry (0 when no retry was needed).
    """
    x, y = (X_train, y_train) if train_rows is None else (X_train[train_rows], y_train[train_rows])
    x_test = X_test if test_rows is None else X_test[test_rows]

    def fit(iterations: int) -> tuple[Pipeline, int]:
        if sample_weight is None:
            model, params = make_probe(C, seed, iterations).set_params(logisticregression__tol=tol), {}
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=iterations, random_state=seed,
                                                                       tol=tol))
            params = {"logisticregression__sample_weight": sample_weight}
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y, **params)
        return model, sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    model, first = fit(max_iter)
    model, left = fit(retry_iter) if first else (model, 0)
    with threadpool_limits(limits=threads):
        proba = np.zeros((len(x_test), n_classes))
        proba[:, model.classes_] = model.predict_proba(x_test)
    return proba, first, left


def rq3_draws(patients: np.ndarray, ks: Sequence[int], draws: int, seed: int) -> list[tuple[int, int, np.ndarray]]:
    """The patient draws of the RQ3 design (scripts/10): (k, draw, patients) for every k, with one
    draw when k is 0 or covers every patient."""
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        if k == 0 or k >= len(patients):
            out.append((min(k, len(patients)), 0, patients if k else patients[:0]))
            continue
        for draw in range(draws):
            out.append((k, draw, np.sort(rng.choice(patients, size=k, replace=False))))
    return out


def rq3_fit(x_hk: np.ndarray, y_hk: np.ndarray, x_sage: np.ndarray, y_sage: np.ndarray, rows: np.ndarray,
            x_test: np.ndarray, regime: str, C: float, seed: int, n_classes: int, tol: float = 1e-6,
            max_iter: int = 5000, threads: int | None = 1) -> tuple[np.ndarray, int, int]:
    """One RQ3 probe as scripts/10 fits it, at a chosen tolerance.

    hk_plus_k: HyperKvasir plus the SAGE `rows`, classes balanced within each source and the two
    sources weighted equally; sage_only: the SAGE rows alone. A draw holding one class predicts it.
    """
    xs, ys = x_sage[rows], y_sage[rows]
    if regime == "hk_plus_k":
        x, y = np.vstack([x_hk, xs]), np.r_[y_hk, ys]
        source = np.r_[np.zeros(len(y_hk)), np.ones(len(ys))]
    else:
        x, y, source = xs, ys, np.zeros(len(ys))
    if len(np.unique(y)) < 2:
        return np.tile(np.eye(n_classes)[y[0]], (len(x_test), 1)), 0, 0
    return fit_tight(x, y, x_test, C, seed, n_classes, sample_weight=balanced_weights(y, source), tol=tol,
                     max_iter=max_iter, threads=threads)

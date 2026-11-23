"""Six-class metrics with optional sample weights (bootstrap resamples are passed as weights).

Macro scores average over the classes present in the truth, so a resample that happens to miss
a class is not penalised for it. With unit weights the results equal scikit-learn's.
"""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted confusion matrix (rows: truth, columns: prediction)."""
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    flat = np.bincount(np.asarray(y_true) * n_classes + np.asarray(y_pred), weights=w, minlength=n_classes ** 2)
    return flat.reshape(n_classes, n_classes)


def label_scores(cm: np.ndarray) -> dict[str, float | np.ndarray]:
    """F1, recall and accuracy from a confusion matrix."""
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    present = support > 0
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    recall = np.divide(tp, support, out=np.full_like(tp, np.nan, dtype=float), where=present)
    total = support.sum()
    return {
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / total),
        "balanced_accuracy": float(np.nanmean(recall)),
        "accuracy": float(tp.sum() / total),
        "f1": np.where(present, f1, np.nan),
    }


def macro_f1_many(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    """Macro-F1 for many weight vectors at once (rows of `weights`), as `label_scores` computes it."""
    pairs = np.asarray(y_true) * n_classes + np.asarray(y_pred)
    counts = (np.asarray(weights, dtype=float) @ np.eye(n_classes * n_classes)[pairs]).reshape(-1, n_classes, n_classes)
    tp = np.diagonal(counts, axis1=1, axis2=2)
    support, predicted = counts.sum(axis=2), counts.sum(axis=1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return (f1 * present).sum(axis=1) / present.sum(axis=1)


def _tie_groups(scores: np.ndarray, *columns: np.ndarray, descending: bool = False) -> list[np.ndarray]:
    """Sum each column within groups of equal score, in score order."""
    order = np.argsort(-scores if descending else scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    return [np.add.reduceat(col[order], starts) for col in columns]


def auroc(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ROC AUC (ties count one half); nan without both classes."""
    pos, neg = _tie_groups(scores, weights * truth, weights * (1 - truth))
    if pos.sum() <= 0 or neg.sum() <= 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def average_precision(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average precision, defined as in scikit-learn; nan without positives."""
    pos, neg = _tie_groups(scores, weights * truth, weights * (1 - truth), descending=True)
    if pos.sum() <= 0:
        return float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_step = pos / pos.sum()
    return float((recall_step * precision).sum())


def evaluate(y_true: np.ndarray, proba: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """All metrics for one set of predictions.

    `y_true` holds class indices (the `class` column), `proba` the predicted probabilities and
    `truth` the 0/1 multi-label matrix used for the one-vs-rest AUROC and AP.
    """
    n_classes = proba.shape[1]
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    y_true, proba, truth, w = np.asarray(y_true)[keep], proba[keep], truth[keep].astype(float), w[keep]
    cm = confusion(y_true, proba.argmax(axis=1), n_classes, w)
    out = label_scores(cm)
    out["auroc"] = np.array([auroc(truth[:, c], proba[:, c], w) for c in range(n_classes)])
    out["ap"] = np.array([average_precision(truth[:, c], proba[:, c], w) for c in range(n_classes)])
    out["macro_auroc"] = float(np.nanmean(out["auroc"]))
    out["macro_ap"] = float(np.nanmean(out["ap"]))
    out["confusion"] = cm
    return out

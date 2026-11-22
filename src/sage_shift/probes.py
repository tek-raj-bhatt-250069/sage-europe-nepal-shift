"""Linear probes on frozen features: standardised multinomial logistic regression (L2, lbfgs)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sage_shift.metrics import confusion, label_scores

Fold = tuple[np.ndarray, np.ndarray]


def make_probe(C: float, seed: int, max_iter: int = 5000) -> Pipeline:
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed))


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, C: float, seed: int,
                n_classes: int, max_iter: int = 5000) -> np.ndarray:
    """Fit a probe and return test probabilities for all classes (zero for classes unseen in training)."""
    model = make_probe(C, seed, max_iter).fit(X_train, y_train)
    proba = np.zeros((len(X_test), n_classes))
    proba[:, model.classes_] = model.predict_proba(X_test)
    return proba


def official_folds(fold: np.ndarray) -> list[Fold]:
    """Train on one official fold and validate on the other, both ways."""
    first, second = np.flatnonzero(fold == 0), np.flatnonzero(fold == 1)
    return [(first, second), (second, first)]


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[Fold]:
    """Stratified folds that keep each patient in one fold."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def select_c(X: np.ndarray, y: np.ndarray, folds: Sequence[Fold], grid: Sequence[float], seed: int, n_classes: int,
             jobs: int = 1, max_iter: int = 5000) -> tuple[float, dict[float, float], np.ndarray]:
    """Pick C by mean validation macro-F1 (ties go to the smaller C).

    Returns the chosen C, the mean score per C and the out-of-fold probabilities at the chosen C.
    """
    tasks = [(C, k) for C in grid for k in range(len(folds))]
    results = Parallel(n_jobs=jobs)(
        delayed(fit_predict)(X[folds[k][0]], y[folds[k][0]], X[folds[k][1]], C, seed, n_classes, max_iter)
        for C, k in tasks)
    scores: dict[float, list[float]] = {C: [] for C in grid}
    oof = {C: np.full((len(y), n_classes), np.nan) for C in grid}
    for (C, k), proba in zip(tasks, results):
        val = folds[k][1]
        oof[C][val] = proba
        scores[C].append(label_scores(confusion(y[val], proba.argmax(axis=1), n_classes))["macro_f1"])
    mean = {C: float(np.mean(values)) for C, values in scores.items()}
    best = min(grid, key=lambda C: (-round(mean[C], 10), C))
    return best, mean, oof[best]

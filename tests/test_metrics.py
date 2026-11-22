"""Weighted metrics and the bootstrap, checked against scikit-learn."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from sage_shift.bootstrap import group_weights, percentile_ci
from sage_shift.metrics import auroc, average_precision, evaluate


def sample(n: int = 80, k: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    proba = rng.dirichlet(np.ones(k), size=n)
    proba[np.arange(n), y] += rng.random(n)  # some signal
    proba = np.round(proba / proba.sum(axis=1, keepdims=True), 2)  # rounding creates ties
    truth = np.eye(k)[y]
    truth[rng.random((n, k)) < 0.1] = 1  # a few extra labels
    return y, proba, truth


class MetricTests(unittest.TestCase):
    def test_unit_weights_match_sklearn(self) -> None:
        y, proba, truth = sample()
        out = evaluate(y, proba, truth)
        pred = proba.argmax(axis=1)
        self.assertAlmostEqual(out["macro_f1"], f1_score(y, pred, average="macro", zero_division=0))
        self.assertAlmostEqual(out["weighted_f1"], f1_score(y, pred, average="weighted", zero_division=0))
        self.assertAlmostEqual(out["balanced_accuracy"], balanced_accuracy_score(y, pred))
        for c in range(4):
            self.assertAlmostEqual(out["auroc"][c], roc_auc_score(truth[:, c], proba[:, c]))
            self.assertAlmostEqual(out["ap"][c], average_precision_score(truth[:, c], proba[:, c]))

    def test_weights_equal_duplicated_samples(self) -> None:
        y, proba, truth = sample(seed=1)
        weights = np.random.default_rng(2).integers(0, 3, size=len(y)).astype(float)
        rows = np.repeat(np.arange(len(y)), weights.astype(int))
        weighted, duplicated = evaluate(y, proba, truth, weights), evaluate(y[rows], proba[rows], truth[rows])
        for key in ("macro_f1", "weighted_f1", "balanced_accuracy", "macro_auroc", "macro_ap"):
            self.assertAlmostEqual(weighted[key], duplicated[key])

    def test_absent_class_is_left_out_of_macro_scores(self) -> None:
        y, proba = np.array([0, 0, 1, 1]), np.array([[0.9, 0.1, 0], [0.8, 0.2, 0], [0.3, 0.7, 0], [0.6, 0.4, 0]])
        out = evaluate(y, proba, np.eye(3)[y])
        self.assertAlmostEqual(out["macro_f1"], f1_score(y, proba.argmax(1), labels=[0, 1], average="macro"))
        self.assertTrue(np.isnan(out["f1"][2]) and np.isnan(out["auroc"][2]))

    def test_degenerate_ranking_scores(self) -> None:
        self.assertTrue(np.isnan(auroc(np.ones(3), np.array([0.1, 0.2, 0.3]), np.ones(3))))
        self.assertTrue(np.isnan(average_precision(np.zeros(3), np.array([0.1, 0.2, 0.3]), np.ones(3))))


class BootstrapTests(unittest.TestCase):
    def test_group_weights(self) -> None:
        groups = np.array([5, 5, 7, 9, 9, 9])
        weights = group_weights(groups, 200, seed=1)
        self.assertEqual(weights.shape, (200, 6))
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])  # images of one patient move together
        np.testing.assert_array_equal(weights[:, 3], weights[:, 5])
        np.testing.assert_array_equal(weights[:, [0, 2, 3]].sum(axis=1), np.full(200, 3))  # three patients drawn
        np.testing.assert_array_equal(weights, group_weights(groups, 200, seed=1))

    def test_percentile_ci_ignores_nan(self) -> None:
        low, high = percentile_ci(np.r_[np.arange(101.0), np.nan])
        self.assertAlmostEqual(low, 2.5)
        self.assertAlmostEqual(high, 97.5)


if __name__ == "__main__":
    unittest.main()

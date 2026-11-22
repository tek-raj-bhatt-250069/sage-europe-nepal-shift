"""Probes, fold builders and the choice of C, on synthetic features."""

from __future__ import annotations

import unittest

import numpy as np

from sage_shift.probes import fit_predict, grouped_folds, official_folds, select_c


def blobs(n_per_class: int = 30, k: int = 3, dim: int = 8, spread: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3.0, size=(k, dim))
    y = np.repeat(np.arange(k), n_per_class)
    return centres[y] + rng.normal(scale=spread, size=(len(y), dim)), y


class ProbeTests(unittest.TestCase):
    def test_fit_predict_fills_unseen_classes_with_zero(self) -> None:
        x, y = blobs()
        keep = y != 1
        proba = fit_predict(x[keep], y[keep], x, C=1.0, seed=0, n_classes=3)
        self.assertEqual(proba.shape, (len(y), 3))
        np.testing.assert_array_equal(proba[:, 1], 0)
        np.testing.assert_allclose(proba.sum(axis=1), 1)

    def test_official_folds(self) -> None:
        folds = official_folds(np.array([0, 1, 0, 1, 1]))
        self.assertEqual([f[0].tolist() for f in folds], [[0, 2], [1, 3, 4]])
        self.assertEqual([f[1].tolist() for f in folds], [[1, 3, 4], [0, 2]])

    def test_grouped_folds_keep_patients_together(self) -> None:
        x, y = blobs(n_per_class=40)
        groups = np.arange(len(y)) // 4
        for train, val in grouped_folds(y, groups, 5, seed=0):
            self.assertFalse(set(groups[train]) & set(groups[val]))

    def test_select_c(self) -> None:
        x, y = blobs(spread=0.5)
        folds = official_folds(np.arange(len(y)) % 2)
        best, scores, oof = select_c(x, y, folds, [0.001, 1.0], seed=0, n_classes=3)
        self.assertEqual(set(scores), {0.001, 1.0})
        self.assertIn(best, scores)
        self.assertEqual(scores[best], max(scores.values()))
        self.assertFalse(np.isnan(oof).any())
        same, _, _ = select_c(x, y, folds, [1.0, 0.5], seed=0, n_classes=3)  # perfectly separable: tie
        self.assertEqual(same, 0.5)  # ties go to the smaller C


if __name__ == "__main__":
    unittest.main()

"""Prior correction, bootstrap scoring and the meaning-group statistic, on synthetic data."""

from __future__ import annotations

import unittest

import numpy as np

from sage_shift.rq2 import boot_scores, ci, meaning_gap_difference, saerens_em


def gaussian_posteriors(n: int, target: list[float], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Posteriors of a two-class problem computed under a uniform training prior."""
    rng = np.random.default_rng(seed)
    y = rng.choice(2, size=n, p=target)
    x = rng.normal(loc=np.where(y == 1, 1.5, -1.5), scale=1.0)
    likelihood = np.stack([np.exp(-0.5 * (x + 1.5) ** 2), np.exp(-0.5 * (x - 1.5) ** 2)], axis=1)
    return likelihood / likelihood.sum(axis=1, keepdims=True), y


class PriorTests(unittest.TestCase):
    def test_em_recovers_a_shifted_prior(self) -> None:
        proba, _ = gaussian_posteriors(20000, [0.8, 0.2])
        adjusted, prior, iterations = saerens_em(proba, [0.5, 0.5])
        np.testing.assert_allclose(prior, [0.8, 0.2], atol=0.02)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)
        self.assertGreater(adjusted[:, 0].mean(), proba[:, 0].mean())
        self.assertGreater(iterations, 1)

    def test_integer_weights_equal_duplicated_rows(self) -> None:
        proba, _ = gaussian_posteriors(300, [0.7, 0.3], seed=1)
        weights = np.random.default_rng(2).integers(0, 3, size=300).astype(float)
        _, weighted, _ = saerens_em(proba, [0.5, 0.5], weights)
        _, duplicated, _ = saerens_em(np.repeat(proba, weights.astype(int), axis=0), [0.5, 0.5])
        np.testing.assert_allclose(weighted, duplicated, atol=1e-6)


class ScoringTests(unittest.TestCase):
    def test_meaning_gap_difference(self) -> None:
        shifted = np.array([False, False, True, True])
        ceiling = np.array([0.8, 0.8, 0.8, 0.8])
        transfer = np.array([0.6, 0.7, 0.2, 0.4])
        self.assertAlmostEqual(float(meaning_gap_difference(ceiling, transfer, shifted)), 0.5 - 0.15)
        rows = np.stack([transfer, np.array([0.6, np.nan, 0.2, np.nan])])
        np.testing.assert_allclose(meaning_gap_difference(np.stack([ceiling, ceiling]), rows, shifted), [0.35, 0.4])

    def test_boot_scores_and_ci(self) -> None:
        rng = np.random.default_rng(0)
        y = np.arange(60) % 3
        proba = np.eye(3)[y] * 0.6 + rng.dirichlet(np.ones(3), size=60) * 0.4
        truth = np.eye(3)[y]
        weights = rng.integers(0, 3, size=(50, 60)).astype(float)
        point, boot = boot_scores(y, proba, truth, weights, polyp=1)
        self.assertAlmostEqual(point["macro_f1"], 1.0)
        self.assertEqual(boot["f1"].shape, (50, 3))
        self.assertEqual(boot["polyp_share"].shape, (50,))
        em_point, em_boot = boot_scores(y, proba, truth, weights, polyp=1, em_prior=[1 / 3] * 3)
        self.assertEqual(em_boot["macro_f1"].shape, (50,))
        entry = ci(0.5, np.linspace(0, 1, 101))
        self.assertEqual((entry["estimate"], entry["ci_low"], entry["ci_high"]), (0.5, 0.025, 0.975))


if __name__ == "__main__":
    unittest.main()

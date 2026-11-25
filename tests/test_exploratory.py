"""Exploratory analysis helpers, on synthetic data."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from sage_shift import exploratory as ex
from sage_shift.metrics import auroc, evaluate
from sage_shift.probes import grouped_folds
from sage_shift.rq2 import saerens_em


def sampled_classes(z: np.ndarray, seed: int) -> np.ndarray:
    """One class per row drawn from softmax(z)."""
    p = softmax(z, axis=1)
    u = np.random.default_rng(seed).random((len(z), 1))
    return np.minimum((p.cumsum(axis=1) < u).sum(axis=1), z.shape[1] - 1)


class InterceptTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.z = rng.normal(scale=1.5, size=(20000, 4))
        self.b_true = np.array([0.8, -0.4, 0.3, -0.7])
        self.y = sampled_classes(self.z + self.b_true, seed=1)

    def test_recovers_known_intercepts(self) -> None:
        b = ex.fit_intercepts(self.z, self.y, np.zeros(4), penalty=1e-8, sample_weight=np.ones(len(self.y)))
        np.testing.assert_allclose(b - b.mean(), self.b_true - self.b_true.mean(), atol=0.05)

    def test_balanced_weights_remove_the_class_prior(self) -> None:
        b = ex.fit_intercepts(self.z, self.y, np.zeros(4), penalty=1e-8)
        expected = self.b_true - np.log(np.bincount(self.y) / len(self.y))
        np.testing.assert_allclose(b - b.mean(), expected - expected.mean(), atol=0.08)

    def test_absent_classes_and_penalty(self) -> None:
        b0 = np.array([0.1, 0.2, 0.3, 0.4])
        keep = self.y != 2
        b = ex.fit_intercepts(self.z[keep], self.y[keep], b0)
        self.assertEqual(b[2], b0[2])  # absent class keeps its intercept
        self.assertGreater(np.abs(b - b0).max(), 0.1)
        held = ex.fit_intercepts(self.z[keep], self.y[keep], b0, penalty=1e6)
        np.testing.assert_allclose(held, b0, atol=1e-4)


class CalibrationTests(unittest.TestCase):
    def test_bcts_reduces_nll_on_miscalibrated_probabilities(self) -> None:
        rng = np.random.default_rng(2)
        z = rng.normal(scale=1.0, size=(8000, 3))
        y = sampled_classes(z, seed=3)
        miscalibrated = log_softmax(3.0 * z + np.array([0.0, 1.0, -1.0]), axis=1)  # overconfident and biased
        temperature, bias = ex.fit_bcts(miscalibrated, y)
        calibrated = np.log(ex.apply_bcts(miscalibrated, temperature, bias))
        self.assertLess(ex.nll(calibrated, y), ex.nll(miscalibrated, y) - 0.1)
        self.assertAlmostEqual(temperature, 3.0, delta=0.3)
        self.assertEqual(bias[0], 0.0)

    def test_oracle_prior_gives_bayes_posteriors(self) -> None:
        x = np.linspace(-3, 3, 13)
        likelihood = np.stack([np.exp(-0.5 * (x + 1) ** 2), np.exp(-0.5 * (x - 1) ** 2)], axis=1)
        uniform_posterior = likelihood / likelihood.sum(axis=1, keepdims=True)
        prior = np.array([0.8, 0.2])
        bayes = likelihood * prior / (likelihood * prior).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(ex.reweight_prior(uniform_posterior, prior, [0.5, 0.5]), bayes)
        stacked = ex.reweight_prior(uniform_posterior, np.stack([prior, [0.5, 0.5]]), [0.5, 0.5])
        self.assertEqual(stacked.shape, (2, 13, 2))
        np.testing.assert_allclose(stacked[1], uniform_posterior)

    def test_em_many_matches_saerens_em_per_resample(self) -> None:
        rng = np.random.default_rng(7)
        proba = rng.dirichlet(np.ones(3), size=40)
        weights = rng.integers(0, 3, size=(4, 40)).astype(float)
        preds, priors = ex.em_many(proba, np.full(3, 1 / 3), weights)
        self.assertEqual((preds.shape, priors.shape), ((4, 40), (4, 3)))
        for b, w in enumerate(weights):
            adjusted, prior, _ = saerens_em(proba, np.full(3, 1 / 3), w)
            np.testing.assert_array_equal(preds[b], adjusted.argmax(axis=1))
            np.testing.assert_allclose(priors[b], prior)

    def test_class_shares_follow_the_weights(self) -> None:
        y = np.array([0, 0, 1, 2])
        shares = ex.class_shares(y, 3, np.array([[1, 1, 1, 1], [0, 2, 1, 1]]))
        np.testing.assert_allclose(shares, [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
        np.testing.assert_allclose(ex.class_shares(y, 3, np.array([0, 0, 3, 1])), [[0.0, 0.75, 0.25]])


class FoldTests(unittest.TestCase):
    def test_source_block_always_trains_and_target_patients_stay_apart(self) -> None:
        rng = np.random.default_rng(4)
        patients = np.repeat(np.arange(30), 3)
        y = rng.integers(0, 3, size=len(patients))
        n_source = 50
        target_folds = grouped_folds(y, patients, 5, seed=0)
        folds = ex.with_source_block(n_source, target_folds)
        seen = []
        for (train, val), (target_train, target_val) in zip(folds, target_folds):
            self.assertTrue(set(range(n_source)) <= set(train.tolist()))
            self.assertTrue((val >= n_source).all())
            np.testing.assert_array_equal(val - n_source, target_val)
            train_patients = set(patients[train[train >= n_source] - n_source])
            self.assertFalse(train_patients & set(patients[val - n_source]))
            seen.extend((val - n_source).tolist())
        self.assertEqual(sorted(seen), list(range(len(patients))))

    def test_weighted_fold_score_and_choice_of_c(self) -> None:
        rng = np.random.default_rng(5)
        centres = rng.normal(scale=3.0, size=(3, 6))
        y = np.tile(np.arange(3), 40)
        x = centres[y] + rng.normal(size=(len(y), 6))
        source = (np.arange(len(y)) >= 60).astype(float)
        fold = (np.arange(0, 100), np.arange(100, 120))
        score = ex.weighted_fold_score(x, y, source, fold, 1.0, 0, 3, 1000)
        self.assertGreater(score, 0.8)
        self.assertEqual(ex.choose_c({0.1: 0.5, 1.0: 0.7, 10.0: 0.7}), 1.0)  # ties go to the smaller C

    def test_patient_draws(self) -> None:
        patients = np.arange(10, 20)
        draws = ex.patient_draws(patients, [0, 3, 10], 2, seed=0)
        self.assertEqual([(k, d) for k, d, _ in draws], [(0, 0), (3, 0), (3, 1), (10, 0)])
        self.assertEqual(len(draws[0][2]), 0)
        np.testing.assert_array_equal(draws[-1][2], patients)
        self.assertEqual(len(set(draws[1][2].tolist())), 3)
        np.testing.assert_array_equal(draws[1][2], ex.patient_draws(patients, [0, 3, 10], 2, seed=0)[1][2])


class GroupTests(unittest.TestCase):
    def test_group_contrasts(self) -> None:
        groups = {"strict": [0, 1], "boundary": [2, 3], "meaning": [4, 5]}
        gap = np.array([0.1, 0.3, 0.2, 0.2, 0.6, 0.4])
        out = ex.group_contrasts(gap, groups, [("meaning", "strict"), ("boundary", "strict"), ("meaning", "boundary")])
        self.assertAlmostEqual(float(out["meaning_minus_strict"]), 0.3)
        self.assertAlmostEqual(float(out["boundary_minus_strict"]), 0.0)
        self.assertAlmostEqual(float(out["meaning_minus_boundary"]), 0.3)
        rows = np.stack([gap, [np.nan, 0.3, 0.2, 0.2, np.nan, np.nan]])
        out = ex.group_contrasts(rows, groups, [("boundary", "strict"), ("meaning", "strict")])
        np.testing.assert_allclose(out["boundary_minus_strict"], [0.0, -0.1])  # nan classes left out
        self.assertTrue(np.isnan(out["meaning_minus_strict"][1]))  # a group with no class gives nan

    def test_smallest_k(self) -> None:
        self.assertEqual(ex.smallest_k([0, 5, 10, 20], [0.2, 0.5, np.nan, 0.9], 0.5), 5)
        self.assertIsNone(ex.smallest_k([0, 5], [0.2, 0.3], 0.5))


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(6)
        self.y = rng.integers(0, 4, size=90)
        proba = rng.dirichlet(np.ones(4), size=90)
        proba[np.arange(90), self.y] += rng.random(90)
        self.proba = np.round(proba / proba.sum(axis=1, keepdims=True), 2)  # ties
        self.truth = np.eye(4)[self.y]
        self.truth[rng.random((90, 4)) < 0.1] = 1
        self.weights = rng.integers(0, 3, size=(20, 90)).astype(float)

    def test_resample_scores_match_evaluate_and_sklearn(self) -> None:
        many = ex.resample_scores(self.y, self.proba, self.truth, self.weights, polyp=2)
        pred = self.proba.argmax(axis=1)
        for b, w in enumerate(self.weights):
            ref = evaluate(self.y, self.proba, self.truth, w)
            for key in ("macro_f1", "balanced_accuracy", "macro_auroc"):
                self.assertAlmostEqual(many[key][b], ref[key])
            np.testing.assert_allclose(many["f1"][b], ref["f1"])
            np.testing.assert_allclose(many["auroc"][b], ref["auroc"])
            self.assertAlmostEqual(many["mcc"][b], matthews_corrcoef(self.y, pred, sample_weight=w))
            self.assertAlmostEqual(many["polyp_share"][b], (w * (pred == 2)).sum() / w.sum())
        one = ex.resample_scores(self.y, self.proba, self.truth, np.ones(90), polyp=2)
        self.assertAlmostEqual(one["macro_f1"][0], f1_score(self.y, pred, average="macro"))
        self.assertAlmostEqual(one["balanced_accuracy"][0], balanced_accuracy_score(self.y, pred))

    def test_per_resample_predictions_and_best_constant(self) -> None:
        preds = np.stack([self.proba.argmax(axis=1), np.zeros(90, dtype=int)])
        cm = ex.confusion_many(self.y, preds, 4, self.weights[:2])
        np.testing.assert_allclose(cm[0], ex.confusion_many(self.y, preds[0], 4, self.weights[0])[0])
        constant = ex.confusion_scores(cm[1])["macro_f1"]
        self.assertLessEqual(constant, ex.best_constant_many(self.y, 4, self.weights[1])[0] + 1e-12)
        self.assertAlmostEqual(float(ex.auroc_many(self.truth[:, 1], self.proba[:, 1], self.weights[3])[0]),
                               auroc(self.truth[:, 1], self.proba[:, 1], self.weights[3]))
        self.assertTrue(np.isnan(ex.auroc_many(np.ones(5), np.arange(5.0), np.ones(5))[0]))


if __name__ == "__main__":
    unittest.main()

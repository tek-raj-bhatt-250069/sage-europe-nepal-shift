"""Sensitivity-analysis helpers on synthetic clustered data (no dataset file is read)."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import matthews_corrcoef

from sage_shift.baselines import constant_predictor_table
from sage_shift.bootstrap import group_weights, percentile_ci
from sage_shift.metrics import evaluate
from sage_shift.probes import balanced_weights, official_folds, select_c
from sage_shift.sensitivity import (PatientScores, bca_interval, best_constant_macro_f1, bootstrap_p,
                                    bootstrap_t_interval, class_f1, cv_choice, draw_counts, fit_tight, holm,
                                    inverse_size_weights, jackknife_se, kish_effective, macro_f1, mcc, mde,
                                    metric_replicates, min_swap_p, patient_bootstrap_rows, patient_weighted, power_at,
                                    resample_deletions, rq3_draws, rq3_fit, select_c_with, stratified_bootstrap_rows,
                                    swap_test, wilson_interval)


def sample(n_patients: int = 20, k: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Patients with 1-5 images each (non-contiguous codes), classes, rounded probabilities, multi-label truth."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(100, 100 + 3 * n_patients, 3), rng.integers(1, 6, size=n_patients))
    n = len(groups)
    y = rng.integers(0, k, size=n)
    proba = rng.dirichlet(np.ones(k), size=n)
    proba[np.arange(n), y] += rng.random(n)  # some signal
    proba = np.round(proba / proba.sum(axis=1, keepdims=True), 2)  # rounding creates ties
    truth = np.eye(k)[y]
    truth[rng.random((n, k)) < 0.1] = 1  # a few extra labels
    return groups, y, proba, truth


class WeightTests(unittest.TestCase):
    def test_inverse_size_weights(self) -> None:
        w = inverse_size_weights(np.array([7, 7, 7, 2, 9, 9]))
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3, 1, 0.5, 0.5])
        self.assertAlmostEqual(w.sum(), 3)  # every patient weighs 1

    def test_draw_counts_and_patient_weights(self) -> None:
        groups = np.array([7, 7, 7, 2, 9, 9])
        weights = group_weights(groups, 50, seed=1)
        counts = draw_counts(weights, groups)  # patients in sorted order: 2, 7, 9
        np.testing.assert_array_equal(counts.sum(axis=1), 3)
        np.testing.assert_array_equal(counts[:, [1, 0, 2]], weights[:, [0, 3, 4]])
        pw = patient_weighted(weights, groups)
        np.testing.assert_allclose(pw[:, :3].sum(axis=1), counts[:, 1])  # a patient's images sum to its draws
        np.testing.assert_allclose(pw.sum(axis=1), 3)
        with self.assertRaises(ValueError):
            draw_counts(weights + np.arange(6), groups)  # weights that differ within a patient


class PatientScoreTests(unittest.TestCase):
    def test_match_evaluate_for_both_estimands(self) -> None:
        groups, y, proba, truth = sample()
        weights = group_weights(groups, 30, seed=1)
        counts = draw_counts(weights, groups)
        inverse = inverse_size_weights(groups)
        patients = np.unique(groups)
        for image_weights, boot_weights in ((None, weights), (inverse, patient_weighted(weights, groups))):
            reps = metric_replicates(PatientScores(y, proba, truth, groups, image_weights), counts)
            full = np.ones(len(y)) if image_weights is None else image_weights
            for metric in ("macro_f1", "macro_auroc", "balanced_accuracy"):
                self.assertAlmostEqual(float(reps[metric].point), evaluate(y, proba, truth, full)[metric], places=12)
                for b in range(len(boot_weights)):
                    self.assertAlmostEqual(reps[metric].boot[b], evaluate(y, proba, truth, boot_weights[b])[metric],
                                           places=12)
                for p, code in enumerate(patients):  # leave-one-patient-out
                    kept = full * (groups != code)
                    self.assertAlmostEqual(reps[metric].jack[p], evaluate(y, proba, truth, kept)[metric], places=12)

    def test_resample_deletions_match_recomputation(self) -> None:
        groups, y, proba, truth = sample(seed=2)
        scores = PatientScores(y, proba, truth, groups)
        counts = draw_counts(group_weights(groups, 10, seed=3), groups)
        deleted = resample_deletions(scores.parts, counts)
        for b in range(10):
            for p in np.flatnonzero(counts[b] > 0):
                one_less = counts[b].copy()
                one_less[p] -= 1
                np.testing.assert_allclose(deleted[b, p], scores.confusion(one_less))

    def test_best_constant_matches_the_baseline_table(self) -> None:
        y = np.repeat(np.arange(4), [9, 4, 2, 5])
        cm = PatientScores(y, np.eye(4)[y], None, np.arange(len(y))).confusion(np.ones(len(y)))
        table = constant_predictor_table(y.tolist(), list(range(4)))
        self.assertAlmostEqual(float(best_constant_macro_f1(cm)), table["macro_f1"].max())


class MccTests(unittest.TestCase):
    def test_matches_sklearn(self) -> None:
        groups, y, proba, truth = sample(seed=4)
        scores = PatientScores(y, proba, None, groups)
        pred = proba.argmax(axis=1)
        self.assertAlmostEqual(float(mcc(scores.confusion(np.ones(scores.n_patients)))), matthews_corrcoef(y, pred))
        weights = group_weights(groups, 5, seed=5)
        counts = draw_counts(weights, groups)
        for b in range(5):
            self.assertAlmostEqual(float(mcc(scores.confusion(counts[b]))),
                                   matthews_corrcoef(y, pred, sample_weight=weights[b]))

    def test_constant_predictor_gives_zero(self) -> None:
        y = np.array([0, 1, 2, 2, 1])
        cm = PatientScores(y, np.tile([0.1, 0.8, 0.1], (5, 1)), None, np.arange(5)).confusion(np.ones(5))
        self.assertEqual(float(mcc(cm)), 0.0)
        self.assertEqual(matthews_corrcoef(y, np.ones(5, dtype=int)), 0.0)


class IntervalTests(unittest.TestCase):
    def test_bca_equals_percentile_without_bias_or_acceleration(self) -> None:
        boot = np.arange(-50.0, 51.0)  # symmetric around the estimate 0, so z0 = 0
        out = bca_interval(0.0, boot, np.array([-2.0, -1.0, 1.0, 2.0]))  # symmetric deletions, so a = 0
        self.assertAlmostEqual(out["z0"], 0.0)
        self.assertAlmostEqual(out["acceleration"], 0.0)
        low, high = percentile_ci(boot)
        self.assertAlmostEqual(out["low"], low)
        self.assertAlmostEqual(out["high"], high)

    def test_bca_moves_towards_the_estimate(self) -> None:
        boot = np.random.default_rng(0).normal(0.0, 1.0, 4000)
        out = bca_interval(0.5, boot, np.array([-1.0, 1.0]))  # most resamples lie below the estimate
        low, high = percentile_ci(boot)
        self.assertGreater(out["z0"], 0)
        self.assertGreater(out["low"], low)
        self.assertGreater(out["high"], high)

    def test_bootstrap_t_with_constant_se_is_the_basic_interval(self) -> None:
        boot = np.random.default_rng(1).gamma(2.0, 1.0, 2000)
        out = bootstrap_t_interval(2.0, 0.5, boot, np.full(2000, 0.5))
        q_low, q_high = np.quantile(boot, [0.025, 0.975])
        self.assertAlmostEqual(out["low"], 4.0 - q_high)
        self.assertAlmostEqual(out["high"], 4.0 - q_low)
        se = np.full(2000, 0.5)
        se[:60] = 0.0  # 3% of resamples with SE 0 and above the estimate: unbounded below
        out = bootstrap_t_interval(2.0, 0.5, np.r_[np.full(60, 9.0), boot[60:]], se)
        self.assertEqual(out["low"], -np.inf)
        self.assertEqual(out["zero_se"], 60)

    def test_jackknife_se_of_the_mean(self) -> None:
        x = np.random.default_rng(2).normal(size=25)
        loo = (x.sum() - x) / (len(x) - 1)
        self.assertAlmostEqual(float(jackknife_se(loo)), x.std(ddof=1) / np.sqrt(len(x)))

    def test_jackknife_se_multiplicity_and_nan(self) -> None:
        self.assertAlmostEqual(float(jackknife_se(np.array([1.0, 2.0, 4.0]), np.array([2.0, 1.0, 0.0]))),
                               float(jackknife_se(np.array([1.0, 1.0, 2.0]))))
        self.assertAlmostEqual(float(jackknife_se(np.array([1.0, np.nan, 2.0, 4.0]))),
                               float(jackknife_se(np.array([1.0, 2.0, 4.0]))))
        self.assertTrue(np.isnan(jackknife_se(np.array([3.0, np.nan]))))

    def test_bootstrap_p_holm_wilson_power(self) -> None:
        self.assertEqual(bootstrap_p(np.arange(1.0, 101.0)), 0.0)
        self.assertEqual(bootstrap_p(np.arange(-50.0, 51.0)), 1.0)
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.2775, places=4)
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low + high, 1.0)
        self.assertAlmostEqual(mde(1.0), 2.8016, places=4)
        self.assertAlmostEqual(power_at(mde(0.1), 0.1), 0.8, places=3)

    def test_kish(self) -> None:
        self.assertAlmostEqual(kish_effective([3, 3, 3, 3]), 4.0)
        self.assertAlmostEqual(kish_effective([8, 2, 2, 1, 1, 1]), 15 ** 2 / 75)
        self.assertLess(kish_effective([100, 1, 1]), 1.05)


class SwapTests(unittest.TestCase):
    @staticmethod
    def parts(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return PatientScores(y, np.eye(2)[pred], None, np.arange(len(y))).parts

    def test_identical_predictions_give_p_one(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        same = self.parts(y, np.array([0, 1, 1, 1, 0]))
        observed, p = swap_test(same, same, lambda r, t: macro_f1(r) - macro_f1(t), 500, seed=0)
        self.assertEqual(float(observed), 0.0)
        self.assertEqual(float(p), 1.0)

    def test_complete_difference_reaches_the_sign_flip_floor(self) -> None:
        y = np.array([0, 1, 0, 1])  # four patients, one image each
        reference, transfer = self.parts(y, y), self.parts(y, 1 - y)
        observed, p = swap_test(reference, transfer, lambda r, t: macro_f1(r) - macro_f1(t), 20000, seed=0)
        self.assertEqual(float(observed), 1.0)
        self.assertEqual(min_swap_p(4), 0.125)  # only no swap and all swaps reach |T| = 1: 2 of 16
        self.assertGreaterEqual(float(p), 1 / 20001)
        self.assertAlmostEqual(float(p), 0.125, delta=0.01)
        _, per_class = swap_test(reference, transfer, lambda r, t: class_f1(r) - class_f1(t), 2000, seed=0)
        self.assertEqual(per_class.shape, (2,))
        self.assertTrue(np.all((per_class > 0) & (per_class <= 1)))


class TrainingResampleTests(unittest.TestCase):
    def test_resampled_rows(self) -> None:
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], [5, 3, 7])
        rows = stratified_bootstrap_rows(y, rng)
        np.testing.assert_array_equal(np.bincount(y[rows]), [5, 3, 7])
        groups = np.repeat([4, 8, 15, 16], [2, 1, 3, 2])
        rows = patient_bootstrap_rows(groups, rng)
        for code in np.unique(groups[rows]):  # a drawn patient brings all its rows, the same number of times each
            members = np.flatnonzero(groups == code)
            self.assertEqual(len(set(np.bincount(rows, minlength=len(groups))[members])), 1)


def blobs(n_per_class: int = 30, k: int = 3, dim: int = 6, spread: float = 1.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n_per_class)
    return rng.normal(scale=3.0, size=(k, dim))[y] + rng.normal(scale=spread, size=(len(y), dim)), y


class RefitTests(unittest.TestCase):
    def test_cv_choice_ties_go_to_the_smaller_c(self) -> None:
        y = np.array([0, 1, 0, 1])
        folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
        perfect = np.eye(2)[y]
        probas = {(C, k): perfect[folds[k][1]] for C in (1.0, 0.5) for k in range(2)}
        best, mean = cv_choice(y, folds, [1.0, 0.5], probas, 2)
        self.assertEqual(best, 0.5)
        self.assertEqual(mean, {1.0: 1.0, 0.5: 1.0})

    def test_select_c_with_default_tolerance_matches_probes(self) -> None:
        x, y = blobs()
        folds = official_folds(np.arange(len(y)) % 2)
        best, mean = select_c_with(x, y, folds, [0.01, 1.0], 0, 3)
        expected_best, expected_mean, _ = select_c(x, y, folds, [0.01, 1.0], 0, 3)
        self.assertEqual(best, expected_best)
        self.assertEqual(mean, expected_mean)

    def test_fit_tight_counts_warnings_and_retries(self) -> None:
        x, y = blobs()
        proba, first, left = fit_tight(x, y, x, 10.0, 0, 3, max_iter=1, retry_iter=5000)
        self.assertGreater(first, 0)  # one iteration cannot converge
        self.assertEqual(left, 0)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        self.assertEqual(fit_tight(x, y, x, 0.1, 0, 3)[1:], (0, 0))

    def test_tight_weighted_fit_matches_tight_balanced_fit(self) -> None:
        x, y = blobs()
        keep = np.r_[np.arange(0, 30), np.arange(30, 40), np.arange(60, 90)]  # unequal class sizes
        x, y = x[keep], y[keep]
        balanced = fit_tight(x, y, x, 0.5, 0, 3)[0]
        weighted = fit_tight(x, y, x, 0.5, 0, 3, sample_weight=balanced_weights(y, np.zeros(len(y))))[0]
        np.testing.assert_allclose(weighted, balanced, atol=1e-5)

    def test_rq3_draws(self) -> None:
        patients = np.arange(10, 30)
        draws = rq3_draws(patients, [0, 5, 20, 145], 3, seed=0)
        self.assertEqual([(k, d) for k, d, _ in draws], [(0, 0), (5, 0), (5, 1), (5, 2), (20, 0), (20, 0)])
        self.assertEqual(len(draws[0][2]), 0)
        for k, _, members in draws[1:4]:
            self.assertEqual(len(np.unique(members)), k)
            np.testing.assert_array_equal(members, np.sort(members))
        np.testing.assert_array_equal(draws[-1][2], patients)
        np.testing.assert_array_equal(rq3_draws(patients, [5], 1, seed=0)[0][2], draws[1][2])  # same stream

    def test_rq3_fit(self) -> None:
        x, y = blobs()
        x_hk, y_hk, x_sage, y_sage = x[::2], y[::2], x[1::2], y[1::2]
        alone = rq3_fit(x_hk, y_hk, x_sage, y_sage, np.flatnonzero(y_sage == 2)[:3], x, "sage_only", 1.0, 0, 3)
        np.testing.assert_array_equal(alone[0], np.tile([0.0, 0.0, 1.0], (len(x), 1)))  # one class: predict it
        hk_only = rq3_fit(x_hk, y_hk, x_sage, y_sage, np.array([], dtype=int), x, "hk_plus_k", 0.5, 0, 3)[0]
        np.testing.assert_allclose(hk_only, fit_tight(x_hk, y_hk, x, 0.5, 0, 3)[0], atol=1e-5)
        mixed = rq3_fit(x_hk, y_hk, x_sage, y_sage, np.arange(10), x, "hk_plus_k", 0.5, 0, 3)[0]
        self.assertEqual(mixed.shape, (len(x), 3))
        np.testing.assert_allclose(mixed.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()

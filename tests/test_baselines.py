"""Tests for the constant-predictor baselines (synthetic labels with SAGE's six-class test counts)."""

import unittest

from sage_shift.baselines import closed_form_constant, constant_predictor_table, matching_constants
from sage_shift.labelspace import SIX_CLASSES

# Class counts of the SAGE six-class test split.
COUNTS = {"pylorus-normal": 57, "polyp": 43, "z-line-normal": 17, "fundus-retroflexion": 16,
          "esophagitis": 15, "rectum-retroflexion": 10}
Y_TRUE = [cls for cls, n in COUNTS.items() for _ in range(n)]


class BaselinesTest(unittest.TestCase):
    def setUp(self):
        self.table = constant_predictor_table(Y_TRUE, SIX_CLASSES).set_index("predict_always")

    def test_known_values(self):
        self.assertAlmostEqual(self.table.loc["polyp", "weighted_f1"], 0.116444, places=5)
        self.assertAlmostEqual(self.table.loc["pylorus-normal", "weighted_f1"], 0.191286, places=5)
        self.assertAlmostEqual(self.table.loc["pylorus-normal", "accuracy"], 57 / 158, places=12)

    def test_closed_form_agrees_with_scikit_learn(self):
        for cls, row in self.table.iterrows():
            expected = closed_form_constant(row["support"] / len(Y_TRUE), len(SIX_CLASSES))
            for metric, value in expected.items():
                self.assertAlmostEqual(row[metric], value, places=12, msg=f"{cls} {metric}")

    def test_matching(self):
        table = self.table.reset_index()
        self.assertEqual(matching_constants(table, 0.1164), ["polyp"])
        self.assertEqual(matching_constants(table, 0.1691), [])


if __name__ == "__main__":
    unittest.main()

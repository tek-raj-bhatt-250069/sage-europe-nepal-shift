"""CIELAB conversion and Reinhard normalisation, on synthetic colours only."""

from __future__ import annotations

import unittest

import numpy as np

from sage_shift.colour import (ColourReference, lab_to_srgb, reference_from_stats, reinhard, srgb_to_lab,
                               transfer)


class LabTests(unittest.TestCase):
    def test_known_values(self) -> None:
        np.testing.assert_allclose(srgb_to_lab(np.array([1.0, 1.0, 1.0])), [100, 0, 0], atol=1e-3)
        np.testing.assert_allclose(srgb_to_lab(np.array([0.0, 0.0, 0.0])), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(srgb_to_lab(np.array([1.0, 0.0, 0.0])), [53.2408, 80.0925, 67.2032], atol=0.01)

    def test_round_trip(self) -> None:
        rgb = np.random.default_rng(0).random((500, 3))
        np.testing.assert_allclose(lab_to_srgb(srgb_to_lab(rgb)), rgb, atol=1e-6)


class ReinhardTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.rgb = np.clip(np.stack([rng.normal(0.6, 0.08, (64, 64)), rng.normal(0.35, 0.06, (64, 64)),
                                     rng.normal(0.3, 0.05, (64, 64))], axis=-1), 0, 1).astype(np.float32)
        self.mask = np.zeros((64, 64), dtype=bool)
        self.mask[8:56, 8:56] = True
        lab = srgb_to_lab(self.rgb[self.mask])
        mean, std = lab.mean(axis=0), lab.std(axis=0)
        self.reference = ColourReference(tuple(mean + [2.0, -3.0, 1.5]), tuple(std * [1.1, 0.9, 1.05]), 1, "test")

    def test_transfer_hits_the_reference_exactly(self) -> None:
        pixels, scale = transfer(srgb_to_lab(self.rgb[self.mask]), self.reference)
        np.testing.assert_allclose(pixels.mean(axis=0), self.reference.mean, atol=1e-9)
        np.testing.assert_allclose(pixels.std(axis=0), self.reference.std, atol=1e-9)
        self.assertEqual(scale.shape, (3,))

    def test_reinhard_changes_only_the_field(self) -> None:
        out = reinhard(self.rgb, self.mask, self.reference)
        np.testing.assert_array_equal(out[~self.mask], self.rgb[~self.mask])
        lab = srgb_to_lab(out[self.mask])
        np.testing.assert_allclose(lab.mean(axis=0), self.reference.mean, atol=0.05)
        self.assertEqual(out.dtype, np.float32)

    def test_empty_field_is_left_unchanged(self) -> None:
        out = reinhard(self.rgb, np.zeros((64, 64), dtype=bool), self.reference)
        np.testing.assert_array_equal(out, self.rgb)

    def test_reference_from_stats_and_round_trip(self) -> None:
        reference = reference_from_stats(np.array([[50, 10, 5], [60, 20, 15]]), np.array([[10, 4, 2], [20, 6, 4]]), "x")
        self.assertEqual(reference.mean, (55.0, 15.0, 10.0))
        self.assertEqual(reference.std, (15.0, 5.0, 3.0))
        self.assertEqual(ColourReference.from_dict(reference.to_dict()), reference)
        with self.assertRaises(ValueError):
            ColourReference.from_dict({"mean": [1, 2, 3], "std": [1, 0, 1], "images": 1, "source": "x"})


if __name__ == "__main__":
    unittest.main()

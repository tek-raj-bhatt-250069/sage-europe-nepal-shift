"""Image variants, the preprocessing reference file, worker helpers and contact sheets (synthetic only)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sage_shift.colour import ColourReference
from sage_shift.fieldcheck import detect_task, green_mask, has_problem, unpack_detection
from sage_shift.fov import FovLimits, FovParams, build_layouts, detect_field
from sage_shift.preprocess import (PreprocessReference, check_variants, frame_field, make_variants, resize_mask,
                                   resize_rgb)
from sage_shift.sheets import compose, draw_field
from synthetic import frame, octagon

REFERENCE = ColourReference((50.0, 20.0, 18.0), (17.0, 8.0, 7.5), 10, "test")


class VariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = frame(octagon(64, 22, 702, 556, 60))
        self.rgb[40:52, 4:50] = 255  # text in the border
        fields = [detect_field(frame(octagon(64, 22, 702, 556, 60), seed=i)) for i in range(5)]
        layouts, _ = build_layouts(fields, FovLimits(), prefix="t-")
        self.reference = PreprocessReference(FovParams(), FovLimits(), 224, {"sage": tuple(layouts)}, REFERENCE)
        self.choice = frame_field(self.rgb, self.reference, "sage")

    def test_all_variants(self) -> None:
        out = make_variants(self.rgb, self.choice, 224, REFERENCE, ["fov_colour", "raw", "colour", "fov"])
        self.assertEqual(list(out), ["raw", "fov", "colour", "fov_colour"])
        for image in out.values():
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(0.0 <= image.min() and image.max() <= 1.0)
        np.testing.assert_array_equal(out["raw"], resize_rgb(self.rgb, 224).astype(np.float32) / 255)
        self.assertLess(out["fov"][:3, :3].max(), 0.05)  # cut-off corner of the octagon is black
        self.assertGreater(out["raw"][:, :10].max(), 0.9)  # the border text is still in the raw frame

    def test_missing_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            make_variants(self.rgb, None, 224, REFERENCE, ["fov"])
        with self.assertRaises(ValueError):
            make_variants(self.rgb, self.choice, 224, None, ["colour"])
        with self.assertRaises(ValueError):
            check_variants(["sepia"])
        self.assertEqual(list(make_variants(self.rgb, None, 224, None, ["raw"])), ["raw"])

    def test_reference_round_trip(self) -> None:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        self.reference.save(folder / "ref.json")
        loaded = PreprocessReference.load(folder / "ref.json")
        self.assertEqual(loaded.params, self.reference.params)
        self.assertEqual(loaded.limits, self.reference.limits)
        self.assertEqual(loaded.colour, REFERENCE)
        self.assertEqual(loaded.layouts["sage"], self.reference.layouts["sage"])

    def test_resize_mask(self) -> None:
        mask = np.zeros((576, 768), dtype=bool)
        mask[:, :384] = True
        small = resize_mask(mask, 224)
        self.assertEqual(small.shape, (224, 224))
        self.assertTrue(small[:, :100].all() and not small[:, 124:].any())


class HelperTests(unittest.TestCase):
    def test_detect_task_round_trip(self) -> None:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        rgb = frame(octagon(64, 22, 702, 556, 60))
        Image.fromarray(rgb).save(folder / "f.png")
        item = detect_task((str(folder / "f.png"), FovParams()))
        field = unpack_detection(item)
        np.testing.assert_array_equal(field.mask, detect_field(rgb).mask)
        self.assertEqual(field.bbox, detect_field(rgb).bbox)

    def test_green_and_flags(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[0, 0] = (20, 200, 30)
        rgb[1, 1] = (200, 200, 200)
        self.assertEqual(green_mask(rgb).sum(), 1)
        self.assertTrue(has_problem("not_convex|fallback"))
        self.assertFalse(has_problem(""))

    def test_sheet_layout(self) -> None:
        panel = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        marked = draw_field(panel, mask, (2, 2, 18, 18))
        self.assertTrue((marked[5, 5] == (0, 255, 255)).all())
        sheet = compose([("1", [marked, panel])] * 3, per_row=2, gap=4, label_height=10)
        self.assertEqual(sheet.size, (2 * (2 * 20 + 2) + 3 * 4, 2 * (10 + 20) + 3 * 4))


if __name__ == "__main__":
    unittest.main()

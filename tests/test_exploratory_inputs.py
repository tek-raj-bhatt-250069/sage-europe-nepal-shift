"""Exploratory model inputs on synthetic frames: unchanged variants, grey fill, letterbox, sageify, colour_global."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sage_shift.colour import ColourReference, masked_lab_stats, reinhard, srgb_to_lab
from sage_shift.fov import FovLimits, FovParams, Layout, apply_field, build_layouts, detect_field
from sage_shift.preprocess import (EXPLORATORY, GREY, GREY_RGB, INPUTS, VARIANTS, Placement, PreprocessReference,
                                   check_dataset, check_variants, frame_field, geometry_images, global_transfer,
                                   make_variants, pad_square, place, resize_mask, resize_rgb, sage_placement)
from synthetic import frame, octagon

try:
    import torch

    from sage_shift.backbones import IMAGENET_MEAN, IMAGENET_STD
    from sage_shift.features import FrameDataset
    from sage_shift.recipe import cache_field_frames
except ImportError:  # torch is installed on Colab; locally it is optional
    torch = None

REFERENCE = ColourReference((50.0, 20.0, 18.0), (17.0, 8.0, 7.5), 10, "test")
SAGE_STATS = ColourReference((47.7, 21.5, 19.9), (16.5, 8.0, 7.8), 857, "test")
PLACEMENT = Placement((576, 768), (64, 22, 702, 556))
STD = (0.229, 0.224, 0.225)


def hk_like(height: int = 530, width: int = 622) -> np.ndarray:
    """A HyperKvasir-like frame: tissue texture filling the frame, black cut corners."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:height, 0:width]
    cut = width // 8
    corners = (x + y < cut) | ((width - 1 - x) + y < cut) | (x + (height - 1 - y) < cut) | \
              ((width - 1 - x) + (height - 1 - y) < cut)
    out = rng.integers(0, 10, size=(height, width, 3)).astype(np.uint8)
    out[~corners] = (190, 95, 80)
    return out


def sage_reference() -> PreprocessReference:
    fields = [detect_field(frame(octagon(64, 22, 702, 556, 60), seed=i)) for i in range(5)]
    layouts, _ = build_layouts(fields, FovLimits(), prefix="t-")
    return PreprocessReference(FovParams(), FovLimits(), 224, {"sage": tuple(layouts)}, REFERENCE,
                               {"sage": True, "hk": False})


class ExploratoryInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = sage_reference()

    def setUp(self) -> None:
        self.rgb = frame(octagon(64, 22, 702, 556, 60))
        self.rgb[40:52, 4:50] = 255  # text in the border
        self.choice = frame_field(self.rgb, self.reference, "sage")

    def test_pre_registered_variants_are_unchanged(self) -> None:
        raw = resize_rgb(self.rgb, 224).astype(np.float32) / 255
        crop, crop_mask = apply_field(self.rgb, self.choice.field)
        fov = resize_rgb(crop, 224).astype(np.float32) / 255
        expected = {"raw": raw, "fov": fov,
                    "colour": reinhard(raw, resize_mask(self.choice.field.full_mask(), 224), REFERENCE),
                    "fov_colour": reinhard(fov, resize_mask(crop_mask, 224), REFERENCE)}
        self.assertEqual(VARIANTS, ("raw", "fov", "colour", "fov_colour"))
        alone = make_variants(self.rgb, self.choice, 224, REFERENCE, list(VARIANTS))
        together = make_variants(self.rgb, self.choice, 224, REFERENCE, list(INPUTS), SAGE_STATS, PLACEMENT)
        for name, image in expected.items():
            self.assertEqual(alone[name].dtype, np.float32)
            self.assertEqual(alone[name].tobytes(), image.tobytes())
            self.assertEqual(together[name].tobytes(), image.tobytes())
        self.assertEqual(check_variants(["fov_colour", "colour", "fov", "raw"]), VARIANTS)

    def test_shapes_and_range(self) -> None:
        out = make_variants(self.rgb, self.choice, 224, REFERENCE, list(EXPLORATORY), SAGE_STATS, PLACEMENT)
        self.assertEqual(tuple(out), EXPLORATORY)
        for image in out.values():
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(0.0 <= image.min() and image.max() <= 1.0)

    def test_grey_fill(self) -> None:
        self.assertEqual(GREY.tolist(), [124, 116, 104])
        normalised = (GREY / 255 - np.array(GREY_RGB)) / np.array(STD)
        self.assertLess(float(np.abs(normalised).max()), 0.01)
        names = ["masked", "masked_grey", "fov", "fov_grey", "letterbox", "letterbox_grey"]
        images = geometry_images(self.rgb, self.choice.field, 224, names)
        for name, image in images.items():  # the top-left corner is outside the field in all of them
            np.testing.assert_array_equal(image[0, 0], GREY if name.endswith("_grey") else [0, 0, 0])
        self.assertGreater(int(geometry_images(self.rgb, None, 224, ["raw"])["raw"][:, :10].max()), 200)
        self.assertLess(int(images["masked"][:, :10].max()), 20)  # border text removed
        np.testing.assert_array_equal(images["masked"][100:120, 100:120], images["masked_grey"][100:120, 100:120])

    def test_letterbox_is_square_and_centred(self) -> None:
        small = np.full((2, 4, 3), 200, dtype=np.uint8)
        padded = pad_square(small, GREY)
        self.assertEqual(padded.shape, (4, 4, 3))
        np.testing.assert_array_equal(padded[1:3], small)
        np.testing.assert_array_equal(padded[[0, 3]], np.broadcast_to(GREY, (2, 4, 3)))
        image = geometry_images(self.rgb, self.choice.field, 224, ["letterbox"])["letterbox"]
        left, top, right, bottom = self.choice.field.bbox
        height = (bottom - top) / (right - left) * 224  # the box is wider than tall: full width, height kept
        rows = np.flatnonzero(image.max(axis=(1, 2)) > 40)
        cols = np.flatnonzero(image.max(axis=(0, 2)) > 40)
        self.assertAlmostEqual(rows.max() - rows.min() + 1, height, delta=2)
        self.assertAlmostEqual(rows.min(), (224 - height) / 2, delta=2)
        self.assertTrue(cols.min() <= 2 and cols.max() >= 221)

    def test_sageify_placement(self) -> None:
        canvas = place(np.full((10, 20, 3), 200, dtype=np.uint8), PLACEMENT, GREY)
        self.assertEqual(canvas.shape, (576, 768, 3))
        inside = np.zeros((576, 768), dtype=bool)
        inside[22:556, 64:702] = True
        self.assertTrue((canvas[inside] == 200).all())
        self.assertTrue((canvas[~inside] == GREY).all())
        rgb = hk_like()
        field = detect_field(rgb)
        images = geometry_images(rgb, field, 224, ["sageify", "sageify_grey"], PLACEMENT)
        for name, image in images.items():
            fill = GREY if name == "sageify_grey" else [0, 0, 0]
            for row, col in ((0, 0), (112, 5), (112, 220), (2, 112), (221, 112)):  # margins around the SAGE box
                np.testing.assert_array_equal(image[row, col], fill)
            np.testing.assert_array_equal(image[112, 112], (190, 95, 80))  # tissue in the box
        with self.assertRaises(ValueError):
            geometry_images(rgb, field, 224, ["sageify"])  # no placement

    def test_sage_placement_takes_the_layout_with_most_frames(self) -> None:
        layouts = (Layout("a", (576, 768), 2, (36, 14, 732, 564), (), 10),
                   Layout("b", (576, 768), 2, (64, 22, 702, 556), (), 20))
        reference = PreprocessReference(FovParams(), FovLimits(), 224, {"sage": layouts})
        self.assertEqual(sage_placement(reference), PLACEMENT)
        with self.assertRaises(ValueError):
            sage_placement(PreprocessReference(FovParams(), FovLimits(), 224, {}))

    def test_dataset_rules_and_names(self) -> None:
        check_dataset("sage", ["raw", "masked", "letterbox_grey", "colour_global", "fov_colour_global"])
        check_dataset("hk", ["raw", "fov", "sageify", "sageify_grey"])
        with self.assertRaises(ValueError):
            check_dataset("hk", ["raw", "colour_global"])
        with self.assertRaises(ValueError):
            check_dataset("sage", ["sageify_grey"])
        with self.assertRaises(ValueError):
            check_variants(["sepia"])
        self.assertEqual(check_variants(["sageify", "raw", "masked"]), ("raw", "masked", "sageify"))

    def test_missing_settings_raise(self) -> None:
        with self.assertRaises(ValueError):
            make_variants(self.rgb, None, 224, REFERENCE, ["masked"])
        with self.assertRaises(ValueError):
            make_variants(self.rgb, self.choice, 224, REFERENCE, ["colour_global"])
        with self.assertRaises(ValueError):
            make_variants(self.rgb, self.choice, 224, None, ["fov_colour_global"], SAGE_STATS)
        with self.assertRaises(ValueError):
            make_variants(self.rgb, self.choice, 224, REFERENCE, ["sageify"])

    def test_global_transfer(self) -> None:
        raw = resize_rgb(self.rgb, 224).astype(np.float32) / 255
        mask = resize_mask(self.choice.field.full_mask(), 224)
        mean, std = masked_lab_stats(srgb_to_lab(raw), mask)
        own = ColourReference(tuple(mean.tolist()), tuple(std.tolist()), 1, "this frame")
        # with the frame's own statistics, the fixed transfer is per-image Reinhard
        np.testing.assert_allclose(global_transfer(raw, mask, own, REFERENCE), reinhard(raw, mask, REFERENCE), atol=1e-6)
        out = global_transfer(raw, mask, SAGE_STATS, REFERENCE)
        np.testing.assert_array_equal(out[~mask], raw[~mask])  # outside the field nothing changes
        made = make_variants(self.rgb, self.choice, 224, REFERENCE, ["colour_global"], SAGE_STATS)["colour_global"]
        np.testing.assert_array_equal(made, out)
        np.testing.assert_allclose(global_transfer(raw, mask, REFERENCE, REFERENCE), raw, atol=1e-4)


@unittest.skipIf(torch is None, "torch is not installed")
class ModelInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = sage_reference()

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.rgb = frame(octagon(64, 22, 702, 556, 60), seed=7)
        self.path = str(self.folder / "frame.png")
        Image.fromarray(self.rgb).save(self.path)

    def test_grey_is_the_imagenet_mean(self) -> None:
        self.assertEqual(GREY_RGB, IMAGENET_MEAN)
        self.assertEqual(STD, IMAGENET_STD)

    def test_masked_and_fov_equal_the_recipe_cache(self) -> None:
        field = frame_field(self.rgb, self.reference, "sage").field
        images = geometry_images(self.rgb, field, 224, ["masked", "fov"])
        np.testing.assert_array_equal(images["masked"], cache_field_frames([self.path], "sage", self.reference,
                                                                           workers=1, crop=False)[0])
        np.testing.assert_array_equal(images["fov"], cache_field_frames([self.path], "sage", self.reference,
                                                                        workers=1, crop=True)[0])

    def test_frame_dataset(self) -> None:
        with self.assertRaises(ValueError):
            FrameDataset([self.path], "hk", ["colour_global"], self.reference, 64, SAGE_STATS)
        with self.assertRaises(ValueError):
            FrameDataset([self.path], "sage", ["sageify"], self.reference, 64)
        stack, index = FrameDataset([self.path], "sage", ["masked_grey", "raw", "colour_global"], self.reference, 64,
                                    SAGE_STATS)[0]
        self.assertEqual((tuple(stack.shape), index), ((3, 3, 64, 64), 0))
        self.assertLess(float(stack[1, :, 0, 0].abs().max()), 0.01)  # masked_grey corner: about 0 after normalisation


if __name__ == "__main__":
    unittest.main()

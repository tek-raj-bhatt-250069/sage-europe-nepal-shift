"""Field-of-view detection, layouts and field choice, on synthetic frames only."""

from __future__ import annotations

import unittest

import numpy as np

from sage_shift.fov import (FovLimits, FovParams, Layout, apply_field, build_layouts, choose_field, cluster_boxes,
                            detect_field, green_mask, reduce_frame, remove_windows, solidity, window_mask)
from synthetic import HEIGHT, WIDTH, frame, octagon

GREEN = (20, 170, 130)  # ScopeGuide-like teal

class DetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = octagon(64, 22, 702, 556, 60)

    def test_finds_the_field_despite_text_boxes_and_lumen(self) -> None:
        rgb = frame(self.truth)
        rgb[40:52, 4:50] = 255            # a text line in the border
        rgb[60:63, 4:70] = 255            # a thin stroke that touches the field
        rgb[300:340, 0:40] = 0            # a black redaction box
        rgb[250:330, 300:400] = 3         # dark lumen inside the field
        field = detect_field(rgb, FovParams())
        mask = field.full_mask()
        iou = (mask & self.truth).sum() / (mask | self.truth).sum()
        self.assertGreater(iou, 0.98)
        for found, expected in zip(field.bbox, (64, 22, 702, 556)):
            self.assertLessEqual(abs(found - expected), 4)
        self.assertGreater(field.solidity, 0.97)

    def test_notch_lowers_solidity(self) -> None:
        notched = self.truth.copy()
        notched[150:450, 64:300] = False
        rgb = frame(notched)
        self.assertLess(detect_field(rgb).solidity, 0.9)
        self.assertGreater(solidity(self.truth), 0.98)

    def test_black_frame_gives_an_empty_field(self) -> None:
        field = detect_field(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        self.assertEqual(field.area_fraction, 0.0)
        self.assertEqual(field.bbox, (0, 0, 0, 0))

    def test_rejects_non_rgb_input(self) -> None:
        with self.assertRaises(ValueError):
            detect_field(np.zeros((10, 10), dtype=np.uint8))

    def test_apply_field_blacks_out_and_crops(self) -> None:
        rgb = frame(self.truth)
        rgb[40:52, 4:50] = 255
        field = detect_field(rgb)
        crop, crop_mask = apply_field(rgb, field)
        left, top, right, bottom = field.bbox
        self.assertEqual(crop.shape[:2], (bottom - top, right - left))
        self.assertTrue((crop[~crop_mask] == 0).all())


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = FovLimits()
        boxes_a = [(64 + d, 22, 702 + d, 556) for d in (0, 2, -2, 0, 2, 0)]
        boxes_b = [(36, 14 + d, 732, 564 + d) for d in (0, 2, 0, -2, 0)]
        self.fields = [detect_field(frame(octagon(*box, 60), seed=i)) for i, box in enumerate(boxes_a + boxes_b)]
        self.layouts, self.assignment = build_layouts(self.fields, self.limits, prefix="t-")

    def test_two_layouts_found(self) -> None:
        self.assertEqual(len(self.layouts), 2)
        self.assertEqual([layout.frames for layout in self.layouts], [6, 5])
        self.assertTrue((self.assignment >= 0).all())
        restored = Layout.from_dict(self.layouts[0].to_dict())
        self.assertEqual(restored, self.layouts[0])
        self.assertTrue((restored.mask == self.layouts[0].mask).all())

    def test_cluster_boxes_leaves_rare_boxes_out(self) -> None:
        boxes = np.array([(10, 10, 100, 100)] * 6 + [(300, 300, 400, 400)] * 2)
        labels = cluster_boxes(boxes, (HEIGHT, WIDTH), tolerance=0.02, min_frames=5)
        self.assertEqual(labels.tolist(), [0] * 6 + [-1] * 2)

    def test_clean_frame_keeps_its_own_detection(self) -> None:
        detected = detect_field(frame(octagon(66, 22, 704, 556, 60), seed=50))
        choice = choose_field(detected, self.layouts, self.limits)
        self.assertEqual(choice.flags, ())
        self.assertEqual(choice.layout, "t-768x576-1")
        self.assertEqual(choice.field.bbox, detected.bbox)

    def test_notched_frame_falls_back_to_its_layout(self) -> None:
        notched = octagon(36, 14, 732, 564, 60)
        notched[100:480, 36:330] = False  # dark tissue covering the left side
        choice = choose_field(detect_field(frame(notched, seed=51)), self.layouts, self.limits)
        self.assertIn("not_convex", choice.flags)
        self.assertIn("fallback", choice.flags)
        self.assertEqual(choice.layout, "t-768x576-2")
        self.assertEqual(choice.field.bbox, self.layouts[1].bbox)

    def test_black_frame_takes_the_most_frequent_layout(self) -> None:
        detected = detect_field(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        choice = choose_field(detected, self.layouts, self.limits)
        self.assertEqual(choice.layout, "t-768x576-1")
        self.assertIn("small", choice.flags)
        self.assertIn("fallback", choice.flags)

    def test_frame_size_without_layouts_keeps_its_detection(self) -> None:
        small_frame = frame(octagon(64, 22, 702, 556, 60))[:480, :720].copy()
        detected = detect_field(small_frame)
        choice = choose_field(detected, self.layouts, self.limits)
        self.assertEqual(choice.layout, "")
        self.assertEqual(choice.flags, ())

    def test_fallback_can_be_switched_off(self) -> None:
        notched = octagon(36, 14, 732, 564, 60)
        notched[250:564, 36:380] = False  # black picture-in-picture window in the lower-left corner
        detected = detect_field(frame(notched, seed=52))
        choice = choose_field(detected, self.layouts, self.limits, fallback=False)
        self.assertIn("not_convex", choice.flags)
        self.assertIn("no_fallback", choice.flags)
        self.assertNotIn("fallback", choice.flags)
        self.assertIs(choice.field, detected)


class WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = FovLimits()
        self.truth = octagon(64, 22, 702, 556, 60)

    def choice_for(self, rgb: np.ndarray):
        detected = detect_field(rgb)
        choice = choose_field(detected, (), self.limits)
        return detected, remove_windows(choice, reduce_frame(rgb, detected.factor), self.limits)

    def test_window_touching_the_picture_is_removed(self) -> None:
        rgb = frame(self.truth)
        rgb[400:570, 20:250] = GREEN
        detected, choice = self.choice_for(rgb)
        self.assertTrue(detected.full_mask()[450:550, 100:200].all())  # merged into the detected field
        self.assertIn("window", choice.flags)
        mask = choice.field.full_mask()
        self.assertFalse((green_mask(rgb) & mask).any())
        self.assertTrue(mask[200:300, 300:400].all())  # the rest of the picture stays
        self.assertLess(choice.field.area_fraction, detected.area_fraction)

    def test_small_or_ragged_green_areas_stay(self) -> None:
        rgb = frame(self.truth)
        rgb[250:270, 300:320] = GREEN  # small blob (bile-like)
        detected, choice = self.choice_for(rgb)
        self.assertNotIn("window", choice.flags)
        self.assertIs(choice.field, detected)
        ring = np.zeros((288, 384, 3), dtype=np.uint8)
        ring[20:220, 50:300] = GREEN
        ring[26:214, 56:294] = 0  # large but hollow: fills about a tenth of its box
        self.assertFalse(window_mask(ring, self.limits).any())
        ring[26:214, 56:294] = GREEN
        self.assertTrue(window_mask(ring, self.limits)[20:220, 50:300].all())


if __name__ == "__main__":
    unittest.main()

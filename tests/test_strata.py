"""Tests for the text-derived stratification flags."""

import unittest

from sage_shift.strata import affirms, location_context, nbi_flag, visibility_bin, vqa_answer


class StrataTest(unittest.TestCase):
    def test_visibility_bins(self):
        cases = {
            "The visibility in the image is good, with no obstructions.": "good",
            "The visibility in the image is clear.": "good",
            "The visibility in the image is moderate due to bubbles.": "fair",
            "The visibility in the image is partially obstructed by fluid.": "fair",
            "The visibility in the image is poor.": "poor",
            "The visibility in the image is not optimal.": "poor",
            "Visibility seems fine.": "unbinned",
            "": "missing",
        }
        for answer, expected in cases.items():
            self.assertEqual(visibility_bin(answer), expected, answer)

    def test_affirms_uses_clause_level_negation(self):
        self.assertTrue(affirms("The endoscope is visible, but it is not targeting anything.", r"endoscop"))
        self.assertFalse(affirms("There is no endoscope in view.", r"endoscop"))
        self.assertFalse(affirms("No instruments are present in the image.", r"instrument"))

    def test_nbi_flag(self):
        self.assertTrue(nbi_flag("The special finding in this image is the use of narrow-band imaging."))
        self.assertTrue(nbi_flag("Image taken in NBI mode."))
        self.assertFalse(nbi_flag("No narrow-band imaging was used."))
        self.assertFalse(nbi_flag("A pinkish mucosa."))

    def test_location_context(self):
        self.assertEqual(location_context(["polyp", "stomach"]), "upper GI")
        self.assertEqual(location_context(["polyp", "colon"]), "lower GI")
        self.assertEqual(location_context(["stomach", "colon"]), "both")
        self.assertEqual(location_context(["polyp"]), "no location label")

    def test_vqa_answer(self):
        pairs = [{"question": "Does this image show a specific section?", "answer": "The stomach."},
                 {"question": "What is the visibility in the image? If there are any ...", "answer": "Good."}]
        self.assertEqual(vqa_answer(pairs, "What is the visibility in the image?"), "Good.")
        self.assertEqual(vqa_answer(pairs, "Are there any polyps"), "")


if __name__ == "__main__":
    unittest.main()

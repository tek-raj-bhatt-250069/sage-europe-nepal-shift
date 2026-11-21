"""Tests for the HyperKvasir -> SAGE label space."""

import json
import unittest
from pathlib import Path

from sage_shift.labelspace import (
    BASE_LABEL, HK_FINDING_TO_SAGE, HK_SPLIT_SPELLING, MEANING_GROUP, SIX_CLASSES, hk_finding, hk_to_sage,
)

ROOT = Path(__file__).resolve().parents[1]

# Class names in HyperKvasir's official_splits/2_fold_split.csv (commit 21cc366).
SPLIT_FILE_NAMES = [
    "barretts", "bbps-0-1", "bbps-2-3", "dyed-lifted-polyps", "dyed-resection-margins", "hemorroids", "ileum",
    "impacted-stool", "normal-cecum", "normal-pylorus", "normal-z-line", "oesophagitis-a", "oesophagitis-b-d",
    "polyp", "retroflex-rectum", "retroflex-stomach", "short-segment-barretts", "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1", "ulcerative-colitis-grade-1-2", "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3", "ulcerative-colitis-grade-3",
]


class LabelSpaceTest(unittest.TestCase):
    def test_six_classes_are_sorted_and_match_the_mapping(self):
        self.assertEqual(list(SIX_CLASSES), sorted(SIX_CLASSES))
        self.assertEqual(set(SIX_CLASSES), {v for v in HK_FINDING_TO_SAGE.values() if v})

    def test_authors_mapping_has_23_classes_and_7_mapped(self):
        self.assertEqual(len(HK_FINDING_TO_SAGE), 23)
        self.assertEqual(sum(v is not None for v in HK_FINDING_TO_SAGE.values()), 7)
        self.assertEqual(hk_to_sage("esophagitis-a"), hk_to_sage("esophagitis-b-d"))

    def test_every_split_file_name_maps(self):
        mapped = {name: hk_to_sage(name) for name in SPLIT_FILE_NAMES}
        self.assertEqual(len({hk_finding(name) for name in SPLIT_FILE_NAMES}), 23)
        self.assertEqual(sum(v is not None for v in mapped.values()), 7)
        self.assertEqual(mapped["normal-pylorus"], "pylorus-normal")
        self.assertEqual(mapped["polyp"], "polyp")
        self.assertIsNone(mapped["normal-cecum"])

    def test_both_spellings_agree(self):
        for split_name, finding in HK_SPLIT_SPELLING.items():
            self.assertEqual(hk_to_sage(split_name), hk_to_sage(finding))

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            hk_finding("not-a-class")

    def test_base_labels_drop_the_suffix(self):
        self.assertEqual(set(BASE_LABEL), set(SIX_CLASSES))
        for cls, base in BASE_LABEL.items():
            self.assertEqual(cls.removesuffix("-normal").removesuffix("-retroflexion"), base)

    def test_meaning_groups(self):
        self.assertEqual(set(MEANING_GROUP), set(SIX_CLASSES))
        self.assertEqual({c for c, g in MEANING_GROUP.items() if g == "shifted"}, {"polyp", "pylorus-normal"})

    def test_config_repeats_the_same_classes(self):
        cfg = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(cfg["six_classes"]), SIX_CLASSES)


if __name__ == "__main__":
    unittest.main()

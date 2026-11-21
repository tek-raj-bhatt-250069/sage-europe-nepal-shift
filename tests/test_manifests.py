"""Tests for the manifest builders: synthetic HyperKvasir files, plus a SAGE check that needs data/."""

import shutil
import tempfile
import unittest
from pathlib import Path

from sage_shift.data import SagePaths
from sage_shift.manifests import build_hk_manifest, build_sage_manifest, sage_summary

ROOT = Path(__file__).resolve().parents[1]
DATA = SagePaths(ROOT / "data")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class HyperKvasirManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.split2 = write(self.tmp / "2_fold_split.csv", "file-name;class-name;split-index\n"
                            "a.jpg;normal-pylorus;0\nb.jpg;polyp;1\nc.jpg;normal-cecum;0\n")
        self.split5 = write(self.tmp / "5_fold_split.csv", "file-name;class-name;split-index\n"
                            "a.jpg;normal-pylorus;3\nb.jpg;polyp;4\nc.jpg;normal-cecum;0\n")

    def test_classes_and_folds(self):
        manifest = build_hk_manifest(self.split2, self.split5).set_index("file_name")
        self.assertEqual(manifest.loc["a.jpg", "hk_class"], "pylorus")
        self.assertEqual(manifest.loc["a.jpg", "sage_class"], "pylorus-normal")
        self.assertEqual(manifest.loc["c.jpg", "sage_class"], "")
        self.assertEqual(manifest.loc["b.jpg", "fold5"], 4)

    def test_paths_from_image_labels(self):
        labels = write(self.tmp / "image-labels.csv", "Video file,Organ,Finding,Classification\n"
                       "a,Upper GI,pylorus,anatomical-landmarks\nb,Lower GI,polyps,pathological-findings\n"
                       "c,Lower GI,cecum,anatomical-landmarks\n")
        image = self.tmp / "upper-gi-tract" / "anatomical-landmarks" / "pylorus" / "a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"")
        manifest = build_hk_manifest(self.split2, None, labels, self.tmp).set_index("file_name")
        self.assertEqual(manifest.loc["b.jpg", "rel_path"], "lower-gi-tract/pathological-findings/polyps/b.jpg")
        self.assertEqual(manifest.loc["a.jpg", "exists"], 1)
        self.assertEqual(manifest.loc["b.jpg", "exists"], 0)

    def test_wrong_columns_or_disagreeing_classes_raise(self):
        bad = write(self.tmp / "bad.csv", "file,class,fold\na.jpg;normal-pylorus;0\n")
        with self.assertRaises(ValueError):
            build_hk_manifest(bad)
        labels = write(self.tmp / "image-labels.csv", "Video file,Organ,Finding,Classification\n"
                       "a,Upper GI,z-line,anatomical-landmarks\nb,Lower GI,polyps,pathological-findings\n"
                       "c,Lower GI,cecum,anatomical-landmarks\n")
        with self.assertRaises(ValueError):
            build_hk_manifest(self.split2, None, labels)


@unittest.skipUnless((DATA.files_dir / "samples.csv").is_file(), "SAGE data not available")
class SageManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_sage_manifest(DATA)

    def test_no_patient_or_visit_identifiers(self):
        self.assertNotIn("patient_uuid", self.manifest.columns)
        self.assertNotIn("visit_uuid", self.manifest.columns)

    def test_six_class_counts(self):
        six = sage_summary(self.manifest)["six_class"]
        self.assertEqual((six["train"]["images"], six["train"]["patients"]), (314, 145))
        self.assertEqual((six["test"]["images"], six["test"]["patients"]), (158, 62))

    def test_flags_match_the_dataset_review(self):
        self.assertEqual(int(self.manifest["nbi"].sum()), 47)
        bins = self.manifest["visibility_bin"].value_counts().to_dict()
        self.assertEqual((bins["good"], bins["fair"], bins["poor"]), (1122, 140, 25))


if __name__ == "__main__":
    unittest.main()

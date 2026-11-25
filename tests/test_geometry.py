"""Geometry controls on synthetic data: input preparation, bootstrap scores, reading rules, scripts 15 and 16."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sage_shift.colour import ColourReference
from sage_shift.fov import FovLimits, FovParams, build_layouts, detect_field
from sage_shift.geometry import (HK_INPUTS, SAGE_INPUTS, class_shares, hk_scores, pair_verdict, prepare_inputs,
                                 sage_scores, shortcut_shown)
from sage_shift.labelspace import SIX_CLASSES
from sage_shift.manifests import LABEL_COLUMNS
from sage_shift.metrics import evaluate
from sage_shift.preprocess import PreprocessReference, frame_field, geometry_images
from synthetic import frame, octagon

ROOT = Path(__file__).resolve().parents[1]
K = len(SIX_CLASSES)
P, R, Z = SIX_CLASSES.index("polyp"), SIX_CLASSES.index("rectum-retroflexion"), SIX_CLASSES.index("z-line-normal")
UPPER = [SIX_CLASSES.index(name) for name in ("esophagitis", "fundus-retroflexion", "pylorus-normal", "z-line-normal")]


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], capture_output=True, text=True)


class InputTests(unittest.TestCase):
    def test_prepare_inputs_matches_geometry_images(self) -> None:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        fields = [detect_field(frame(octagon(64, 22, 702, 556, 60), seed=i)) for i in range(5)]
        layouts, _ = build_layouts(fields, FovLimits(), prefix="t-")
        reference = PreprocessReference(FovParams(), FovLimits(), 64, {"sage": tuple(layouts)},
                                        ColourReference((50.0, 20.0, 18.0), (17.0, 8.0, 7.5), 10, "test"),
                                        {"sage": True, "hk": False})
        paths = []
        for i in range(3):
            Image.fromarray(frame(octagon(64, 22, 702, 556, 60), seed=10 + i)).save(folder / f"{i}.png")
            paths.append(str(folder / f"{i}.png"))
        one = prepare_inputs(paths, "sage", SAGE_INPUTS, reference, size=64, workers=1)
        two = prepare_inputs(paths, "sage", SAGE_INPUTS, reference, size=64, workers=2)
        self.assertEqual(tuple(one), SAGE_INPUTS)
        for i, path in enumerate(paths):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            expected = geometry_images(rgb, frame_field(rgb, reference, "sage").field, 64, SAGE_INPUTS)
            for name in SAGE_INPUTS:
                self.assertEqual(one[name].shape, (3, 64, 64, 3))
                np.testing.assert_array_equal(one[name][i], expected[name])
                np.testing.assert_array_equal(two[name][i], expected[name])
        hk = prepare_inputs(paths[:1], "hk", HK_INPUTS, reference, size=64, workers=1)
        self.assertEqual({name: images.shape for name, images in hk.items()}, {name: (1, 64, 64, 3) for name in HK_INPUTS})
        with self.assertRaises(ValueError):
            prepare_inputs(paths, "sage", ["sageify"], reference, size=64, workers=1)


class ScoreTests(unittest.TestCase):
    def test_sage_scores(self) -> None:
        rng = np.random.default_rng(0)
        y = np.arange(48) % K
        proba = np.eye(K)[y] * 0.5 + rng.dirichlet(np.ones(K), size=48) * 0.5
        proba[:12] = np.eye(K)[P] * 0.9 + 0.1 / K  # some frames go to polyp
        truth = np.eye(K)[y]
        truth[rng.random((48, K)) < 0.1] = 1
        weights = np.vstack([np.ones(48), rng.integers(0, 3, size=(20, 48))]).astype(float)
        point, boot = sage_scores(y, proba, truth, weights)
        reference = evaluate(y, proba, truth)
        self.assertAlmostEqual(point["macro_f1"], reference["macro_f1"])
        self.assertAlmostEqual(point["macro_auroc"], reference["macro_auroc"])
        pred = proba.argmax(axis=1)
        self.assertAlmostEqual(point["polyp_share"], float(np.mean(pred == P)))
        self.assertAlmostEqual(point["top_class_share"], float(np.bincount(pred, minlength=K).max() / 48))
        self.assertEqual(point["top_class"], SIX_CLASSES[int(np.bincount(pred, minlength=K).argmax())])
        for key in ("macro_f1", "macro_auroc", "polyp_share", "top_class_share"):
            self.assertEqual(boot[key].shape, (21,))
            self.assertAlmostEqual(float(boot[key][0]), point[key])  # unit weights give the point value
        self.assertAlmostEqual(float(boot["macro_auroc"][3]), evaluate(y, proba, truth, weights[3])["macro_auroc"])
        np.testing.assert_allclose(class_shares(pred, weights[:1])[0], class_shares(pred))

    def test_hk_scores(self) -> None:
        y = np.array([P, R, UPPER[0], UPPER[1], UPPER[2], UPPER[3]])
        pred = np.array([P, P, P, UPPER[1], R, UPPER[3]])
        upper = np.isin(y, UPPER)
        hd = np.array([False, False, True, True, False, True])
        point, boot = hk_scores(y, pred, upper, hd, np.ones((1, 6)))
        self.assertEqual(point["accuracy_polyp"], 1.0)
        self.assertEqual(point["accuracy_rectum-retroflexion"], 0.0)
        self.assertEqual(point["upper_gi_polyp_or_rectum_share"], 0.5)  # 2 of 4 upper-GI frames
        self.assertAlmostEqual(point["hd_upper_gi_polyp_or_rectum_share"], 1 / 3)  # 1 of 3 HD upper-GI frames
        for key, value in point.items():
            self.assertAlmostEqual(float(boot[key][0]), value)
        _, weighted = hk_scores(y, pred, upper, hd, np.array([[0, 0, 2, 1, 0, 1]], dtype=float))
        self.assertAlmostEqual(float(weighted["upper_gi_polyp_or_rectum_share"][0]), 0.5)  # 2 of 4 weighted frames
        self.assertTrue(np.isnan(weighted["accuracy_polyp"][0]))  # no polyp in this resample


class RuleTests(unittest.TestCase):
    def test_pair_verdict(self) -> None:
        self.assertIn("black area", pair_verdict([True, True, True], [False, False, True])["verdict"])
        self.assertIn("scale, aspect or position", pair_verdict([True, True, True], [True, True, False])["verdict"])
        self.assertEqual(pair_verdict([False] * 3, [False] * 3)["verdict"], "neither reading applies")
        self.assertIn("undecided", pair_verdict([True], [False])["verdict"])
        self.assertEqual(pair_verdict([False, False], [False, False])["verdict"], "neither reading applies")
        out = pair_verdict([True, True, False], [False, True, False])
        self.assertEqual((out["black_only"], out["both"], out["black_collapses"], out["grey_collapses"]), (1, 1, 2, 1))
        with self.assertRaises(ValueError):
            pair_verdict([True], [True, False])

    def test_shortcut_rule(self) -> None:
        self.assertTrue(shortcut_shown({"estimate": 0.3}, {"ci_low": 0.1}))
        self.assertFalse(shortcut_shown({"estimate": 0.3}, {"ci_low": -0.01}))
        self.assertFalse(shortcut_shown({"estimate": -0.1}, {"ci_low": 0.05}))


class ScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def write_inputs(self) -> dict[str, list[str]]:
        """Synthetic manifests, features, RQ1 summaries, scripts/14 outputs and a reduced config."""
        rng = np.random.default_rng(0)
        f = self.folder
        for sub in ("manifests", "rq1", "geometry", "preprocess", "features/sage/resnet50_imagenet",
                    "features/hk/resnet50_imagenet"):
            (f / sub).mkdir(parents=True)
        sage_ids = [f"uuid-{i:04d}" for i in range(84)]
        split = ["test"] * 36 + ["train"] * 48
        y_sage = np.arange(84) % K
        sage = pd.DataFrame({"sample_uuid": sage_ids, "patient_group": [i // 3 for i in range(84)], "split": split,
                             "class": [SIX_CLASSES[c] for c in y_sage], "in_six": 1})
        for j, name in enumerate(SIX_CLASSES):
            sage[LABEL_COLUMNS[name]] = (y_sage == j).astype(int)
        sage.to_csv(f / "manifests" / "sage_manifest.csv", index=False)
        hk_ids = [f"file-{i:04d}.jpg" for i in range(72)]
        y_hk = np.arange(72) % K
        fold = (np.arange(72) // K) % 2  # blocks of six, so each fold holds every class
        pd.DataFrame({"file_name": hk_ids, "sage_class": [SIX_CLASSES[c] for c in y_hk],
                      "fold2": fold}).to_csv(f / "manifests" / "hk_manifest.csv", index=False)
        pd.DataFrame({"id": hk_ids, "width": [1225 if i % 4 else 622 for i in range(72)]}).to_csv(
            f / "preprocess" / "hk_fov_quality.csv", index=False)
        centres = rng.normal(scale=3.0, size=(K, 8))

        def features(y: np.ndarray, shift: float) -> np.ndarray:
            return centres[y] + shift + rng.normal(size=(len(y), 8))

        for name in (*SAGE_INPUTS, "colour_global", "fov_colour_global"):
            np.savez(f / "features/sage/resnet50_imagenet" / f"{name}.npz", features=features(y_sage, 0.3), ids=sage_ids)
        for name in ("raw", "fov", "sageify", "sageify_grey"):
            np.savez(f / "features/hk/resnet50_imagenet" / f"{name}.npz", features=features(y_hk, 0.0), ids=hk_ids)
        for variant in ("raw", "fov"):
            (f / "rq1" / f"rq1_{variant}_summary.json").write_text(json.dumps(
                {"backbones": {"resnet50_imagenet": {"C_hyperkvasir": 1.0, "C_sage": 1.0}}}))

        test_ids, y_test = sage_ids[:36], y_sage[:36]
        fold1_ids, y_fold1 = [i for i, k in zip(hk_ids, fold) if k == 1], y_hk[fold == 1]
        for seed in range(3):
            tag = f"adjusted_seed{seed}"
            (f / "geometry" / f"{tag}_geometry_summary.json").write_text(json.dumps(
                {"tag": tag, "recipe": "adjusted", "seed": seed, "best_epoch": 5}))
            rows = []
            for name in SAGE_INPUTS:  # black-fill inputs and raw collapse to polyp, grey and fov ones do not
                pred = np.full(36, P) if name in ("raw", "masked", "letterbox") else y_test
                rows += [{"sample_uuid": i, "input": name, **dict(zip(SIX_CLASSES, np.eye(K)[c] * 0.9 + 0.1 / K))}
                         for i, c in zip(test_ids, pred)]
            pd.DataFrame(rows).to_csv(f / "geometry" / f"{tag}_geometry_sage_probs.csv", index=False)
            rows = []
            for name in HK_INPUTS:  # sageify sends upper-GI frames to polyp in seeds 0 and 1
                pred = np.where(np.isin(y_fold1, UPPER), P, y_fold1) if name == "sageify" and seed < 2 else y_fold1
                rows += [{"file_name": i, "input": name, **dict(zip(SIX_CLASSES, np.eye(K)[c] * 0.9 + 0.1 / K))}
                         for i, c in zip(fold1_ids, pred)]
            pd.DataFrame(rows).to_csv(f / "geometry" / f"{tag}_geometry_hk_probs.csv", index=False)
        (f / "config.json").write_text(json.dumps({
            "seeds": {"global": 0, "bootstrap": 1}, "bootstrap": {"n_resamples": 100, "ci": 0.95},
            "probe": {"C_grid": [0.1, 1.0], "max_iter": 1000},
            "paper_recipe_rerun": {"collapse_criterion": {"top_class_share_of_predictions": 0.8}}}))
        return {"ids": sage_ids + hk_ids}

    def test_script_15_runs_on_synthetic_inputs(self) -> None:
        made = self.write_inputs()
        f = self.folder
        done = run_script("15_geometry_analysis.py", "--features", str(f / "features"), "--manifests", str(f / "manifests"),
                          "--rq1", str(f / "rq1"), "--geometry", str(f / "geometry"), "--hk-quality",
                          str(f / "preprocess" / "hk_fov_quality.csv"), "--config", str(f / "config.json"),
                          "--out-dir", str(f / "out"), "--jobs", "1")
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])
        summary = json.loads((f / "out" / "geometry_summary.json").read_text())
        verdicts = summary["verdicts"]["adjusted"]
        self.assertIn("black area", verdicts["masked_vs_masked_grey"]["verdict"])
        self.assertIn("black area", verdicts["letterbox_vs_letterbox_grey"]["verdict"])
        self.assertEqual(verdicts["fov_vs_fov_grey"]["verdict"], "neither reading applies")
        self.assertEqual((verdicts["source_shortcut"]["seeds_shown"], verdicts["source_shortcut"]["seeds"]), (2, 3))
        self.assertIn("resnet50_imagenet", summary["verdicts"]["probes_source_shortcut"])
        self.assertEqual(set(summary["probes"]["resnet50_imagenet"]["colour_global"]), {"colour_global", "fov_colour_global"})
        table = pd.read_csv(f / "out" / "geometry_table.csv")
        self.assertEqual(set(table["section"]), {"probe_sage", "probe_hk", "colour_global", "finetuned_sage", "finetuned_hk"})
        runs = pd.read_csv(f / "out" / "geometry_runs.csv")
        self.assertEqual(len(runs), 3 * len(SAGE_INPUTS))
        self.assertTrue(runs.loc[runs["input"] == "masked", "collapse"].all())
        self.assertFalse(runs.loc[runs["input"] == "masked_grey", "collapse"].any())
        text = "".join((f / "out" / name).read_text() for name in ("geometry_summary.json", "geometry_table.csv",
                                                                   "geometry_runs.csv"))
        self.assertFalse([i for i in made["ids"] if i in text])  # aggregate outputs only

    def test_script_16_colour_statistics(self) -> None:
        f = self.folder
        quality = pd.DataFrame({
            "id": ["a", "b", "c", "d", "e"], "split": ["train", "train", "train", "train", "test"],
            "class": ["polyp", "", "polyp", "", "polyp"], "in_six": [True, False, True, False, True],
            "flags": ["", "", "not_convex|fallback", "", ""],
            "L_mean": [40.0, 50.0, 99.0, 60.0, 99.0], "a_mean": [20.0, 22.0, 99.0, 24.0, 99.0],
            "b_mean": [18.0, 20.0, 99.0, 22.0, 99.0], "L_sd": [10.0, 12.0, 99.0, 14.0, 99.0],
            "a_sd": [5.0, 6.0, 99.0, 7.0, 99.0], "b_sd": [4.0, 5.0, 99.0, 6.0, 99.0]})
        quality.to_csv(f / "quality.csv", index=False)
        reference = PreprocessReference(FovParams(), FovLimits(), 224, {},
                                        ColourReference((53.0, 22.0, 16.0), (15.0, 7.0, 5.0), 9, "test"))
        reference.save(f / "reference.json")
        args = ["--quality", str(f / "quality.csv"), "--reference", str(f / "reference.json"),
                "--out", str(f / "out" / "colour.json"), "--data-dir", str(f / "data")]
        done = run_script("16_colour_global_stats.py", *args)
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])
        out = json.loads((f / "out" / "colour.json").read_text())
        np.testing.assert_allclose(out["mean"], [50.0, 22.0, 20.0])  # clean train frames a, b, d
        np.testing.assert_allclose(out["std"], [12.0, 6.0, 5.0])
        self.assertEqual((out["images"], out["train_frames"], out["train_frames_with_field_flags"]), (3, 4, 1))
        self.assertEqual(out["six_class_train_only"]["images"], 1)
        self.assertEqual(out["target"]["mean"], [53.0, 22.0, 16.0])
        self.assertEqual(ColourReference.from_dict(out).images, 3)
        self.assertNotIn('"a"', (f / "out" / "colour.json").read_text())
        quality.loc[0, "L_mean"] = 41.0
        quality.to_csv(f / "quality.csv", index=False)
        self.assertNotEqual(run_script("16_colour_global_stats.py", *args).returncode, 0)  # other values: kept
        self.assertEqual(run_script("16_colour_global_stats.py", *args, "--overwrite").returncode, 0)


if __name__ == "__main__":
    unittest.main()

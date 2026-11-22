"""The paper-recipe helpers: split, transforms, training step and scoring (skipped without torch)."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

try:
    import torch

    from sage_shift.recipe import (CachedFrames, faithful_split, make_loader, make_model, predict, prediction_summary,
                                   rng_state, scores, set_rng_state, test_transform, train_epoch, train_transform)
except ImportError:  # torch is installed on Colab; locally it is optional
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class RecipeTests(unittest.TestCase):
    def test_faithful_split_is_stratified_and_repeatable(self) -> None:
        labels = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        train, test = faithful_split(labels, seed=0)
        self.assertEqual((len(train), len(test)), (80, 20))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(sorted(np.array(labels)[test].tolist()), ["a"] * 10 + ["b"] * 6 + ["c"] * 4)
        again, _ = faithful_split(labels, seed=0)
        np.testing.assert_array_equal(train, again)

    def test_transforms(self) -> None:
        image = np.full((224, 224, 3), 128, dtype=np.uint8)
        data = CachedFrames(image[None], np.array([2]), test_transform())
        tensor, label = data[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(label, 2)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), (128 / 255 - 0.485) / 0.229, places=5)
        augmented = train_transform()(Image.fromarray(image))
        self.assertEqual(tuple(augmented.shape), (3, 224, 224))

    def test_scores_and_collapse(self) -> None:
        y = np.array([0, 1, 2, 2, 3, 4, 5, 2])
        proba = np.zeros((8, 6))
        proba[:, 2] = 1.0  # always "polyp"
        out = prediction_summary(y, proba, collapse_share=0.8)
        self.assertTrue(out["collapse"])
        self.assertEqual(out["top_class"], "polyp")
        self.assertEqual(out["top_class_share"], 1.0)
        self.assertAlmostEqual(out["weighted_f1"], round(f1_score(y, proba.argmax(1), average="weighted",
                                                                  zero_division=0), 4))
        self.assertEqual(sum(out["predicted_counts"].values()), 8)
        self.assertAlmostEqual(scores(y, np.eye(6)[y])["macro_f1"], 1.0)

    def test_one_training_epoch_and_rng_round_trip(self) -> None:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(8, 64, 64, 3), dtype=np.uint8)
        labels = np.arange(8) % 6
        generator = torch.Generator().manual_seed(0)
        loader = make_loader(CachedFrames(images, labels, train_transform()), 4, True, 0, generator)
        model = make_model(pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_epoch(model, loader, optimizer, torch.device("cpu"))
        self.assertTrue(np.isfinite(loss))
        proba = predict(model, make_loader(CachedFrames(images, labels, test_transform()), 4, False, 0),
                        torch.device("cpu"))
        self.assertEqual(proba.shape, (8, 6))
        np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)
        state = rng_state(generator)
        first = torch.rand(3), torch.randperm(10, generator=generator)
        set_rng_state(state, generator)
        second = torch.rand(3), torch.randperm(10, generator=generator)
        self.assertTrue(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()

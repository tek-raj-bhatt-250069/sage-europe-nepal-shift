"""Backbone loaders and the extraction loop, with random weights and synthetic frames (skipped without torch)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn

    from sage_shift.backbones import Backbone, clean_state_dict, dinov2_features, load_resnet50_gastronet, resnet_trunk
    from sage_shift.features import FrameDataset, extract, load_features, save_features
except ImportError:  # torch is installed on Colab; locally it is optional
    torch = None

from synthetic import frame, octagon


@unittest.skipIf(torch is None, "torch is not installed")
class GastroNetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        torch.manual_seed(0)
        self.source = resnet_trunk()
        self.state = {f"module.backbone.{k}": v for k, v in self.source.state_dict().items()}
        self.state["head.mlp.0.weight"] = torch.zeros(4, 4)

    def check_same(self, model: nn.Module) -> None:
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value)

    def test_teacher_checkpoint_with_prefixes(self) -> None:
        torch.save({"teacher": self.state, "student": {}}, self.folder / "gastronet.pth")
        backbone = load_resnet50_gastronet(self.folder / "gastronet.pth")
        self.check_same(backbone.model)
        self.assertEqual(len(backbone.record["sha256"]), 64)
        self.assertNotIn("head.mlp.0.weight", clean_state_dict({"teacher": self.state}))

    def test_missing_weights_raise(self) -> None:
        partial = {k: v for k, v in self.state.items() if "layer4" not in k}
        torch.save(partial, self.folder / "partial.pth")
        with self.assertRaises(RuntimeError):
            load_resnet50_gastronet(self.folder / "partial.pth")

    def test_pickle_checkpoint_needs_permission(self) -> None:
        torch.save({"teacher": self.state, "args": argparse.Namespace(arch="resnet50")}, self.folder / "full.pth")
        with self.assertRaises(RuntimeError):
            load_resnet50_gastronet(self.folder / "full.pth")
        self.check_same(load_resnet50_gastronet(self.folder / "full.pth", allow_pickle=True).model)


class _StubViT(nn.Module if torch is not None else object):
    """Mimics DINOv2's forward_features output."""

    def forward_features(self, x):  # noqa: D401 - torch-style method
        batch = x.shape[0]
        return {"x_norm_clstoken": torch.ones(batch, 768), "x_norm_patchtokens": torch.arange(256.0).repeat(batch, 768, 1).transpose(1, 2)}


@unittest.skipIf(torch is None, "torch is not installed")
class ExtractionTests(unittest.TestCase):
    def test_dinov2_feature_is_cls_plus_mean_patch(self) -> None:
        out = dinov2_features(_StubViT(), torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 1536))
        torch.testing.assert_close(out[:, :768], torch.ones(2, 768))
        torch.testing.assert_close(out[:, 768:], torch.full((2, 768), 127.5))

    def test_extract_keeps_order_and_shapes(self) -> None:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        paths = []
        for i in range(5):
            Image.fromarray(frame(octagon(64, 22, 702, 556, 60), seed=i)).save(folder / f"{i}.png")
            paths.append(str(folder / f"{i}.png"))
        net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
        stub = Backbone("stub", net, lambda model, x: model(x), 4)
        dataset = FrameDataset(paths, "sage", ["raw"], None, 64)
        features = extract(dataset, [stub], torch.device("cpu"), batch_size=2, workers=0)
        self.assertEqual(list(features), [("stub", "raw")])
        values = features[("stub", "raw")]
        self.assertEqual(values.shape, (5, 4))
        single = extract(FrameDataset(paths[3:4], "sage", ["raw"], None, 64), [stub], torch.device("cpu"), 1, 0)
        np.testing.assert_allclose(values[3], single[("stub", "raw")][0], rtol=1e-5)
        save_features(folder / "out" / "raw.npz", values, [f"id{i}" for i in range(5)], variant="raw")
        loaded, ids = load_features(folder / "out" / "raw.npz")
        np.testing.assert_array_equal(loaded, values)
        self.assertEqual(ids.tolist(), [f"id{i}" for i in range(5)])
        with self.assertRaises(ValueError):
            FrameDataset(paths, "sage", ["fov"], None, 64)


if __name__ == "__main__":
    unittest.main()

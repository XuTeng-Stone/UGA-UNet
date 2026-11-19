"""Small CPU smoke tests for the core implementation.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ugaunet.losses import LossCEDice, anchor_loss_from_maps
from ugaunet.metrics import binary_dice_iou, calculate_advanced_metrics, count_parameters
from ugaunet.models import UGAUNet


class TestUGAUNet(unittest.TestCase):
    def test_parameter_count(self) -> None:
        model = UGAUNet()
        self.assertEqual(count_parameters(model), 3_913_474)

    def test_forward_shapes(self) -> None:
        model = UGAUNet()
        logits, features = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))
        self.assertEqual(tuple(features.shape), (2, 32, 64, 64))


class TestAnchorLoss(unittest.TestCase):
    def test_backward(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(2, 8, 64, 64, requires_grad=True)
        gt = torch.zeros(2, 64, 64)
        gt[:, 10:40, 10:40] = 1
        uncertainty = torch.rand(2, 64, 64)
        loss = anchor_loss_from_maps(features, gt, uncertainty)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_empty_batch_is_safe(self) -> None:
        features = torch.randn(1, 8, 32, 32, requires_grad=True)
        gt = torch.zeros(1, 32, 32)
        uncertainty = torch.rand(1, 32, 32)
        loss = anchor_loss_from_maps(features, gt, uncertainty)
        self.assertEqual(float(loss), 0.0)
        loss.backward()

    def test_segmentation_loss(self) -> None:
        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (2, 32, 32))
        loss = LossCEDice()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class TestMetrics(unittest.TestCase):
    def test_dice_iou(self) -> None:
        pred = np.zeros((10, 10), dtype=np.uint8)
        target = np.zeros((10, 10), dtype=np.uint8)
        pred[2:6, 2:6] = 1
        target[2:6, 2:6] = 1
        dice, iou = binary_dice_iou(pred, target)
        self.assertAlmostEqual(dice, 1.0, places=4)
        self.assertAlmostEqual(iou, 1.0, places=4)

    def test_assd_hd95(self) -> None:
        logits = torch.zeros(1, 2, 32, 32)
        logits[0, 1, 8:20, 8:20] = 10.0
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[0, 8:20, 8:20] = 1
        assd, hd95 = calculate_advanced_metrics(logits, target)
        self.assertGreaterEqual(assd, 0.0)
        self.assertGreaterEqual(hd95, 0.0)


if __name__ == "__main__":
    unittest.main()

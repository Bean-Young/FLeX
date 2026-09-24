from __future__ import annotations

import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flex.frequency import FrequencyPrompt, apply_frequency_prompt, decompose_frequency_bands
from flex.method import FlexAdapter, FlexConfig
from flex.metrics import classification_metrics, segmentation_metrics, summarize_records
from tools.summarize_runs import summarize


class PaperMetricTests(unittest.TestCase):
    def test_weighted_f1_differs_from_macro_when_target_support_is_imbalanced(self):
        score = classification_metrics([0, 0, 0, 1, 2], [0, 0, 1, 1, 1])
        self.assertAlmostEqual(score["cls_f1"], (3 * 0.8 + 1 * 0.5) / 5)
        self.assertAlmostEqual(score["cls_f1_macro"], (0.8 + 0.5 + 0.0) / 3)
        self.assertEqual(score["confusion_matrix"], [[2, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(score["per_class"]["Malignant"]["recall"], 1.0)

    def test_empty_masks_are_reported_separately(self):
        empty = np.zeros((8, 8), dtype=bool)
        lesion = empty.copy()
        lesion[2:4, 2:4] = True
        rows = []
        for truth, prediction in ((empty, empty), (empty, lesion), (lesion, empty)):
            rows.append({"true_class": 2 if not truth.any() else 1,
                         "pred_class": 2 if not prediction.any() else 1,
                         **segmentation_metrics(prediction, truth)})
        score = summarize_records(rows)
        self.assertAlmostEqual(score["empty_mask_false_positive_rate"], 0.5)
        self.assertAlmostEqual(score["empty_mismatch_rate"], 2 / 3)
        self.assertEqual(score["n_lesion"], 1)
        self.assertEqual(score["n_empty"], 2)
        self.assertAlmostEqual(score["lesion_iou"], 0.0)

    def test_three_run_two_level_aggregation_and_dispersion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "runs.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["setting", "method", "source", "target", "backbone", "seed", "result"])
                for target, count, base in (("B", 1, 0.2), ("C", 3, 0.4)):
                    for backbone, offset in (("UNet", 0.0), ("TransUNet", 0.4)):
                        for seed in range(3):
                            iou = base + offset + seed * 0.1
                            record = {"true_class": 1, "pred_class": 1,
                                      "iou": iou, "dice": iou, "hd95": 1.0,
                                      "lesion_iou": iou, "lesion_dice": iou,
                                      "empty_mismatch": 0.0,
                                      "true_empty": False, "pred_empty": False}
                            path = root / f"{target}-{backbone}-{seed}.json"
                            path.write_text(json.dumps({"n": count, "records": [record] * count}))
                            writer.writerow(["TTA", "FLeX", "A", target, backbone, seed, path.name])
            report = summarize(manifest, expected_runs=3, expected_backbones=2)
            overall = report["source_means"][0]
            self.assertAlmostEqual(overall["metrics"]["iou"], 0.65)
            self.assertAlmostEqual(overall["run_sd"]["iou"], 0.1)
            self.assertEqual(overall["n"], 4)


class TinyModel(nn.Module):
    def forward(self, images):
        segmentation = (images[:, :1] - 0.5) * 8
        mean = images.mean(dim=(1, 2, 3))
        classes = torch.stack((mean, -mean, 1 - mean), dim=1)
        return segmentation, classes


class PaperModeTests(unittest.TestCase):
    def test_zero_prompt_is_fixed_frequency_weighting(self):
        image = torch.rand(2, 3, 16, 16)
        prompt = FrequencyPrompt(image_size=16)
        with torch.no_grad():
            prompt.low.fill_(2)
            prompt.mid.fill_(3)
            prompt.high.fill_(4)
        bands = decompose_frequency_bands(image, prompt)
        expected = 0.25 * bands[0] + 0.50 * bands[1] + 0.25 * bands[2]
        observed = apply_frequency_prompt(image, prompt, zero_prompt=True)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6))
        self.assertFalse(torch.allclose(observed, image))

    def test_no_adapt_returns_raw_source_and_fixed_control_does_not_update(self):
        image = torch.rand(2, 3, 16, 16)
        adapter = FlexAdapter(TinyModel(), image_size=16, cfg=FlexConfig())
        source_seg, source_cls = adapter.source_model(image)
        raw = adapter.predict(image, adapt=False)
        self.assertTrue(torch.allclose(raw["seg_probs"], source_seg.sigmoid()))
        self.assertTrue(torch.allclose(raw["cls_probs"], source_cls.softmax(dim=1)))

        with torch.no_grad():
            adapter.prompt.low.fill_(1.0)
        before = [parameter.detach().clone() for parameter in adapter.prompt.parameters()]
        self.assertEqual(len(adapter.optimizer.state), 0)
        fixed = adapter.predict(image, adapt=False, fixed_frequency=True)
        self.assertEqual(fixed["seg_probs"].shape, raw["seg_probs"].shape)
        self.assertEqual(len(adapter.optimizer.state), 0)
        for old, current in zip(before, adapter.prompt.parameters()):
            self.assertTrue(torch.equal(old, current))


if __name__ == "__main__":
    unittest.main()

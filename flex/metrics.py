from __future__ import annotations

import numpy as np

try:
    from medpy import metric as medpy_metric
except Exception:  # pragma: no cover
    medpy_metric = None


def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float | None]:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    pred_sum = pred.sum()
    true_sum = true.sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if pred_sum + true_sum == 0 else float(2 * intersection / (pred_sum + true_sum))
    hd95 = None
    if medpy_metric is not None and pred_sum > 0 and true_sum > 0:
        hd95 = float(medpy_metric.binary.hd95(pred, true))
    return {"iou": iou, "dice": dice, "hd95": hd95}


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    for label in labels:
        tp = int(((true_arr == label) & (pred_arr == label)).sum())
        fp = int(((true_arr != label) & (pred_arr == label)).sum())
        fn = int(((true_arr == label) & (pred_arr != label)).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores)) if scores else 0.0

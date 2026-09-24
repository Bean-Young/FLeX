from __future__ import annotations

import numpy as np

try:
    from medpy import metric as medpy_metric
except Exception:  # pragma: no cover
    medpy_metric = None


def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float | bool | None]:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    pred_sum = pred.sum()
    true_sum = true.sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if pred_sum + true_sum == 0 else float(2 * intersection / (pred_sum + true_sum))
    empty_mismatch = bool((pred_sum == 0) != (true_sum == 0))
    if pred_sum == 0 and true_sum == 0:
        hd95 = 0.0
    elif empty_mismatch:
        # Penalize a complete miss or false lesion by the image diagonal.
        hd95 = float(np.hypot(*pred.shape[-2:]))
    elif medpy_metric is not None:
        hd95 = float(medpy_metric.binary.hd95(pred, true))
    else:
        hd95 = None
    return {
        "iou": iou,
        "dice": dice,
        "hd95": hd95,
        "lesion_iou": None if true_sum == 0 else iou,
        "lesion_dice": None if true_sum == 0 else dice,
        "empty_mismatch": float(empty_mismatch),
        "true_empty": bool(true_sum == 0),
        "pred_empty": bool(pred_sum == 0),
    }


def macro_f1(
    y_true: list[int],
    y_pred: list[int],
    labels: tuple[int, ...] = (0, 1, 2),
) -> float:
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


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    labels: tuple[int, ...] = (0, 1, 2),
    names: tuple[str, ...] = ("Benign", "Malignant", "Normal"),
) -> dict:
    """Paper F1 uses target support; macro F1 and class detail are diagnostics."""

    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and prediction counts differ")
    if len(labels) != len(names):
        raise ValueError("Each class label needs a name")
    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique")
    if set(y_true + y_pred) - set(labels):
        raise ValueError("Classification label outside the fixed class set")

    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    lookup = {label: index for index, label in enumerate(labels)}
    for truth, prediction in zip(y_true, y_pred):
        confusion[lookup[truth], lookup[prediction]] += 1

    classes = {}
    for index, name in enumerate(names):
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        tp = int(confusion[index, index])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        classes[name] = {
            "label": labels[index],
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    count = len(y_true)
    weighted = sum(item["support"] * item["f1"] for item in classes.values()) / count if count else 0.0
    macro = sum(item["f1"] for item in classes.values()) / len(labels) if labels else 0.0
    return {
        "cls_acc": float(np.trace(confusion) / count) if count else 0.0,
        "cls_f1": weighted,
        "cls_f1_weighted": weighted,
        "cls_f1_macro": macro,
        "malignant_recall": classes["Malignant"]["recall"],
        "per_class": classes,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": list(names),
    }


def summarize_records(records: list[dict]) -> dict:
    """Recompute the manuscript metrics from image-level evaluation records."""

    count = len(records)
    y_true = [int(row["true_class"]) for row in records]
    y_pred = [int(row["pred_class"]) for row in records]
    summary = classification_metrics(y_true, y_pred)

    def mean_of(key: str, rows: list[dict]) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    lesion_rows = [row for row in records if row.get("lesion_iou") is not None]
    empty_rows = [row for row in records if row.get("true_empty") is True]
    # Older JSON records lack true_empty; lesion_iou still identifies empty GT.
    if not empty_rows and records and all("true_empty" not in row for row in records):
        empty_rows = [row for row in records if row.get("lesion_iou") is None]
    hd95_rows = [row for row in records if row.get("hd95") is not None]

    summary.update({
        "iou": mean_of("iou", records),
        "dice": mean_of("dice", records),
        "hd95": mean_of("hd95", records) if len(hd95_rows) == count else None,
        "hd95_coverage": len(hd95_rows) / count if count else 0.0,
        "lesion_iou": mean_of("lesion_iou", lesion_rows),
        "lesion_dice": mean_of("lesion_dice", lesion_rows),
        "empty_mismatch_rate": mean_of("empty_mismatch", records),
        "empty_mask_false_positive_rate": None,
        "n": count,
        "n_lesion": len(lesion_rows),
        "n_empty": len(empty_rows),
    })
    if empty_rows:
        summary["empty_mask_false_positive_rate"] = sum(
            not bool(row["pred_empty"]) if "pred_empty" in row else bool(row["empty_mismatch"])
            for row in empty_rows
        ) / len(empty_rows)
    return summary

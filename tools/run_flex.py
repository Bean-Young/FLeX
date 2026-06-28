from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flex.data import UltrasoundCsvDataset
from flex.method import FlexAdapter, FlexConfig
from flex.metrics import macro_f1, segmentation_metrics


def load_factory(spec: str):
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def load_model(factory_spec: str, checkpoint: Path, image_size: int, device: torch.device):
    model = load_factory(factory_spec)(image_size=image_size)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    return model.to(device)


def evaluate(adapter: FlexAdapter, loader: DataLoader, device: torch.device, threshold: float, adapt: bool):
    ious, dices, hd95s = [], [], []
    y_true, y_pred = [], []
    records = []
    for images, masks, labels, paths in tqdm(loader, desc="eval", ncols=100):
        images = images.to(device)
        outputs = adapter.predict(images, adapt=adapt)
        seg_probs = outputs["seg_probs"].detach().cpu().numpy()
        cls_probs = outputs["cls_probs"].detach().cpu().numpy()
        masks_np = masks.numpy()
        labels_np = labels.numpy()
        preds = cls_probs.argmax(axis=1)
        pred_masks = seg_probs > threshold
        for idx, path in enumerate(paths):
            metrics = segmentation_metrics(pred_masks[idx, 0], masks_np[idx, 0] > 0.5)
            ious.append(metrics["iou"])
            dices.append(metrics["dice"])
            if metrics["hd95"] is not None:
                hd95s.append(metrics["hd95"])
            y_true.append(int(labels_np[idx]))
            y_pred.append(int(preds[idx]))
            records.append(
                {
                    "image": str(path),
                    "true_class": int(labels_np[idx]),
                    "pred_class": int(preds[idx]),
                    "probabilities": cls_probs[idx].astype(float).tolist(),
                    **metrics,
                }
            )
    return {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "hd95": float(np.mean(hd95s)) if hd95s else None,
        "cls_acc": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
        "cls_f1": macro_f1(y_true, y_pred),
        "n": len(y_true),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with image, mask, label columns.")
    parser.add_argument("--root", default=None, help="Root for relative CSV paths.")
    parser.add_argument("--model-factory", required=True, help="Python factory, for example my_models:build_unet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = UltrasoundCsvDataset(args.manifest, root=args.root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = load_model(args.model_factory, Path(args.checkpoint), args.image_size, device)
    adapter = FlexAdapter(model, image_size=args.image_size, cfg=FlexConfig()).to(device)
    result = evaluate(adapter, loader, device, args.threshold, adapt=not args.no_adapt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

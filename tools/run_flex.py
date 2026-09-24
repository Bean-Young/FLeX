from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flex.data import UltrasoundCsvDataset
from flex.method import FlexAdapter, FlexConfig, refine_mask
from flex.metrics import segmentation_metrics, summarize_records


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


def evaluate(adapter: FlexAdapter, loader: DataLoader, device: torch.device, mode: str):
    if mode not in {"adapt", "source", "source_hierarchy", "fixed_frequency"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    records = []
    for images, masks, labels, paths in tqdm(loader, desc="eval", ncols=100):
        images = images.to(device)
        outputs = adapter.predict(
            images,
            adapt=mode == "adapt",
            fixed_frequency=mode == "fixed_frequency",
            source_hierarchy=mode == "source_hierarchy",
        )
        seg_probs = outputs["seg_probs"].detach().cpu().numpy()
        cls_probs = outputs["cls_probs"].detach().cpu().numpy()
        masks_np = masks.numpy()
        labels_np = labels.numpy()
        preds = cls_probs.argmax(axis=1)
        pred_masks = seg_probs > adapter.cfg.mask_threshold
        for idx, path in enumerate(paths):
            pred_mask = pred_masks[idx, 0]
            if mode != "source":
                pred_mask = refine_mask(pred_mask, adapter.cfg)
            metrics = segmentation_metrics(pred_mask, masks_np[idx, 0] > 0.5)
            records.append(
                {
                    "image": str(path),
                    "true_class": int(labels_np[idx]),
                    "pred_class": int(preds[idx]),
                    "probabilities": cls_probs[idx].astype(float).tolist(),
                    **metrics,
                }
            )
    return {"mode": mode, **summarize_records(records), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with image, mask, label columns.")
    parser.add_argument("--root", default=None, help="Root for relative CSV paths.")
    parser.add_argument("--model-factory", required=True, help="Python factory, for example my_models:build_unet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--low-radius", type=float, default=2.0, help="Low/mid boundary in Fourier-grid pixels at the selected image size.")
    parser.add_argument("--mid-radius", type=float, default=8.0, help="Mid/high boundary in Fourier-grid pixels at the selected image size.")
    parser.add_argument("--prompt-amplitude", type=float, default=0.05)
    parser.add_argument("--frequency-weights", default="0.25,0.50,0.25", help="Comma-separated low,mid,high fusion weights.")
    parser.add_argument("--prompt-mode", choices=("bands", "full"), default="bands", help="Use three frequency bands or the full-band ablation.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rho-seg", type=float, default=0.50)
    parser.add_argument("--rho-cls", type=float, default=0.50)
    parser.add_argument("--disable-source-fusion", action="store_true")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta-min", type=float, default=0.50)
    parser.add_argument("--eta-max", type=float, default=0.95)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--lambda-src", type=float, default=0.10)
    parser.add_argument("--lambda-pres", type=float, default=0.50)
    parser.add_argument("--lambda-neg", type=float, default=0.50)
    parser.add_argument("--beta-area", type=float, default=1.0)
    parser.add_argument("--disable-morphology", action="store_true")
    parser.add_argument("--min-component-area-ratio", type=float, default=5e-4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-adapt", action="store_true", help="Raw frozen source model baseline; no FLeX postprocessing.")
    mode.add_argument("--source-hierarchy", action="store_true", help="Diagnostic control: frozen source with the FLeX lesion hierarchy.")
    mode.add_argument("--fixed-frequency-only", action="store_true", help="Zero-prompt weighted-band control with the same FLeX fusion and refinement.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = UltrasoundCsvDataset(args.manifest, root=args.root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    frequency_weights = tuple(float(item) for item in args.frequency_weights.split(","))
    if len(frequency_weights) != 3:
        raise ValueError("--frequency-weights must contain low,mid,high values")
    cfg = FlexConfig(
        mask_threshold=args.threshold,
        frequency_alpha_low=args.low_radius / args.image_size,
        frequency_alpha_mid=args.mid_radius / args.image_size,
        prompt_amplitude=args.prompt_amplitude,
        frequency_weights=frequency_weights,
        prompt_mode=args.prompt_mode,
        steps=args.steps,
        lr=args.lr,
        rho_seg=args.rho_seg,
        rho_cls=args.rho_cls,
        reliability_fusion=not args.disable_source_fusion,
        kappa=args.kappa,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        delta=args.delta,
        lambda_src=args.lambda_src,
        lambda_pres=args.lambda_pres,
        lambda_neg=args.lambda_neg,
        beta_area=args.beta_area,
        morphology_refinement=not args.disable_morphology,
        min_component_area_ratio=args.min_component_area_ratio,
    )
    model = load_model(args.model_factory, Path(args.checkpoint), args.image_size, device)
    adapter = FlexAdapter(model, image_size=args.image_size, cfg=cfg).to(device)
    evaluation_mode = (
        "source" if args.no_adapt else
        "source_hierarchy" if args.source_hierarchy else
        "fixed_frequency" if args.fixed_frequency_only else "adapt"
    )
    result = evaluate(adapter, loader, device, mode=evaluation_mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

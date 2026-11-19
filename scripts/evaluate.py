#!/usr/bin/env python
"""Evaluate a trained checkpoint on one of the paper datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ugaunet.datasets import build_dataloaders  # noqa: E402
from ugaunet.engine import evaluate, load_checkpoint, resolve_device, seed_everything  # noqa: E402
from ugaunet.models import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a UGA-UNet or baseline checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["busi", "breast", "qata_v1", "qata_v2"])
    parser.add_argument(
        "--model",
        type=str,
        default="ugaunet",
        choices=["ugaunet", "unet", "unetplusplus", "deeplabv3", "anato", "medt", "transunet", "cmunext", "ukan"],
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--splits-dir", type=str, default=str(REPO_ROOT / "splits"))
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lgb-mode", type=str, default="fusion")
    parser.add_argument("--lgb-blocks", type=int, default=2)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)

    bundle = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_uncertainty=False,
        uncertainty_dir=args.weights_dir,
    )

    kwargs = {}
    if args.model == "ugaunet":
        kwargs = {"mode": args.lgb_mode, "num_bottleneck_blocks": args.lgb_blocks}
    model = build_model(args.model, num_classes=2, image_size=args.image_size, **kwargs)
    model = load_checkpoint(model, args.checkpoint, device).to(device)

    metrics = evaluate(model, bundle.val_loader, device)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

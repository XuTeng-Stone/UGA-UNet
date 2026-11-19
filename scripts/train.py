#!/usr/bin/env python
"""Unified training entry point for UGA-UNet and the paper baselines.

Examples
--------
UGA-UNet on BUSI with the anchor loss (paper setting)::

    python scripts/train.py --dataset busi --model ugaunet --anchor \
        --data-root /path/to/data --epochs 100 --device cuda:0

U-Net baseline without the anchor loss::

    python scripts/train.py --dataset busi --model unet --no-anchor \
        --data-root /path/to/data --epochs 100

Use a YAML config::

    python scripts/train.py --config configs/busi_ugaunet.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ugaunet.datasets import build_dataloaders  # noqa: E402
from ugaunet.engine import TrainConfig, load_checkpoint, seed_everything, train  # noqa: E402
from ugaunet.metrics import cal_params_flops  # noqa: E402
from ugaunet.models import AnchorFeatureWrapper, build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train UGA-UNet or a baseline on BUSI / BrEaST / QaTa-COV19."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="busi",
        choices=["busi", "breast", "qata_v1", "qata_v2"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ugaunet",
        choices=[
            "ugaunet",
            "unet",
            "unetplusplus",
            "deeplabv3",
            "anato",
            "medt",
            "transunet",
            "cmunext",
            "ukan",
        ],
    )
    parser.add_argument(
        "--anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the Uncertainty-Guided Anchor Loss. Default: on for UGA-UNet.",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--splits-dir", type=str, default=str(REPO_ROOT / "splits"))
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lgb-mode", type=str, default="fusion", choices=["fusion", "local", "global", "conv"])
    parser.add_argument("--lgb-blocks", type=int, default=2)

    parser.add_argument("--alpha-max", type=float, default=0.1)
    parser.add_argument("--anchor-margin", type=float, default=0.3)
    parser.add_argument("--lambda-bg", type=float, default=0.25)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--rin", type=int, default=3)
    parser.add_argument("--rout", type=int, default=8)
    parser.add_argument("--q-low", type=float, default=0.2)
    parser.add_argument("--q-high", type=float, default=0.8)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--w-ce", type=float, default=1.0)
    parser.add_argument("--w-dice", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)

    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = args.anchor if args.anchor is not None else (args.model == "ugaunet")
    seed_everything(args.seed)

    config = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        anchor=anchor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        image_size=args.image_size,
        num_workers=args.num_workers,
        eval_every=args.eval_every,
        alpha_max=args.alpha_max,
        anchor_margin=args.anchor_margin,
        lambda_bg=args.lambda_bg,
        warmup_ratio=args.warmup_ratio,
        rin=args.rin,
        rout=args.rout,
        q_low=args.q_low,
        q_high=args.q_high,
        min_points=args.min_points,
        w_ce=args.w_ce,
        w_dice=args.w_dice,
        lgb_mode=args.lgb_mode,
        lgb_blocks=args.lgb_blocks,
        pretrained=args.pretrained,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )

    print("=" * 72)
    print("UGA-UNet training")
    print("=" * 72)
    print(f"dataset={config.dataset} model={config.model} anchor={config.anchor}")
    print(f"device={config.device} seed={config.seed} epochs={config.epochs}")
    print(f"data_root={args.data_root} splits={args.splits_dir}")

    bundle = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_uncertainty=anchor,
        uncertainty_dir=args.weights_dir,
    )

    if anchor:
        missing = getattr(bundle.train_set, "missing_uncertainty", [0])[0]
        total = len(bundle.train_set)
        if missing:
            print(
                f"[warning] {missing}/{total} training samples have no uncertainty map. "
                "Those samples contribute no anchor signal."
            )

    model_kwargs = {}
    if args.model == "ugaunet":
        model_kwargs = {
            "mode": args.lgb_mode,
            "num_bottleneck_blocks": args.lgb_blocks,
        }
    model = build_model(
        args.model,
        num_classes=2,
        pretrained=args.pretrained,
        image_size=args.image_size,
        **model_kwargs,
    )
    if anchor and args.model != "ugaunet":
        model = AnchorFeatureWrapper(model, num_classes=2)

    try:
        params_m, flops_g = cal_params_flops(model, (1, 3, args.image_size, args.image_size), args.device)
        print(f"[complexity] {params_m:.3f}M params, {flops_g:.2f} GFLOPs (thop)")
    except Exception as exc:
        print(f"[complexity] skipped: {exc}")

    if args.resume:
        model = load_checkpoint(model, args.resume, torch.device(args.device))

    final = train(model, bundle.train_loader, bundle.val_loader, config)
    print("\nFinal metrics")
    for key, value in final.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

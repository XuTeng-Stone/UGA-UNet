"""Training and evaluation loop.

The training schedule follows the paper:

* 100 epochs, AdamW, learning rate 1e-4, batch size 8, 256x256 inputs.
* Anchor weight ``alpha(t) = 0`` for the first 10% of epochs, then a linear
  ramp from 0 to ``alpha_max = 0.1``.
* Standard logit loss ``CE + Dice`` plus ``alpha(t) * AnchorLoss``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import LossCEDice, anchor_loss_from_maps
from .metrics import calculate_advanced_metrics, binary_dice_iou, count_parameters


@dataclass
class TrainConfig:
    dataset: str = "busi"
    model: str = "ugaunet"
    anchor: bool = True
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda"
    image_size: int = 256
    num_workers: int = 4
    eval_every: int = 5

    # Anchor-loss hyperparameters (paper Sec. 4.3 / Appendix B).
    alpha_max: float = 0.1
    anchor_margin: float = 0.3
    lambda_bg: float = 0.25
    warmup_ratio: float = 0.1
    rin: int = 3
    rout: int = 8
    q_low: float = 0.2
    q_high: float = 0.8
    min_points: int = 10
    w_ce: float = 1.0
    w_dice: float = 1.0

    # Architecture / model factory.
    lgb_mode: str = "fusion"
    lgb_blocks: int = 2
    pretrained: bool = True

    out_dir: str = "runs"
    run_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def unwrap_logits(output: Any) -> torch.Tensor:
    """Return logits from a tensor, dict, or (logits, features) tuple."""
    if isinstance(output, dict):
        output = output.get("out", output.get("logits"))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected model output: {type(output)!r}")
    return output


def model_forward(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass returning ``(2-channel logits, features or None)``."""
    output = model(images)
    features = None
    if isinstance(output, (tuple, list)) and len(output) > 1:
        features = output[1]
    logits = unwrap_logits(output)
    if logits.shape[1] == 1:
        logits = torch.cat([-logits, logits], dim=1)
    return logits, features


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
) -> dict[str, float]:
    """Compute mIoU, Dice (mF1), ASSD, and HD95.

    Dice/IoU are averaged per image.  ASSD/HD95 follow the original
    implementation: a batch-level value is appended for each batch and the
    values are then averaged.
    """
    model.eval()
    ious: list[float] = []
    dices: list[float] = []
    assds: list[float] = []
    hd95s: list[float] = []

    for images, masks, _uncertainty in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = torch.clamp(masks, 0, 1).long().to(device, non_blocking=True)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        logits, _ = model_forward(model, images)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        target = masks.cpu().numpy()

        for p, t in zip(pred, target):
            if np.sum(t) > 0:
                dice, iou = binary_dice_iou(p, t)
                dices.append(dice)
                ious.append(iou)

        assd, hd95 = calculate_advanced_metrics(logits, masks)
        if assd > 0:
            assds.append(assd)
        if hd95 > 0:
            hd95s.append(hd95)

    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "mF1": float(np.mean(dices)) if dices else 0.0,
        "assd": float(np.mean(assds)) if assds else 0.0,
        "hd95": float(np.mean(hd95s)) if hd95s else 0.0,
        "n_images": len(ious),
    }


def anchor_weight(epoch: int, epochs: int, warmup_ratio: float, alpha_max: float) -> float:
    """Piecewise-linear warm-up used in the paper."""
    warmup_epochs = int(warmup_ratio * epochs)
    if epoch < warmup_epochs:
        return 0.0
    return alpha_max * (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)


def _make_run_dir(config: TrainConfig) -> Path:
    if config.run_name:
        run_dir = Path(config.out_dir) / config.run_name
    else:
        suffix = "anchor" if config.anchor else "noanchor"
        run_dir = (
            Path(config.out_dir)
            / config.dataset
            / f"{config.model}_{suffix}_seed{config.seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train one model and write ``best_model.pth`` / metrics under ``out_dir``."""
    device = resolve_device(config.device)
    model = model.to(device)
    run_dir = _make_run_dir(config)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    direct_params = count_parameters(model)
    print(f"[model] {config.model} | parameters: {direct_params / 1e6:.3f}M")
    print(f"[run]   {run_dir}")

    criterion_seg = LossCEDice(w_ce=config.w_ce, w_dice=config.w_dice).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_iou = -1.0
    best_alpha = 0.0
    history: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        model.train()
        alpha = (
            anchor_weight(epoch, config.epochs, config.warmup_ratio, config.alpha_max)
            if config.anchor
            else 0.0
        )
        stage = "warmup" if config.anchor and alpha <= 0 else ("ramp" if config.anchor else "seg")
        epoch_loss = 0.0
        n_batches = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} [{stage}] alpha={alpha:.4f}",
            leave=False,
        )
        for images, masks, uncertainty in loop:
            images = images.to(device, non_blocking=True)
            masks = torch.clamp(masks, 0, 1).long().to(device, non_blocking=True)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            uncertainty = uncertainty.float().to(device, non_blocking=True)
            if uncertainty.ndim == 4:
                uncertainty = uncertainty.squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                logits, features = model_forward(model, images)
                loss_seg = criterion_seg(logits, masks)

                if config.anchor and alpha > 1e-6:
                    if features is None:
                        raise RuntimeError(
                            "Anchor loss requested but the model returned no feature map. "
                            "Use UGAUNet or wrap the baseline with AnchorFeatureWrapper."
                        )
                    loss_anchor = anchor_loss_from_maps(
                        feat=features.float(),
                        gt=masks,
                        u_fg=uncertainty,
                        w_fg=None,
                        rin=config.rin,
                        rout=config.rout,
                        q_low=config.q_low,
                        q_high=config.q_high,
                        margin=config.anchor_margin,
                        lam_bg=config.lambda_bg,
                        min_pts=config.min_points,
                    )
                else:
                    loss_anchor = logits.new_tensor(0.0)

                loss = loss_seg + alpha * loss_anchor

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())
            n_batches += 1
            loop.set_postfix(loss=f"{float(loss.detach()):.4f}")

        do_eval = ((epoch + 1) % config.eval_every == 0) or (
            epoch >= config.epochs - 10
        )
        if do_eval:
            metrics = evaluate(model, val_loader, device)
            metrics.update(
                {
                    "epoch": epoch + 1,
                    "alpha": alpha,
                    "train_loss": epoch_loss / max(1, n_batches),
                }
            )
            history.append(metrics)
            print(
                f"Epoch {epoch + 1:3d} | loss {metrics['train_loss']:.4f} | "
                f"mIoU {metrics['miou']:.4f} | Dice {metrics['dice']:.4f} | "
                f"ASSD {metrics['assd']:.2f} | HD95 {metrics['hd95']:.2f}"
            )
            if metrics["miou"] > best_iou:
                best_iou = metrics["miou"]
                best_alpha = alpha
                torch.save(model.state_dict(), run_dir / "best_model.pth")

    # Final evaluation with the best checkpoint.
    ckpt = run_dir / "best_model.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
    final = evaluate(model, val_loader, device, desc="Final test")
    final.update(
        {
            "best_val_miou": max(best_iou, 0.0),
            "best_alpha": best_alpha,
            "params_m_direct": direct_params / 1e6,
        }
    )

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"final": final, "history": history}, f, indent=2)

    with open(run_dir / "final_report.txt", "w", encoding="utf-8") as f:
        f.write(
            f"model: {config.model}\n"
            f"dataset: {config.dataset}\n"
            f"anchor: {config.anchor}\n"
            f"seed: {config.seed}\n"
            f"params(M, direct): {direct_params / 1e6:.4f}\n"
            f"mIoU: {final['miou']:.6f}\n"
            f"Dice (mF1): {final['dice']:.6f}\n"
            f"ASSD: {final['assd']:.6f}\n"
            f"HD95: {final['hd95']:.6f}\n"
            f"Best_Alpha: {best_alpha:.6f}\n"
            f"Best_Val_mIoU: {max(best_iou, 0.0):.6f}\n"
        )

    return final


def load_checkpoint(
    model: torch.nn.Module, checkpoint: str | os.PathLike, device: torch.device
) -> torch.nn.Module:
    """Load a checkpoint, tolerating the ``model.`` prefix used by old runs."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if all(k.startswith("model.") for k in state):
        state = {k[len("model.") :]: v for k, v in state.items()}
    elif all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warning] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[warning] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    return model

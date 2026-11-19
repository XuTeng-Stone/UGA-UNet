#!/usr/bin/env python
"""Generate MedSAM-based uncertainty maps for anchor-loss training.

For each training image, ``N`` perturbed point prompts are sampled around the
centroid(s) of the ground-truth lesion.  The pixel-wise variance across the
``N`` MedSAM predictions is smoothed and min-max normalized to ``[0, 1]`` and
stored as an 8-bit PNG.  The maps are computed once, offline, and are not used
at inference time.

Examples
--------
::

    python scripts/generate_uncertainty.py --dataset busi \
        --data-root /path/to/data --device cuda:0

    python scripts/generate_uncertainty.py --dataset qata_v2 \
        --data-root /path/to/data --device cuda:0 --strategy bilateral

Prompt strategies
-----------------
``centroid``
    One positive point per connected component (used for BUSI / BrEaST).
``bilateral``
    Split each mask into left/right halves and take one point per component in
    each half (used for the large, diffuse QaTa infection regions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ugaunet.datasets import BrEaSTDataset, BUSIDataset, QaTaV1Dataset, QaTaV2Dataset  # noqa: E402


class MedSAMUncertainty:
    def __init__(
        self,
        model_name: str = "wanglab/medsam-vit-base",
        device: str = "cuda",
        n_prompts: int = 12,
        jitter: int = 5,
        min_area: float = 20,
        strategy: str = "centroid",
        seed: int = 42,
    ) -> None:
        from transformers import SamModel, SamProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[MedSAM] loading {model_name} on {self.device}")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()
        self.n_prompts = n_prompts
        self.jitter = jitter
        self.min_area = min_area
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def _component_points(self, mask: np.ndarray) -> list[list[int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points: list[list[int]] = []
        for contour in contours:
            if cv2.contourArea(contour) <= self.min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = contour[0][0].tolist()
            points.append([cx, cy])
        return points

    def prompts(self, mask: np.ndarray) -> list[list[int]]:
        if self.strategy == "bilateral":
            h, w = mask.shape
            mid = w // 2
            left = mask.copy()
            left[:, mid:] = 0
            right = mask.copy()
            right[:, :mid] = 0
            return self._component_points(left) + self._component_points(right)
        return self._component_points(mask)

    @torch.no_grad()
    def uncertainty_map(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        points = self.prompts(mask)
        if not points:
            return np.zeros(mask.shape, dtype=np.float32)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = mask.shape
        predictions = []
        for _ in range(self.n_prompts):
            run_points, run_labels = [], []
            for x, y in points:
                dx = int(self.rng.integers(-self.jitter, self.jitter + 1))
                dy = int(self.rng.integers(-self.jitter, self.jitter + 1))
                run_points.append([int(np.clip(x + dx, 0, w - 1)), int(np.clip(y + dy, 0, h - 1))])
                run_labels.append(1)

            inputs = self.processor(
                image,
                input_points=[run_points],
                input_labels=[run_labels],
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            pred = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0][0][0].numpy()
            predictions.append(pred)

        variance = np.var(np.stack(predictions), axis=0)
        variance = cv2.GaussianBlur(variance, (3, 3), 0)
        max_value = float(variance.max()) + 1e-6
        return (variance / max_value).astype(np.float32)


def _resolve_busi(ds: BUSIDataset, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
    row = ds.df.iloc[idx]
    image_path = ds._resolve(row["image"], ds.image_index)
    mask_path = ds._resolve(row["mask"], ds.mask_index)
    if image_path is None or mask_path is None:
        raise FileNotFoundError(f"Could not resolve BUSI row {idx}")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return image, mask, image_path.stem


def _resolve_breast(ds: BrEaSTDataset, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
    row = ds.df.iloc[idx]
    image_path = ds.root_dir / str(row["image"])
    mask_path = ds.root_dir / str(row["label"])
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return image, mask, image_path.stem


def _resolve_qata_v1(ds: QaTaV1Dataset, idx: int):
    row = ds.df.iloc[idx]
    image_path = ds._path(row["image_path"])
    mask_path = ds._path(row["mask_path"])
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return image, mask, Path(str(row["img_name"])).stem


def _resolve_qata_v2(ds: QaTaV2Dataset, idx: int):
    image_path, mask_path, image_name = ds.samples[idx]
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    return image, mask, Path(image_name).stem


def build_jobs(args):
    """Return a list of (dataset, resolver, prefix) jobs."""
    root = Path(args.data_root)
    splits = Path(args.splits_dir)
    jobs = []
    if args.dataset == "busi":
        ds = BUSIDataset(splits / "busi_train.csv", root / "BUSI", transform=None)
        jobs.append((ds, _resolve_busi, ""))
    elif args.dataset == "breast":
        ds = BrEaSTDataset(splits / "breast_train.csv", root / "BrEaST", transform=None)
        jobs.append((ds, _resolve_breast, ""))
    elif args.dataset == "qata_v1":
        ds = QaTaV1Dataset(splits / "qata_v1_train.csv", root / "QaTa-COV19", transform=None)
        jobs.append((ds, _resolve_qata_v1, "v1_"))
    elif args.dataset == "qata_v2":
        base = root / "QaTa-COV19" / "QaTa-COV19-v2"
        train_ds = QaTaV2Dataset(
            base / "Train Set" / "Images",
            base / "Train Set" / "Ground-truths",
            transform=None,
        )
        test_ds = QaTaV2Dataset(
            base / "Test Set" / "Images",
            base / "Test Set" / "Ground-truths",
            transform=None,
        )
        jobs.append((train_ds, _resolve_qata_v2, "v2_train_"))
        jobs.append((test_ds, _resolve_qata_v2, "v2_test_"))
    else:
        raise ValueError(args.dataset)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MedSAM uncertainty maps.")
    parser.add_argument("--dataset", required=True, choices=["busi", "breast", "qata_v1", "qata_v2"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "splits"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default="wanglab/medsam-vit-base")
    parser.add_argument("--n-prompts", type=int, default=12)
    parser.add_argument("--jitter", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--strategy", choices=["centroid", "bilateral"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    if args.out_dir is None:
        if args.dataset in {"busi", "breast"}:
            out_dir = root / ("BUSI" if args.dataset == "busi" else "BrEaST") / "uncertainty_weights"
        else:
            out_dir = root / "QaTa-COV19" / "uncertainty_weights"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = args.strategy or ("bilateral" if args.dataset.startswith("qata") else "centroid")
    generator = MedSAMUncertainty(
        model_name=args.model_name,
        device=args.device,
        n_prompts=args.n_prompts,
        jitter=args.jitter,
        min_area=args.min_area,
        strategy=strategy,
        seed=args.seed,
    )

    for dataset, resolver, prefix in build_jobs(args):
        print(f"[{args.dataset}] processing {len(dataset)} images -> {out_dir}")
        for idx in tqdm(range(len(dataset))):
            image, mask, stem = resolver(dataset, idx)
            out_path = out_dir / f"{prefix}{stem}_weight.png"
            if out_path.exists() and not args.overwrite:
                continue
            try:
                u_map = generator.uncertainty_map(image, mask)
                cv2.imwrite(str(out_path), (u_map * 255).astype(np.uint8))
            except Exception as exc:
                print(f"[error] {stem}: {exc}")


if __name__ == "__main__":
    main()

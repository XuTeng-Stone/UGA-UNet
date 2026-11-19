#!/usr/bin/env python
"""Recreate the fixed QaTa-v1 80/20 train/test CSV split (seed 42).

The paper uses a fixed split; this script documents and reproduces it from the
public QaTa-COV19-v1 directory.  Paths in the output CSV are relative to the
``QaTa-COV19`` data root so the split is portable.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing QaTa-COV19-v1/")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root)
    v1 = root / "QaTa-COV19-v1"
    image_dir, mask_dir = v1 / "Images", v1 / "Ground-truths"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mask_path in sorted(glob.glob(str(mask_dir / "mask_*.png"))):
        mask_name = Path(mask_path).name
        image_name = mask_name.replace("mask_", "", 1)
        image_path = image_dir / image_name
        if image_path.exists():
            rows.append(
                {
                    "image_path": f"QaTa-COV19-v1/Images/{image_name}",
                    "mask_path": f"QaTa-COV19-v1/Ground-truths/{mask_name}",
                    "img_name": image_name,
                }
            )

    if not rows:
        raise SystemExit(f"No paired QaTa-v1 samples found under {v1}")

    train_rows, test_rows = train_test_split(rows, test_size=0.2, random_state=args.seed)
    pd.DataFrame(train_rows).to_csv(out_dir / "qata_v1_train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "qata_v1_test.csv", index=False)
    print(f"train={len(train_rows)} test={len(test_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()

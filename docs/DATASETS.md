# Datasets

The repository does **not** redistribute images or masks.  Download each dataset
from its original source and cite the corresponding paper.

## Benchmark settings

| Dataset | Train | Test | Total | Modality | Task |
|---|---:|---:|---:|---|---|
| BUSI | 517 | 130 | 647 | Breast ultrasound | Tumour segmentation |
| BrEaST | 201 | 51 | 252 | Breast ultrasound | Tumour segmentation |
| QaTa-v1 | 2,360 | 591 | 2,951 | Chest X-ray | COVID-19 lesion segmentation |
| QaTa-v2 | 7,145 | 2,113 | 9,258 | Chest X-ray | COVID-19 lesion segmentation |

References:

- **BUSI** — Al-Dhabyani et al., *Dataset of Breast Ultrasound Images*,
  Data in Brief, 2020.  The common Kaggle mirror is
  `aryashah2k/breast-ultrasound-images-dataset`.
- **BrEaST** — Pawłowska et al., *Curated benchmark dataset for ultrasound
  based breast lesion analysis*, Scientific Data, 2024.
- **QaTa-COV19 v1 / v2** — Degerli et al., *COVID-19 infection map generation
  and detection from chest X-ray images*, Health Information Science and
  Systems, 2021; and *OSegNet*, ICIP 2022.

## Preprocessing

- All images are resized to `256 × 256`.
- Masks are binarized: any positive pixel value is foreground.
- BUSI and BrEaST: lesion-free / normal images are excluded, matching the
  paper.  This avoids inflated overlap metrics from empty masks.
- QaTa-v1 and QaTa-v2: all images contain infection masks and are used in full.
- Augmentation during training: horizontal flip with probability 0.5 and
  rotation within `±30°` with probability 0.5 (Albumentations).
- Normalization uses Albumentations' default ImageNet mean/std.

## Splits

Fixed split files are provided in [`../splits/`](../splits/):

| File | Description |
|---|---|
| `busi_train.csv`, `busi_test.csv` | `image,mask` filenames (resolved from `BUSI/images` and `BUSI/labels`). |
| `breast_train.csv`, `breast_test.csv` | `image,label` paths relative to `BrEaST/`. |
| `qata_v1_train.csv`, `qata_v1_test.csv` | 80/20 split, seed 42; paths relative to `QaTa-COV19/`. |

QaTa-v2 uses the official `Train Set` / `Test Set` directories.

The split files are file lists only.  They contain no image content and are
included so that the exact evaluation protocol is reproducible.

## Uncertainty maps

The anchor loss expects one uncertainty map per training image:

| Dataset | File name | Location |
|---|---|---|
| BUSI | `<image-stem>_weight.png` | `BUSI/uncertainty_weights/` |
| BrEaST | `<image-stem>_weight.png` | `BrEaST/uncertainty_weights/` |
| QaTa-v1 | `v1_<image-stem>_weight.png` | `QaTa-COV19/uncertainty_weights/` |
| QaTa-v2 train | `v2_train_<image-stem>_weight.png` | `QaTa-COV19/uncertainty_weights/` |
| QaTa-v2 test | `v2_test_<image-stem>_weight.png` | `QaTa-COV19/uncertainty_weights/` |

Generate them once with
[`../scripts/generate_uncertainty.py`](../scripts/generate_uncertainty.py); see
[`REPRODUCE.md`](REPRODUCE.md).  The maps are 8-bit PNGs where `0` means stable
and `255` means maximally uncertain.

At test time, uncertainty maps are not used.  `scripts/evaluate.py` always
passes `use_uncertainty=False`.

## Licensing

Each dataset retains its original license and terms of use.  Check the source
before redistributing derived artifacts such as uncertainty maps.

# UGA-UNet

**Lightweight Segmentation with Uncertainty-Guided Boundary Anchoring**  
BMVC 2026 Submission #308

This repository is the cleaned research code for **UGA-UNet**, a compact
3.9 M-parameter segmentation network with a Local–Global Bottleneck (LGB) and
an **Uncertainty-Guided Anchor Loss**.  The loss uses offline MedSAM uncertainty
maps to identify ambiguous boundary pixels during training and aligns their
decoder features with stable class prototypes.  At inference, only UGA-UNet is
used; MedSAM and the uncertainty maps are not required.

> **Review status.** The paper is under review.  The PDF is included under
> [`paper/`](paper/).  Please do not redistribute this repository or the
> manuscript before the review process is complete.

---

## Highlights

- **3.9 M parameters / 12.74 GFLOPs** — a strong accuracy–efficiency trade-off.
- **Local–Global Bottleneck** — depthwise separable 7×7 convolution plus pooled
  multi-head self-attention.
- **Uncertainty-Guided Anchor Loss** — margin-based prototype anchoring in
  feature space, using stable foreground and local background-ring pixels.
- **Cross-modality evaluation** — BUSI, BrEaST, QaTa-v1, and QaTa-v2.
- **Reproducible pipeline** — fixed splits, YAML configs, standalone training
  and evaluation entry points, and the original experiment scripts archived for
  provenance.

## Results

Dice / HD95 over three random seeds (paper Table 4).  Lightweight models have
fewer than 10 M parameters.

| Model | Params | BUSI | BrEaST | QaTa-v2 | QaTa-v1 |
|---|---:|---:|---:|---:|---:|
| MedT | 1.4 M | .613±.010 / 52.0 | .630±.131 / 30.5 | .753±.002 / 40.5 | .707±.003 / 44.0 |
| U-KAN | 4.2 M | .655±.004 / 43.8 | .653±.166 / 37.8 | .782±.001 / 34.9 | .753±.003 / 37.0 |
| CMUNeXt-L | 8.3 M | .756±.018 / 29.8 | .756±.066 / 15.2 | .787±.003 / 33.3 | .756±.003 / 37.0 |
| **UGA-UNet (Ours)** | **3.9 M** | **.768±.010 / 29.6** | **.770±.063 / 23.6** | **.794±.002 / 32.0** | **.765±.004 / 36.4** |
| U-Net | 34.5 M | .734±.009 / 33.0 | .779±.061 / 15.5 | .794±.001 / 32.6 | .766±.001 / 36.3 |
| UNet++ | 26.9 M | .770±.005 / 28.0 | .765±.048 / 15.5 | .783±.001 / 33.4 | .754±.003 / 35.3 |
| DeepLabV3+ | 39.6 M | .780±.003 / 29.2 | .694±.143 / 20.7 | .800±.002 / 30.5 | .776±.002 / 31.5 |
| TransUNet | 93.2 M | .773±.009 / 31.1 | .775±.015 / 14.5 | .796±.000 / 31.7 | .765±.001 / 35.8 |
| AnatoNet | 116.4 M | .764±.042 / 29.0 | .806±.015 / 10.7 | .769±.012 / 34.5 | .760±.015 / 35.1 |

Full tables, ablation results, and available per-run CSVs are in
[`results/`](results/) and [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository layout

```
UGA-UNet/
├── ugaunet/                 # clean, importable implementation
│   ├── models/              # UGAUNet, LGB, baselines, third-party sources
│   ├── losses/              # CE+Dice and Uncertainty-Guided Anchor Loss
│   ├── datasets/            # BUSI, BrEaST, QaTa-v1, QaTa-v2 loaders
│   ├── metrics.py           # Dice/IoU, ASSD, HD95, FLOPs
│   └── engine.py            # training / evaluation loop
├── scripts/
│   ├── train.py             # unified training entry point
│   ├── evaluate.py          # checkpoint evaluation
│   ├── generate_uncertainty.py  # offline MedSAM uncertainty maps
│   ├── prepare_qata_v1_split.py
│   └── run_all_seeds.sh     # three-seed reproduction
├── configs/                 # YAML configs for the four datasets
├── splits/                  # fixed train/test file lists
├── results/                 # paper tables, sensitivity, raw per-run CSVs
├── figures/                 # qualitative figures from the paper
├── paper/                   # manuscript PDF
├── docs/                    # method, datasets, reproduction, code map
└── reproduce/original/      # original experiment scripts (archival)
```

## Installation

Python 3.10+ is recommended.

```bash
git clone <this-repository>
cd UGA-UNet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For MedSAM uncertainty generation, `transformers` must be able to load
[`wanglab/medsam-vit-base`](https://huggingface.co/wanglab/medsam-vit-base).
The MedSAM checkpoint is **not** included in this repository.

## Data

The datasets are not redistributed.  Download them from their original sources
and arrange them as:

```
data/
├── BUSI/
│   ├── images/  labels/
│   ├── train_split.csv  test_split.csv
│   └── uncertainty_weights/       # generated in step 2
├── BrEaST/
│   ├── images/  labels/
│   ├── train.csv  test.csv
│   └── uncertainty_weights/
└── QaTa-COV19/
    ├── QaTa-COV19-v1/Images/  QaTa-COV19-v1/Ground-truths/
    ├── QaTa-COV19-v2/Train Set/Images/  Ground-truths/
    ├── QaTa-COV19-v2/Test Set/Images/   Ground-truths/
    ├── train_v1.csv  test_v1.csv
    └── uncertainty_weights/
```

Copies of the fixed split CSVs are in [`splits/`](splits/) and can be copied
into the corresponding dataset directories.  See
[`docs/DATASETS.md`](docs/DATASETS.md) for statistics and preprocessing.

### 1. Offline uncertainty maps (once)

```bash
python scripts/generate_uncertainty.py \
  --dataset busi --data-root data --device cuda:0

python scripts/generate_uncertainty.py \
  --dataset breast --data-root data --device cuda:0

python scripts/generate_uncertainty.py \
  --dataset qata_v1 --data-root data --device cuda:0 --strategy bilateral

python scripts/generate_uncertainty.py \
  --dataset qata_v2 --data-root data --device cuda:0 --strategy bilateral
```

The defaults match the paper: `N = 12` prompt perturbations, centroid jitter
`±5` pixels, 3×3 Gaussian smoothing, min–max normalization.

### 2. Train UGA-UNet

```bash
# Paper setting: anchor loss enabled (default for --model ugaunet).
python scripts/train.py --config configs/busi_ugaunet.yaml --data-root data --device cuda:0

python scripts/train.py --config configs/breast_ugaunet.yaml --data-root data --device cuda:0
python scripts/train.py --config configs/qata_v1_ugaunet.yaml --data-root data --device cuda:0
python scripts/train.py --config configs/qata_v2_ugaunet.yaml --data-root data --device cuda:0
```

Equivalent explicit command:

```bash
python scripts/train.py \
  --dataset busi --model ugaunet --anchor \
  --data-root data --epochs 100 --batch-size 8 --lr 1e-4 \
  --device cuda:0 --out-dir runs
```

All three seeds:

```bash
DATA_ROOT=data DEVICE=cuda:0 bash scripts/run_all_seeds.sh
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint runs/busi/ugaunet_anchor_seed42/best_model.pth \
  --dataset busi --model ugaunet --data-root data --device cuda:0
```

## Baselines and ablations

The same entry point trains the paper baselines (without the anchor loss by
default):

```bash
python scripts/train.py --dataset busi --model unet --no-anchor --data-root data
```

UGA-UNet ablations:

```bash
# Bottleneck ablation: conv / local / global / fusion, k = 1..4
python scripts/train.py --dataset busi --model ugaunet --no-anchor \
  --lgb-mode local --lgb-blocks 1 --data-root data

# Anchor-loss ablation
python scripts/train.py --config configs/busi_ugaunet_no_anchor.yaml --data-root data
```

The original experiment scripts that produced the reported numbers are kept in
[`reproduce/original/`](reproduce/original/) with their hard-coded paths
unchanged.  See [`docs/CODE_MAP.md`](docs/CODE_MAP.md) for the mapping between
the original scripts and the cleaned package.

## Tests

```bash
python -m unittest discover -s tests -v
```

The smoke tests check the 3.9M parameter count, forward shapes, anchor-loss
backward, and the metric implementations.

## Method recap

For a decoder feature map `F` and normalized uncertainty map `U`:

1. The background region is the local ring `dilate(Y, r_out) \ dilate(Y, r_in)`,
   with `r_in = 3`, `r_out = 8`.
2. Within each class region, pixels below the `q_low = 0.2` uncertainty
   quantile are **stable**; pixels above the `q_high = 0.8` quantile are
   **ambiguous**.
3. Stable pixels form stop-gradient class prototypes `p_fg`, `p_bg`.
4. Ambiguous pixels are anchored with
   `softplus(d(f, p_c) - d(f, p_opposite) + m)`, `m = 0.3`, and the background
   term is weighted by `λ_bg = 0.25`.
5. The total objective is `L = L_CE + L_Dice + α(t) L_anchor`, where `α(t) = 0`
   for the first 10% of epochs and then ramps linearly to `α_max = 0.1`.

See [`docs/METHOD.md`](docs/METHOD.md) for details and the code map.

## Citation

```bibtex
@inproceedings{ugaunet2026,
  title     = {UGA-UNet: Lightweight Segmentation with Uncertainty-Guided Boundary Anchoring},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026},
  note      = {Submission \#308}
}
```

## Acknowledgements

The baseline implementations are adapted from the corresponding public
projects (U-Net, UNet++, DeepLabV3+, AnatoNet, TransUNet, MedT, CMUNeXt, U-KAN).
MedSAM is used only as an offline uncertainty estimator.  See
[`ugaunet/models/third_party/`](ugaunet/models/third_party/) for the model
sources and [`docs/CODE_MAP.md`](docs/CODE_MAP.md) for provenance notes.

## Contact

For questions about the paper or code, please contact the authors.  The
manuscript is anonymous during review.

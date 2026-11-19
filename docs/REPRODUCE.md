# Reproduction guide

## 1. Environment

The paper reports PyTorch 2.8.0 on Ubuntu 20.04 with eight RTX 3090 GPUs.  The
cleaned code was smoke-tested with:

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- Albumentations 2.0.8
- Transformers 4.46.3

Newer PyTorch versions should work.  Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the matching PyTorch build from
<https://pytorch.org/get-started/locally/> first, then `pip install -r
requirements.txt`.

## 2. Data layout

Set `DATA_ROOT` to a directory containing the three dataset families:

```text
data/
├── BUSI/
├── BrEaST/
└── QaTa-COV19/
```

Copy the provided split files into the expected places, or pass
`--splits-dir` to the scripts:

```bash
cp splits/busi_train.csv splits/busi_test.csv data/BUSI/
cp splits/breast_train.csv splits/breast_test.csv data/BrEaST/
cp splits/qata_v1_train.csv splits/qata_v1_test.csv data/QaTa-COV19/
```

See [`DATASETS.md`](DATASETS.md) for the dataset sources, statistics, and
preprocessing.

## 3. Generate uncertainty maps (once)

MedSAM is loaded from Hugging Face (`wanglab/medsam-vit-base`).  The first run
downloads the checkpoint.  The maps are written next to each dataset:

```bash
python scripts/generate_uncertainty.py --dataset busi     --data-root data --device cuda:0
python scripts/generate_uncertainty.py --dataset breast   --data-root data --device cuda:0
python scripts/generate_uncertainty.py --dataset qata_v1  --data-root data --device cuda:0 --strategy bilateral
python scripts/generate_uncertainty.py --dataset qata_v2  --data-root data --device cuda:0 --strategy bilateral
```

The default is `N = 12`, jitter `±5`, 3×3 Gaussian smoothing, and min–max
normalization.  Existing maps are skipped; pass `--overwrite` to regenerate.

## 4. Use released checkpoints (optional)

The [v1.0.0 release](https://github.com/XuTeng-Stone/UGA-UNet/releases/tag/v1.0.0)
contains the 13 UGA-UNet checkpoints used for the paper's four datasets and
three seeds, plus the BUSI no-anchor ablation checkpoint.  Download the assets
and evaluate directly:

```bash
python scripts/evaluate.py \
  --checkpoint ugaunet_busi_seed42_anchor.pth \
  --dataset busi --model ugaunet --data-root data --device cuda:0
```

## 5. Train

The paper setting is 100 epochs, batch size 8, AdamW with learning rate `1e-4`,
input `256 × 256`, and the anchor loss enabled.  The YAML configs encode these
values:

```bash
python scripts/train.py --config configs/busi_ugaunet.yaml     --data-root data --device cuda:0
python scripts/train.py --config configs/breast_ugaunet.yaml   --data-root data --device cuda:0
python scripts/train.py --config configs/qata_v1_ugaunet.yaml  --data-root data --device cuda:0
python scripts/train.py --config configs/qata_v2_ugaunet.yaml  --data-root data --device cuda:0
```

The output directory is:

```text
runs/<dataset>/ugaunet_anchor_seed<seed>/
├── best_model.pth
├── config.json
├── metrics.json
└── final_report.txt
```

For the three seeds used in Table 4:

```bash
DATA_ROOT=data DEVICE=cuda:0 bash scripts/run_all_seeds.sh
```

## 6. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint runs/busi/ugaunet_anchor_seed42/best_model.pth \
  --dataset busi --model ugaunet --data-root data --device cuda:0
```

This writes the metrics to stdout and, with `--out`, to a JSON file.

## 7. Expected metrics

Paper Table 4, mean±std over seeds 42, 123, and 999:

| Dataset | Dice | HD95 |
|---|---:|---:|
| BUSI | .768 ± .010 | 29.6 ± 6.4 |
| BrEaST | .770 ± .063 | 23.6 ± 19.5 |
| QaTa-v2 | .794 ± .002 | 32.0 ± 0.4 |
| QaTa-v1 | .765 ± .004 | 36.4 ± 0.6 |

Small differences are expected from GPU nondeterminism, library versions, and
the exact pretrained baseline weights.

## 8. Original experiment scripts

The scripts that produced the reported numbers are archived unchanged in
[`../reproduce/original/`](../reproduce/original/).  They contain the original
author's absolute paths, for example:

```python
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
```

They are provided for provenance, not as the recommended entry point.  If you
need to rerun them, edit those paths or create a symlink/`DATA_ROOT` matching
the expected layout.  See [`CODE_MAP.md`](CODE_MAP.md).

## 9. Troubleshooting

**`FileNotFoundError` in a dataset loader.** Check `--data-root` and
`--splits-dir`.  BUSI expects `images/` and `labels/`; BrEaST expects paths in
its CSV to be relative to `BrEaST/`; QaTa-v1 expects paths relative to
`QaTa-COV19/`.

**Anchor loss is inactive.** Check that the uncertainty map directory exists
and that file names match the table in [`DATASETS.md`](DATASETS.md).  The
trainer prints a warning with the number of missing maps.

**CUDA out of memory.** Reduce `--batch-size`, or use a larger GPU.  The paper
uses batch size 8 at 256×256.

**Baseline pretrained weights fail to download.** Use `--no-pretrained` for
offline runs, or pre-populate the torchvision cache.  This changes
initialization but not the architecture.

**TransUNet / MedT at a non-256 resolution.** The public implementations used
here are configured for 256×256.  Keep `--image-size 256` for those baselines.

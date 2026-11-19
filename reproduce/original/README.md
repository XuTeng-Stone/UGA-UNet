# Original experiment scripts (archival)

This folder contains the original scripts from the paper workspace, copied
without rewriting their logic.  They are kept for provenance: the cleaned
package in `ugaunet/` and `scripts/` is the recommended way to run the method,
while these files document exactly what produced the reported numbers.

## Layout

```text
reproduce/original/
├── step1_*.py                 # MedSAM uncertainty-map generation
├── step2_dataset_*.py         # BUSI / BrEaST / QaTa loaders
├── step2_loss*.py             # CE+Dice and anchor loss
├── step3_train_*.py           # per-model / per-dataset training scripts
├── step3_train_sensitivity_busi.py
├── run_all_models_overall_seeds.py
├── generate_*.py              # paper figure scripts
├── extract_results.py
├── split_v1_to_csv.py
├── small_unet_ablation.py     # UGA-UNet architecture
├── metrics_utils.py
├── models/                    # baseline model definitions
└── networks/                  # TransUNet definitions
```

## Important: hard-coded paths

The original scripts assume the authors' machine layout:

```python
DATA_ROOT = "/data/xuten/Breast Ultrasound Datasets/BUSI"
```

To run one, either edit `DATA_ROOT` or create an equivalent directory.  The
scripts are flat: run them from this directory (or add it to `PYTHONPATH`) so
that imports such as `from step2_dataset_busi import BUSIDataset` resolve.

Example:

```bash
cd reproduce/original
python step3_train_depth_busi_proposed.py
```

The cleaned entry point is preferred:

```bash
python scripts/train.py --config configs/busi_ugaunet.yaml --data-root /path/to/data
```

## What is not included

- `*.pth` checkpoints (the original result tree was ~19 GB).
- Raw datasets.
- `figures_sets/` intermediate visualization folders.
- Dataset-specific duplicate copies of the scripts (`BUSI/BUSI_code`,
  `BUSI/BUSI_Anchor_code`, `BrEaST/BrEaST_Anchor_code`, etc.).  Those files were
  checksum-identical to the root scripts for the paper datasets.
- The `UCLM` experiment (not part of the paper's four benchmark settings).

## Figure scripts

`generate_fig5_qualitative.py`, `generate_fig6_anchor_effect.py`,
`generate_fig7_failure.py`, `generate_fig8_scatter.py`,
`generate_fig_uncertainty_diagram.py`, and `generate_visualization_sets.py`
expect checkpoints and result directories from the original workspace.  The
final rendered figures are available in `../../figures/`.

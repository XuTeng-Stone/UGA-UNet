# Code map

The original workspace was a collection of flat experiment scripts with
dataset-specific copies and hard-coded paths.  This repository keeps the exact
original scripts under [`../reproduce/original/`](../reproduce/original/) and
provides a cleaned, importable implementation under
[`../ugaunet/`](../ugaunet/).

## Original → cleaned

| Original script / file | Cleaned location | Notes |
|---|---|---|
| `small_unet_ablation.py` | `ugaunet/models/ugaunet.py` | `MobileUVitUNet_Ablation` → `UGAUNet`; same state-dict keys. |
| `step2_loss_anchor.py` | `ugaunet/losses/anchor.py`, `ugaunet/losses/segmentation.py` | Anchor loss and CE+Dice. |
| `step2_loss.py` | `ugaunet/losses/segmentation.py` | Only `DiceLoss` is used; the old unused `AnchorLoss` is not carried over. |
| `step1_busi_medsam_multitumor.py` | `scripts/generate_uncertainty.py` | Generalized to all datasets; `centroid` strategy. |
| `step1_qata_medsam_bilateral.py` | `scripts/generate_uncertainty.py --strategy bilateral` | Left/right prompt strategy. |
| `step2_dataset_busi.py` | `ugaunet/datasets/busi.py` | |
| `step2_dataset_breast.py` | `ugaunet/datasets/breast.py` | |
| `step2_dataset_qata.py` | `ugaunet/datasets/qata.py` (`QaTaV2Dataset`) | |
| `step2_dataset_qata_v1.py` | `ugaunet/datasets/qata.py` (`QaTaV1Dataset`) | |
| `metrics_utils.py` | `ugaunet/metrics.py` | |
| `models/unet.py` | `ugaunet/models/third_party/unet.py` | |
| `models/unetplus.py` | `ugaunet/models/third_party/unetplus.py` | |
| `models/AnatoNet.py` | `ugaunet/models/third_party/AnatoNet.py` | |
| `models/axialnet.py` | `ugaunet/models/third_party/axialnet.py` | MedT. |
| `models/CMUNeXt.py` | `ugaunet/models/third_party/CMUNeXt.py` | |
| `models/ukan.py` | `ugaunet/models/third_party/ukan.py` | |
| `networks/vit_seg_*.py` | `ugaunet/models/third_party/transunet/` | |
| `step3_train_depth_busi_proposed.py` | `scripts/train.py --dataset busi --model ugaunet --anchor` | Main UGA-UNet run. |
| `step3_train_depth_breast.py` | `scripts/train.py --dataset breast --model ugaunet --anchor` | |
| `step3_train_depth_qata_v1.py` | `scripts/train.py --dataset qata_v1 --model ugaunet --anchor` | |
| `step3_train_depth_qata_v2.py` | `scripts/train.py --dataset qata_v2 --model ugaunet --anchor` | |
| `step3_train_*_busi_proposed.py` | `scripts/train.py --model <baseline> --anchor` | Baseline + anchor loss. |
| `step3_train_*_baseline.py` | `scripts/train.py --model <baseline> --no-anchor` | Baseline, CE+Dice only. |
| `step3_train_sensitivity_busi.py` | `scripts/train.py` with `--rin/--rout/--q-low/--q-high` | |
| `run_all_models_overall_seeds.py` | `scripts/run_all_seeds.sh` | Simplified; the original script injected seeds by rewriting temporary files. |
| `split_v1_to_csv.py` | `scripts/prepare_qata_v1_split.py` | |
| `extract_results.py` | `results/available_runs.csv` | Result aggregation. |
| `generate_fig5_qualitative.py` | `reproduce/original/` | Figure script; expects checkpoints. |
| `generate_fig6_anchor_effect.py` | `reproduce/original/` | Figure script; trains a no-anchor model and compares. |
| `generate_fig7_failure.py` | `reproduce/original/` | Failure-case figure. |
| `generate_fig8_scatter.py` | `reproduce/original/` | Scatter plot. |
| `generate_fig_uncertainty_diagram.py` | `reproduce/original/` | Uncertainty diagram. |
| `generate_visualization_sets.py` | `reproduce/original/` | Per-sample visualization sets. |
| `models/uresnet.py`, `networks/*` (unused variants) | `reproduce/original/` | Kept for completeness. |

## What was not duplicated

The original workspace contained dataset-specific copies:

```text
BUSI/BUSI_code/                 BrEaST/BrEaST_Anchor_code/
BUSI/BUSI_Anchor_code/          QaTa/QaTa_v1_Anchor_code/
QaTa/QaTa_v2_Anchor_code/       UCLM/...
```

The Python files in these folders were checksum-identical to the corresponding
root scripts for the paper datasets.  They are not duplicated in this
repository; the single canonical originals are in `reproduce/original/`.
`UCLM` is an additional dataset that is not part of the paper's four benchmark
settings; its code and result CSVs are archived under
[`../additional/UCLM/`](../additional/UCLM/) and are not included in the paper
tables.

## Original results

The original result folders contained 19 GB of checkpoints and intermediate
images.  This repository includes:

- the small per-run CSV files (`results/raw/`, `results/available_runs.csv`),
- the paper tables (`results/paper_table4.csv`, ablations, sensitivity),
- the qualitative figures from the paper (`figures/`).

The 13 UGA-UNet checkpoints are available in the
[`v1.0.0` release](https://github.com/XuTeng-Stone/UGA-UNet/releases/tag/v1.0.0).
The much larger baseline checkpoints (TransUNet, AnatoNet, etc.), raw datasets,
and the remaining experiment outputs are not included.  Keep those in your
local backup or an external artifact store.

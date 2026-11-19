# Results

This directory contains the reported paper tables and the per-run CSVs that were
available in the original experiment workspace.

## Files

| File | Contents |
|---|---|
| `paper_table4.csv` | Main comparison over three seeds (paper Table 4). Dice and HD95 are mean±std. |
| `ablation_bottleneck.csv` | Bottleneck design / depth sweep on BUSI without the anchor loss (paper Table 2). |
| `ablation_anchor.csv` | Anchor-loss ablation on BUSI (paper Table 3). |
| `sensitivity.csv` | Anchor hyperparameter sensitivity on BUSI (ring radii and quantiles). |
| `sensitivity_logs/` | Training logs for the four sensitivity runs. |
| `anchor_hyperparameter_sweep.csv` | Anchor weight/margin/background-weight sweep on BUSI. |
| `bottleneck_component_ablation.csv` | Conv / local / global / fusion bottleneck comparison. |
| `available_runs.csv` | Every per-run CSV found in the original workspace, normalized to `dataset, model, setting, seed, params, FLOPs, mIoU, Dice, ASSD, HD95`. |
| `raw/` | Raw per-run CSV files, copied unchanged from the original result folders. |

## Notes on provenance

- `paper_table4.csv`, `ablation_bottleneck.csv`, and `ablation_anchor.csv` are
  transcribed from the manuscript.  They are the authoritative numbers.
- `available_runs.csv` is a convenience index over the files that were present
  in the experiment workspace.  It is **not** a complete rerun matrix: some
  baseline/dataset combinations (notably BrEaST and QaTa baselines without the
  anchor loss) were stored in other locations or are no longer available.
- The `Params` column for U-KAN in the original CSVs comes from `thop` and
  differs from a direct `sum(p.numel())` count.  The paper's 3.9 M for UGA-UNet
  is the direct count (3,913,474 parameters).
- `setting` values in `available_runs.csv`:
  - `segmentation-loss-only` — CE + Dice, no anchor loss.
  - `anchor` — baseline trained with the uncertainty-guided anchor loss.
  - `depth-ablation (no anchor)` / `depth-ablation (anchor)` — UGA-UNet LGB
    depth/architecture sweeps.
  - `hyperparameter-ablation` — anchor hyperparameter sweep.
  - `sensitivity` — ring/quantile sensitivity summary.

## Reproducing the tables

See [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).  The main three-seed runs
are launched with:

```bash
DATA_ROOT=/path/to/data DEVICE=cuda:0 bash scripts/run_all_seeds.sh
```

The archived original scripts used to produce the numbers are in
[`../reproduce/original/`](../reproduce/original/).

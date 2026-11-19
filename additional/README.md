# Additional / archival material

This folder preserves experiments that are **not part of the BIBM 2026 paper's
four benchmark settings**.  It is included for completeness and reproducibility
of the broader project, not as part of the official paper release.

## Contents

| Folder | Contents | Notes |
|---|---|---|
| `UCLM/code/` | BUS-UCLM training scripts (baseline and anchor variants) | The UCLM dataset is not one of the paper's four benchmarks. |
| `UCLM/results/` | UCLM result CSVs and `final_report.txt` files | No checkpoints. |
| `legacy_code/` | Earlier BUSI/BrEaST experiment scripts from the original workspace | Superseded by the cleaned `ugaunet/` package; kept for provenance. |

## Running the UCLM scripts

The UCLM scripts use the same shared modules as the original experiments.  Run
them from `reproduce/original/` (or put it on `PYTHONPATH`) and edit the
hard-coded `DATA_ROOT` paths in the scripts:

```bash
cd reproduce/original
PYTHONPATH=. python ../../additional/UCLM/code/step3_train_depth_uclm.py
```

As with the archived originals, these scripts expect the author's original
directory layout.  They are not maintained as part of the cleaned API.

## What is not included

- `*.pth` checkpoints.
- Raw datasets.
- The `UCLM_results`/`UCLM_Anchor_results` intermediate images and `.npy` files.

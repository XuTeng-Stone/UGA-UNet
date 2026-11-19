# Paper figures

Rendered qualitative figures from the manuscript.  The file prefixes follow the
internal experiment numbering, which differs from the final manuscript
numbering:

| File | Manuscript figure | Contents |
|---|---|---|
| `fig5_qualitative.png` | Fig. 4 | Qualitative segmentation results on the four datasets. |
| `fig6_anchor_effect.png/.pdf` | Fig. 5 | Effect of the anchor loss on boundary delineation. |
| `fig7_failure.png/.pdf` | Fig. 6 | Representative failure cases on BUSI. |

These are the final rendered artifacts.  The scripts that generated them are in
[`../reproduce/original/`](../reproduce/original/) and require the original
checkpoints/result folders.


## Per-sample visualization sets

`sets/` contains the intermediate visualizations for three BUSI cases:

- `01_input_image.png`
- `02_ground_truth.png`
- `03_pred_k01.png` … `03_pred_k12.png` — the 12 perturbed MedSAM prompts
- `04_uncertainty_U.png` — raw variance map
- `05_uncertainty_Ubar.npy` / `.png` — normalized uncertainty map
- `06_segmentation_Z.npy` / `.png` — segmentation logits/probabilities
- `06b_final_segmentation.png`, `06c_pred_vs_gt.png`

These are useful for regenerating the uncertainty/prompt-sensitivity diagrams.

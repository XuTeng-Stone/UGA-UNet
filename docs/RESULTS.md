# Results

The canonical numbers are in the manuscript.  This page summarizes the tables
and points to the machine-readable files.

## Main comparison (paper Table 4)

Dice / HD95, mean±std over three seeds.  Best overall in bold in the paper;
best among lightweight models underlined.

See [`../results/paper_table4.csv`](../results/paper_table4.csv) for the full
machine-readable table.

| Model | Params | BUSI | BrEaST | QaTa-v2 | QaTa-v1 |
|---|---:|---:|---:|---:|---:|
| MedT | 1.4 M | .613±.010 / 52.0 | .630±.131 / 30.5 | .753±.002 / 40.5 | .707±.003 / 44.0 |
| U-KAN | 4.2 M | .655±.004 / 43.8 | .653±.166 / 37.8 | .782±.001 / 34.9 | .753±.003 / 37.0 |
| CMUNeXt-L | 8.3 M | .756±.018 / 29.8 | .756±.066 / 15.2 | .787±.003 / 33.3 | .756±.003 / 37.0 |
| **UGA-UNet** | **3.9 M** | **.768±.010 / 29.6** | **.770±.063 / 23.6** | **.794±.002 / 32.0** | **.765±.004 / 36.4** |
| U-Net | 34.5 M | .734±.009 / 33.0 | .779±.061 / 15.5 | .794±.001 / 32.6 | .766±.001 / 36.3 |
| UNet++ | 26.9 M | .770±.005 / 28.0 | .765±.048 / 15.5 | .783±.001 / 33.4 | .754±.003 / 35.3 |
| DeepLabV3+ | 39.6 M | .780±.003 / 29.2 | .694±.143 / 20.7 | .800±.002 / 30.5 | .776±.002 / 31.5 |
| TransUNet | 93.2 M | .773±.009 / 31.1 | .775±.015 / 14.5 | .796±.000 / 31.7 | .765±.001 / 35.8 |
| AnatoNet | 116.4 M | .764±.042 / 29.0 | .806±.015 / 10.7 | .769±.012 / 34.5 | .760±.015 / 35.1 |

UGA-UNet has the best Dice among models under 10 M parameters on all four
datasets, at 3.9 M parameters.

## Bottleneck ablation (paper Table 2, BUSI, no anchor)

| Bottleneck | k | Dice | HD95 | Params (M) | FLOPs (G) |
|---|---:|---:|---:|---:|---:|
| Conv only | 1 | .685 | 51.99 | 3.88 | 13.52 |
| Local only | 1 | .697 | 40.62 | 2.78 | 12.39 |
| Global only | 1 | .742 | 38.22 | 3.23 | 12.44 |
| Local+Global | 1 | .749 | 33.49 | 3.31 | 12.52 |
| Local+Global | 2 | .761 | 32.42 | 3.91 | 12.74 |
| Local+Global | 3 | .765 | 30.62 | 4.52 | 12.96 |
| Local+Global | 4 | .774 | 29.75 | 5.12 | 13.17 |

CSV: [`../results/ablation_bottleneck.csv`](../results/ablation_bottleneck.csv).

## Anchor-loss ablation (paper Table 3, BUSI)

| Configuration | Dice | HD95 |
|---|---:|---:|
| LGB (k=2) | .761 | 32.42 |
| LGB (k=2) + Anchor Loss | .775 | 22.23 |

CSV: [`../results/ablation_anchor.csv`](../results/ablation_anchor.csv).

## Hyperparameter sensitivity (BUSI)

| Setting | r_in | r_out | q_low | q_high | mIoU | Dice | ASSD | HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wider_ring | 4 | 10 | .2 | .8 | .6806 | .7755 | 10.9938 | 31.6359 |
| less_selective | 3 | 8 | .3 | .7 | .6795 | .7721 | 11.0956 | 32.9614 |
| narrow_ring | 2 | 6 | .2 | .8 | .6805 | .7711 | 9.8340 | 29.6749 |
| more_selective | 3 | 8 | .1 | .9 | .6747 | .7616 | 10.2012 | 27.5037 |

CSV: [`../results/sensitivity.csv`](../results/sensitivity.csv).

## Available per-run results

[`../results/available_runs.csv`](../results/available_runs.csv) indexes every
small result CSV found in the original workspace (159 runs), with the dataset,
model, setting, seed, and metrics.  Raw files are in
[`../results/raw/`](../results/raw/).

The original workspace did not contain a complete matrix for all
baseline × dataset × seed combinations.  In particular, BrEaST and QaTa
baseline rows without the anchor loss were not all present.  The paper tables
remain the authoritative source.

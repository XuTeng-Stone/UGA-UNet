# Method

This document summarizes the method and maps the equations to the cleaned
implementation.  The paper is the authoritative description.

## 1. Overview

UGA-UNet has two training-time components:

1. **Offline uncertainty estimation.** A frozen MedSAM model is prompted
   multiple times around each ground-truth lesion.  The variance of the
   resulting masks gives a pixel-wise uncertainty map `U`.
2. **Uncertainty-guided feature anchoring.** UGA-UNet predicts logits and a
   full-resolution decoder feature map.  The logits are supervised by
   cross-entropy + Dice; the features are additionally supervised by the
   anchor loss on high-uncertainty pixels.

At inference, only UGA-UNet is used.

## 2. Uncertainty map generation

For each connected foreground component, the centroid is computed and `N = 12`
point prompts are sampled by jittering the centroid within `±5` pixels.  Each
prompt produces a binary MedSAM prediction `Ŷ(j)`.  The uncertainty is

```
U(x) = Var(Ŷ(1)(x), ..., Ŷ(N)(x)).
```

`U` is smoothed with a `3 × 3` Gaussian filter and min–max normalized to
`[0, 1]`.  The maps are stored as 8-bit PNGs and loaded only during training.

Implementation: [`scripts/generate_uncertainty.py`](../scripts/generate_uncertainty.py).
For QaTa, the default prompt strategy splits each mask into left/right halves
before computing component centroids, following the original experiments
(`--strategy bilateral`).

## 3. Stable / ambiguous partition

Let `Y` be the ground-truth mask.  The foreground region is `Ω_fg = {Y = 1}`.
The background region is the local ring

```
Ω_bg = dilate(Y, r_out) \ dilate(Y, r_in),     r_in = 3, r_out = 8.
```

For each region `c ∈ {fg, bg}`, let `U_c` be the uncertainty values in that
region.  Pixels are partitioned by region-specific quantiles:

```
Ω_stable(c)    = {x ∈ Ω_c | U(x) ≤ q_low(c)}     (low uncertainty)
Ω_ambiguous(c) = {x ∈ Ω_c | U(x) ≥ q_high(c)}    (high uncertainty)
```

with `q_low = 0.2` and `q_high = 0.8` for both regions.  The original code also
uses a confidence map `W = 1 - U` for stable foreground selection; both
formulations select the same low-uncertainty pixels.

Implementation: [`ugaunet/losses/anchor.py`](../ugaunet/losses/anchor.py),
`make_outer_ring_from_gt` and `anchor_loss_from_maps`.

## 4. Prototype anchoring

Features are L2-normalized per pixel.  The class prototype is the normalized
mean of stable-pixel features, with stop-gradient:

```
p_c = sg( normalize( mean_{x ∈ Ω_stable(c)} f̃(x) ) ).
```

For each ambiguous pixel, the loss encourages its feature to be closer to its
own prototype than to the opposite one:

```
L_anchor = Σ_c λ_c / |Ω_ambiguous(c)| Σ_{x ∈ Ω_ambiguous(c)}
           softplus( d(f̃(x), p_c) - d(f̃(x), p_opposite) + m ),
```

where `d` is cosine distance, `m = 0.3`, and `λ_fg = 1`, `λ_bg = 0.25`.
The `softplus` is the smooth margin relaxation.

The total objective is

```
L = L_CE + L_Dice + α(t) L_anchor.
```

`α(t) = 0` for the first 10% of epochs and then increases linearly to
`α_max = 0.1`.  In the original code, `α` reaches `α_max` at the final epoch
(the ramp denominator is `epochs - warmup_epochs`).

Implementation: `anchor_loss_from_maps` returns the per-batch anchor loss; the
weighting and warm-up are in [`ugaunet/engine.py`](../ugaunet/engine.py),
`anchor_weight` and `train`.

## 5. Architecture

UGA-UNet is a U-Net-style encoder–decoder:

- Encoder channels: `32, 64, 128, 256`.
  - Stage 1: two `3 × 3` convolutions, stride 1.
  - Stages 2–4: `3 × 3` stride-2 downsampling followed by two convolutions.
- Bottleneck: `k = 2` stacked Local–Global Bottleneck (LGB) blocks.
- Decoder: three transposed-convolution upsampling blocks with skip
  connections to `e3`, `e2`, `e1`.
- Output: `1 × 1` convolution to two logits.
- The full-resolution decoder feature `d1` is returned for the anchor loss.

Each LGB block has:

- **Local branch:** depthwise separable `7 × 7` convolution, BN, GELU.
- **Global branch:** adaptive average pooling (`r = 2`), flatten to tokens,
  4-head self-attention, linear projection, MLP, reshape, bilinear upsample.
- **Fusion:** residual addition `z_out = z + z_local + z_global`.

Implementation: [`ugaunet/models/ugaunet.py`](../ugaunet/models/ugaunet.py).
The default model has **3,913,474** parameters and **12.74 GFLOPs** at
`256 × 256` (thop).

## 6. Hyperparameters

| Parameter | Value |
|---|---|
| Input size | 256 × 256 |
| Epochs | 100 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Augmentation | horizontal flip p=0.5, rotation ±30° p=0.5 |
| `α_max` | 0.1 |
| Warm-up | first 10% of epochs |
| Margin `m` | 0.3 |
| `λ_bg` | 0.25 |
| `q_low`, `q_high` | 0.2, 0.8 |
| `r_in`, `r_out` | 3, 8 |
| MedSAM prompts `N` | 12 |
| Prompt jitter | ±5 pixels |
| Minimum prototype pixels | 10 (training code) |

## 7. Implementation notes and caveats

- **Uncertainty maps are uncertainty, not confidence.** The files are named
  `*_weight.png` in the original code, but high values mean *high uncertainty*.
  The datasets return them as `uncertainty` tensors; the anchor loss uses
  `1 - U` only to select stable foreground pixels.
- **Missing maps.** The cleaned loaders treat a missing map as zero
  uncertainty (no ambiguous pixels), so anchor training does not get a spurious
  signal.  A warning is printed when maps are missing.  The original loaders
  used an all-ones map, which had the opposite effect; do not rely on that
  behavior.
- **Batch aggregation.** The original evaluation averages Dice/IoU per image
  and averages ASSD/HD95 per batch.  The cleaned `evaluate` follows the same
  convention so numbers remain comparable.
- **Parameter counts.** The cleaned code prints a direct
  `sum(p.numel())` count.  The original result CSVs use thop, which can
  undercount models with custom operations (notably U-KAN).  The paper's
  UGA-UNet 3.9 M is the direct count.
- **Baseline features for anchoring.** For baselines, the anchor loss uses the
  input to the last `nn.Conv2d` as the feature map, matching the original
  `UniversalAnchorWrapper`.  For UGA-UNet, the model returns `d1` directly.

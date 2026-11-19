"""UGA-UNet backbone and Local-Global Bottleneck (LGB).

Architecture (paper Sec. 3.4 / Fig. 2):

    input 3x256x256
      -> four encoder stages (32, 64, 128, 256 channels)
      -> k stacked Local-Global Bottleneck blocks at the deepest feature
      -> three skip-connected transposed-convolution decoder stages
      -> 1x1 convolution for logits

The full-resolution decoder feature ``d1`` is returned alongside the logits;
it is the feature map used by the Uncertainty-Guided Anchor Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise k x k convolution followed by a pointwise 1 x 1 convolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """Two 3 x 3 convolutions with BatchNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Stride-2 convolutional downsampling followed by a ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        return self.conv(x)


class UpBlock(nn.Module):
    """Transposed-convolution upsampling, skip concatenation, and ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_dec = self.up(x_dec)
        diff_y = x_skip.size(2) - x_dec.size(2)
        diff_x = x_skip.size(3) - x_dec.size(3)
        x_dec = F.pad(
            x_dec,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x)


class LocalGlobalBlock(nn.Module):
    """Local-Global Bottleneck block.

    Parameters
    ----------
    channels:
        Number of input/output channels.
    mode:
        One of ``"fusion"`` (paper default), ``"local"``, ``"global"``, or
        ``"conv"``.  The latter three are used only by the bottleneck ablation.
    kernel_size:
        Kernel size of the depthwise separable local branch (paper: 7).
    pool_ratio:
        Spatial reduction factor of the global branch (paper: 2).
    num_heads:
        Number of self-attention heads (paper: 4).
    mlp_ratio:
        Hidden dimension ratio of the token MLP.
    """

    def __init__(
        self,
        channels: int,
        mode: str = "fusion",
        kernel_size: int = 7,
        pool_ratio: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in {"fusion", "local", "global"}:
            raise ValueError(f"Unsupported LGB mode: {mode!r}")
        self.mode = mode
        self.pool_ratio = pool_ratio

        # Local branch: depthwise separable 7x7 convolution.
        if mode in {"local", "fusion"}:
            self.local_conv = DepthwiseSeparableConv(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        # Global branch: pooled multi-head self-attention + MLP.
        if mode in {"global", "fusion"}:
            self.attn_dim = channels
            self.num_heads = num_heads
            self.scale = (channels // num_heads) ** -0.5
            self.qkv = nn.Linear(channels, channels * 3, bias=True)
            self.proj = nn.Linear(channels, channels)
            hidden_dim = int(channels * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, channels),
            )
            # Kept for compatibility with the original checkpoints.  The
            # original implementation also constructs this layer but does not
            # apply it in the forward pass.
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_out = x

        if self.mode in {"local", "fusion"}:
            x_out = x_out + self.local_conv(x)

        if self.mode in {"global", "fusion"}:
            pr = self.pool_ratio
            hp, wp = max(1, h // pr), max(1, w // pr)
            x_pooled = F.adaptive_avg_pool2d(x, output_size=(hp, wp))
            tokens = x_pooled.flatten(2).transpose(1, 2)  # B, N, C
            n = tokens.shape[1]

            qkv = self.qkv(tokens).reshape(b, n, 3, self.num_heads, c // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(b, n, c)

            x_tokens = self.proj(x_attn)
            x_tokens = x_tokens + self.mlp(self.norm2(x_tokens))
            x_global = x_tokens.transpose(1, 2).reshape(b, c, hp, wp)
            x_global_up = F.interpolate(
                x_global, size=(h, w), mode="bilinear", align_corners=False
            )
            x_out = x_out + x_global_up

        return x_out


class UGAUNet(nn.Module):
    """Compact encoder-decoder with stacked Local-Global Bottlenecks.

    The default configuration (``base_ch=32``, ``num_bottleneck_blocks=2``,
    ``mode="fusion"``) has 3,913,474 parameters, matching the 3.9M reported in
    the paper.

    Forward returns ``(logits, decoder_features)`` where ``decoder_features`` is
    the full-resolution ``d1`` feature map used by the anchor loss.
    """

    def __init__(
        self,
        mode: str = "fusion",
        in_channels: int = 3,
        n_classes: int = 2,
        base_ch: int = 32,
        num_bottleneck_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.mode = mode
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)

        blocks: list[nn.Module] = []
        for _ in range(num_bottleneck_blocks):
            if mode == "conv":
                # Ablation variant: two standard 3x3 convolutions.
                blocks.append(ConvBlock(c4, c4))
            else:
                blocks.append(LocalGlobalBlock(channels=c4, mode=mode))
        self.bottleneck = nn.Sequential(*blocks)

        self.up3 = UpBlock(c4, c3)
        self.up2 = UpBlock(c3, c2)
        self.up1 = UpBlock(c2, c1)
        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out_conv(d1), d1


# Backward-compatible name used by the original experiment scripts.
MobileUVitUNet_Ablation = UGAUNet

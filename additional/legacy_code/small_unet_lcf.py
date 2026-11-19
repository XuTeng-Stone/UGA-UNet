import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Basic conv blocks
# =========================================================
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable conv: depthwise (k x k) + pointwise (1 x 1).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """Two standard 3x3 convs with BN + GELU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsample (stride-2 conv) + ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.act(self.bn(self.down(x)))
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Up-convolution + skip connection + ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x_dec, x_skip):
        x_dec = self.up(x_dec)
        # Align shapes
        diffY = x_skip.size(2) - x_dec.size(2)
        diffX = x_skip.size(3) - x_dec.size(3)
        x_dec = F.pad(
            x_dec,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x_skip, x_dec], dim=1)
        x = self.conv(x)
        return x


# =========================================================
# Mobile U-ViT-ish LocalGlobal Block
# =========================================================
class LocalGlobalBlock(nn.Module):
    """
    Local+Global fusion block (U-ViT-inspired).
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        pool_ratio: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.channels = channels

        # Local path
        self.local_conv = DepthwiseSeparableConv(
            in_ch=channels,
            out_ch=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # Global path components
        self.attn_dim = channels
        self.qkv = nn.Linear(self.attn_dim, self.attn_dim * 3, bias=True)
        self.num_heads = num_heads
        head_dim = self.attn_dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(self.attn_dim, self.attn_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        hidden_dim = int(self.attn_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.attn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, self.attn_dim),
            nn.Dropout(proj_dropout),
        )

        self.norm1 = nn.LayerNorm(self.attn_dim)
        self.norm2 = nn.LayerNorm(self.attn_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) Local branch
        x_local = self.local_conv(x)

        # 2) Global branch
        pr = self.pool_ratio
        Hp = max(1, H // pr)
        Wp = max(1, W // pr)
        x_pooled = F.adaptive_avg_pool2d(x, output_size=(Hp, Wp))

        tokens = x_pooled.flatten(2).transpose(1, 2)
        N = tokens.shape[1]

        qkv = self.qkv(tokens)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.attn_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, self.attn_dim)

        x_tokens = self.proj(x_attn)
        x_tokens = self.proj_drop(x_tokens)

        x_tokens = x_tokens + self.mlp(self.norm2(x_tokens))

        x_global = x_tokens.transpose(1, 2).reshape(B, self.attn_dim, Hp, Wp)
        x_global_up = F.interpolate(x_global, size=(H, W), mode="bilinear", align_corners=False)

        # 3) Fuse
        x_out = x + x_local + x_global_up
        return x_out


# =========================================================
# MobileUVitUNet (Fixed: Returns both logits and features)
# =========================================================
class MobileUVitUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        base_ch: int = 32,
        num_bottleneck_blocks: int = 2,
        bottleneck_kernel_size: int = 7,
        bottleneck_heads: int = 4,
        pool_ratio: int = 2,
    ):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)

        # Bottleneck
        blocks = []
        for _ in range(num_bottleneck_blocks):
            blocks.append(
                LocalGlobalBlock(
                    channels=c4,
                    kernel_size=bottleneck_kernel_size,
                    pool_ratio=pool_ratio,
                    num_heads=bottleneck_heads,
                    mlp_ratio=2.0,
                    attn_dropout=0.0,
                    proj_dropout=0.0,
                )
            )
        self.bottleneck = nn.Sequential(*blocks)

        # Decoder
        self.up3 = UpBlock(c4, c3)
        self.up2 = UpBlock(c3, c2)
        self.up1 = UpBlock(c2, c1)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)  # [Features for Anchor Loss]

        out = self.out_conv(d1)
        
        # [Fix] Return both logits and features
        return out, d1
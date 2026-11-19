import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Basic conv blocks (保持不变)
# =========================================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None: padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class ConvBlock(nn.Module):
    """Two standard 3x3 convs with BN + GELU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.act(self.bn(self.down(x)))
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x_dec, x_skip):
        x_dec = self.up(x_dec)
        diffY = x_skip.size(2) - x_dec.size(2)
        diffX = x_skip.size(3) - x_dec.size(3)
        x_dec = F.pad(x_dec, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x)

# =========================================================
# LocalGlobal Block (不变)
# =========================================================
class LocalGlobalBlock(nn.Module):
    def __init__(self, channels, mode='fusion', kernel_size=7, pool_ratio=2, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.mode = mode 
        self.pool_ratio = pool_ratio

        # Local path
        if self.mode in ['local', 'fusion']:
            self.local_conv = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)

        # Global path
        if self.mode in ['global', 'fusion']:
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
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_out = x

        if self.mode in ['local', 'fusion']:
            x_local = self.local_conv(x)
            x_out = x_out + x_local

        if self.mode in ['global', 'fusion']:
            pr = self.pool_ratio
            Hp, Wp = max(1, H // pr), max(1, W // pr)
            x_pooled = F.adaptive_avg_pool2d(x, output_size=(Hp, Wp))
            tokens = x_pooled.flatten(2).transpose(1, 2)
            N = tokens.shape[1]
            qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x_tokens = self.proj(x_attn)
            x_tokens = x_tokens + self.mlp(self.norm2(x_tokens))
            x_global = x_tokens.transpose(1, 2).reshape(B, C, Hp, Wp)
            x_global_up = F.interpolate(x_global, size=(H, W), mode="bilinear", align_corners=False)
            x_out = x_out + x_global_up

        return x_out

# =========================================================
# Updated Main Model
# =========================================================
class MobileUVitUNet_Ablation(nn.Module):
    def __init__(self, mode='fusion', in_channels=3, n_classes=1, base_ch=32, num_bottleneck_blocks=1):
        super().__init__()
        self.mode = mode
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)

        blocks = []
        for _ in range(num_bottleneck_blocks):
            if self.mode == 'conv':
                # [新增] 纯卷积 Bottleneck: 使用标准的 ConvBlock (2个 3x3 Conv)
                # 保持通道数 c4 -> c4
                blocks.append(ConvBlock(in_ch=c4, out_ch=c4))
            else:
                # 原有的 Local/Global/Fusion
                blocks.append(LocalGlobalBlock(channels=c4, mode=mode))
        
        self.bottleneck = nn.Sequential(*blocks)

        self.up3 = UpBlock(c4, c3)
        self.up2 = UpBlock(c3, c2)
        self.up1 = UpBlock(c2, c1)
        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        
        return self.out_conv(d1), d1
# dual_swin_unet_dualblock.py
# -----------------------------------------------------------
# Small U-Net + Dual Swin Bottleneck using YOUR DualSwinTransformerBlock
#
# - Encoder: lightweight U-Net encoder (no ResNet)
# - Bottleneck: DualSwinBottleneck
#       * learns a soft lesion mask M from bottleneck features F
#       * F_les = F * M, F_ctx = F * (1 - M)
#       * passes each branch through your DualSwinTransformerBlock
#       * fuses lesion + context back to C channels
# - Decoder: lightweight U-Net decoder
#
# Base channels and window size are configurable.
# -----------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utility: DropPath (stochastic depth)
# =========================================================
class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


# =========================================================
# Swin window ops
# =========================================================
def window_partition(x, window_size: int):
    """
    x: (B, H, W, C)
    return: (B*nW, ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
              .view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    windows: (B*nW, ws, ws, C)
    return: (B, H, W, C)
    """
    L = (H // window_size) * (W // window_size)
    B = windows.shape[0] // L
    x = windows.view(
        B,
        H // window_size, W // window_size,
        window_size, window_size, -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
         .view(B, H, W, -1)
    return x


# =========================================================
# MLP (FFN) used by the Transformer blocks
# =========================================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_features, out_features)

        # optional second dropout after fc2 is also embedded in self.drop
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# =========================================================
# Dual-Branch Window Attention (your code)
# =========================================================
class DualBranchWindowAttention(nn.Module):
    """
    Local tokens (ws×ws) + global tokens (L windows per image).
    Relative bias only for local-local. Includes output proj.
    """
    def __init__(self, dim, window_size, num_heads, dropout=0.0, beta=1.5, gamma=1.5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.beta = beta
        self.gamma = gamma

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.qkv_global = nn.Linear(dim, dim * 3, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # relative bias for local-local
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x_local, x_global, use_global: bool = True):
        """
        x_local:  (B_local, N_local, C)
        x_global: (B_local, L, C)
        return:   (B_local, N_local+L, C) if use_global else (B_local, N_local, C)
        """
        B_local, N_local, C = x_local.shape

        # local qkv
        qkv_local = self.qkv(x_local)
        qkv_local = qkv_local.reshape(B_local, N_local, 3, self.num_heads, C // self.num_heads)
        qkv_local = qkv_local.permute(2, 0, 3, 1, 4)
        q_local, k_local, v_local = qkv_local[0], qkv_local[1], qkv_local[2]

        if not use_global:
            q = q_local * self.scale
            attn = q @ k_local.transpose(-2, -1)  # (B_local, heads, N, N)
            relative_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N_local, N_local, -1).permute(2, 0, 1)
            attn = attn + relative_bias.unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x_out = attn @ v_local
            x_out = x_out.transpose(1, 2).reshape(B_local, N_local, C)
            x_out = self.proj_drop(self.proj(x_out))
            return x_out

        # global qkv
        _, L, Cg = x_global.shape
        assert Cg == C
        qkv_global = self.qkv_global(x_global)
        qkv_global = qkv_global.reshape(B_local, L, 3, self.num_heads, C // self.num_heads)
        qkv_global = qkv_global.permute(2, 0, 3, 1, 4)
        q_global, k_global, v_global = qkv_global[0], qkv_global[1], qkv_global[2]

        # concat tokens
        q = torch.cat([q_local, q_global], dim=2) * self.scale
        k = torch.cat([k_local, k_global], dim=2)
        v = torch.cat([v_local, v_global], dim=2)

        attn = q @ k.transpose(-2, -1)  # (B_local, heads, N+L, N+L)

        total_tokens = N_local + L
        bias = torch.zeros((self.num_heads, total_tokens, total_tokens), device=x_local.device)

        # relative bias for local-local
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N_local, N_local, -1).permute(2, 0, 1)
        bias[:, :N_local, :N_local] = relative_bias

        # simple structured bias among global tokens
        pos_y = torch.arange(0, L, device=x_local.device).unsqueeze(0)
        G_L = self.beta - self.gamma * torch.abs(pos_y - pos_y.T)  # (L, L)
        G_L = (G_L / G_L.abs().max()) * 0.04
        bias[:, N_local:, N_local:] = G_L

        attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x_out = attn @ v
        x_out = x_out.transpose(1, 2).reshape(B_local, total_tokens, C)
        x_out = self.proj_drop(self.proj(x_out))
        return x_out


# =========================================================
# Dynamic SE merge (token fusion) – your code
# =========================================================
class DynamicSEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)

    def forward(self, x):
        b, t, f = x.size()
        y = x.mean(dim=1)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, 1, f)
        return x * y


class DynamicMergeAndSEReduce(nn.Module):
    def __init__(self, input_dim, output_dim, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
        )
        self.se_block = DynamicSEBlock(input_dim=input_dim, reduction=reduction)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.mlp(x)
        x = self.se_block(x)
        return self.proj(x)


# =========================================================
# Dual-Swin Transformer Block (your code, slightly wrapped)
# =========================================================
class DualSwinTransformerBlock(nn.Module):
    """
    Dual-branch Swin block with global token aggregation and SE fusion.
    Uses GroupNorm in global_fuse for small-batch stability.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., dropout=0.0, attn_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # nominal (H,W), not strictly used
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = DualBranchWindowAttention(dim, window_size=window_size,
                                               num_heads=num_heads, dropout=attn_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

        self.merge_q = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(dim // 4, 1, bias=True),
        )

        self.global_fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.deconv = nn.ConvTranspose2d(dim, dim, kernel_size=window_size, stride=window_size)
        self.merge_block = DynamicMergeAndSEReduce(input_dim=dim * 2, output_dim=dim, reduction=16)

    def forward(self, x):
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.window_size, W // self.window_size
        L = grid_h * grid_w

        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)     # (B, H*W, C)
        x_norm = self.norm1(x_flat).view(B, H, W, C)

        do_shift = (self.shift_size > 0) and (min(H, W) > self.window_size)
        shifted_x = torch.roll(x_norm,
                               shifts=(-self.shift_size, -self.shift_size),
                               dims=(1, 2)) if do_shift else x_norm

        x_windows = window_partition(shifted_x, self.window_size)      # (B*L, ws, ws, C)
        B_local = x_windows.shape[0]
        N_local = self.window_size * self.window_size
        x_windows = x_windows.view(B_local, N_local, C)
        assert B_local == B * L, f"B_local {B_local} must equal B*L {B*L}"

        # global tokens built per image and repeated for each window
        x_global = self.aggregate_global_tokens(x_windows, B)          # (B_local, L, C)

        # attention over local+global
        attn_out  = self.attn(x_windows, x_global)                     # (B_local, N_local+L, C)
        local_out = attn_out[:, :N_local, :]
        global_out = attn_out[:, N_local:, :]                          # (B_local, L, C)

        # weight and fuse global tokens per image
        global_out = global_out.view(B, L, L, C)
        m = global_out.mean(dim=2)                                     # (B, L, C)
        w = self.merge_q(m)                                            # (B, L, 1)
        w = F.softmax(w, dim=1).unsqueeze(2)                           # (B, L, 1, 1)
        merged = (global_out * w).sum(dim=1)                           # (B, L, C)

        merged = merged.view(B, grid_h, grid_w, C).permute(0, 3, 1, 2).contiguous()
        merged = self.global_fuse(merged)
        global_map = self.deconv(merged)                                # (B, C, H, W)

        x_local = window_reverse(local_out, self.window_size, H, W)     # (B, H, W, C)
        x_local = x_local.view(B, H * W, C)
        global_flat = global_map.flatten(2).transpose(1, 2)

        fused = self.merge_block(x_local, global_flat)                  # (B, H*W, C)
        shortcut = shortcut.flatten(2).transpose(1, 2)
        x_out = shortcut + self.drop_path(fused)
        x_out = x_out + self.drop_path(self.mlp(self.norm2(x_out)))
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        return x_out

    @staticmethod
    def aggregate_global_tokens(x_windows, B):
        B_local, N_local, C = x_windows.shape
        L = B_local // B
        per_win = x_windows.mean(dim=1)           # (B_local, C)
        tokens_per_img = per_win.view(B, L, C)    # (B, L, C)
        x_global = tokens_per_img.repeat_interleave(L, dim=0)
        return x_global


# =========================================================
# Pad-to-window wrapper so H,W need not be multiple of ws
# =========================================================
class PadToWindow(nn.Module):
    def __init__(self, block: nn.Module, window_size: int):
        super().__init__()
        self.block = block
        self.ws = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (self.ws - H % self.ws) % self.ws
        pw = (self.ws - W % self.ws) % self.ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        y = self.block(x)
        if ph or pw:
            y = y[:, :, :H, :W]
        return y


# =========================================================
# U-Net basic blocks
# =========================================================
class ConvBlock(nn.Module):
    """Two 3x3 convs with BN + GELU."""
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
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x_dec, x_skip):
        x_dec = self.up(x_dec)
        # Align shapes (for odd sizes)
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
# Dual Swin Bottleneck using YOUR DualSwinTransformerBlock
# =========================================================
class DualSwinBottleneck(nn.Module):
    """
    Dual-branch Swin bottleneck:

    Input:
        F: (B, C, H, W)

    Internally:
        1) Predict soft lesion mask M from F
        2) F_les = F * M
           F_ctx = F * (1 - M)
        3) Apply DualSwinTransformerBlock to each branch
        4) Fuse: concat + 1x1 conv -> (B, C, H, W)

    Returns:
        F_fused:  (B, C, H, W)  -- to decoder
        F_les_o:  (B, C, H, W)  -- lesion branch features
        F_ctx_o:  (B, C, H, W)  -- context branch features
        M:        (B, 1, H, W)  -- learned soft mask
    """
    def __init__(
        self,
        channels: int,
        window_size: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.ws = window_size

        # Soft lesion mask predictor
        mid = max(channels // 4, 16)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),  # -> (B,1,H,W)
        )

        # One DualSwinTransformerBlock per branch, wrapped with PadToWindow
        swin_les = DualSwinTransformerBlock(
            dim=channels,
            input_resolution=(0, 0),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path_prob,
        )
        swin_ctx = DualSwinTransformerBlock(
            dim=channels,
            input_resolution=(0, 0),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path_prob,
        )

        self.swin_les = PadToWindow(swin_les, window_size=window_size)
        self.swin_ctx = PadToWindow(swin_ctx, window_size=window_size)

        # Fuse lesion+context: 2C -> C
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, F):
        B, C, H, W = F.shape

        # 1) Predict soft lesion mask
        M = torch.sigmoid(self.mask_conv(F))         # (B,1,H,W)
        M_ctx = 1.0 - M

        # 2) Split into lesion / context
        F_les = F * M
        F_ctx = F * M_ctx

        # 3) Apply Dual Swin to each branch
        F_les_out = self.swin_les(F_les)
        F_ctx_out = self.swin_ctx(F_ctx)

        # 4) Fuse lesion + context
        F_cat   = torch.cat([F_les_out, F_ctx_out], dim=1)  # (B, 2C, H, W)
        F_fused = self.fuse(F_cat)                          # (B, C, H, W)

        return F_fused, F_les_out, F_ctx_out, M


# =========================================================
# Full model: small U-Net + Dual Swin Bottleneck
# =========================================================
class DualSwinUNet(nn.Module):
    """
    Small U-Net-like architecture with a dual Swin bottleneck.

    Encoder:
        C1 = base_ch
        C2 = base_ch * 2
        C3 = base_ch * 4
        C4 = base_ch * 8

    Bottleneck:
        DualSwinBottleneck at C4

    Decoder:
        Mirror of encoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        base_ch: int = 32,
        bottleneck_window_size: int = 4,
        bottleneck_heads: int = 4,
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

        # Dual Swin bottleneck
        self.bottleneck = DualSwinBottleneck(
            channels=c4,
            window_size=bottleneck_window_size,
            num_heads=bottleneck_heads,
            mlp_ratio=4.0,
            dropout=0.0,
            attn_dropout=0.0,
            drop_path_prob=0.0,
        )

        # Decoder
        self.up3 = UpBlock(c4, c3)
        self.up2 = UpBlock(c3, c2)
        self.up1 = UpBlock(c2, c1)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x, return_dual_feats: bool = False):
        # Encoder
        e1 = self.enc1(x)  # (B, c1, H,   W)
        e2 = self.enc2(e1) # (B, c2, H/2, W/2)
        e3 = self.enc3(e2) # (B, c3, H/4, W/4)
        e4 = self.enc4(e3) # (B, c4, H/8, W/8)

        # Bottleneck
        b, F_les, F_ctx, M = self.bottleneck(e4)  # (B, c4, H/8, W/8)

        # Decoder
        d3 = self.up3(b, e3)    # (B, c3, H/4, W/4)
        d2 = self.up2(d3, e2)   # (B, c2, H/2, W/2)
        d1 = self.up1(d2, e1)   # (B, c1, H,   W)

        out = self.out_conv(d1) # (B, n_classes, H, W)

        if return_dual_feats:
            return out, F_les, F_ctx, M
        return out


# =========================================================
# Helper: count trainable parameters
# =========================================================
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# Quick sanity test
# =========================================================
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    model = DualSwinUNet(
        in_channels=3,
        n_classes=1,
        base_ch=32,               # set 24 or 16 if you want fewer params
        bottleneck_window_size=4, # for 256x256 -> bottleneck 32x32, ws=4 divides 32
        bottleneck_heads=4,
    ).eval()

    x = torch.randn(1, 3, 256, 256)
    y, F_les, F_ctx, M = model(x, return_dual_feats=True)

    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Bottleneck lesion feats:", F_les.shape)
    print("Bottleneck context feats:", F_ctx.shape)
    print("Learned mask M:", M.shape)
    print("Trainable params (M):", count_trainable_params(model) / 1e6)

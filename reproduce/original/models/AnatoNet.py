import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import copy


class DifferentialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, beta=2.0, gamma=1.0):
        super(DifferentialTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.beta = beta
        self.gamma = gamma

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V transformation layers
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)


    def compute_differentials(self, feature_map):
        """
        Compute horizontal and vertical differentials of the 2D feature map.
        Args:
            feature_map (torch.Tensor): Input feature map (B, N, C)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Horizontal and Vertical gradients
        """
        B, N, C = feature_map.shape
        H = W = int(math.sqrt(N))  # Assume feature map is square for simplicity

        # Reshape into 2D spatial format (B, C, H, W)
        feature_map_2d = feature_map.reshape(B, C, H, W)

        # Compute gradients along the height and width (vertical and horizontal)
        horizontal_diff = torch.diff(feature_map_2d, n=1, dim=-1, append=feature_map_2d[:, :, :, -1:])
        vertical_diff = torch.diff(feature_map_2d, n=1, dim=-2, append=feature_map_2d[:, :, -1:, :])

        # Flatten back into (B, N, C)
        horizontal_diff = horizontal_diff.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        vertical_diff = vertical_diff.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        return horizontal_diff, vertical_diff


    def forward(self, x):
        """
        Compute the differential-aware self-attention.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape  # Batch size, number of tokens, channels (C = embed_dim)

        # Compute differentials in horizontal and vertical directions
        horizontal_diff, vertical_diff = self.compute_differentials(x)

        # Generate Q, K, V matrices from the input feature and its differentials
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Queries, keys, values (B, num_heads, N, head_dim)

        # Generate Q, K, V matrices for horizontal and vertical differentials
        h_qkv = self.qkv(horizontal_diff).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        v_qkv = self.qkv(vertical_diff).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        h_q, h_k, _ = h_qkv[0], h_qkv[1], h_qkv[2]  # Horizontal queries, keys, values
        v_q, v_k, _ = v_qkv[0], v_qkv[1], v_qkv[2]  # Vertical queries, keys, values

        # Horizontal bias matrix G_L and vertical bias G_T
        pos_y = torch.arange(0, N, device=x.device).unsqueeze(0)  # Vertical positions
        G_L = self.beta - self.gamma * torch.abs(pos_y - pos_y.T)  # Horizontal bias matrix G_L (N x N)
        # G_T = self.beta - self.gamma * torch.abs(pos_y - pos_y.T).T  # Vertical bias matrix G_T (N x N)

        # Differential-aware Attention computation
        attn = (q @ k.transpose(-2, -1)) / self.scale
        h_attn = (h_q @ h_k.transpose(-2, -1)) / self.scale
        v_attn = (v_q @ v_k.transpose(-2, -1)) / self.scale

        # Aggregate attentions and apply softmax
        attn = F.softmax(attn + h_attn + v_attn + G_L.unsqueeze(0).unsqueeze(0), dim=-1)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out


class HorizontalBiasAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, beta=2.0, gamma=1.5):
        super(HorizontalBiasAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.beta = beta
        self.gamma = gamma

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # Batch size, number of tokens, channels (C = embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Queries, keys, values (B, num_heads, N, head_dim)

        # Horizontal bias matrix G_L
        pos_y = torch.arange(0, N, device=x.device).unsqueeze(0)  # Vertical positions
        G_L = self.beta - self.gamma * torch.abs(pos_y - pos_y.T)  # Horizontal bias matrix G_L (N x N)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) / self.scale + G_L.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class WindowMHSAWithHorizontalBias(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, beta=1.0, gamma=1.0):
        super(WindowMHSAWithHorizontalBias, self).__init__()
        self.window_size = window_size
        self.attention = HorizontalBiasAttention(embed_dim, num_heads, beta, gamma)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        # Divide input into windows
        x = x.view(B, H, W, C)
        x_windows = self.window_partition(x)  # Shape: [B*num_windows, window_size, window_size, C]

        # Perform attention on each window
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attention(x_windows)  # Shape: [B*num_windows, window_size*window_size, C]

        # Merge windows back into original image shape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W)  # Shape: [B, H, W, C]

        return x.view(B, H*W, C)

    def window_partition(self, x):
        # Get dimensions
        B, H, W, C = x.shape

        # Check that H and W are divisible by the window size
        assert H % self.window_size == 0, "H must be divisible by window_size"
        assert W % self.window_size == 0, "W must be divisible by window_size"

        # Reshape into windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        # Merge windows back into the original image size
        B = int(windows.shape[0] / (H / self.window_size * W / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class ShiftedWindowMHSAWithHorizontalBias(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift_size, beta=1.0, gamma=1.0):
        super(ShiftedWindowMHSAWithHorizontalBias, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.attention = HorizontalBiasAttention(embed_dim, num_heads, beta, gamma)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        # Apply cyclic shift
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = self.window_partition(shifted_x)

        # Perform attention
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attention(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x.view(B, H * W, C)

    def window_partition(self, x):
        # Same as WMHSA
        return WindowMHSAWithHorizontalBias.window_partition(self, x)

    def window_reverse(self, windows, H, W):
        # Same as WMHSA
        return WindowMHSAWithHorizontalBias.window_reverse(self, windows, H, W)


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = x.view(B, H, W, C)

        # Split input into patches of 2x2
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # Concatenate along the channel dimension
        x = x.view(B, -1, 4 * C)  # Reshape
        x = self.norm(x)
        x = self.reduction(x).view(B, 2 * C, H // 2, W // 2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        return x


class UResNetWithAttention(nn.Module):
    def __init__(self, n_classes, pretrained=True, embed_dim=2048, num_heads=8, beta=2.0, gamma=1.0):
        super(UResNetWithAttention, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)
        self.base_layers = list(self.base_model.children())

        self.enc1 = nn.Sequential(*self.base_layers[:3])
        self.enc2 = nn.Sequential(*self.base_layers[3:5])
        self.enc3 = self.base_layers[5]
        self.enc4 = self.base_layers[6]
        self.enc5 = self.base_layers[7]
        # self.attention3 = HorizontalBiasAttention(embed_dim=512, num_heads=num_heads)
        self.attention4 = DifferentialTransformer(embed_dim=1024, num_heads=num_heads, beta=beta, gamma=gamma)

        #  self.center = ConvBlock(2048, 2048)
        self.center_attention = DifferentialTransformer(embed_dim=embed_dim, num_heads=num_heads, beta=beta, gamma=gamma)

        self.up4 = UpBlock(2048, 1024)
        self.up3 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up1 = UpBlock(256, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
       # enc3 = self.attention3(enc3.view(enc3.size(0), enc3.size(1), -1).permute(0, 2, 1)).permute(0, 2, 1).view(enc3.size())

        enc4 = self.enc4(enc3)
        enc4 = self.attention4(enc4.view(enc4.size(0), enc4.size(1), -1).permute(0, 2, 1)).permute(0, 2, 1).view(enc4.size())
        enc5 = self.enc5(enc4)

        # Center - Attention block replaces ConvBlock
        center = self.center_attention(enc5.view(enc5.size(0), enc5.size(1), -1).permute(0, 2, 1))
        center = center.permute(0, 2, 1).view(enc5.size())

        dec4 = self.up4(center, enc4)
        dec3 = self.up3(dec4, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)

        dec1 = self.final_up(dec1)

        out = self.final(dec1)
        return out

    def extract_features(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        return enc5, enc1, enc2, enc3, enc4

    def decoder(self, features, enc1, enc2, enc3, enc4):
        center = self.center(features)

        dec4 = self.up4(center, enc4)
        dec3 = self.up3(dec4, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)

        dec1 = self.final_up(dec1)

        out = self.final(dec1)
        return out



if __name__ == "__main__":
    model = UResNetWithAttention(n_classes=1, pretrained=True)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)

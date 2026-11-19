import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 核心组件: KAN Layer (Kolmogorov-Arnold Network Layer)
# ==========================================
class KANLayer(nn.Module):
    """
    KAN Layer 实现。
    高效的 B-Spline 版本。
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 基础权重
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = base_activation()
        
        # B-Spline 参数
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)
        
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # 1. 生成噪声目标值 noise
            # 形状: (Num_Points, In, Out)
            num_points = self.grid_size + self.spline_order
            noise = (torch.rand(num_points, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            
            # 2. [关键修复] 生成对应数量的输入点 x
            # 我们需要在 [-1, 1] 区间内均匀生成 num_points 个点，确保方程组 Ax=B 的维度匹配 (N=M)
            # grid[0, spline_order] 是起始点 (-1), grid[0, -spline_order-1] 是终点 (1)
            x = torch.linspace(
                self.grid[0, self.spline_order], 
                self.grid[0, -self.spline_order - 1], 
                num_points,
                device=self.grid.device,
                dtype=self.grid.dtype
            ).unsqueeze(1).expand(num_points, self.in_features).contiguous()
            
            # 3. 计算系数
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.scale_spline is None else 0.001) 
                * self.curve2coeff(x, noise)
            )

    def b_splines(self, x: torch.Tensor):
        """
        计算 B-Spline 基函数
        x: (Batch, In_Features)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        根据曲线点计算系数 (初始化用)
        x: (N, In)
        y: (N, In, Out)
        """
        # [关键修复] x 已经是 (N, In) 格式，不需要转置即可传入 b_splines
        # A shape: (In, N, Coeff)
        A = self.b_splines(x).transpose(0, 1) 
        # B shape: (In, N, Out)
        B = y.transpose(0, 1)
        
        # 求解线性方程组 AX = B
        # solution shape: (In, Coeff, Out)
        solution = torch.linalg.lstsq(A, B).solution 
        
        # 转换回权重形状: (Out, In, Coeff)
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x):
        # 基础线性变换
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 样条变换
        x_clamped = torch.clamp(x, min=self.grid[0, 0] + self.grid_eps, max=self.grid[0, -1] - self.grid_eps)
        spline_basis = self.b_splines(x_clamped)
        spline_output = F.linear(spline_basis.view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        
        return base_output + spline_output

# ==========================================
# 2. Tokenized KAN Block
# ==========================================
class TokenizedKANBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, no_kan_layers=3):
        super(TokenizedKANBlock, self).__init__()
        
        self.tokenization = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, stride=1)
        
        self.kan_layers = nn.ModuleList([
            KANLayer(embed_dim, embed_dim) for _ in range(no_kan_layers)
        ])
        
        self.dw_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)
        
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        
        tokens = self.tokenization(x) 
        identity = tokens 
        
        B, C, H, W = tokens.shape
        x_flat = tokens.permute(0, 2, 3, 1).reshape(-1, C)
        
        for layer in self.kan_layers:
            x_flat = layer(x_flat)
            
        x_processed = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        out = self.dw_conv(x_processed)
        out = self.bn(out)
        out = self.act(out)
        
        out = out + identity 
        
        out = out.permute(0, 2, 3, 1) 
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2) 
        
        if residual.shape == out.shape:
             out = out + residual

        return out

# ==========================================
# 3. 基础卷积块
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 4. U-KAN 主模型
# ==========================================
class UKAN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, 
                 c_list=[32, 64, 128, 256, 512]): 
        super(UKAN, self).__init__()
        
        c1, c2, c3, d1, d2 = c_list 
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        
        # Encoder (Tokenized KAN)
        self.enc4 = TokenizedKANBlock(c3, d1)
        self.pool4 = nn.MaxPool2d(2) 
        
        # Bottleneck
        self.bottleneck = TokenizedKANBlock(d1, d2)
        
        # Decoder (Tokenized KAN)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = TokenizedKANBlock(d2 + d1, d1) 
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = TokenizedKANBlock(d1 + c3, c3)
        
        # Decoder (Conv)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(c3 + c2, c2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(c2 + c1, c1)
        
        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(x)              # [B, C1, H, W]
        p1 = self.pool1(x1)            # [B, C1, H/2, W/2]
        
        x2 = self.enc2(p1)             # [B, C2, H/2, W/2]
        p2 = self.pool2(x2)            # [B, C2, H/4, W/4]
        
        x3 = self.enc3(p2)             # [B, C3, H/4, W/4]
        p3 = self.pool3(x3)            # [B, C3, H/8, W/8]
        
        x4 = self.enc4(p3)             # [B, D1, H/8, W/8]
        p4 = self.pool4(x4)            # [B, D1, H/16, W/16]
        
        bn = self.bottleneck(p4)       # [B, D2, H/16, W/16]
        
        u4 = self.up4(bn)              # [B, D2, H/8, W/8]
        cat4 = torch.cat([u4, x4], dim=1) 
        d4 = self.dec4(cat4)
        
        u3 = self.up3(d4)              # [B, D1, H/4, W/4]
        cat3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(cat3)           # [B, C3, H/4, W/4]
        
        u2 = self.up2(d3)              # [B, C3, H/2, W/2]
        cat2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(cat2)           # [B, C2, H/2, W/2]
        
        u1 = self.up1(d2)              # [B, C2, H, W]
        cat1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(cat1)           # [B, C1, H, W]
        
        out = self.final_conv(d1)      # [B, Num_Classes, H, W]
        
        return out
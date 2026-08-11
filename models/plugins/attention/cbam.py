"""
CBAM: Convolutional Block Attention Module
-------------------------------------------
结合通道注意力和空间注意力的混合注意力机制。
先通过通道注意力筛选重要通道，再通过空间注意力定位关键空间位置。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "cbam_channel",
    category="attention",
    description="CBAM 通道注意力子模块：AvgPool + MaxPool → 共享MLP → 相加 → Sigmoid",
)
class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块

    同时使用平均池化和最大池化聚合空间信息，共享一个 MLP 处理两种池化结果，
    最后相加并通过 Sigmoid 生成通道权重。

    Args:
        channels: 输入通道数
        reduction: 压缩比例（默认 16）
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        reduced = max(1, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            通道注意力权重，shape (B, C, 1, 1)
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


@PLUGIN_REGISTRY.register(
    "cbam_spatial",
    category="attention",
    description="CBAM 空间注意力子模块：沿通道concat(AvgPool, MaxPool) → Conv7x7 → Sigmoid",
)
class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块

    沿通道维度分别做平均池化和最大池化，拼接后通过 7×7 卷积生成空间权重图。

    Args:
        kernel_size: 卷积核大小（默认 7）
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, f"kernel_size 必须为奇数，实际为 {kernel_size}"

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            空间注意力权重，shape (B, 1, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return torch.sigmoid(self.conv(combined))


@PLUGIN_REGISTRY.register(
    "cbam",
    category="attention",
    description="CBAM 完整模块：通道注意力 → 空间注意力，串行增强特征",
)
class CBAM(nn.Module):
    """Convolutional Block Attention Module

    依次应用通道注意力和空间注意力，增强特征表示。
    两部分串联：先对通道加权，再对空间位置加权。

    Args:
        channels: 输入通道数
        reduction: 通道注意力的压缩比例（默认 16）
        kernel_size: 空间注意力的卷积核大小（默认 7）
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== CBAM 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)

    # 分别测试子模块
    ca = ChannelAttention(channels=64, reduction=16).to(device)
    sa = SpatialAttention(kernel_size=7).to(device)
    cbam = CBAM(channels=64, reduction=16, kernel_size=7).to(device)

    ca.eval()
    sa.eval()
    cbam.eval()

    with torch.no_grad():
        w_c = ca(x)
        w_s = sa(x)
        y = cbam(x)

    print(f"输入 shape: {x.shape}")
    print(f"通道注意力权重 shape: {w_c.shape}")
    print(f"空间注意力权重 shape: {w_s.shape}")
    print(f"CBAM 输出 shape: {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"CBAM 参数量: {sum(p.numel() for p in cbam.parameters()):,}")
    print(f"全部通过!")

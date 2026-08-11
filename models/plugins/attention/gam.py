"""
GAM: Global Attention Mechanism
--------------------------------
全局注意力机制：通过 3D 排列和 MLP 捕获通道维度的全局交互，
再通过空间卷积捕获空间维度的全局交互，增强特征表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "gam",
    category="attention",
    description="全局注意力机制 GAM：通道注意力（3D排列+MLP）+ 空间注意力（卷积），串行增强",
)
class GAM(nn.Module):
    """Global Attention Mechanism 模块

    先通过通道注意力模块捕获跨通道的全局依赖，
    再通过空间注意力模块捕获跨空间的全局依赖，两者串联增强特征。

    Args:
        channels: 输入通道数
        reduction: 基础压缩比例（默认 16）
        channel_ratio: 通道注意力中的额外压缩倍数（默认 4）
        spatial_ratio: 空间注意力中的压缩倍数（默认 4）
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        channel_ratio: int = 4,
        spatial_ratio: int = 4,
    ):
        super(GAM, self).__init__()
        self.channels = channels
        reduced = max(1, channels // reduction)

        # ===== 通道注意力 =====
        # 3D 排列 + MLP：将 (B, C, H, W) 视为序列处理
        channel_hidden = max(1, channels // channel_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hidden, channels, bias=False),
        )

        # ===== 空间注意力 =====
        spatial_hidden = max(1, reduced // spatial_ratio)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(reduced, spatial_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_hidden, 1, kernel_size=3, padding=1, bias=False),
        )
        self.down_sample = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()

        # ---- 通道注意力 ----
        # 3D 排列： (B, C, H, W) → (B, H*W, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        # MLP 处理每个空间位置的通道向量
        x_att = self.channel_mlp(x_perm)  # (B, H*W, C)
        # 还原 → (B, C, H, W)
        x_channel_att = x_att.view(b, h, w, c).permute(0, 3, 1, 2)
        x_channel_att = torch.sigmoid(x_channel_att)
        x_out = x * x_channel_att

        # ---- 空间注意力 ----
        # 降维
        x_spatial = self.down_sample(x_out)  # (B, reduced, H, W)
        # 空间卷积 → (B, 1, H, W)
        spatial_att = torch.sigmoid(self.spatial_conv(x_spatial))
        x_out = x_out * spatial_att

        return x_out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== GAM 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = GAM(channels=64, reduction=16, channel_ratio=4, spatial_ratio=4).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"全部通过!")

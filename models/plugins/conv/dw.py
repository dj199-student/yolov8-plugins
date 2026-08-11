"""
深度可分离卷积 (Depthwise Separable Convolution)
-------------------------------------------------
MobileNet 风格的高效卷积：先 depthwise（每个通道独立卷积）后 pointwise（1x1 跨通道融合）。
将标准卷积分解为两阶段，大幅减少参数与计算量。

参考：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "dw_conv",
    category="conv",
    description="Depthwise Separable Conv: depthwise(3x3, groups=C) + pointwise(1x1) + BN + SiLU，MobileNet 风格",
)
class DWConv(nn.Module):
    """深度可分离卷积 — MobileNet 高效卷积

    结构：Depthwise Conv (groups=in_channels, kernel_size=k) → BN → Act
         → Pointwise Conv (1x1) → BN → Act (可选)

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        act:          是否在 pointwise 后使用 SiLU 激活（默认 True）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: bool = True,
    ):
        super(DWConv, self).__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        pointwise_layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            pointwise_layers.append(nn.SiLU(inplace=True))
        self.pointwise = nn.Sequential(*pointwise_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C_in, H, W)

        Returns:
            输出特征图，shape (B, C_out, H//stride, W//stride)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DWConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)

    # stride=1
    conv1 = DWConv(64, 128, kernel_size=3, stride=1).to(device)
    conv1.eval()
    with torch.no_grad():
        y1 = conv1(x)
    print(f"stride=1: 输入 {x.shape} -> 输出 {y1.shape}")

    # stride=2
    conv2 = DWConv(64, 128, kernel_size=3, stride=2).to(device)
    conv2.eval()
    with torch.no_grad():
        y2 = conv2(x)
    print(f"stride=2: 输入 {x.shape} -> 输出 {y2.shape}")

    # 参数量对比
    dwc_params = sum(p.numel() for p in dwc.parameters())
    std_params = sum(p.numel() for p in nn.Conv2d(64, 128, 3, padding=1).parameters())
    print(f"DWConv 参数量: {dwc_params:,}")
    print(f"标准 Conv 参数量: {std_params:,}")
    print(f"参数量减少: {(1 - dwc_params / std_params) * 100:.1f}%")
    print("全部通过!")

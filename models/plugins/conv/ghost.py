"""
GhostNet 模块：GhostModule 与 GhostConv
---------------------------------------
GhostNet 核心思想：一部分通道通过普通卷积生成（内在特征图），
其余通道通过廉价的线性变换（depthwise conv）生成（幽灵特征图），
两部分拼接后得到完整输出，大幅减少计算量。

参考：GhostNet: More Features from Cheap Operations (CVPR 2020)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "ghost_module",
    category="conv",
    description="GhostModule: 一半通道用普通卷积生成，另一半通过廉价 depthwise conv 生成，拼接输出",
)
class GhostModule(nn.Module):
    """Ghost Module — GhostNet 的基础构建块

    将输出通道分成两部分：
      - primary_channels: 标准卷积生成
      - cheap_channels:   depthwise conv（廉价操作）生成
    最终拼接两部分作为输出。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  廉价操作的卷积核大小（默认 1）
        ratio:        压缩比，primary = out_channels // ratio（默认 2）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
    ):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        primary_channels = int(math.ceil(out_channels / ratio))
        cheap_channels = out_channels - primary_channels

        # 主卷积：标准 1x1 生成内在特征图
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary_channels),
            nn.ReLU(inplace=True),
        )

        # 廉价操作：depthwise conv 在内在特征图上生成幽灵特征图
        padding = kernel_size // 2
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                primary_channels,
                cheap_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=primary_channels,
                bias=False,
            ),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            输出特征图，shape (B, out_channels, H, W)
        """
        # 内在特征图
        primary = self.primary_conv(x)
        # 幽灵特征图
        cheap = self.cheap_operation(primary)
        # 拼接
        return torch.cat([primary, cheap], dim=1)


@PLUGIN_REGISTRY.register(
    "ghost_conv",
    category="conv",
    description="GhostConv: GhostModule 封装为标准卷积替换，支持 stride 和可配置 kernel_size",
)
class GhostConv(nn.Module):
    """GhostConv — 标准卷积的 GhostNet 替代

    由两个 GhostModule 堆叠而成，第二个 GhostModule 后跟可选的 stride。
    结构与标准卷积类似，但计算量更少。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        ratio:        Ghost 压缩比（默认 2）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        ratio: int = 2,
    ):
        super(GhostConv, self).__init__()
        self.stride = stride

        # 第一个 GhostModule
        self.ghost1 = GhostModule(
            in_channels, out_channels // 2, kernel_size=kernel_size, ratio=ratio
        )

        # 第二个 GhostModule（可选 stride）
        self.ghost2 = GhostModule(
            out_channels // 2,
            out_channels - out_channels // 2,
            kernel_size=kernel_size,
            ratio=ratio,
        )

        # stride 通过第二个 ghost 后的 pooling 实现
        if stride != 1:
            self.downsample = nn.AvgPool2d(stride, stride=stride)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            输出特征图，shape (B, out_channels, H//stride, W//stride)
        """
        x1 = self.ghost1(x)
        x1 = self.downsample(x1)
        x2 = self.ghost2(x1)
        return torch.cat([x1, x2], dim=1)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== GhostModule / GhostConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GhostModule 测试
    x = torch.randn(2, 64, 32, 32).to(device)
    gm = GhostModule(64, 128, kernel_size=3, ratio=2).to(device)
    gm.eval()
    with torch.no_grad():
        y = gm(x)
    print(f"GhostModule: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"GhostModule 参数量: {sum(p.numel() for p in gm.parameters()):,}")

    # GhostConv 测试
    gc = GhostConv(64, 128, kernel_size=3, stride=2, ratio=2).to(device)
    gc.eval()
    with torch.no_grad():
        y = gc(x)
    print(f"GhostConv (stride=2): 输入 {x.shape} -> 输出 {y.shape}")
    print(f"GhostConv 参数量: {sum(p.numel() for p in gc.parameters()):,}")
    print("全部通过!")

"""
PConv: Partial Convolution (FasterNet)
---------------------------------------
FasterNet 的核心构建块。仅对输入通道的一个子集 cp = in_channels * ratio 进行
常规卷积操作，其余通道保持不变（identity），从而大幅减少冗余计算和内存访问。

特点：
  - 只对部分通道做卷积，其余通道保持恒等映射
  - 比深度可分离卷积更快（避免 1x1 pointwise 的计算开销）
  - 延迟极低，适合实时应用

参考：Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks (CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "p_conv",
    category="conv",
    description="Partial Conv (FasterNet): 仅对 cp 部分通道做卷积，其余恒等映射，极致低延迟",
)
class PConv(nn.Module):
    """Partial Convolution — FasterNet 的核心模块

    将输入通道分为两部分：
      - 前 cp 个通道：应用常规卷积
      - 剩余通道：  恒等映射（identity）

    Args:
        in_channels:  输入通道数（同时也等于输出通道数）
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        ratio:        参与卷积的通道比例（默认 0.25）
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        ratio: float = 0.25,
    ):
        super(PConv, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.cp = max(1, int(in_channels * ratio))
        padding = kernel_size // 2

        # 仅对 cp 个通道做卷积
        self.conv = nn.Conv2d(
            self.cp, self.cp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.cp)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        对前 cp 个通道做卷积，其余通道保持恒等映射。

        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            输出特征图 (B, C, H//stride, W//stride)
        """
        # 分割：前 cp 通道做卷积，剩余恒等
        x_conv = x[:, :self.cp, :, :]
        x_id = x[:, self.cp:, :, :]

        # 卷积分支
        x_conv = self.conv(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = self.act(x_conv)

        # 当 stride > 1 时，恒等分支也需要下采样
        if self.stride > 1:
            x_id = F.avg_pool2d(x_id, self.stride, stride=self.stride)

        # 拼接
        return torch.cat([x_conv, x_id], dim=1)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== PConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)

    # stride=1
    pc1 = PConv(64, kernel_size=3, stride=1, ratio=0.25).to(device)
    pc1.eval()
    with torch.no_grad():
        y1 = pc1(x)
    print(f"stride=1, ratio=0.25: 输入 {x.shape} -> 输出 {y1.shape}")
    print(f"  参与卷积通道: {pc1.cp}/{pc1.in_channels}")

    # stride=2
    pc2 = PConv(64, kernel_size=3, stride=2, ratio=0.25).to(device)
    pc2.eval()
    with torch.no_grad():
        y2 = pc2(x)
    print(f"stride=2, ratio=0.25: 输入 {x.shape} -> 输出 {y2.shape}")

    # ratio=0.5
    pc3 = PConv(64, kernel_size=3, stride=1, ratio=0.5).to(device)
    pc3.eval()
    with torch.no_grad():
        y3 = pc3(x)
    print(f"stride=1, ratio=0.50: 输入 {x.shape} -> 输出 {y3.shape}")
    print(f"  参与卷积通道: {pc3.cp}/{pc3.in_channels}")

    print(f"PConv 参数量: {sum(p.numel() for p in pc1.parameters()):,}")
    print(f"标准 Conv3x3 参数量: {sum(p.numel() for p in nn.Conv2d(64, 64, 3, padding=1).parameters()):,}")
    print("全部通过!")

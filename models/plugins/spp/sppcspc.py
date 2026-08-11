"""
SPPCSPC: Spatial Pyramid Pooling with Cross-Stage Partial Connection
--------------------------------------------------------------------
SPP 与 CSP（Cross-Stage Partial）连接的结合体。
将输入一分为二，一半走 SPP 多尺度池化路径，一半直通，
最后拼接并通过 1x1 卷积融合，兼顾多尺度感受野和梯度流。

参考：YOLOv5/v7 SPPCSPC 模块设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "sppcspc",
    category="spp",
    description="SPPCSPC: SPP 与 CSP 结合，多尺度并行池化 [kxk maxpool] → concat + 分支拼接融合",
)
class SPPCSPC(nn.Module):
    """Spatial Pyramid Pooling with Cross-Stage Partial Connection

    将输入通道按比例 e 一分为二：
      - Part1: SPP（多个并行最大池化核 + 原始特征拼接）→ 1x1 conv
      - Part2: 1x1 conv（保持梯度直通）
      - 最终：Concat(Part1, Part2) → 1x1 conv

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        e:            通道分割比例（默认 0.5，各取一半）
        kernels:      SPP 最大池化的核尺寸列表（默认 [5, 9, 13]）
    """

    def __init__(self, in_channels: int, out_channels: int, e: float = 0.5, kernels=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(out_channels * e)  # hidden channels per branch
        c1 = int(in_channels * e)   # channels for part 2

        # SPP branch: multiple parallel maxpools with different kernel sizes
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in kernels
        ])

        # Part 1: SPP branch convs
        # c_ * (len(kernels) + 1) because we concat all pool outputs + original
        self.cv1 = nn.Sequential(
            nn.Conv2d(c_ * (len(kernels) + 1), c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.ReLU(inplace=True),
        )

        # Part 2: direct branch
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.ReLU(inplace=True),
        )

        # Final fusion conv
        self.cv3 = nn.Sequential(
            nn.Conv2d(c_ * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.c_ = c_
        self.c1 = c1

    def forward(self, x):
        """Forward pass of SPPCSPC.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Split input into two parts along channel dimension
        x1 = x[:, :self.c_, :, :]    # Part 1 for SPP
        x2 = x[:, self.c_:, :, :]    # Part 2 for direct

        # Part 1: SPP — parallel maxpools + original
        pool_outs = [x1] + [pool(x1) for pool in self.maxpools]
        spp_out = torch.cat(pool_outs, dim=1)
        spp_out = self.cv1(spp_out)

        # Part 2: direct 1x1 conv
        direct_out = self.cv2(x2)

        # Concat and fuse
        fused = torch.cat([spp_out, direct_out], dim=1)
        return self.cv3(fused)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== SPPCSPC 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 512, 20, 20).to(device)

    model = SPPCSPC(in_channels=512, out_channels=512, e=0.5, kernels=[5, 9, 13]).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
    assert y.shape[0] == x.shape[0], "Batch size mismatch"
    assert y.shape[2] == x.shape[2], "Height changed (should be same)"
    assert y.shape[3] == x.shape[3], "Width changed (should be same)"
    print(f"空间尺寸不变: {x.shape[2:]} == {y.shape[2:]}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'sppcspc'")
    print(f"全部通过!")

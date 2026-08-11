"""
SPPFCSPC: Spatial Pyramid Pooling Fast with Cross-Stage Partial Connection
--------------------------------------------------------------------------
SPPF（快速空间金字塔池化）+ CSP 连接。
用 3 次串联的 5x5 最大池化替代并行多个大核池化，速度更快，
效果等效。适用于 YOLOv8 风格的特征增强。

参考：YOLOv5/v8 SPPFCSPC 模块设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "sppfcspc",
    category="spp",
    description="SPPFCSPC: SPPF 串联小核池化 + CSP 连接，速度更快，YOLOv8 风格",
)
class SPPFCSPC(nn.Module):
    """Spatial Pyramid Pooling Fast with Cross-Stage Partial Connection

    与 SPPCSPC 对比，用 3 个串联的 5x5 最大池化替代并行的大核池化。
    结构：
      - 输入一分为二
      - Part1: 3x5x5 串联maxpool → concat(origin + 3 intermediates) → 1x1 conv
      - Part2: 1x1 conv（直通分支）
      - Concat → 1x1 conv

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        e:            通道分割比例（默认 0.5）
    """

    def __init__(self, in_channels: int, out_channels: int, e: float = 0.5):
        super(SPPFCSPC, self).__init__()
        c_ = int(out_channels * e)  # hidden channels per branch
        c1 = int(in_channels * e)   # channels for part 2

        # SPPF uses sequential 5x5 maxpools
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # Part 1: SPPF branch conv
        # Concat 4 tensors: original + 3 intermediate pool outputs
        self.cv1 = nn.Sequential(
            nn.Conv2d(c_ * 4, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.ReLU(inplace=True),
        )

        # Part 2: direct branch
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.ReLU(inplace=True),
        )

        # Final fusion
        self.cv3 = nn.Sequential(
            nn.Conv2d(c_ * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.c_ = c_
        self.c1 = c1

    def forward(self, x):
        """Forward pass of SPPFCSPC.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Split
        x1 = x[:, :self.c_, :, :]
        x2 = x[:, self.c_:, :, :]

        # Part 1: SPPF — sequential maxpools, collect intermediate outputs
        p1 = self.maxpool(x1)
        p2 = self.maxpool(p1)
        p3 = self.maxpool(p2)
        sppf_out = torch.cat([x1, p1, p2, p3], dim=1)
        sppf_out = self.cv1(sppf_out)

        # Part 2: direct
        direct_out = self.cv2(x2)

        # Fuse
        fused = torch.cat([sppf_out, direct_out], dim=1)
        return self.cv3(fused)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== SPPFCSPC 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 512, 20, 20).to(device)

    model = SPPFCSPC(in_channels=512, out_channels=512, e=0.5).to(device)
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
    print(f"已注册名称: 'sppfcspc'")
    print(f"全部通过!")

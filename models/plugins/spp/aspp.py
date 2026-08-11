"""
ASPP: Atrous Spatial Pyramid Pooling
-------------------------------------
来自 DeepLab 系列的空洞空间金字塔池化。
通过不同空洞率（dilation rate）的并行空洞卷积捕获多尺度上下文，
辅以全局平均池化分支获取图像级特征。

参考：DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
      Atrous Convolution, and Fully Connected CRFs (TPAMI 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _ASPPConv(nn.Module):
    """单个 ASPP 空洞卷积分支：3x3 conv(dilation) + BN + ReLU"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super(_ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _ASPPPooling(nn.Module):
    """ASPP 全局平均池化分支：AdaptiveAvgPool2d(1) → 1x1 conv → BN → ReLU → upsample"""

    def __init__(self, in_channels: int, out_channels: int):
        super(_ASPPPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        out = self.gap(x)
        out = self.conv(out)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return out


@PLUGIN_REGISTRY.register(
    "aspp",
    category="spp",
    description="ASPP: 空洞空间金字塔池化，多个并行的空洞卷积分支 + 全局池化分支，多尺度上下文捕获",
)
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling（来自 DeepLab）

    5 个并行分支：
      - 1x1 卷积
      - 3x3 空洞卷积，dilation=6
      - 3x3 空洞卷积，dilation=12
      - 3x3 空洞卷积，dilation=18
      - 全局平均池化 → 1x1 conv → 上采样
    所有分支输出拼接后通过 1x1 卷积融合。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        dilations:    空洞率列表（默认 [6, 12, 18]）
    """

    def __init__(self, in_channels: int, out_channels: int, dilations=(6, 12, 18)):
        super(ASPP, self).__init__()
        # Each branch output channels (equal split)
        branch_channels = out_channels // (len(dilations) + 2)
        remainder = out_channels - branch_channels * (len(dilations) + 2)

        # Branch 1: 1x1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + remainder, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels + remainder),
            nn.ReLU(inplace=True),
        )

        # Branches 2-N: 3x3 dilated convs
        self.aspp_convs = nn.ModuleList([
            _ASPPConv(in_channels, branch_channels, dilation=d)
            for d in dilations
        ])

        # Global pooling branch
        self.aspp_pool = _ASPPPooling(in_channels, branch_channels)

        # Final fusion: concat all branches -> 1x1 conv
        total_channels = (branch_channels + remainder) + branch_channels * len(dilations) + branch_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass of ASPP.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # All branches are spatial-size preserving
        branches = [self.conv1x1(x)]
        branches.extend([conv(x) for conv in self.aspp_convs])
        branches.append(self.aspp_pool(x))

        concat = torch.cat(branches, dim=1)
        return self.fusion(concat)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== ASPP 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(2, 512, 20, 20).to(device)

    model = ASPP(in_channels=512, out_channels=512, dilations=[6, 12, 18]).to(device)
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
    print(f"已注册名称: 'aspp'")
    print(f"全部通过!")

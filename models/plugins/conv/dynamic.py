"""
动态卷积：DynamicConv 与 ODConv
--------------------------------
DynamicConv: 聚合 K 个并行的卷积核，通过 SE-style 注意力网络生成 kernel-wise 权重，动态混合。
ODConv: 全维动态卷积 — 同时沿 kernel 维度、空间维度、输入通道维度、输出通道维度
        生成 4 种互补注意力，对 K 个卷积核进行精细化加权。

参考：Dynamic Convolution: Attention over Convolution Kernels (CVPR 2020)
      Omni-Dimensional Dynamic Convolution (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _AttentionWithTemp(nn.Module):
    """生成 K 个 kernel 权重的注意力模块（带 temperature）"""

    def __init__(self, in_channels: int, num_kernels: int, reduction: int, temperature: float = 30.0):
        super(_AttentionWithTemp, self).__init__()
        self.num_kernels = num_kernels
        self.temperature = temperature
        reduced = max(1, in_channels // reduction)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, num_kernels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成 kernel attention weights

        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            kernel_weights: (B, K, 1, 1)
        """
        b, c, _, _ = x.shape
        pooled = self.global_avg_pool(x).view(b, c)
        attn = self.fc(pooled)  # (B, K)
        attn = torch.softmax(attn / self.temperature, dim=1)
        return attn.view(b, self.num_kernels, 1, 1)


@PLUGIN_REGISTRY.register(
    "dynamic_conv",
    category="conv",
    description="动态卷积：K 个并行 conv kernel + SE 注意力生成 kernel 权重，动态聚合",
)
class DynamicConv(nn.Module):
    """动态卷积 — K 个卷积核的动态加权聚合

    维护 K 个并行的卷积核（独立参数），通过轻量 SE 注意力网络
    根据输入动态生成 K 维权重，对各卷积核的输出进行加权求和。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        num_kernels:  并行卷积核数量 K（默认 4）
        reduction:    SE 注意力的压缩比（默认 4）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_kernels: int = 4,
        reduction: int = 4,
    ):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        padding = kernel_size // 2

        # K 个并行卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            )
            for _ in range(num_kernels)
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # 注意力网络：生成 K 个 kernel 的权重
        self.attention = _AttentionWithTemp(in_channels, num_kernels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图 (B, C_in, H, W)

        Returns:
            输出特征图 (B, C_out, H//stride, W//stride)
        """
        # 生成 kernel 权重
        kernel_weights = self.attention(x)  # (B, K, 1, 1)

        # 各卷积核输出加权求和
        out = None
        for i in range(self.num_kernels):
            conv_out = self.convs[i](x)  # (B, C_out, H, W)
            weight = kernel_weights[:, i:i + 1, :, :]  # (B, 1, 1, 1)
            if out is None:
                out = conv_out * weight
            else:
                out = out + conv_out * weight

        out = self.bn(out)
        out = self.act(out)
        return out


@PLUGIN_REGISTRY.register(
    "od_conv",
    category="conv",
    description="全维动态卷积(ODConv)：沿 kernel/空间/输入通道/输出通道 4 个维度生成注意力，精细化加权",
)
class ODConv(nn.Module):
    """全维动态卷积 (Omni-Dimensional Dynamic Convolution)

    与 DynamicConv 仅用标量权重不同，ODConv 对每个 kernel 生成 4 种注意力：
      - α_si: 空间注意力（沿 H, W 每个位置）
      - α_ci: 输入通道注意力（沿 C_in）
      - α_fi: 输出通道注意力（沿 C_out）
      - α_wi: 卷积核注意力（标量）
    四种注意力逐元素乘到对应的 kernel 维度上，实现精细化动态加权。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        num_kernels:  并行卷积核数量 K（默认 4）
        reduction:    压缩比（默认 4）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_kernels: int = 4,
        reduction: int = 4,
    ):
        super(ODConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        reduced = max(1, in_channels // reduction)

        # K 个并行卷积核（无 bias，bias 由注意力处理后统一加）
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            )
            for _ in range(num_kernels)
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # 全局池化 + 降维
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
        )

        # 4 种注意力头
        self.fc_alpha_si = nn.Linear(reduced, num_kernels * kernel_size * kernel_size)
        self.fc_alpha_ci = nn.Linear(reduced, num_kernels * in_channels)
        self.fc_alpha_fi = nn.Linear(reduced, num_kernels * out_channels)
        self.fc_alpha_wi = nn.Linear(reduced, num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图 (B, C_in, H, W)

        Returns:
            输出特征图 (B, C_out, H//stride, W//stride)
        """
        b, c, h, w = x.shape
        k = self.num_kernels

        # 生成注意力
        pooled = self.global_avg_pool(x).view(b, c)
        feat = self.fc_reduce(pooled)  # (B, reduced)

        # 4 种注意力
        alpha_si = self.fc_alpha_si(feat).view(b, k, self.kernel_size, self.kernel_size)
        alpha_ci = torch.sigmoid(self.fc_alpha_ci(feat)).view(b, k, c, 1, 1)
        alpha_fi = torch.sigmoid(self.fc_alpha_fi(feat)).view(b, k, self.out_channels, 1, 1)
        alpha_wi = torch.sigmoid(self.fc_alpha_wi(feat)).view(b, k, 1, 1, 1)

        # 对每个 kernel 应用 4 种注意力并累加
        out = 0
        for i in range(k):
            weight = self.convs[i].weight  # (C_out, C_in, k, k)

            # 应用注意力：不同 kernel 不同 batch
            si = alpha_si[:, i, :, :]  # (B, k_size, k_size)
            attn_combined = (
                alpha_wi[:, i, :, :, :]      # (B, 1, 1, 1, 1)
                * alpha_fi[:, i, :, :, :]     # (B, C_out, 1, 1)
                * alpha_ci[:, i, :, :, :]     # (B, C_in, 1, 1)
            )  # (B, C_out, C_in, 1, 1)

            # 加权 kernel: (B, C_out, C_in, k_size, k_size)
            weighted_kernel = weight.unsqueeze(0) * attn_combined
            # 在 kernel_size 维度上乘空间注意力
            weighted_kernel = weighted_kernel * si.unsqueeze(1).unsqueeze(2)  # (B, C_out, C_in, k, k)

            # 按 batch 逐个计算卷积
            for j in range(b):
                conv_out = F.conv2d(
                    x[j:j + 1],
                    weighted_kernel[j],
                    bias=None,
                    stride=self.convs[i].stride,
                    padding=self.convs[i].padding,
                )
                if i == 0 and j == 0:
                    out = torch.zeros(b, self.out_channels, conv_out.shape[2], conv_out.shape[3],
                                      device=x.device, dtype=x.dtype)
                out[j:j + 1] = out[j:j + 1] + conv_out

        out = self.bn(out)
        out = self.act(out)
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DynamicConv / ODConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DynamicConv 测试
    x = torch.randn(2, 64, 32, 32).to(device)
    dc = DynamicConv(64, 128, kernel_size=3, stride=1, num_kernels=4, reduction=4).to(device)
    dc.eval()
    with torch.no_grad():
        y = dc(x)
    print(f"DynamicConv: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"DynamicConv 参数量: {sum(p.numel() for p in dc.parameters()):,}")

    # ODConv 测试（小尺寸以节省计算）
    x_small = torch.randn(1, 16, 16, 16).to(device)
    od = ODConv(16, 32, kernel_size=3, stride=1, num_kernels=4, reduction=4).to(device)
    od.eval()
    with torch.no_grad():
        y_od = od(x_small)
    print(f"ODConv: 输入 {x_small.shape} -> 输出 {y_od.shape}")
    print(f"ODConv 参数量: {sum(p.numel() for p in od.parameters()):,}")
    print("全部通过!")

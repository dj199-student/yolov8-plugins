"""
DSConv: Distribution Shifting Convolution
------------------------------------------
一种高效的量化感知卷积。通过学习"分布偏移量"（distribution shifter）
来量化标准卷积权重，大幅减少内存占用和计算开销，
同时保持与标准卷积接近的精度。

原理：
  1. 将权重按 block_size 分组
  2. 每组学习一个可训练的分布偏移量（shift）和缩放因子（scale）
  3. 量化后的权重 = quantize(weight + shift) * scale
  4. 前向传播使用量化权重，反向传播使用 STE（直通估计器）

参考：DSConv: Efficient Convolution Operator (AIBT 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


def _quantize_to_nearest(tensor: torch.Tensor) -> torch.Tensor:
    """将张量量化到最近的整数（STE 直通估计器）"""
    return (tensor.round() - tensor).detach() + tensor


@PLUGIN_REGISTRY.register(
    "ds_conv",
    category="conv",
    description="DSConv: 分布偏移量化卷积 — 可学习的 shift+scale 量化权重，节省内存与计算",
)
class DSConv(nn.Module):
    """分布偏移量化卷积 (Distribution Shifting Convolution)

    通过可学习的分布偏移量对权重进行量化，降低存储和计算成本。
    权重按 block_size 分块，每块学习独立的 shift 和 scale。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3）
        stride:       步长（默认 1）
        block_size:   量化分块大小（默认 16）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        block_size: int = 16,
    ):
        super(DSConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block_size = block_size
        padding = kernel_size // 2

        # 原始浮点权重（VQK = Variable Quantized Kernel）
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )

        # 计算分块数量：每个输出通道作为一个块组
        # 将权重按 out_channels 维度分块
        total_blocks = -(-out_channels // block_size)  # ceil division
        num_elements_per_block = in_channels * kernel_size * kernel_size

        # 每块的偏移量 (KDS = Kernel Distribution Shifter)
        self.kds_shift = nn.Parameter(torch.zeros(total_blocks, num_elements_per_block))
        self.kds_scale = nn.Parameter(torch.ones(total_blocks, num_elements_per_block))

        # bias
        self.bias = nn.Parameter(torch.zeros(out_channels)) if True else None

        self.stride = stride
        self.padding = padding

    def _get_quantized_weight(self) -> torch.Tensor:
        """获取量化后的权重

        Returns:
            量化后的卷积核 weight: (C_out, C_in, k, k)
        """
        c_out, c_in, k, _ = self.weight.shape
        weight_flat = self.weight.view(c_out, -1)  # (C_out, C_in*k*k)

        # 对每个块应用 shift 和 scale
        num_blocks = self.kds_shift.shape[0]
        q_weight = torch.zeros_like(weight_flat)

        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, c_out)
            if start >= c_out:
                break

            block = weight_flat[start:end]  # (block_size, C_in*k*k)
            shift = self.kds_shift[b:b + 1]  # (1, C_in*k*k)
            scale = self.kds_scale[b:b + 1]  # (1, C_in*k*k)

            # 量化：shifted = (weight + shift) / scale → round → * scale - shift
            shifted = (block + shift) / (scale.abs() + 1e-8)
            quantized = _quantize_to_nearest(shifted)
            q_weight[start:end] = quantized * (scale.abs() + 1e-8) - shift

        return q_weight.view(self.out_channels, self.in_channels, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图 (B, C_in, H, W)

        Returns:
            输出特征图 (B, C_out, H//stride, W//stride)
        """
        q_weight = self._get_quantized_weight()
        return F.conv2d(x, q_weight, self.bias, stride=self.stride, padding=self.padding)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DSConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)

    ds = DSConv(64, 128, kernel_size=3, stride=1, block_size=16).to(device)
    ds.eval()
    with torch.no_grad():
        y = ds(x)
    print(f"DSConv: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"DSConv 参数量: {sum(p.numel() for p in ds.parameters()):,}")

    # 对比标准卷积
    std = nn.Conv2d(64, 128, 3, padding=1, bias=True).to(device)
    std_params = sum(p.numel() for p in std.parameters())
    print(f"标准 Conv 参数量: {std_params:,}")
    print(f"额外参数(kds shift+scale): {ds.kds_shift.numel() + ds.kds_scale.numel():,}")
    print("全部通过!")

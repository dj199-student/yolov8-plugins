"""
Involution: 空间特异、通道无关的算子
--------------------------------------
与标准卷积（通道特异、空间共享 kernel）互补：
Involution 对每个空间位置生成独有的 kernel（空间特异），
但在通道维度上共享 kernel（通道无关），大幅减少参数。

通过一个轻量生成网络（reduction → BN → ReLU → 展开）在
每个空间位置上动态生成 kernel weight。

参考：Involution: Inverting the Inherence of Convolution for Visual Recognition (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "involution",
    category="conv",
    description="Involution: 空间特异、通道无关 — 每空间位置生成 kernel，通道共享参数",
)
class Involution(nn.Module):
    """Involution — 反转卷积的内建特性

    标准卷积：kernel 在空间上共享（translation equivariance），在通道上独立
    Involution：kernel 在空间上独立（spatial specific），在通道上共享（channel agnostic）

    每个空间位置 (h, w) 通过生成网络动态产生一个 kernel_size x kernel_size x groups 的核。

    Args:
        in_channels:  输入通道数
        kernel_size:  卷积核大小（默认 7）
        stride:       步长（默认 1）
        reduction:    生成网络中的通道压缩比（默认 4）
        groups:       通道分组数，kernel 在每组内共享（默认 1）
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        reduction: int = 4,
        groups: int = 1,
    ):
        super(Involution, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.channels_per_group = in_channels // groups

        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        assert in_channels % groups == 0, "in_channels 必须能被 groups 整除"

        # 生成网络：输入 → reduction → kernel 展开
        reduced = max(1, in_channels // reduction)
        self.generate = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        # 输出 kernel_size * kernel_size * groups 个值，代表每个空间位置每组生成一个 kernel_size x kernel_size 的核
        self.kernel_gen = nn.Conv2d(
            reduced, kernel_size * kernel_size * groups, kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            输出特征图 (B, C, H//stride, W//stride)
        """
        b, c, h, w = x.shape
        g = self.groups
        k = self.kernel_size
        c_per_g = self.channels_per_group
        pad = k // 2

        # Step 1: 在 stride 点生成 kernel
        # 先对输入下采样到输出分辨率
        if self.stride > 1:
            x_out = F.avg_pool2d(x, self.stride, stride=self.stride)
        else:
            x_out = x

        # Step 2: 生成网络 → kernel
        kernel_feat = self.generate(x_out)  # (B, reduced, H', W')
        kernel_weight = self.kernel_gen(kernel_feat)  # (B, k*k*g, H', W')

        oh, ow = kernel_weight.shape[2], kernel_weight.shape[3]

        # reshape kernel: (B, k*k*g, H', W') → (B, G, k*k, H', W')
        kernel_weight = kernel_weight.view(b, g, k * k, oh, ow)

        # Step 3: 将 kernel 转换为 unfold 用的权重
        # 对每个输出位置，用 unfold 取出 k*k 邻域，然后与对应 kernel 做点积

        # 使用 unfold 提取每个输出位置对应的邻域
        # unfold 输出: (B, C * k * k, H' * W')
        x_unfold = F.unfold(x, kernel_size=k, padding=pad, stride=self.stride)
        x_unfold = x_unfold.view(b, g, c_per_g, k * k, oh * ow)  # (B, G, C/G, k*k, N)

        # kernel_weight: (B, G, k*k, H', W') → (B, G, 1, k*k, N)
        kernel_weight = kernel_weight.view(b, g, 1, k * k, oh * ow)
        kernel_weight = torch.softmax(kernel_weight, dim=3)  # 可选：在 k*k 上归一化

        # 逐位置点积：(B, G, C/G, k*k, N) * (B, G, 1, k*k, N) → sum over k*k → (B, G, C/G, N)
        out = (x_unfold * kernel_weight).sum(dim=3)  # (B, G, C/G, N)

        # reshape 回空间
        out = out.view(b, c, oh, ow)

        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== Involution 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 小输入测试
    x = torch.randn(2, 32, 16, 16).to(device)
    inv = Involution(32, kernel_size=7, stride=1, reduction=4, groups=1).to(device)
    inv.eval()
    with torch.no_grad():
        y = inv(x)
    print(f"Involution (k=7, g=1): 输入 {x.shape} -> 输出 {y.shape}")

    # stride=2
    inv2 = Involution(32, kernel_size=7, stride=2, reduction=4, groups=1).to(device)
    inv2.eval()
    with torch.no_grad():
        y2 = inv2(x)
    print(f"Involution (k=7, stride=2): 输入 {x.shape} -> 输出 {y2.shape}")

    # groups=4
    inv3 = Involution(32, kernel_size=5, stride=1, reduction=4, groups=4).to(device)
    inv3.eval()
    with torch.no_grad():
        y3 = inv3(x)
    print(f"Involution (k=5, g=4): 输入 {x.shape} -> 输出 {y3.shape}")

    print(f"Involution 参数量: {sum(p.numel() for p in inv.parameters()):,}")
    print("全部通过!")

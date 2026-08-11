"""
ECA: Efficient Channel Attention
---------------------------------
高效通道注意力：使用 1D 卷积捕获局部跨通道交互，避免全连接层的降维操作。
自适应计算 1D 卷积核大小，保持通道间的局部交互能力。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "eca",
    category="attention",
    description="ECA 高效通道注意力：自适应核大小的 1D 卷积，无需降维，轻量高效",
)
class ECAAttention(nn.Module):
    """Efficient Channel Attention 模块

    通过 1D 卷积（无降维）学习通道注意力权重。
    卷积核大小根据通道数自适应计算：k = |(log2(C)/γ + b/γ)|_{odd}

    Args:
        channels: 输入通道数（如果传入则自动计算 kernel_size）
        gamma: 自适应核大小公式中的 γ 参数（默认 2）
        b: 自适应核大小公式中的 b 参数（默认 1）
        kernel_size: 手动指定 1D 卷积核大小（如果提供则覆盖自适应计算）
    """

    def __init__(
        self,
        channels: int = None,
        gamma: int = 2,
        b: int = 1,
        kernel_size: int = None,
    ):
        super(ECAAttention, self).__init__()

        if kernel_size is None:
            if channels is None:
                raise ValueError("channels 和 kernel_size 必须至少提供一个")
            # 自适应计算 kernel_size: k = |(log2(C) / γ + b / γ)|_odd
            t = int(abs(math.log2(channels) / gamma + b / gamma))
            kernel_size = t if t % 2 == 1 else t + 1

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            加权后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        # GAP → (B, C, 1) → transpose → (B, 1, C)
        y = x.mean(dim=[2, 3]).view(b, 1, c)
        # 1D 卷积 → (B, 1, C) → Sigmoid
        y = self.conv(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        # Scale
        return x * y


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== ECAAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for c in [64, 128, 256, 512]:
        model = ECAAttention(channels=c, gamma=2, b=1).to(device)
        x = torch.randn(2, c, 32, 32).to(device)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"  C={c:4d} | kernel_size={model.kernel_size} | "
              f"shape={tuple(y.shape)} | 参数量={sum(p.numel() for p in model.parameters())}")

    print(f"\n全部通过!")

"""
Squeeze-and-Excitation Attention (SE Attention)
------------------------------------------------
通道注意力机制：通过全局平均池化压缩空间信息，再通过两个全连接层
学习通道间的依赖关系，最后用 Sigmoid 生成通道权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "se_attention",
    category="attention",
    description="Squeeze-and-Excitation 通道注意力：GAP → FC → ReLU → FC → Sigmoid，自适应通道加权",
)
class SEAttention(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块

    对输入特征图的每个通道计算全局重要性权重，增强有用通道，抑制无用通道。

    Args:
        channels: 输入通道数
        reduction: 压缩比例，控制中间层通道数（默认 16）
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEAttention, self).__init__()
        reduced_channels = max(1, channels // reduction)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            加权后的特征图，shape (B, C, H, W)
        """
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化 → (B, C, 1, 1)
        y = self.global_avg_pool(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid → (B, C)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 逐通道加权
        return x * y


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== SEAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = SEAttention(channels=64, reduction=16).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"全部通过!")

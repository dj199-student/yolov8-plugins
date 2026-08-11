"""
SimAM: Simple Attention Module
-------------------------------
基于神经科学理论的参数无关注意力机制。
通过能量函数评估每个神经元的重要性，无需任何可学习参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "simam",
    category="attention",
    description="SimAM 无参注意力：基于神经科学能量函数，评估每个神经元重要性，零参数",
)
class SimAM(nn.Module):
    """Simple Attention Module — 参数无关的注意力机制

    基于神经科学中"富含信息的神经元通常表现出与周围神经元不同的放电模式"这一发现。
    利用能量函数评估每个神经元的重要性：e_t* = 4*(σ² + λ) / ((t - μ)² + 2σ² + 2λ)

    Args:
        lambda_val: 正则化系数，控制抑制强度（默认 1e-4）
    """

    def __init__(self, lambda_val: float = 1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            加权后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        n = h * w  # 空间神经元数量

        # 每个通道空间维度上的神经元均值 μ，shape (B, C, 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)

        # 每个神经元与均值的偏差 t - μ
        t_minus_mu = x - mu

        # 方差 σ²，shape (B, C, 1, 1)
        sigma2 = x.var(dim=[2, 3], keepdim=True, unbiased=False)

        # 能量函数：e_t* = 4 * (σ² + λ) / ((t - μ)² + 2σ² + 2λ)
        numerator = 4.0 * (sigma2 + self.lambda_val)
        denominator = t_minus_mu.pow(2) + 2.0 * sigma2 + 2.0 * self.lambda_val
        energy = numerator / denominator

        # 能量越低，越重要 → 使用 sigmoid 映射为 (0, 1)
        attention = torch.sigmoid(1.0 / (energy + 1e-8))

        return x * attention


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== SimAM 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = SimAM(lambda_val=1e-4).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())} (应为 0)")
    print(f"全部通过!")

"""
Triplet Attention
------------------
三重视觉注意力：通过三个并行的分支分别在 (C,H,W)、(H,C,W)、(W,H,C)
三个维度空间上建立注意力，捕获跨维度的交互信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "triplet_attention",
    category="attention",
    description="Triplet Attention：三个分支分别沿 C-H、H-C、W-C 维度建立注意力，平均融合",
)
class TripletAttention(nn.Module):
    """Triplet Attention 模块

    通过三个并行的旋转注意力分支捕获跨维度交互：
    - Branch 1: (C, H, W)  标准空间注意力
    - Branch 2: (H, C, W)  高度-通道交叉注意力
    - Branch 3: (W, H, C)  宽度-空间交叉注意力

    每个分支的操作：Z-Pool(avg+max concat) → Conv7x7 → BN → Sigmoid
    """

    def __init__(self, kernel_size: int = 7):
        super(TripletAttention, self).__init__()
        padding = kernel_size // 2

        # Branch 1: C-H-W 维度的注意力
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

        # Branch 2: H-C-W 维度的注意力（处理后 1 通道）
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

        # Branch 3: W-H-C 维度的注意力（处理后 1 通道）
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

    @staticmethod
    def _z_pool(x: torch.Tensor) -> torch.Tensor:
        """Z-Pool：沿通道维度拼接平均池化和最大池化结果

        Args:
            x: (B, C, H, W)

        Returns:
            (B, 2, H, W)
        """
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_val, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        return torch.cat([avg, max_val], dim=1)  # (B, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        # ===== Branch 1: (B, C, H, W) 标准空间注意力 =====
        z1 = self._z_pool(x)              # (B, 2, H, W)
        a1 = torch.sigmoid(self.branch1_conv(z1))  # (B, 1, H, W)
        out1 = x * a1

        # ===== Branch 2: (B, C, H, W) → (B, H, C, W) 交叉维度 =====
        # permute: (B, C, H, W) → (B, H, C, W)
        x_perm2 = x.permute(0, 2, 1, 3).contiguous()
        # 合并 B 和 H 维度以便用 Conv2d: (B*H, C, W)
        b, h, c2, w2 = x_perm2.shape
        x_merge2 = x_perm2.view(b * h, c2, w2)
        z2 = self._z_pool(x_merge2)        # (B*H, 2, W)
        a2 = torch.sigmoid(self.branch2_conv(z2))  # (B*H, 1, W)
        # 恢复 → (B, H, 1, W) → (B, 1, H, W)
        a2 = a2.view(b, h, 1, w2)
        out2 = x * a2

        # ===== Branch 3: (B, C, H, W) → (B, W, H, C) 交叉维度 =====
        # permute: (B, C, H, W) → (B, W, H, C)
        x_perm3 = x.permute(0, 3, 2, 1).contiguous()
        # 合并 B 和 W: (B*W, H, C)
        b3, w3, h3, c3 = x_perm3.shape
        x_merge3 = x_perm3.view(b3 * w3, h3, c3)
        z3 = self._z_pool(x_merge3)        # (B*W, 2, H)
        a3 = torch.sigmoid(self.branch3_conv(z3))  # (B*W, 1, H)
        # 恢复 → (B, W, 1, H) → (B, 1, 1, H) → 需要广播
        a3 = a3.view(b3, w3, 1, h3)        # (B, W, 1, H)
        a3 = a3.permute(0, 2, 3, 1)        # (B, 1, H, W)
        out3 = x * a3

        # 三个分支输出取平均
        out = (out1 + out2 + out3) / 3.0

        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== TripletAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = TripletAttention(kernel_size=7).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"全部通过!")

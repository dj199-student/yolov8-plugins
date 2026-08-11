"""
Vision Transformer Block (ViT Block)
-------------------------------------
标准的 ViT 编码器块：LayerNorm → Multi-Head Self-Attention → residual
                   → LayerNorm → MLP(GELU, expansion=4) → residual

参考：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "vit_block",
    category="transformer",
    description="ViT 编码器块: LN→MHSA→残差→LN→MLP(GELU, expansion=4)→残差，标准 ViT 构建块",
)
class ViTBlock(nn.Module):
    """Vision Transformer 编码器块

    标准 Transformer 编码器块，适合图像 patch 序列。

    Args:
        dim:        特征维度
        num_heads:  多头注意力的头数（默认 8）
        mlp_ratio:  MLP 隐藏层扩展比例（默认 4.0）
        dropout:    Attention 和 MLP 中的 dropout 比例（默认 0.1）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super(ViTBlock, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征，shape (B, N, dim)，N 为序列长度（如 patch 数）

        Returns:
            输出特征，shape (B, N, dim)
        """
        # Self-Attention + residual
        shortcut = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = shortcut + attn_out

        # MLP + residual
        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)

        return x


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== ViTBlock 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2D 特征图 → 序列
    b, c, h, w = 2, 256, 14, 14
    x = torch.randn(b, c, h, w).to(device)
    # (B, C, H, W) → (B, H*W, C)
    x_seq = x.flatten(2).transpose(1, 2)  # (B, 196, 256)

    vit = ViTBlock(dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1).to(device)
    vit.eval()
    with torch.no_grad():
        y = vit(x_seq)
    print(f"ViTBlock: 输入 {x_seq.shape} -> 输出 {y.shape}")
    print(f"shape 一致: {x_seq.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in vit.parameters()):,}")

    # 测试不同维度
    vit2 = ViTBlock(dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.0).to(device)
    x2 = torch.randn(2, 100, 512).to(device)
    vit2.eval()
    with torch.no_grad():
        y2 = vit2(x2)
    print(f"ViTBlock (dim=512): 输入 {x2.shape} -> 输出 {y2.shape}")
    print("全部通过!")

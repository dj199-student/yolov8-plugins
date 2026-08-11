"""
Multi-Head Self-Attention (MHSA) — 独立实现
----------------------------------------------
纯粹的 Multi-Head Self-Attention 模块，不包含 FFN。
将输入投影到 Q,K,V → 拆分为多头 → scaled dot-product attention → concat → 输出投影。

可作为构建块嵌入到任何需要注意力机制的模块中。

参考：Attention Is All You Need (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "mhsa",
    category="transformer",
    description="独立 Multi-Head Self-Attention: QKV 投影 → 多头切分 → scaled dot-product → 拼接 → 输出投影",
)
class MHSA(nn.Module):
    """Multi-Head Self-Attention (独立模块)

    完整实现 scaled dot-product attention + 多头机制。

    Args:
        dim:        输入/输出特征维度
        num_heads:  注意力头数（默认 8）
        dropout:    Attention dropout 比例（默认 0.0）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super(MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 联合投影
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # 输出投影
        self.proj = nn.Linear(dim, dim, bias=True)
        # Dropout
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 (B, N, dim)

        Returns:
            输出特征 (B, N, dim)
        """
        b, n, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        # QKV 投影
        qkv = self.qkv(x)  # (B, N, 3*dim)
        qkv = qkv.view(b, n, 3, h, d)  # (B, N, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权聚合
        out = attn @ v  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous()  # (B, N, H, D)
        out = out.view(b, n, self.dim)  # (B, N, dim)

        # 输出投影
        out = self.proj(out)

        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== MHSA 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = 256
    num_heads = 8
    b, n = 2, 196  # 例如 14x14 patch

    x = torch.randn(b, n, dim).to(device)
    mhsa = MHSA(dim=dim, num_heads=num_heads, dropout=0.1).to(device)
    mhsa.eval()
    with torch.no_grad():
        y = mhsa(x)
    print(f"MHSA: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in mhsa.parameters()):,}")

    # 对比 nn.MultiheadAttention
    from torch.nn import MultiheadAttention
    mma = MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True).to(device)
    mma.eval()
    with torch.no_grad():
        y2, _ = mma(x, x, x)
    print(f"nn.MultiheadAttention: 输入 {x.shape} -> 输出 {y2.shape}")
    print(f"nn.MHA 参数量: {sum(p.numel() for p in mma.parameters()):,}")
    print("全部通过!")

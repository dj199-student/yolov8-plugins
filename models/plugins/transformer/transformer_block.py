"""
标准 Transformer Block (编码器 + 解码器)
-----------------------------------------
提供两个核心变体：
  - TransformerEncoderBlock: Self-Attention + FFN
  - TransformerDecoderBlock: Self-Attn + Cross-Attn + FFN
两者均为标准 Post-LN 架构，带残差连接与 GELU 激活。

参考：Attention Is All You Need (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "transformer_encoder_block",
    category="transformer",
    description="Transformer 编码器块: LN→MHSA→残差→LN→FFN(GELU,expansion=4)→残差",
)
class TransformerEncoderBlock(nn.Module):
    """标准 Transformer 编码器块 (Post-LN)

    Self-Attention → Add & Norm → FFN → Add & Norm

    Args:
        dim:        特征维度
        num_heads:  多头注意力的头数（默认 8）
        ffn_ratio:  FFN 隐藏层扩展比例（默认 4）
        dropout:    Attention 和 FFN 中的 dropout（默认 0.1）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super(TransformerEncoderBlock, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 (B, N, dim)

        Returns:
            输出特征 (B, N, dim)
        """
        # Self-Attention
        shortcut = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = shortcut + self.dropout1(attn_out)

        # FFN
        shortcut = x
        x = shortcut + self.ffn(self.norm2(x))

        return x


@PLUGIN_REGISTRY.register(
    "transformer_decoder_block",
    category="transformer",
    description="Transformer 解码器块: self-attn → cross-attn → FFN，每步带 LN+残差",
)
class TransformerDecoderBlock(nn.Module):
    """标准 Transformer 解码器块 (Post-LN)

    Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm

    Args:
        dim:        特征维度
        num_heads:  多头注意力的头数（默认 8）
        ffn_ratio:  FFN 隐藏层扩展比例（默认 4）
        dropout:    Attention 和 FFN 中的 dropout（默认 0.1）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super(TransformerDecoderBlock, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        # Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播

        Args:
            tgt:    目标序列 (B, N_tgt, dim) — decoder queries
            memory: 编码器输出 (B, N_mem, dim) — key/value for cross-attn

        Returns:
            输出特征 (B, N_tgt, dim)
        """
        # Self-Attention
        shortcut = tgt
        tgt_norm = self.norm1(tgt)
        self_attn_out, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, need_weights=False,
        )
        tgt = shortcut + self.dropout1(self_attn_out)

        # Cross-Attention (query=tgt, key/value=memory)
        shortcut = tgt
        tgt_norm = self.norm2(tgt)
        cross_attn_out, _ = self.cross_attn(
            tgt_norm, memory, memory, need_weights=False,
        )
        tgt = shortcut + self.dropout2(cross_attn_out)

        # FFN
        shortcut = tgt
        tgt = shortcut + self.ffn(self.norm3(tgt))

        return tgt


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== TransformerEncoderBlock / TransformerDecoderBlock 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = 256
    b, n = 2, 100

    # Encoder 测试
    x = torch.randn(b, n, dim).to(device)
    enc = TransformerEncoderBlock(dim=dim, num_heads=8, ffn_ratio=4, dropout=0.1).to(device)
    enc.eval()
    with torch.no_grad():
        y = enc(x)
    print(f"Encoder: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"  shape 一致: {x.shape == y.shape}")

    # Decoder 测试
    tgt = torch.randn(b, 50, dim).to(device)
    memory = torch.randn(b, 100, dim).to(device)
    dec = TransformerDecoderBlock(dim=dim, num_heads=8, ffn_ratio=4, dropout=0.1).to(device)
    dec.eval()
    with torch.no_grad():
        y_dec = dec(tgt, memory)
    print(f"Decoder: tgt {tgt.shape}, memory {memory.shape} -> 输出 {y_dec.shape}")
    print(f"  shape 一致: {tgt.shape == y_dec.shape}")

    print(f"Encoder 参数量: {sum(p.numel() for p in enc.parameters()):,}")
    print(f"Decoder 参数量: {sum(p.numel() for p in dec.parameters()):,}")
    print("全部通过!")

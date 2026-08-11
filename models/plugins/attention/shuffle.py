"""
Shuffle Attention
------------------
分组注意力 + 通道混洗：将通道分为多个组，每组内一半做通道注意力，
一半做空间注意力，最后通过通道混洗促进组间信息交互。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "shuffle_attention",
    category="attention",
    description="Shuffle Attention：分组通道+空间注意力 → Concat → Channel Shuffle，轻量高效",
)
class ShuffleAttention(nn.Module):
    """Shuffle Attention 模块

    将通道等分成 G 组，每组内再将通道对半分：
    - 前一半做通道注意力（GAP → FC → Sigmoid）
    - 后一半做空间注意力（GroupNorm → FC → Sigmoid）
    合并后通过 Channel Shuffle 促进组间交互。

    Args:
        channels: 输入通道数
        groups: 分组数量（默认 8）
    """

    def __init__(self, channels: int, groups: int = 8):
        super(ShuffleAttention, self).__init__()
        self.groups = groups
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels)

        # channels 必须能被 2 * groups 整除
        self.channels_per_group = channels // groups
        self.half_channels = self.channels_per_group // 2

        # 通道注意力的全连接层（组内前一半通道）
        if self.half_channels > 0:
            self.channel_gate = nn.Sequential(
                nn.Linear(self.half_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.channel_gate = None

        # 空间注意力的全连接层（组内后一半通道）
        if self.half_channels > 0:
            self.spatial_gate = nn.Sequential(
                nn.Linear(self.half_channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

    def _channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """通道混洗操作

        Args:
            x: (B, C, H, W)

        Returns:
            混洗后 (B, C, H, W)
        """
        b, c, h, w = x.size()
        g = self.groups
        # reshape → (B, G, C//G, H, W) → transpose → (B, C//G, G, H, W) → reshape
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        g = self.groups
        cpg = self.channels_per_group  # channels per group

        # 分成 G 组: (B, G, cpg, H, W)
        x_grouped = x.view(b, g, cpg, h, w)

        # 每组内前半 / 后半
        x1 = x_grouped[:, :, :self.half_channels, :, :]  # 通道注意力分支
        x2 = x_grouped[:, :, self.half_channels:, :, :]  # 空间注意力分支

        # ---- 通道注意力分支 ----
        # GAP → (B, G, half, 1, 1) → squeeze → (B, G, half)
        x1_pool = x1.mean(dim=[3, 4])
        if self.channel_gate is not None:
            # 对每个位置的 G, half 做 FC → (B, G, 1) → unsqueeze
            x1_att = self.channel_gate(x1_pool).unsqueeze(-1).unsqueeze(-1)  # (B, G, 1, 1, 1)
            x1_out = x1 * x1_att
        else:
            x1_out = x1

        # ---- 空间注意力分支 ----
        # 先对整个 x 做 GroupNorm → 取 x2 对应部分
        x_norm = self.gn(x)  # (B, C, H, W)
        x_norm_grouped = x_norm.view(b, g, cpg, h, w)
        x2_norm = x_norm_grouped[:, :, self.half_channels:, :, :]  # (B, G, half, H, W)

        # 空间注意力：FC(gate) 对 channel 维度做 attention
        # 将 (B, G, half, H, W) → (B, G, H, W, half) → linear(channel) → sigmoid
        if self.spatial_gate is not None and self.half_channels > 0:
            x2_perm = x2_norm.permute(0, 1, 3, 4, 2)  # (B, G, H, W, half)
            x2_att = self.spatial_gate(x2_perm).permute(0, 1, 4, 2, 3)  # (B, G, 1, H, W)
            x2_out = x2 * x2_att
        else:
            x2_out = x2

        # ---- 合并并通道混洗 ----
        x_out = torch.cat([x1_out, x2_out], dim=2)  # (B, G, cpg, H, W)
        x_out = x_out.view(b, c, h, w)
        x_out = self._channel_shuffle(x_out)

        return x_out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== ShuffleAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 通道数需要能被 2 * groups 整除
    channels = 64
    groups = 8
    print(f"  channels={channels}, groups={groups}")
    print(f"  channels_per_group={channels // groups}, half={channels // groups // 2}")

    x = torch.randn(2, channels, 32, 32).to(device)
    model = ShuffleAttention(channels=channels, groups=groups).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"全部通过!")

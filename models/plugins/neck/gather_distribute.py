"""
Gather-and-Distribute Mechanism
--------------------------------
来自 Gold-YOLO / DAMO-YOLO 的 Gather-and-Distribute（GD）机制。
先 Gather（收集）多尺度特征通过注意力进行融合，
再 Distribute（分发）融合后的全局信息回各尺度。

参考：Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism (NeurIPS 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _GatherLow(nn.Module):
    """低层 Gather：融合 P3 和 P4 特征（高分辨率层）

    P3 和 P4 通过交叉注意力进行信息聚合。

    Args:
        in_channels_list: P3 和 P4 的通道数
        out_channels:     输出通道数
    """

    def __init__(self, in_channels_list, out_channels: int):
        super(_GatherLow, self).__init__()
        c_low, c_high = in_channels_list[0], in_channels_list[1]

        # Project both to unified channels
        self.proj_low = nn.Sequential(
            nn.Conv2d(c_low, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(c_high, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Attention fusion: query from low, key/value from high
        self.attn_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, p3, p4):
        """Gather low-level features.

        Args:
            p3: Feature at P3 level (B, C3, H3, W3)
            p4: Feature at P4 level (B, C4, H4, W4)

        Returns:
            Gathered feature at P3 spatial size (B, out_channels, H3, W3).
        """
        f_low = self.proj_low(p3)
        f_high = self.proj_high(p4)

        # Upsample high to match low spatial size
        f_high_up = F.interpolate(f_high, size=f_low.shape[2:], mode='nearest')

        # Cross-attention: high features attend to low features
        concat = torch.cat([f_low, f_high_up], dim=1)
        attn = self.attn_conv(concat)

        # Fuse with attention
        fused = torch.cat([f_low, f_high_up * attn], dim=1)
        return self.fusion(fused)


class _GatherHigh(nn.Module):
    """高层 Gather：融合 P4 和 P5 特征（低分辨率层）

    Args:
        in_channels_list: P4 和 P5 的通道数
        out_channels:     输出通道数
    """

    def __init__(self, in_channels_list, out_channels: int):
        super(_GatherHigh, self).__init__()
        c_mid, c_high = in_channels_list[1], in_channels_list[2]

        self.proj_mid = nn.Sequential(
            nn.Conv2d(c_mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(c_high, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, p4, p5):
        """Gather high-level features.

        Args:
            p4: Feature at P4 level (B, C4, H4, W4)
            p5: Feature at P5 level (B, C5, H5, W5)

        Returns:
            Gathered feature at P5 spatial size (B, out_channels, H5, W5).
        """
        f_mid = self.proj_mid(p4)
        f_high = self.proj_high(p5)

        # Downsample mid to match high spatial size
        f_mid_down = F.interpolate(f_mid, size=f_high.shape[2:], mode='nearest')

        concat = torch.cat([f_high, f_mid_down], dim=1)
        attn = self.attn_conv(concat)

        fused = torch.cat([f_high, f_mid_down * attn], dim=1)
        return self.fusion(fused)


class _Distribute(nn.Module):
    """Distribute：将 gathered 信息分发回各尺度

    通过可学习权重将 gathered 特征注入到原始特征中。

    Args:
        out_channels: 输出通道数
        num_levels:   特征层数
    """

    def __init__(self, out_channels: int, num_levels: int = 3):
        super(_Distribute, self).__init__()
        self.out_channels = out_channels
        self.num_levels = num_levels

        # Gate weights: learnable modulation per level
        self.gates = nn.Parameter(torch.zeros(num_levels, dtype=torch.float32), requires_grad=True)

    def forward(self, orig_feats, gathered_low, gathered_high):
        """Distribute gathered features to each level.

        Args:
            orig_feats:    List of original feature tensors [P3, P4, P5]
            gathered_low:  Gathered low-level feature (B, C, H3, W3)
            gathered_high: Gathered high-level feature (B, C, H5, W5)

        Returns:
            List of enhanced feature tensors.
        """
        gates = torch.sigmoid(self.gates)

        # P3: inject low gather
        p3 = orig_feats[0] + gates[0] * gathered_low

        # P4: inject both gathers (resized)
        g_low_to_p4 = F.interpolate(gathered_low, size=orig_feats[1].shape[2:], mode='nearest')
        g_high_to_p4 = F.interpolate(gathered_high, size=orig_feats[1].shape[2:], mode='nearest')
        p4 = orig_feats[1] + gates[1] * (g_low_to_p4 + g_high_to_p4) * 0.5

        # P5: inject high gather
        p5 = orig_feats[2] + gates[2] * gathered_high

        return [p3, p4, p5]


@PLUGIN_REGISTRY.register(
    "gather_distribute",
    category="neck",
    description="Gather-and-Distribute: Gold-YOLO 特征收集与分发，注意力融合 + 门控注入",
)
class GatherDistribute(nn.Module):
    """Gather-and-Distribute 机制（来自 Gold-YOLO / DAMO-YOLO）

    两阶段操作：
      1. Gather:  分别融合低层（P3+P4）和高层（P4+P5）的特征
      2. Distribute: 通过可学习门控将收集到的全局信息注入回各尺度

    Args:
        in_channels_list: 输入各层通道数列表 [C0, C1, C2]
        out_channels:     中间特征通道数（默认 256）
    """

    def __init__(self, in_channels_list, out_channels: int = 256):
        super(GatherDistribute, self).__init__()
        self.num_levels = len(in_channels_list)

        # Input projections to unify channels per level
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels_list
        ])

        # Gather modules
        self.gather_low = _GatherLow(in_channels_list, out_channels)
        self.gather_high = _GatherHigh(in_channels_list, out_channels)

        # Distribute module
        self.distribute = _Distribute(out_channels, self.num_levels)

        # Output projections back to original channels
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, inc, kernel_size=1, bias=False),
                nn.BatchNorm2d(inc),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels_list
        ])

    def forward(self, xs):
        """Forward pass of Gather-and-Distribute.

        Args:
            xs: List of feature maps [P3, P4, P5].

        Returns:
            List of enhanced feature maps with same spatial sizes and original channels.
        """
        projected = [proj(x) for proj, x in zip(self.input_projs, xs)]

        # Gather
        g_low = self.gather_low(xs[0], xs[1])      # from raw P3, P4
        g_high = self.gather_high(xs[1], xs[2])    # from raw P4, P5

        # Project gathered features to unified channels for distribution
        g_low_proj = projected[0].new_zeros(projected[0].shape)
        g_high_proj = projected[0].new_zeros(projected[0].shape)
        # Use gather outputs directly (they are already in out_channels)
        g_low_proj = g_low
        g_high_proj = g_high

        # Distribute to each level
        enhanced = self.distribute(projected, g_low_proj, g_high_proj)

        # Project back
        out = [proj(f) for proj, f in zip(self.output_projs, enhanced)]
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== GatherDistribute 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs = [
        torch.randn(2, 256, 80, 80).to(device),
        torch.randn(2, 512, 40, 40).to(device),
        torch.randn(2, 1024, 20, 20).to(device),
    ]

    model = GatherDistribute(in_channels_list=[256, 512, 1024], out_channels=256).to(device)
    model.eval()

    with torch.no_grad():
        outs = model(xs)

    print(f"输入 shapes: {[x.shape for x in xs]}")
    print(f"输出 shapes: {[o.shape for o in outs]}")
    for i, (x, o) in enumerate(zip(xs, outs)):
        assert x.shape == o.shape, f"Level {i} shape mismatch: {x.shape} vs {o.shape}"
        print(f"  Level {i}: {x.shape} -> {o.shape} OK")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'gather_distribute'")
    print(f"全部通过!")

"""
SDLI: Scale-Decoupled Layer Interaction
----------------------------------------
来自 DAMO-YOLO 的尺度解耦层间交互模块。
核心思想：按尺度解耦特征，再通过交叉尺度注意力进行层间交互。
先统一通道，再对每层施加 CBAM 式注意力，最后跨层可学习混合。

参考：DAMO-YOLO: A Report on Real-Time Object Detection Design (arXiv 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _ChannelAttention(nn.Module):
    """通道注意力（CBAM 风格的轻量版）"""

    def __init__(self, channels: int, reduction: int = 8):
        super(_ChannelAttention, self).__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class _SpatialAttention(nn.Module):
    """空间注意力（CBAM 风格）"""

    def __init__(self, kernel_size: int = 7):
        super(_SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(combined))


class _CBAMBottleneck(nn.Module):
    """带 CBAM 注意力的 bottleneck 块：1x1 conv -> CBAM -> 1x1 conv"""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8):
        super(_CBAMBottleneck, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ca = _ChannelAttention(mid_channels, reduction=reduction)
        self.sa = _SpatialAttention(kernel_size=7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out * self.ca(out)
        out = out * self.sa(out)
        out = self.conv2(out)
        return out


class _ScaleInteraction(nn.Module):
    """跨尺度交互：将其他层的特征缩放后以可学习权重混合到当前层。"""

    def __init__(self, level: int, num_levels: int, channels: int):
        super(_ScaleInteraction, self).__init__()
        self.level = level
        # Learnable mixing weights for contributions from each level
        self.mix_weights = nn.Parameter(torch.ones(num_levels, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feats):
        """Mix features from all levels into current level.

        Args:
            feats: List of tensors [F0, F1, F2], all in unified channels.

        Returns:
            Mixed tensor at this level's spatial size.
        """
        target_shape = feats[self.level].shape[2:]
        w = self.softmax(self.mix_weights)
        out = 0
        for i, f in enumerate(feats):
            if f.shape[2] != target_shape[0] or f.shape[3] != target_shape[1]:
                f_resized = F.interpolate(f, size=target_shape, mode='nearest')
            else:
                f_resized = f
            out = out + w[i] * f_resized
        return out


@PLUGIN_REGISTRY.register(
    "sdli",
    category="neck",
    description="SDLI: 尺度解耦层间交互，CBAM 注意力增强 + 可学习跨层权重混合",
)
class SDLI(nn.Module):
    """Scale-Decoupled Layer Interaction（来自 DAMO-YOLO）

    对每个尺度独立提取注意力增强特征，再通过可学习权重
    将各层特征混合，实现层间信息交互而不过度耦合。

    Args:
        in_channels_list: 输入各层通道数列表 [C0, C1, C2]
        out_channels:     统一中间通道数（默认 256）
    """

    def __init__(self, in_channels_list, out_channels: int = 256):
        super(SDLI, self).__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels

        # 1x1 input projections to unify channels
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels_list
        ])

        # CBAM-style attention per level
        self.attentions = nn.ModuleList([
            _CBAMBottleneck(out_channels, out_channels)
            for _ in range(self.num_levels)
        ])

        # Cross-scale interaction per level
        self.interactions = nn.ModuleList([
            _ScaleInteraction(level=i, num_levels=self.num_levels, channels=out_channels)
            for i in range(self.num_levels)
        ])

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
        """Forward pass of SDLI.

        Args:
            xs: List of feature maps [P3, P4, P5], each (B, C_i, H_i, W_i).

        Returns:
            List of feature maps with same spatial sizes and original channels.
        """
        # Unify channels
        projected = [proj(x) for proj, x in zip(self.input_projs, xs)]

        # Apply CBAM attention per level
        attended = [attn(p) for attn, p in zip(self.attentions, projected)]

        # Cross-scale mixing
        mixed = [inter(attended) for inter in self.interactions]

        # Fuse original attended + mixed (residual connection)
        fused = [attn + mix for attn, mix in zip(attended, mixed)]

        # Project back to original channels
        out = [proj(f) for proj, f in zip(self.output_projs, fused)]
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== SDLI 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs = [
        torch.randn(2, 256, 80, 80).to(device),
        torch.randn(2, 512, 40, 40).to(device),
        torch.randn(2, 1024, 20, 20).to(device),
    ]

    model = SDLI(in_channels_list=[256, 512, 1024], out_channels=256).to(device)
    model.eval()

    with torch.no_grad():
        outs = model(xs)

    print(f"输入 shapes: {[x.shape for x in xs]}")
    print(f"输出 shapes: {[o.shape for o in outs]}")
    for i, (x, o) in enumerate(zip(xs, outs)):
        assert x.shape == o.shape, f"Level {i} shape mismatch: {x.shape} vs {o.shape}"
        print(f"  Level {i}: {x.shape} -> {o.shape} OK")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'sdli'")
    print(f"全部通过!")

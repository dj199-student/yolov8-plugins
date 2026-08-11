"""
BiFPN: Bi-directional Feature Pyramid Network
----------------------------------------------
来自 EfficientDet 的双向特征金字塔网络。
通过可学习的加权特征融合，在自上而下和自下而上两条路径上
反复融合多尺度特征，同时保留跨尺度直连（skip connection）。

参考：EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _FusionNode(nn.Module):
    """BiFPN 加权融合节点

    对多个输入特征图执行加权求和（learnable per-input weights），
    后接 1x1 卷积 + BN + 激活函数。

    Args:
        num_inputs: 参与融合的输入数量
        channels:   统一的通道数
    """

    def __init__(self, num_inputs: int, channels: int):
        super(_FusionNode, self).__init__()
        # 每个输入一个可学习权重，用 ReLU 确保非负
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.eps = 1e-4

    def forward(self, xs):
        """Weighted fusion of multiple feature tensors.

        Args:
            xs: List of tensors, each (B, C, H, W), all same spatial size.

        Returns:
            Fused tensor (B, C, H, W).
        """
        w = F.relu(self.weights)
        w_sum = w.sum() + self.eps
        w_norm = w / w_sum
        out = sum(wi * xi for wi, xi in zip(w_norm, xs))
        return self.conv(out)


class _BiFPNLayer(nn.Module):
    """BiFPN 单层：一次完整的 top-down + bottom-up 双向融合。

    Args:
        channels:  统一输出通道数
        num_levels: 特征层数（默认 3，对应 P3/P4/P5）
    """

    def __init__(self, channels: int, num_levels: int = 3):
        super(_BiFPNLayer, self).__init__()
        self.channels = channels
        self.num_levels = num_levels

        # Top-down fusion nodes (from high to low, 2 inputs each: lateral + upsample)
        self.td_fusions = nn.ModuleList()
        for i in range(num_levels - 1):  # P4_td, P3_td
            self.td_fusions.append(_FusionNode(num_inputs=2, channels=channels))

        # Bottom-up fusion nodes (from low to high, 3 inputs: input + td + downsample)
        self.bu_fusions = nn.ModuleList()
        for i in range(num_levels - 1):  # P4_out, P5_out
            self.bu_fusions.append(_FusionNode(num_inputs=3 if i == 0 else 2, channels=channels))

    def forward(self, xs):
        """Forward pass of one BiFPN layer.

        Args:
            xs: List of tensors [P3, P4, P5], each (B, C_in_i, H_i, W_i).

        Returns:
            List of tensors [P3_out, P4_out, P5_out], each (B, channels, H_i, W_i).
        """
        # --- Top-down pathway ---
        td_features = [None] * self.num_levels
        # P5_td = P5 (top level is pass-through from input)
        td_current = xs[-1]
        td_features[-1] = td_current
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample td_current to match xs[i] spatial size
            up = F.interpolate(td_current, size=xs[i].shape[2:], mode='nearest')
            td_current = self.td_fusions[i]([xs[i], up])
            td_features[i] = td_current

        # --- Bottom-up pathway ---
        out_features = [None] * self.num_levels
        # P3_out = P3_td initially
        bu_current = td_features[0]
        out_features[0] = bu_current
        for i in range(1, self.num_levels):
            # Downsample bu_current to match xs[i] spatial size
            down = F.interpolate(bu_current, size=xs[i].shape[2:], mode='nearest')
            # P4_out fuses: xs[i] (original input), td_features[i] (top-down), down (bottom-up)
            if i == 1:
                bu_current = self.bu_fusions[i - 1]([xs[i], td_features[i], down])
            else:
                bu_current = self.bu_fusions[i - 1]([xs[i], td_features[i]])
            out_features[i] = bu_current

        return out_features


@PLUGIN_REGISTRY.register(
    "bifpn",
    category="neck",
    description="BiFPN: EfficientDet 双向特征金字塔，可学习加权融合，多层重复 top-down + bottom-up 路径",
)
class BiFPN(nn.Module):
    """Bi-directional Feature Pyramid Network (来自 EfficientDet)

    对多尺度特征图（P3, P4, P5）进行反复的双向加权融合。
    每层包含自顶向下和自底向上两条路径，通过可学习的权重决定
    每个输入的贡献比例。

    Args:
        in_channels_list: 输入各层的通道数列表，如 [256, 512, 1024]
        out_channels:     统一输出通道数（默认 256）
        num_layers:       BiFPN 层重复次数（默认 3）
    """

    def __init__(self, in_channels_list, out_channels: int = 256, num_layers: int = 3):
        super(BiFPN, self).__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels

        # 1x1 projections to unify channel count
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inc, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels_list
        ])

        # Stacked BiFPN layers
        self.layers = nn.ModuleList([
            _BiFPNLayer(out_channels, self.num_levels)
            for _ in range(num_layers)
        ])

        # Output projections back to original channel dimensions
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, inc, kernel_size=1, bias=False),
                nn.BatchNorm2d(inc),
                nn.ReLU(inplace=True),
            )
            for inc in in_channels_list
        ])

    def forward(self, xs):
        """Forward pass of BiFPN.

        Args:
            xs: List of feature maps [P3, P4, P5], each (B, C_i, H_i, W_i).

        Returns:
            List of feature maps [P3_out, P4_out, P5_out] with same spatial sizes
            and original channel dimensions.
        """
        # Project to unified channel count
        projected = [proj(x) for proj, x in zip(self.input_projs, xs)]

        # Apply stacked BiFPN layers
        feats = projected
        for layer in self.layers:
            feats = layer(feats)

        # Project back to original channel counts
        out = [proj(f) for proj, f in zip(self.output_projs, feats)]
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== BiFPN 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate P3, P4, P5 features
    xs = [
        torch.randn(2, 256, 80, 80).to(device),   # P3
        torch.randn(2, 512, 40, 40).to(device),   # P4
        torch.randn(2, 1024, 20, 20).to(device),  # P5
    ]

    model = BiFPN(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        num_layers=3,
    ).to(device)
    model.eval()

    with torch.no_grad():
        outs = model(xs)

    print(f"输入 shapes: {[x.shape for x in xs]}")
    print(f"输出 shapes: {[o.shape for o in outs]}")
    for i, (x, o) in enumerate(zip(xs, outs)):
        assert x.shape == o.shape, f"Level {i} shape mismatch: {x.shape} vs {o.shape}"
        print(f"  Level {i}: {x.shape} -> {o.shape} OK")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'bifpn'")
    print(f"全部通过!")

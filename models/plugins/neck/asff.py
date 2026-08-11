"""
ASFF: Adaptive Spatial Feature Fusion
--------------------------------------
来自 YOLOv3-ASFF 的自适应空间特征融合。
对每一层输出，将其他层的特征图缩放到相同空间尺寸后，
通过学习到的空间权重软选择各层的贡献。

参考：Learning Spatial Fusion for Single-Shot Object Detection (arXiv 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _ASFFLevel(nn.Module):
    """单个 ASFF 层的自适应空间融合

    将三个不同尺度的特征图缩放到当前层尺寸，通过 softmax 学习
    各层在每个空间位置上的融合权重。

    Args:
        level:            当前输出层级索引 (0/1/2 对应 P3/P4/P5)
        in_channels_list: 三层输入通道数列表 [C0, C1, C2]
        out_channels:     输出通道数（None 则使用当前层原始通道数）
    """

    def __init__(self, level: int, in_channels_list, out_channels: int = None):
        super(_ASFFLevel, self).__init__()
        self.level = level
        self.num_levels = len(in_channels_list)

        out_ch = out_channels if out_channels is not None else in_channels_list[level]

        # For each other level, define how to resize to current level's spatial size
        self.compress_convs = nn.ModuleList()
        for i, inc in enumerate(in_channels_list):
            if i == level:
                # Same level: just 1x1 compress
                self.compress_convs.append(
                    nn.Sequential(
                        nn.Conv2d(inc, out_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            elif i < level:
                # Higher resolution -> need to downsample
                # Use stride-2 conv for each level difference
                stride = 2 ** (level - i)
                seq_layers = []
                cur_ch = inc
                # If multiple strides, use intermediate convs
                for s in range(level - i):
                    mid_ch = out_ch
                    seq_layers.extend([
                        nn.Conv2d(cur_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(mid_ch),
                        nn.ReLU(inplace=True),
                    ])
                    cur_ch = mid_ch
                # Final 1x1 to ensure exact channel count
                seq_layers.append(nn.Conv2d(cur_ch, out_ch, kernel_size=1, bias=False))
                seq_layers.append(nn.BatchNorm2d(out_ch))
                seq_layers.append(nn.ReLU(inplace=True))
                self.compress_convs.append(nn.Sequential(*seq_layers))
            else:
                # Lower resolution -> need to upsample
                scale_factor = 2 ** (i - level)
                self.compress_convs.append(
                    nn.Sequential(
                        nn.Conv2d(inc, out_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
                # Store scale factor for this level
                self.register_buffer(f'scale_{i}', torch.tensor(scale_factor), persistent=False)

        # Weight conv: 3*out_ch -> 3-channel spatial weight map
        self.weight_conv = nn.Sequential(
            nn.Conv2d(out_ch * self.num_levels, self.num_levels, kernel_size=1, bias=False),
        )

        self.out_channels = out_ch

    def forward(self, xs):
        """Forward pass of ASFF level.

        Args:
            xs: List of 3 feature tensors [F0, F1, F2] at different scales.

        Returns:
            Fused tensor (B, out_ch, H_level, W_level).
        """
        resized = []
        target_h, target_w = xs[self.level].shape[2], xs[self.level].shape[3]

        for i, (x, compress) in enumerate(zip(xs, self.compress_convs)):
            if i < self.level:
                # Downsample via stride conv in compress_convs
                resized.append(compress(x))
            elif i == self.level:
                resized.append(compress(x))
            else:
                # Upsample: compress first, then interpolate
                compressed = compress(x)
                if compressed.shape[2] != target_h or compressed.shape[3] != target_w:
                    compressed = F.interpolate(compressed, size=(target_h, target_w), mode='nearest')
                resized.append(compressed)

        # Concatenate all resized features -> (B, 3*out_ch, H, W)
        concat = torch.cat(resized, dim=1)

        # Compute spatial weights via 1x1 conv + softmax along level dim
        weight = self.weight_conv(concat)  # (B, 3, H, W)
        weight = F.softmax(weight, dim=1)   # normalize across levels

        # Weighted sum
        out = sum(weight[:, i:i+1] * resized[i] for i in range(self.num_levels))
        return out


@PLUGIN_REGISTRY.register(
    "asff",
    category="neck",
    description="ASFF: 自适应空间特征融合，学习每层每个空间位置的最优融合权重（softmax 归一化）",
)
class ASFF(nn.Module):
    """Adaptive Spatial Feature Fusion（来自 YOLOv3-ASFF）

    为每个输出层级学习一个空间注意力权重图，自适应地融合来自
    三个不同尺度的特征，解决了多尺度特征间的冲突问题。

    Args:
        in_channels_list: 三层输入通道数列表 [C0, C1, C2]
        out_channels:     输出通道数（None 保持各层原始通道数）
    """

    def __init__(self, in_channels_list, out_channels: int = None):
        super(ASFF, self).__init__()
        self.num_levels = len(in_channels_list)
        self.levels = nn.ModuleList([
            _ASFFLevel(level=i, in_channels_list=in_channels_list, out_channels=out_channels)
            for i in range(self.num_levels)
        ])

    def forward(self, xs):
        """Forward pass of ASFF.

        Args:
            xs: List of feature maps [P3, P4, P5], each (B, C_i, H_i, W_i).

        Returns:
            List of fused feature maps, same spatial sizes.
        """
        return [level(xs) for level in self.levels]


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== ASFF 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs = [
        torch.randn(2, 256, 80, 80).to(device),   # P3 (level 0)
        torch.randn(2, 512, 40, 40).to(device),   # P4 (level 1)
        torch.randn(2, 1024, 20, 20).to(device),  # P5 (level 2)
    ]

    model = ASFF(in_channels_list=[256, 512, 1024]).to(device)
    model.eval()

    with torch.no_grad():
        outs = model(xs)

    print(f"输入 shapes: {[x.shape for x in xs]}")
    print(f"输出 shapes: {[o.shape for o in outs]}")
    for i, (x, o) in enumerate(zip(xs, outs)):
        assert x.shape[0] == o.shape[0], f"Level {i} batch mismatch"
        assert x.shape[2] == o.shape[2], f"Level {i} height mismatch: {x.shape[2]} vs {o.shape[2]}"
        assert x.shape[3] == o.shape[3], f"Level {i} width mismatch: {x.shape[3]} vs {o.shape[3]}"
        print(f"  Level {i}: {x.shape} -> {o.shape} OK")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'asff'")
    print(f"全部通过!")

"""
DyHead: Dynamic Head
---------------------
来自 DyHead 论文的注意力驱动动态检测头。
通过统一的三维注意力（scale-aware, spatial-aware, task-aware）
动态增强检测头的特征表示，显著提升检测性能。

参考：Dynamic Head: Unifying Object Detection Heads with Attentions (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


class _ScaleAwareAttention(nn.Module):
    """尺度感知注意力：对每个通道学习一个全局 scale 响应

    AvgPool(H,W) → FC → ReLU → FC → Sigmoid → 逐通道乘回
    """

    def __init__(self, channels: int):
        super(_ScaleAwareAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        b, c = x.shape[:2]
        attn = self.pool(x).view(b, c)  # (B, C)
        attn = self.fc(attn).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * attn


class _SpatialAwareAttention(nn.Module):
    """空间感知注意力：使用可变形感知的简化实现

    用 3x3 卷积预测空间偏移，再通过 3x3 卷积生成空间注意力图。
    """

    def __init__(self, channels: int):
        super(_SpatialAwareAttention, self).__init__()
        # Offset prediction: predict 2*K*K offsets for deformable-like behavior
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
        )
        # Spatial attention conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        offset_feat = self.offset_conv(x)
        attn = self.spatial_conv(offset_feat)  # (B, C, H, W)
        return x * attn


class _TaskAwareAttention(nn.Module):
    """任务感知注意力：动态调整通道对不同任务的响应

    对每个空间位置学习通道加权，用全局统计量归一化。
    """

    def __init__(self, channels: int):
        super(_TaskAwareAttention, self).__init__()
        self.norm = nn.LayerNorm([channels])
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
        )

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        b, c, h, w = x.shape
        # Reshape to (B*H*W, C) for layer norm
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)
        x_norm = self.norm(x_flat)
        # Learnable gating
        gate = self.fc(x_norm.mean(dim=0, keepdim=True))  # (1, C)
        gate = torch.sigmoid(gate).view(1, c, 1, 1)
        return x * gate


class DyHeadBlock(nn.Module):
    """Dynamic Head Block — 单层注意力增强块

    依次应用 scale-aware, spatial-aware, task-aware 三种注意力，
    并以残差连接加强。

    Args:
        channels: 特征通道数
    """

    def __init__(self, channels: int):
        super(DyHeadBlock, self).__init__()
        self.scale_attn = _ScaleAwareAttention(channels)
        self.spatial_attn = _SpatialAwareAttention(channels)
        self.task_attn = _TaskAwareAttention(channels)

    def forward(self, x):
        """Forward pass of DyHeadBlock.

        Args:
            x: Feature tensor (B, C, H, W)

        Returns:
            Enhanced feature tensor (B, C, H, W)
        """
        identity = x
        x = self.scale_attn(x)
        x = self.spatial_attn(x)
        x = self.task_attn(x)
        return x + identity


@PLUGIN_REGISTRY.register(
    "dy_head",
    category="head",
    description="DyHead: 动态检测头，统一 scale/spatial/task 三维注意力，堆叠多块增强",
)
class DyHead(nn.Module):
    """Dynamic Head（来自 DyHead 论文）

    对每个 FPN 层级应用多个 DyHeadBlock，通过三维注意力
    （scale + spatial + task）动态增强检测头特征。

    Args:
        in_channels_list: 各 FPN 层级通道数列表 [C0, C1, C2]
        num_blocks:       DyHeadBlock 堆叠数量（默认 6，平均分配到各层）
    """

    def __init__(self, in_channels_list, num_blocks: int = 6):
        super(DyHead, self).__init__()
        self.num_levels = len(in_channels_list)

        # Compute blocks per level
        blocks_per_level = num_blocks // self.num_levels
        remainder = num_blocks % self.num_levels

        # Create blocks for each level
        self.level_blocks = nn.ModuleList()
        for i, ch in enumerate(in_channels_list):
            n = blocks_per_level + (1 if i < remainder else 0)
            level = nn.ModuleList([DyHeadBlock(ch) for _ in range(n)])
            self.level_blocks.append(level)

        # Store channels list for reference
        self.in_channels_list = in_channels_list

    def forward(self, xs):
        """Forward pass of DyHead.

        Args:
            xs: List of feature maps [P3, P4, P5], each (B, C_i, H_i, W_i).

        Returns:
            List of enhanced feature maps with same shapes.
        """
        out = []
        for i, x in enumerate(xs):
            for block in self.level_blocks[i]:
                x = block(x)
            out.append(x)
        return out


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DyHead 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs = [
        torch.randn(2, 256, 80, 80).to(device),
        torch.randn(2, 512, 40, 40).to(device),
        torch.randn(2, 1024, 20, 20).to(device),
    ]

    model = DyHead(in_channels_list=[256, 512, 1024], num_blocks=6).to(device)
    model.eval()

    with torch.no_grad():
        outs = model(xs)

    print(f"输入 shapes: {[x.shape for x in xs]}")
    print(f"输出 shapes: {[o.shape for o in outs]}")
    for i, (x, o) in enumerate(zip(xs, outs)):
        assert x.shape == o.shape, f"Level {i} shape mismatch: {x.shape} vs {o.shape}"
        print(f"  Level {i}: {x.shape} -> {o.shape} OK")
        print(f"    Block 数量: {len(model.level_blocks[i])}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"已注册名称: 'dy_head'")
    print(f"全部通过!")

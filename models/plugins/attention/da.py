"""
Dual Attention (DANet)
-----------------------
双注意力网络：并行使用位置注意力模块（PAM）和通道注意力模块（CAM），
分别捕获空间维度和通道维度的长距离依赖关系，求和融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "pam",
    category="attention",
    description="Position Attention Module：空间自注意力，捕获任意两个空间位置之间的长距离依赖",
)
class PAM(nn.Module):
    """Position Attention Module — 位置注意力模块

    通过空间自注意力机制建立任意两个空间位置之间的依赖关系：
    - Reshape (B, C, H, W) → (B, C, N),  N = H*W
    - 注意力图 S = softmax(Q^T · K)，shape (N, N)
    - 输出 = V · S^T → reshape + residual

    Args:
        channels: 输入通道数
    """

    def __init__(self, channels: int):
        super(PAM, self).__init__()
        inner_channels = channels // 8

        # 三个 1x1 卷积用于生成 query, key, value
        self.query_conv = nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 可学习的缩放参数 gamma（初始化为 0，让网络逐渐学习残差强度）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        n = h * w  # 空间位置总数

        # Query: (B, C//8, H, W) → (B, C//8, N)
        query = self.query_conv(x).view(b, -1, n)
        # Key: (B, C//8, H, W) → (B, C//8, N)
        key = self.key_conv(x).view(b, -1, n)
        # Value: (B, C, H, W) → (B, C, N)
        value = self.value_conv(x).view(b, -1, n)

        # 注意力图: Q^T · K → (B, N, N)
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)  # 沿每行（每个 query 对所有 key）归一化

        # 加权聚合: V · S^T → (B, C, N)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        # 残差连接（gamma 可学习）
        return self.gamma * out + x


@PLUGIN_REGISTRY.register(
    "cam",
    category="attention",
    description="Channel Attention Module：通道自注意力，捕获任意两个通道之间的长距离依赖",
)
class CAM(nn.Module):
    """Channel Attention Module — 通道注意力模块

    通过通道自注意力机制建立任意两个通道之间的依赖关系：
    - Reshape (B, C, H, W) → (B, C, N),  N = H*W
    - 通道亲和力图 X = softmax(A · A^T)，shape (C, C)
    - 输出 = X · A → reshape + residual

    若设置 reduction > 1，则先降维以节省计算量。

    Args:
        channels: 输入通道数
        reduction: 压缩比例（默认 16），设为 1 表示不降维
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(CAM, self).__init__()

        self.use_reduction = reduction > 1
        in_channels = channels

        if self.use_reduction:
            # 降维和还原
            reduced_channels = max(1, channels // reduction)
            self.squeeze = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
            self.expand = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
            in_channels = reduced_channels

        self.in_channels = in_channels   # 用于前向时的实际通道数

        # 可学习的缩放参数 gamma
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        n = h * w

        # 可选降维
        if self.use_reduction:
            x_proj = self.squeeze(x)  # (B, C//r, H, W)
            cc = x_proj.size(1)
        else:
            x_proj = x
            cc = c

        # Reshape: (B, C', H, W) → (B, C', N)
        x_flat = x_proj.view(b, cc, n)

        # 通道亲和力矩阵: (B, C', N) × (B, N, C') → (B, C', C')
        attention = torch.bmm(x_flat, x_flat.permute(0, 2, 1))
        attention = F.softmax(attention, dim=-1)

        # 聚合: (B, C', C') × (B, C', N) → (B, C', N)
        out = torch.bmm(attention, x_flat)
        out = out.view(b, cc, h, w)

        # 如果做了降维，还原通道数
        if self.use_reduction:
            out = self.expand(out)  # (B, C, H, W)

        # 残差连接
        return self.gamma * out + x


@PLUGIN_REGISTRY.register(
    "dual_attention",
    category="attention",
    description="Dual Attention (DANet)：PAM 位置注意力 + CAM 通道注意力并行，求和融合输出",
)
class DualAttention(nn.Module):
    """Dual Attention Module — 双注意力模块

    并行使用位置注意力（PAM）和通道注意力（CAM），
    将两者的输出求和融合，同时捕获空间维度和通道维度的长距离依赖。

    Args:
        channels: 输入通道数
        reduction: CAM 中的压缩比例（默认 16）
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(DualAttention, self).__init__()
        self.pam = PAM(channels)
        self.cam = CAM(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            增强后的特征图，shape (B, C, H, W)
        """
        out_pam = self.pam(x)
        out_cam = self.cam(x)
        return out_pam + out_cam


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DualAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 较小分辨率避免大矩阵乘法 OOM
    x = torch.randn(2, 64, 16, 16).to(device)

    pam = PAM(channels=64).to(device)
    cam = CAM(channels=64, reduction=16).to(device)
    dual = DualAttention(channels=64, reduction=16).to(device)

    pam.train()
    cam.train()
    dual.train()

    print(f"输入 shape: {x.shape}")

    y_pam = pam(x)
    print(f"PAM 输出 shape:          {y_pam.shape}  | 一致: {x.shape == y_pam.shape}")

    y_cam = cam(x)
    print(f"CAM 输出 shape:          {y_cam.shape}  | 一致: {x.shape == y_cam.shape}")

    y_dual = dual(x)
    print(f"DualAttention 输出 shape: {y_dual.shape}  | 一致: {x.shape == y_dual.shape}")

    print(f"\nPAM 参数量:    {sum(p.numel() for p in pam.parameters()):,}")
    print(f"CAM 参数量:    {sum(p.numel() for p in cam.parameters()):,}")
    print(f"Dual 总参数量:  {sum(p.numel() for p in dual.parameters()):,}")
    print(f"全部通过!")

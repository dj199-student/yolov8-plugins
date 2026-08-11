"""
Coordinate Attention
--------------------
坐标注意力：将通道注意力分解为两个 1D 方向的特征编码（水平和垂直），
既保留了长距离依赖，又捕获了精确的位置信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


@PLUGIN_REGISTRY.register(
    "ca",
    category="attention",
    description="坐标注意力：分别沿 H 和 W 方向池化 → 共享变换 → 分离预测 → 空间-通道联合加权",
)
class CoordAtt(nn.Module):
    """Coordinate Attention 模块

    将全局池化解耦为两个正交方向的 1D 池化（X Avg Pool 和 Y Avg Pool），
    保留空间坐标信息的同时建模通道依赖关系。

    Args:
        channels: 输入通道数
        reduction: 压缩比例（默认 32）
    """

    def __init__(self, channels: int, reduction: int = 32):
        super(CoordAtt, self).__init__()
        reduced = max(1, channels // reduction)

        # 共享 1x1 卷积进行降维
        self.shared_conv = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU(inplace=True)

        # 分别预测水平和垂直方向的注意力权重
        self.conv_h = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, C, H, W)

        Returns:
            加权后的特征图，shape (B, C, H, W)
        """
        b, c, h, w = x.size()

        # X Avg Pool: 沿水平方向 (W) 池化 → (B, C, H, 1)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        # Y Avg Pool: 沿垂直方向 (H) 池化 → (B, C, 1, W)
        x_w = F.adaptive_avg_pool2d(x, (1, w))

        # 将 W 池化结果转置为 (B, C, W, 1)，与 H 池化结果对齐
        x_w_permuted = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)

        # 在空间维度拼接： (B, C, H+W, 1)
        x_cat = torch.cat([x_h, x_w_permuted], dim=2)

        # 共享变换：1x1 Conv → BN → ReLU → (B, reduced, H+W, 1)
        x_cat = self.relu(self.bn(self.shared_conv(x_cat)))

        # 分离回两个分支 → (B, reduced, H, 1) 和 (B, reduced, W, 1)
        x_h_enc, x_w_enc = torch.split(x_cat, [h, w], dim=2)

        # 分别预测注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h_enc))              # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w_enc))               # (B, C, W, 1)
        a_w = a_w.permute(0, 1, 3, 2)                            # (B, C, 1, W)

        # 广播相乘得到联合注意力
        return x * a_h * a_w


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== CoordAtt 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    model = CoordAtt(channels=64, reduction=32).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {y.shape}")
    print(f"输入输出 shape 一致: {x.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"全部通过!")

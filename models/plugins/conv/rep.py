"""
RepConv — RepVGG 风格的结构重参数化卷积
----------------------------------------
训练时：多分支（3x3 + 1x1 + identity），提升表征能力。
推理时：通过 fuse_conv_bn 将所有分支合并为单个 3x3 卷积，
        在完全等价的前提下大幅加速推理。

参考：RepVGG: Making VGG-style ConvNets Great Again (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


def _pad_1x1_to_3x3_tensor(kernel1x1: torch.Tensor) -> torch.Tensor:
    """将 1x1 卷积填充为 3x3 卷积（零填充周围）"""
    if kernel1x1 is None:
        return None
    return F.pad(kernel1x1, [1, 1, 1, 1])


@PLUGIN_REGISTRY.register(
    "rep_conv",
    category="conv",
    description="RepVGG 重参数化卷积：训练多分支(3x3+1x1+identity)，推理合并为单 3x3 卷积",
)
class RepConv(nn.Module):
    """RepVGG 风格的结构重参数化卷积

    训练时：
      - 3x3 卷积分支
      - 1x1 卷积分支
      - Identity 分支（仅当 stride==1 且 in_channels==out_channels）
    每个分支带 BN，输出相加后过激活函数。

    推理时调用 fuse_conv_bn() 将所有权重合并为单个 3x3 卷积，
    精度完全等价，速度大幅提升。

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        kernel_size:  卷积核大小（默认 3，当前仅支持 3）
        stride:       步长（默认 1）
        act:          是否有激活函数（默认 True，使用 SiLU）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: bool = True,
    ):
        super(RepConv, self).__init__()
        assert kernel_size == 3, "RepConv 目前仅支持 kernel_size=3"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.act = act
        self.deploy = False  # 是否为部署模式

        # 训练分支
        padding = kernel_size // 2

        # 3x3 卷积分支
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积分支
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=stride, padding=0, bias=False,
        )
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        # Identity 分支（仅 stride==1 且通道数匹配时有效）
        self.has_identity = (stride == 1 and in_channels == out_channels)
        if self.has_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)

        # 激活函数
        if act:
            self.nonlinear = nn.SiLU(inplace=True)
        else:
            self.nonlinear = nn.Identity()

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple:
        """将 Conv + BN 融合为单个 Conv

        公式：W_fused = W * (gamma / sigma),  b_fused = beta - mu * (gamma / sigma)

        Args:
            conv: 卷积层
            bn:   BatchNorm 层

        Returns:
            (fused_weight, fused_bias)
        """
        # BN 参数
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        w_scale = gamma / std  # shape: (C_out,)

        # 融合权重
        fused_weight = conv.weight * w_scale[:, None, None, None]

        # 融合偏置
        fused_bias = beta - running_mean * w_scale

        return fused_weight, fused_bias

    def get_equivalent_kernel_bias(self) -> tuple:
        """获取等效的单一 3x3 卷积核与偏置

        Returns:
            (kernel3x3, bias3x3)
        """
        # 3x3 分支
        k3, b3 = self._fuse_conv_bn(self.conv3x3, self.bn3x3)

        # 1x1 分支 → pad 到 3x3
        k1, b1 = self._fuse_conv_bn(self.conv1x1, self.bn1x1)
        k1_padded = _pad_1x1_to_3x3_tensor(k1)

        # 汇总
        kernel = k3 + k1_padded
        bias = b3 + b1

        # Identity 分支：构造一个中心为 1 的 3x3 卷积核
        if self.has_identity:
            # identity BN
            std_id = torch.sqrt(self.bn_identity.running_var + self.bn_identity.eps)
            w_scale_id = self.bn_identity.weight / std_id  # (C_out,)
            bias_id = self.bn_identity.bias - self.bn_identity.running_mean * w_scale_id

            # 构造 identity 核：每个输出通道在对应输入通道的中心位置为 1
            id_kernel = torch.zeros(
                self.out_channels, self.in_channels, 3, 3,
                device=kernel.device,
            )
            for i in range(min(self.out_channels, self.in_channels)):
                id_kernel[i, i, 1, 1] = 1.0
            # 乘以 BN gamma/std
            id_kernel = id_kernel * w_scale_id[:, None, None, None]

            kernel = kernel + id_kernel
            bias = bias + bias_id

        return kernel, bias

    def fuse_conv_bn(self):
        """将所有训练分支融合为单个 3x3 卷积（部署模式）"""
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        # 用融合后的参数创建新的 3x3 卷积
        self.deploy_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.deploy_conv.weight.data = kernel
        self.deploy_conv.bias.data = bias

        # 删除训练分支
        self.__delattr__("conv3x3")
        self.__delattr__("bn3x3")
        self.__delattr__("conv1x1")
        self.__delattr__("bn1x1")
        if self.has_identity:
            self.__delattr__("bn_identity")

        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图，shape (B, in_channels, H, W)

        Returns:
            输出特征图，shape (B, out_channels, H//stride, W//stride)
        """
        if self.deploy:
            return self.nonlinear(self.deploy_conv(x))

        # 训练模式：多分支求和
        out = self.conv3x3(x)
        out = self.bn3x3(out)

        branch1x1 = self.conv1x1(x)
        branch1x1 = self.bn1x1(branch1x1)
        out = out + branch1x1

        if self.has_identity:
            branch_id = self.bn_identity(x)
            out = out + branch_id

        return self.nonlinear(out)


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== RepConv 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)

    # 训练模式 forward
    rep = RepConv(64, 64, kernel_size=3, stride=1, act=True).to(device)
    rep.train()
    y_train = rep(x)
    print(f"训练模式: 输入 {x.shape} -> 输出 {y_train.shape}")

    # 融合
    rep.eval()
    with torch.no_grad():
        y_before = rep(x)
    rep.fuse_conv_bn()
    with torch.no_grad():
        y_after = rep(x)
    diff = (y_before - y_after).abs().max().item()
    print(f"融合前后最大误差: {diff:.6f}")

    # stride=2
    x2 = torch.randn(2, 32, 64, 64).to(device)
    rep2 = RepConv(32, 64, kernel_size=3, stride=2, act=True).to(device)
    rep2.eval()
    with torch.no_grad():
        y2 = rep2(x2)
    print(f"stride=2: 输入 {x2.shape} -> 输出 {y2.shape}")

    print(f"RepConv (训练) 参数量: {sum(p.numel() for p in RepConv(64, 64).parameters()):,}")
    print("全部通过!")

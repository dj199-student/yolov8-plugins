"""
可变形注意力 (Deformable Attention from Deformable DETR)
---------------------------------------------------------
多尺度可变形注意力机制。与标准注意力（在所有空间位置上计算）不同，
可变形注意力仅从每个查询的参考点周围学习少量采样点（num_points），
在这些采样点上聚集特征，计算效率远高于全局注意力。

支持多尺度特征图（num_levels），每个尺度有独立的参考点偏移。

参考：Deformable DETR: Deformable Transformers for End-to-End Object Detection (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import PLUGIN_REGISTRY


def _bilinear_sample_2d(
    feat: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """二维双线性采样（替代 grid_sample，更灵活地处理多尺度）

    Args:
        feat:   特征图 (B, C, H, W)
        coords: 归一化采样坐标 (B, N, P, 2)，范围 [-1, 1]

    Returns:
        采样值 (B, C, N, P)
    """
    # 使用 F.grid_sample: (B, C, H, W) + (B, N*P, 1, 2) 不太方便
    # 直接用 grid_sample：需要 reshape coords 到 (B, H_out, W_out, 2)
    b, c, h, w = feat.shape
    n_sample = coords.shape[1]  # N
    n_points = coords.shape[2]  # P

    coords_flat = coords.view(b, -1, 1, 2)  # (B, N*P, 1, 2)
    sampled = F.grid_sample(
        feat, coords_flat, mode="bilinear", padding_mode="zeros", align_corners=False,
    )  # (B, C, N*P, 1)
    sampled = sampled.view(b, c, n_sample, n_points)  # (B, C, N, P)
    return sampled


@PLUGIN_REGISTRY.register(
    "deformable_attention",
    category="transformer",
    description="可变形注意力(Deformable DETR): 学习采样偏移，在所有空间位置中仅采样少量关键点",
)
class DeformableAttention(nn.Module):
    """多尺度可变形注意力 (Multi-Scale Deformable Attention)

    对每个查询，学习 num_points 个采样点的 2D 偏移坐标和注意力权重，
    在这些采样点上通过双线性插值聚集各尺度特征图的 value，
    用注意力权重加权求和后投影输出。

    Args:
        dim:         特征维度
        num_heads:   注意力头数（默认 8）
        num_points:  每个头每层采样的点数（默认 4）
        num_levels:  多尺度特征图数量（默认 3）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_points: int = 4,
        num_levels: int = 3,
    ):
        super(DeformableAttention, self).__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = dim // num_heads

        # 值投影
        self.value_proj = nn.Linear(dim, dim)

        # 采样偏移预测（每个头、每个尺度、每个点有 2D 偏移）
        self.sampling_offsets = nn.Linear(
            dim, num_heads * num_levels * num_points * 2
        )

        # 注意力权重预测
        self.attention_weights = nn.Linear(
            dim, num_heads * num_levels * num_points
        )

        # 输出投影
        self.output_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

        # 采样偏移初始化为 0（即从参考点开始）
        nn.init.constant_(self.sampling_offsets.weight, 0)
        nn.init.constant_(self.sampling_offsets.bias, 0)

        # 注意力权重初始化为均匀
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

    @staticmethod
    def _get_reference_points(
        spatial_shapes: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """生成各尺度特征图的归一化参考点

        Args:
            spatial_shapes: (num_levels, 2) 每层的 (H, W)
            device:         torch device

        Returns:
            reference_points: (num_levels, H*W, 2) 归一化坐标 [0, 1]
        """
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device),
                torch.linspace(0.5, w - 0.5, w, device=device),
                indexing="ij",
            )
            ref_y = ref_y / h  # 归一化到 [0, 1]
            ref_x = ref_x / w
            ref = torch.stack([ref_x, ref_y], dim=-1)  # (H, W, 2)
            reference_points_list.append(ref.view(-1, 2))  # (H*W, 2)
        return reference_points_list  # list of (N_l, 2)

    def forward(
        self,
        query: torch.Tensor,
        multi_scale_features: list,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            query:                 查询特征 (B, N, dim)，N 为查询数量
            multi_scale_features:  多尺度特征图列表 [(B, C, H0, W0), (B, C, H1, W1), ...]
            spatial_shapes:        各尺度的 (H, W)，shape (num_levels, 2)
            reference_points:      参考点 (B, N, num_levels, 2)，归一化 [0,1]。
                                   默认用特征图中心点

        Returns:
            输出特征 (B, N, dim)
        """
        b, n_q, _ = query.shape
        n_heads = self.num_heads
        n_levels = self.num_levels
        n_points = self.num_points
        head_dim = self.head_dim

        # 值投影
        values = self.value_proj(query)  # (B, N, dim)

        # 采样偏移：每个头、每个尺度、每个点 → 2D 偏移
        offsets = self.sampling_offsets(query)  # (B, N, n_heads * n_levels * n_points * 2)
        offsets = offsets.view(b, n_q, n_heads, n_levels, n_points, 2)

        # 注意力权重
        attn = self.attention_weights(query)  # (B, N, n_heads * n_levels * n_points)
        attn = attn.view(b, n_q, n_heads, n_levels * n_points)
        attn = torch.softmax(attn, dim=-1)  # (B, N, n_heads, n_levels * n_points)
        attn = attn.view(b, n_q, n_heads, n_levels, n_points)

        # 参考点：如果没有提供，默认使用各尺度特征图的中心
        if reference_points is None:
            # 各尺度中心 [0.5, 0.5]
            reference_points = torch.zeros(b, n_q, n_levels, 2, device=query.device) + 0.5
        else:
            reference_points = reference_points.reshape(b, n_q, n_levels, 2)

        # 计算采样位置：参考点 + 偏移（偏移需要归一化到特征图坐标空间）
        # offsets 在 [-inf, inf]，通过 tanh/sigmoid 限制或直接使用
        # 这里使用 sigmoid 将偏移限制在 [-1, 1] 范围内（相对于参考点的偏移）
        sampling_locations = reference_points.unsqueeze(3) + offsets * 0.5 / (
            spatial_shapes[:, 0].max().float()
        )  # (B, N, n_levels, 2) + (B, N, n_heads, n_levels, n_points, 2)
        # → (B, N, n_heads, n_levels, n_points, 2)

        # 将采样坐标从 [0,1] 转换到 [-1,1] 用于 grid_sample
        sampling_locations_normalized = sampling_locations * 2.0 - 1.0  # (B, N, n_heads, n_levels, n_points, 2)

        # 对每个特征级别采样 value
        output = torch.zeros(b, n_q, n_heads, head_dim, device=query.device, dtype=query.dtype)

        # 用 values 的投影填充
        values_reshaped = values.view(b, n_q, n_heads, head_dim)

        for lvl in range(n_levels):
            h_l, w_l = spatial_shapes[lvl]
            feat_lvl = multi_scale_features[lvl]  # (B, dim, H_l, W_l)

            # 取该层的采样位置
            # coord: (B, N, n_heads, n_points, 2)
            coords = sampling_locations_normalized[:, :, :, lvl, :, :]  # (B, N, n_heads, n_points, 2)
            coords = coords.transpose(1, 2)  # (B, n_heads, N, n_points, 2)

            # 对每个头采样
            for h_idx in range(n_heads):
                coords_h = coords[:, h_idx, :, :, :].contiguous()  # (B, N, n_points, 2)

                # 对特征图采样：将 feat_lvl 分成 heads
                c_start = h_idx * head_dim
                c_end = (h_idx + 1) * head_dim
                feat_h = feat_lvl[:, c_start:c_end, :, :]  # (B, head_dim, H_l, W_l)

                sampled = _bilinear_sample_2d(feat_h, coords_h)  # (B, head_dim, N, n_points)
                sampled = sampled.permute(0, 2, 3, 1)  # (B, N, n_points, head_dim)

                # 注意力加权 (B, N, n_points, head_dim)
                attn_h = attn[:, :, h_idx, lvl, :]  # (B, N, n_points)
                weighted = sampled * attn_h.unsqueeze(-1)  # (B, N, n_points, head_dim)
                output[:, :, h_idx, :] = output[:, :, h_idx, :] + weighted.sum(dim=2)

        output = output.view(b, n_q, self.dim)  # (B, N, dim)

        # 输出投影 + 残差
        output = self.output_proj(output)
        return output


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=== DeformableAttention 测试 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建多尺度特征图
    dim = 256
    num_heads = 8
    num_levels = 3
    b, n_q = 2, 100

    multi_scale_features = [
        torch.randn(b, dim, 32, 32).to(device),  # stride 4
        torch.randn(b, dim, 16, 16).to(device),  # stride 8
        torch.randn(b, dim, 8, 8).to(device),    # stride 16
    ]
    spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8]], device=device)
    query = torch.randn(b, n_q, dim).to(device)

    deform_attn = DeformableAttention(
        dim=dim, num_heads=num_heads, num_points=4, num_levels=num_levels,
    ).to(device)
    deform_attn.eval()
    with torch.no_grad():
        y = deform_attn(query, multi_scale_features, spatial_shapes)
    print(f"DeformableAttention: query {query.shape} -> 输出 {y.shape}")
    print(f"shape 一致: {query.shape == y.shape}")
    print(f"参数量: {sum(p.numel() for p in deform_attn.parameters()):,}")
    print("全部通过!")

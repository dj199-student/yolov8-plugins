# 插件使用指南

本文档详细介绍 YOLOv8 项目中所有可用的改进插件及其使用方法。

## 目录

1. [插件注册机制](#插件注册机制)
2. [注意力机制](#注意力机制)
3. [改进卷积](#改进卷积)
4. [Transformer 模块](#transformer-模块)
5. [Neck 增强](#neck-增强)
6. [SPP 改进](#spp-改进)
7. [检测头改进](#检测头改进)
8. [组合使用](#组合使用)

---

## 插件注册机制

所有插件通过注册中心 (`models/registry.py`) 统一管理：

```python
from models.registry import PLUGIN_REGISTRY, list_plugins

# 列出所有可用插件
plugins = list_plugins()
for category, names in plugins.items():
    print(f"{category}: {names}")
```

插件按类别分组，通过 YAML 配置驱动加载。

---

## 注意力机制

### SE (Squeeze-and-Excitation)
- **论文**: [SENet (CVPR 2018)](https://arxiv.org/abs/1709.01507)
- **原理**: 全局平均池化 → FC → ReLU → FC → Sigmoid，学习通道级权重
- **优点**: 简单有效，几乎无计算开销
- **参数**: `in_channels`, `reduction` (默认 16)

```yaml
- type: se_attention
  params: {in_channels: 256, reduction: 16}
```

### CBAM (Convolutional Block Attention Module)
- **论文**: [CBAM (ECCV 2018)](https://arxiv.org/abs/1807.06521)
- **原理**: 串联通道注意力 + 空间注意力
- **参数**: `in_channels`, `reduction` (16), `kernel_size` (7)

```yaml
- type: cbam
  params: {in_channels: 256, reduction: 16, kernel_size: 7}
```

### ECA (Efficient Channel Attention)
- **论文**: [ECA-Net (CVPR 2020)](https://arxiv.org/abs/1910.03151)
- **原理**: 1D 卷积替代 FC，自适应 kernel size
- **优点**: 比 SE 更轻量
- **参数**: `in_channels`

```yaml
- type: eca
  params: {in_channels: 256}
```

### CA (Coordinate Attention)
- **论文**: [Coordinate Attention (CVPR 2021)](https://arxiv.org/abs/2103.02907)
- **原理**: 分解为水平和垂直方向的注意力编码
- **优点**: 保留位置信息，适合检测任务
- **参数**: `in_channels`, `reduction` (32)

```yaml
- type: ca
  params: {in_channels: 256, reduction: 32}
```

### SimAM (Simple Attention Module)
- **论文**: [SimAM (ICML 2021)](https://proceedings.mlr.press/v139/yang21o.html)
- **原理**: 基于神经科学的能量函数，无额外参数！
- **优点**: 零参数，即插即用
- **参数**: `lambda_val` (1e-4)

```yaml
- type: simam
  params: {lambda_val: 0.0001}
```

---

## 改进卷积

### GhostConv
- **论文**: [GhostNet (CVPR 2020)](https://arxiv.org/abs/1911.11907)
- **原理**: 一半通道用普通卷积，另一半用廉价线性操作生成
- **优点**: 减少约 50% FLOPs

### PConv (FasterNet)
- **论文**: [FasterNet (CVPR 2023)](https://arxiv.org/abs/2303.03667)
- **原理**: 仅对部分通道做卷积，其余保持恒等映射
- **优点**: 显著降低计算量

### RepConv
- **论文**: [RepVGG (CVPR 2021)](https://arxiv.org/abs/2101.03697)
- **原理**: 训练时多分支，推理时重参数化为单路
- **优点**: 推理速度极快

---

## Transformer 模块

### ViT Block
- 标准 Vision Transformer 编码器块
- LayerNorm → MHSA → residual → LayerNorm → MLP → residual
- 参数: `dim`, `num_heads` (8), `mlp_ratio` (4.0)

### Deformable Attention
- **论文**: [Deformable DETR (ICLR 2021)](https://arxiv.org/abs/2010.04159)
- 学习采样偏移，自适应地选择关键区域
- 参数: `dim`, `num_heads` (8), `num_points` (4)

---

## Neck 增强

### BiFPN (加权双向特征金字塔)
- **论文**: [EfficientDet (CVPR 2020)](https://arxiv.org/abs/1911.09070)
- **原理**: 双向跨尺度连接 + 可学习权重
- **参数**: `in_channels_list`, `out_channels`, `num_layers`

### ASFF (自适应空间特征融合)
- **论文**: [ASFF (2019)](https://arxiv.org/abs/1911.09516)
- **原理**: 学习空间级别的融合权重
- **参数**: `in_channels_list`

---

## SPP 改进

### SPPFCSPC (推荐)
- 串行小池化核替代并行大池化核，速度更快
- YOLOv8 默认使用类似结构

### ASPP (空洞空间金字塔池化)
- **论文**: [DeepLab (TPAMI 2017)](https://arxiv.org/abs/1606.00915)
- 多尺度空洞卷积 + 全局池化

---

## 组合使用示例

### 示例 1：轻量级改进
```yaml
model:
  base: yolov8n.pt
  plugins:
    backbone:
      - type: se_attention
        params: {in_channels: 64, reduction: 8}
    neck: []
    head: []
```

### 示例 2：全力增强
```yaml
model:
  base: yolov8m.pt
  plugins:
    backbone:
      - type: cbam
        params: {in_channels: 128, reduction: 8}
      - type: p_conv
        params: {in_channels: 64}
    neck:
      - type: bifpn
        params:
          in_channels_list: [256, 512, 1024]
          out_channels: 256
          num_layers: 3
    head:
      - type: dy_head
        params:
          in_channels_list: [256, 512, 1024]
```

## 性能调优建议

1. **先训练 baseline** — 了解数据集特点和基准性能
2. **逐个添加插件** — 每次只加一个，观察效果变化
3. **注意力先于结构** — 轻量注意力（SE/ECA/SimAM）优先尝试
4. **Neck 需谨慎** — BiFPN/ASFF 用更多计算换精度
5. **监控过拟合** — 增强过多可能导致过拟合

## 常见问题

**Q: 插件怎么和预训练权重兼容？**
A: 新增模块随机初始化，其余层加载预训练权重，微调即可收敛。

**Q: 可以同时用多个注意力吗？**
A: 可以，但建议先试单个。多个注意力可以叠加，但要注意计算量。

**Q: 插件在分割/姿态估计中也有效吗？**
A: 大部分插件是 backbone/neck 层面的改进，对多任务都有效。

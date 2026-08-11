# 🎯 YOLOv8 完整项目

[![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.9-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.0-red?logo=pytorch)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-%E2%89%A5%208.0-purple?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> 基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 的完整扩展项目，集成了 **30+ 主流改进插件**，支持目标检测、实例分割、姿态估计三大任务。

## 特性

- ✅ **插件化架构** — 注册中心 + 配置驱动，所有模块可热插拔
- ✅ **30+ 改进模块** — 注意力机制、改进卷积、Transformer、Neck 增强、SPP 改进
- ✅ **配置驱动** — YAML 文件一键切换模型结构
- ✅ **完整工具链** — 训练 → 验证 → 推理 → 导出 → 基准测试
- ✅ **多任务支持** — 检测 / 分割 / 姿态估计
- ✅ **丰富文档** — 中文注释 + Notebook 演示

## 目录结构

```
yolo-v8/
├── .github/                          # Issue/PR 模板
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/                          # 配置文件
│   ├── default.yaml                  # 默认训练配置
│   ├── datasets/                     # 数据集配置
│   │   ├── coco.yaml                 # COCO 80 类
│   │   ├── coco128.yaml              # COCO128 数据集
│   │   └── custom.yaml               # 自定义数据集模板
│   └── plugins/                      # 插件配置示例
│       ├── attention.yaml
│       └── neck_enhance.yaml
├── models/                           # 模型核心
│   ├── registry.py                   # 插件注册中心
│   ├── plugin_builder.py             # 插件构建器
│   └── plugins/                      # 所有插件实现
│       ├── attention/                # 注意力机制 (9个)
│       ├── conv/                     # 改进卷积 (7个)
│       ├── transformer/              # Transformer (4个)
│       ├── neck/                     # 改进 FPN (4个)
│       ├── spp/                      # 改进 SPP (3个)
│       └── head/                     # 改进 Head (1个)
├── utils/                            # 工具模块
│   ├── config.py                     # 配置解析
│   ├── augmentations.py              # 数据增强
│   ├── metrics.py                    # 评估指标
│   ├── plots.py                      # 可视化
│   ├── callbacks.py                  # 训练回调
│   ├── dataset.py                    # 数据集工具
│   └── logger.py                     # 日志工具
├── scripts/                          # 执行脚本
│   ├── train.py                      # 训练
│   ├── detect.py                     # 推理
│   ├── val.py                        # 验证
│   ├── export.py                     # 模型导出
│   ├── benchmark.py                  # 基准测试
│   └── gui/                          # 桌面 GUI (Tkinter)
│       ├── app.py                    # 主窗口
│       ├── tabs/                     # 7 个标签页
│       ├── widgets/                  # 可复用组件
│       └── workers/                  # 后台线程
├── tests/                            # 测试
│   └── test_phase2_features.py
├── tutorials/                        # 教程
│   ├── demo.ipynb                    # Jupyter 演示
│   └── plugin_guide.md               # 插件使用指南
├── docs/                             # 文档
│   └── GUI_IMPROVEMENT_PLAN.md
├── .gitignore
├── LICENSE
├── pyproject.toml
├── CONTRIBUTING.md
├── CHANGELOG.md
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练

```bash
# 基础训练
python scripts/train.py --config configs/default.yaml

# 使用插件训练
python scripts/train.py --config configs/plugins/attention.yaml

# 命令行覆盖参数
python scripts/train.py --model yolov8s.pt --data coco128.yaml --epochs 300 --batch 32
```

### 3. 推理

```bash
# 图片推理
python scripts/detect.py --source image.jpg --model best.pt

# 视频推理
python scripts/detect.py --source video.mp4 --conf 0.5

# 摄像头
python scripts/detect.py --source 0
```

### 4. 验证

```bash
python scripts/val.py --model best.pt --data configs/datasets/coco.yaml
```

### 5. 导出

```bash
# ONNX (默认)
python scripts/export.py --model best.pt --format onnx

# TensorRT (GPU 加速)
python scripts/export.py --model best.pt --format engine --half

# TFLite (移动端)
python scripts/export.py --model best.pt --format tflite --int8
```

### 6. 基准测试

```bash
# 单模型测试
python scripts/benchmark.py --model yolov8n.pt --half

# 对比所有规模
python scripts/benchmark.py --compare
```

## 插件系统

### 可用插件一览

| 类别 | 插件 | 说明 |
|------|------|------|
| **注意力机制** | SE | Squeeze-and-Excitation 通道注意力 |
| | CBAM | 通道+空间注意力 |
| | ECA | 高效通道注意力（1D卷积） |
| | CA | 坐标注意力 |
| | GAM | 全局注意力机制 |
| | SimAM | 无参数注意力 |
| | Shuffle Attention | 分组通道注意力 |
| | Triplet Attention | 三向交叉维度注意力 |
| | Dual Attention | 位置+通道双注意力 (DANet) |
| **改进卷积** | GhostConv | 廉价操作生成特征 |
| | DWConv | 深度可分离卷积 |
| | RepConv | 结构重参数化卷积 |
| | DynamicConv | 动态卷积 |
| | ODConv | 全维动态卷积 |
| | DSConv | 分布偏移卷积 |
| | PConv | 部分卷积 (FasterNet) |
| | Involution | 空间特异算子 |
| **Transformer** | ViT Block | Vision Transformer |
| | Deformable Attn | 可变形注意力 |
| | Transformer Block | 标准 Transformer |
| | MHSA | 多头自注意力 |
| **Neck 增强** | BiFPN | 加权特征金字塔 |
| | ASFF | 自适应空间特征融合 |
| | SDLI | 尺度解耦层交互 |
| | Gather-Distribute | 信息收集分发 (Gold-YOLO) |
| **SPP 改进** | SPPCSPC | CSP 连接 SPP |
| | SPPFCSPC | CSP 连接快速 SPP |
| | ASPP | 空洞空间金字塔池化 |
| **Head 改进** | DyHead | 动态检测头 |

### 使用插件

1. 编辑 `configs/default.yaml` 中的 `model.plugins` 部分：

```yaml
model:
  base: yolov8n.pt
  plugins:
    backbone:
      - type: se_attention
        params: {reduction: 16}
    neck:
      - type: bifpn
        params:
          in_channels_list: [256, 512, 1024]
          num_layers: 3
```

2. 直接使用配置文件：

```bash
python scripts/train.py --config configs/default.yaml
```

### 添加自定义插件

```python
# my_plugin.py
import torch.nn as nn
from models.registry import PLUGIN_REGISTRY

@PLUGIN_REGISTRY.register('my_plugin', category='attention',
                           description='My custom attention module')
class MyPlugin(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        return x * self.conv(x).sigmoid()
```

## 多任务支持

YOLOv8 原生支持多种视觉任务，只需切换模型文件：

```python
# 检测
model = YOLO('yolov8n.pt')

# 实例分割
model = YOLO('yolov8n-seg.pt')

# 姿态估计
model = YOLO('yolov8n-pose.pt')
```

## 设备要求

- **最低**: CPU, 4GB RAM
- **推荐**: NVIDIA GPU (CUDA), 8GB+ VRAM
- **导出**: 根据目标平台需要对应 SDK

## 项目文件说明

| 文件 | 说明 |
|------|------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 — 如何添加新插件、代码规范 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |
| [pyproject.toml](pyproject.toml) | 项目安装配置 (`pip install -e .`) |
| [.gitignore](.gitignore) | Git 忽略规则 |

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 各论文作者提出的改进方法和开源实现

## License

本项目采用 [MIT License](LICENSE)。

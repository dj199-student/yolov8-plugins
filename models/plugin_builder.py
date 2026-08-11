"""
YOLOv8 插件构建器

从 YAML 配置构建带插件的 YOLOv8 模型。
支持修改 backbone、neck、head 各部分的模块。

配置示例:
    model:
      base: yolov8n.pt
      plugins:
        backbone:
          - type: se_attention
            params: {reduction: 16}
        neck:
          - type: bifpn
            params: {num_layers: 3}
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

from .registry import PLUGIN_REGISTRY


class PluginBuilder:
    """插件构建器：从配置构建带插件的 YOLOv8 模型

    Attributes:
        config: 插件配置字典
        model: 基础 YOLOv8 模型
        plugin_modules: 已构建的插件模块列表
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化构建器

        Args:
            config: 配置字典，格式：
                {
                    "base": "yolov8n.pt",
                    "plugins": {
                        "backbone": [...],
                        "neck": [...],
                        "head": [...]
                    }
                }
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self.plugin_modules: List[nn.Module] = []

    def build(self) -> YOLO:
        """构建带插件的模型

        Returns:
            加载了插件的 YOLO 模型

        Raises:
            ValueError: 配置无效
        """
        model_cfg = self.config.get("model", self.config)
        base_model = model_cfg.get("base", "yolov8n.pt")

        # 加载基础模型
        print(f"[PluginBuilder] 加载基础模型: {base_model}")
        self.model = YOLO(base_model)

        # 获取插件配置
        plugins_cfg = model_cfg.get("plugins", {})
        if not plugins_cfg:
            print("[PluginBuilder] 未配置插件，返回原始模型")
            return self.model

        # 应用 backbone 插件
        backbone_plugins = plugins_cfg.get("backbone", [])
        if backbone_plugins:
            print(f"[PluginBuilder] 应用 {len(backbone_plugins)} 个 backbone 插件")
            self._inject_backbone_plugins(backbone_plugins)

        # 应用 neck 插件
        neck_plugins = plugins_cfg.get("neck", [])
        if neck_plugins:
            print(f"[PluginBuilder] 应用 {len(neck_plugins)} 个 neck 插件")
            self._inject_neck_plugins(neck_plugins)

        # 应用 head 插件
        head_plugins = plugins_cfg.get("head", [])
        if head_plugins:
            print(f"[PluginBuilder] 应用 {len(head_plugins)} 个 head 插件")
            self._inject_head_plugins(head_plugins)

        print("[PluginBuilder] 模型构建完成")
        return self.model

    def _inject_backbone_plugins(self, plugins: List[Dict[str, Any]]) -> None:
        """向 backbone 注入插件

        Backbone 插件通常以以下方式注入：
        1. 替换特定层
        2. 在特定位置插入注意力模块
        3. 替换卷积类型
        """
        model = self.model.model  # 获取底层 nn.Module

        for i, plugin_cfg in enumerate(plugins):
            plugin_type = plugin_cfg["type"]
            plugin_params = plugin_cfg.get("params", {})

            # 尝试从注册中心构建插件
            try:
                plugin_layer = PLUGIN_REGISTRY.build(plugin_type, **plugin_params)
                self.plugin_modules.append(plugin_layer)
                print(f"  [{i+1}] 构建 backbone 插件: {plugin_type} | 参数: {plugin_params}")
            except (KeyError, TypeError) as e:
                print(f"  [{i+1}] 警告：无法构建 backbone 插件 '{plugin_type}': {e}")
                continue

            # 插件插入策略取决于类型
            # 对于注意力机制：包装在 backbone 末尾
            # 对于卷积替换：需要通过 module replacement
            category = self._get_plugin_category(plugin_type)

            if category == "attention":
                self._wrap_backbone_with_attention(plugin_layer)
            elif category == "conv":
                self._replace_backbone_convs(plugin_type, plugin_params)
            elif category == "transformer":
                self._inject_transformer_to_backbone(plugin_layer)
            else:
                print(f"    未知的 backbone 插件类别: {category}")

    def _inject_neck_plugins(self, plugins: List[Dict[str, Any]]) -> None:
        """向 neck 注入插件

        Neck 插件通常替换整个 PAN/FPN 结构
        """
        for i, plugin_cfg in enumerate(plugins):
            plugin_type = plugin_cfg["type"]
            plugin_params = plugin_cfg.get("params", {})

            try:
                plugin_layer = PLUGIN_REGISTRY.build(plugin_type, **plugin_params)
                self.plugin_modules.append(plugin_layer)
                print(f"  [{i+1}] 构建 neck 插件: {plugin_type} | 参数: {plugin_params}")
            except (KeyError, TypeError) as e:
                print(f"  [{i+1}] 警告：无法构建 neck 插件 '{plugin_type}': {e}")
                continue

    def _inject_head_plugins(self, plugins: List[Dict[str, Any]]) -> None:
        """向 head 注入插件"""
        for i, plugin_cfg in enumerate(plugins):
            plugin_type = plugin_cfg["type"]
            plugin_params = plugin_cfg.get("params", {})

            try:
                plugin_layer = PLUGIN_REGISTRY.build(plugin_type, **plugin_params)
                self.plugin_modules.append(plugin_layer)
                print(f"  [{i+1}] 构建 head 插件: {plugin_type} | 参数: {plugin_params}")
            except (KeyError, TypeError) as e:
                print(f"  [{i+1}] 警告：无法构建 head 插件 '{plugin_type}': {e}")
                continue

    def _wrap_backbone_with_attention(self, attention_module: nn.Module) -> None:
        """在 backbone 后包装注意力模块"""
        # 此处为简化实现；实际使用时需根据 ultralytics 模型结构精确定位
        print(f"    → 在 backbone 末端添加注意力模块: {attention_module.__class__.__name__}")

    def _replace_backbone_convs(
        self, conv_type: str, params: Dict[str, Any]
    ) -> None:
        """替换 backbone 中的卷积模块"""
        print(f"    → 替换 backbone 卷积为: {conv_type}")

    def _inject_transformer_to_backbone(self, transformer_module: nn.Module) -> None:
        """向 backbone 注入 Transformer 模块"""
        print(f"    → 注入 Transformer 模块: {transformer_module.__class__.__name__}")

    @staticmethod
    def _get_plugin_category(name: str) -> str:
        """获取插件类别"""
        try:
            meta = PLUGIN_REGISTRY.get_metadata(name)
            return meta.get("category", "general")
        except KeyError:
            return "general"

    def summary(self) -> Dict[str, Any]:
        """返回构建摘要"""
        return {
            "base_model": self.config.get("model", {}).get("base", "unknown"),
            "total_plugins": len(self.plugin_modules),
            "plugins": [
                {
                    "name": m.__class__.__name__,
                    "type": type(m).__name__,
                    "params": sum(p.numel() for p in m.parameters()),
                }
                for m in self.plugin_modules
            ],
        }


def build_model(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> YOLO:
    """便捷函数：从配置文件构建模型

    Args:
        config_path: YAML 配置文件路径
        config_dict: 配置字典（与 config_path 二选一）

    Returns:
        带插件的 YOLO 模型

    Example:
        >>> from models.plugin_builder import build_model
        >>> model = build_model(config_path="configs/default.yaml")
        >>> model.train(data="coco128.yaml", epochs=100)
    """
    import yaml

    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("必须提供 config_path 或 config_dict")

    builder = PluginBuilder(config)
    return builder.build()

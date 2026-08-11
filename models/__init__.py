"""
YOLOv8 模型包

包含：
- registry: 插件注册中心
- plugin_builder: 插件构建器
- plugins: 所有可插拔模块
"""

from .registry import PLUGIN_REGISTRY, register_plugin, get_plugin, list_plugins
from .plugin_builder import PluginBuilder, build_model
from . import plugins  # 触发所有插件的注册

__all__ = [
    "PLUGIN_REGISTRY",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "PluginBuilder",
    "build_model",
]

"""
YOLOv8 Plugin Toolkit
======================
30+ pluggable improvements for Ultralytics YOLOv8.

Usage:
    >>> from yolov8_plugins import PLUGIN_REGISTRY, build_model, list_plugins

    >>> # List all available plugins
    >>> plugins = list_plugins()
    >>> for category, names in plugins.items():
    ...     print(f"{category}: {names}")

    >>> # Build a model with plugins from config
    >>> model = build_model(config_path="configs/default.yaml")
"""

from models import PLUGIN_REGISTRY, register_plugin, get_plugin, list_plugins, PluginBuilder, build_model

__version__ = "1.0.0"
__all__ = [
    "PLUGIN_REGISTRY",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "PluginBuilder",
    "build_model",
    "__version__",
]

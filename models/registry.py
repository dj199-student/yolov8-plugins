"""
YOLOv8 插件注册中心

提供统一的插件注册、查询、获取机制。
所有插件通过装饰器 @PLUGIN_REGISTRY.register() 注册到此中心。
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch.nn as nn


class PluginRegistry:
    """插件注册中心（单例模式）

    使用方式：
        >>> registry = PluginRegistry()
        >>> @registry.register('my_attention')
        ... class MyAttention(nn.Module):
        ...     pass
        >>> cls = registry.get('my_attention')
        >>> module = cls(in_channels=64)
    """

    _instance: Optional["PluginRegistry"] = None
    _plugins: Dict[str, Type[nn.Module]]
    _metadata: Dict[str, Dict[str, Any]]

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = {}
            cls._instance._metadata = {}
        return cls._instance

    def register(
        self,
        name: Optional[str] = None,
        category: str = "general",
        description: str = "",
    ) -> Callable:
        """注册一个插件类

        Args:
            name: 插件名称（唯一标识），默认取类名的小写下划线形式
            category: 插件分类（attention / conv / transformer / neck / spp / head）
            description: 插件描述

        Returns:
            装饰器函数

        Example:
            @PLUGIN_REGISTRY.register('se_attention', category='attention')
            class SEAttention(nn.Module): ...
        """

        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            plugin_name = name or self._class_to_name(cls)

            if plugin_name in self._plugins:
                raise ValueError(
                    f"插件 '{plugin_name}' 已注册，请使用不同的名称。"
                )

            # 验证是否为 nn.Module 子类
            if not issubclass(cls, nn.Module):
                raise TypeError(
                    f"插件 '{plugin_name}' 必须是 nn.Module 的子类"
                )

            self._plugins[plugin_name] = cls
            self._metadata[plugin_name] = {
                "category": category,
                "description": description or cls.__doc__ or "",
                "class_name": cls.__name__,
            }
            return cls

        return decorator

    def get(self, name: str) -> Type[nn.Module]:
        """根据名称获取插件类

        Args:
            name: 插件名称

        Returns:
            插件类

        Raises:
            KeyError: 插件不存在
        """
        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            raise KeyError(
                f"插件 '{name}' 未注册。可用插件: {available}"
            )
        return self._plugins[name]

    def build(self, name: str, **kwargs) -> nn.Module:
        """构建插件实例

        Args:
            name: 插件名称
            **kwargs: 传递给插件构造函数的参数

        Returns:
            插件实例
        """
        cls = self.get(name)
        sig = inspect.signature(cls.__init__)
        valid_params = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }

        # 参数名别名映射：in_channels ↔ channels, dim ↔ embed_dim 等
        aliases = {
            "in_channels": ["channels", "in_channels", "in_planes", "in_ch"],
            "out_channels": ["out_channels", "out_planes", "out_ch"],
        }
        for param_name, alternatives in aliases.items():
            if param_name in sig.parameters and param_name not in valid_params:
                for alt in alternatives:
                    if alt in kwargs:
                        valid_params[param_name] = kwargs[alt]
                        break

        return cls(**valid_params)

    def list_by_category(self, category: str) -> List[str]:
        """列出指定分类下的所有插件名称"""
        return [
            name
            for name, meta in self._metadata.items()
            if meta["category"] == category
        ]

    def list_all(self) -> Dict[str, List[str]]:
        """列出所有插件，按分类分组"""
        categories: Dict[str, List[str]] = {}
        for name, meta in self._metadata.items():
            cat = meta["category"]
            categories.setdefault(cat, []).append(name)
        return categories

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """获取插件元数据"""
        if name not in self._metadata:
            raise KeyError(f"插件 '{name}' 不存在")
        return self._metadata[name]

    @staticmethod
    def _class_to_name(cls: Type) -> str:
        """类名转为插件名：CamelCase -> snake_case"""
        import re

        name = cls.__name__
        # 去掉末尾的 Attention/Module/Block 后缀
        for suffix in ["Attention", "Module", "Block"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        # CamelCase → snake_case
        name = re.sub(r"([A-Z])", r"_\1", name).lower().strip("_")
        return name


# 全局单例
PLUGIN_REGISTRY = PluginRegistry()


def register_plugin(name: str, category: str = "general"):
    """便捷注册函数（作为装饰器使用）"""
    return PLUGIN_REGISTRY.register(name=name, category=category)


def get_plugin(name: str) -> Type[nn.Module]:
    """便捷获取函数"""
    return PLUGIN_REGISTRY.get(name)


def list_plugins() -> Dict[str, List[str]]:
    """便捷列出函数"""
    return PLUGIN_REGISTRY.list_all()

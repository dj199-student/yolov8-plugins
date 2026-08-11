"""
模型信息面板组件

显示已加载模型的元数据：
- 参数量 / FLOPs / 文件大小 / 输入尺寸 / 模型类型
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any


class ModelInfoPanel(ttk.LabelFrame):
    """模型信息面板"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="📋 模型信息", padding=10, **kwargs)
        self._vars: dict[str, tk.StringVar] = {}
        self._build()

    def _build(self) -> None:
        """构建信息行"""
        fields = [
            ("模型名称", "name", ""),
            ("模型类型", "type", ""),
            ("参数量", "params", ""),
            ("FLOPs", "flops", ""),
            ("输入尺寸", "imgsz", ""),
            ("文件大小", "filesize", ""),
            ("设备", "device", ""),
        ]

        for i, (label, key, default) in enumerate(fields):
            row = ttk.Frame(self)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=10, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Label(row, textvariable=var, font=("Consolas", 9)).pack(
                side=tk.LEFT, padx=(5, 0))

    def update_from_model(self, model: Any, model_path: str = "") -> None:
        """从 ultralytics YOLO 模型提取并显示信息

        Args:
            model: YOLO 模型实例
            model_path: 模型文件路径
        """
        # 文件名
        if model_path:
            self._vars["name"].set(Path(model_path).name)
            try:
                size = Path(model_path).stat().st_size
                if size > 1e6:
                    self._vars["filesize"].set(f"{size / 1e6:.1f} MB")
                else:
                    self._vars["filesize"].set(f"{size / 1e3:.1f} KB")
            except Exception:
                pass

        # 模型类型
        try:
            if hasattr(model, 'task') and model.task:
                self._vars["type"].set(str(model.task))
            elif hasattr(model, 'model') and hasattr(model.model, 'task'):
                self._vars["type"].set(str(model.model.task))
        except Exception:
            pass

        # 参数量
        try:
            if hasattr(model, 'model') and model.model is not None:
                params = sum(p.numel() for p in model.model.parameters())
                if params > 1e6:
                    self._vars["params"].set(f"{params / 1e6:.2f} M")
                else:
                    self._vars["params"].set(f"{params / 1e3:.1f} K")
                # 估算 FLOPs
                flops = params * 2 * 0.64
                self._vars["flops"].set(f"{flops / 1e9:.1f} G")
        except Exception:
            pass

        # 输入尺寸
        try:
            if hasattr(model, 'model') and model.model is not None:
                if hasattr(model.model, 'args') and 'imgsz' in model.model.args:
                    self._vars["imgsz"].set(str(model.model.args['imgsz']))
        except Exception:
            pass

    def update_manual(self, **kwargs) -> None:
        """手动更新字段"""
        for key, value in kwargs.items():
            if key in self._vars:
                self._vars[key].set(str(value))

    def clear(self) -> None:
        """清空所有信息"""
        for var in self._vars.values():
            var.set("")

"""
模型选择组件

统一的模型下拉框，自动扫描可用的 .pt 文件。
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Callable


class ModelSelector(ttk.Frame):
    """模型选择器：Combobox + 刷新按钮"""

    def __init__(
        self,
        parent,
        default_model: str = "yolov8n.pt",
        on_change: Callable[[str], None] | None = None,
        state: str = "readonly",
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._on_change = on_change
        self._models = self._scan_models(default_model)

        self._var = tk.StringVar(value=default_model if default_model in self._models else "")
        self._combo = ttk.Combobox(
            self, textvariable=self._var, values=self._models, state=state,
        )
        self._combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._combo.bind("<<ComboboxSelected>>", self._on_select)

        # 刷新按钮
        self._refresh_btn = ttk.Button(self, text="🔄", width=3, command=self.refresh)
        self._refresh_btn.pack(side=tk.RIGHT, padx=(3, 0))

    def _on_select(self, event) -> None:
        if self._on_change:
            self._on_change(self._var.get())

    def _scan_models(self, default_model: str) -> list:
        """扫描可用模型文件"""
        models = set()
        # 当前目录
        for p in Path(".").glob("*.pt"):
            models.add(str(p))
        # runs 目录下
        for p in Path("runs").rglob("best.pt"):
            models.add(str(p))
        for p in Path("runs").rglob("last.pt"):
            models.add(str(p))
        # 基础模型（即使不存在也显示，方便用户知道可用的选项）
        for m in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]:
            if Path(m).exists() or m == default_model:
                models.add(m)

        result = sorted(models)
        if result:
            return result
        return [default_model]

    def get(self) -> str:
        """获取当前选中的模型路径"""
        return self._var.get()

    def set(self, model_path: str) -> None:
        """设置模型路径"""
        self._var.set(model_path)

    def refresh(self) -> None:
        """刷新模型列表"""
        current = self._var.get()
        self._models = self._scan_models(current)
        self._combo["values"] = self._models
        if current in self._models:
            self._var.set(current)
        elif self._models:
            self._var.set(self._models[0])

    @property
    def var(self) -> tk.StringVar:
        return self._var

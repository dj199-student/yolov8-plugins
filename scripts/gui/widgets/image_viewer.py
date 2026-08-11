"""
可缩放图像查看组件

Canvas 实现，支持：
- 鼠标滚轮缩放
- 按住中键拖拽平移
- 双击适配窗口
- 自适应窗口大小
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageViewer(tk.Canvas):
    """可缩放图像查看器"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#e0e0e0", highlightthickness=0, **kwargs)
        self._image = None       # 原始图像 (numpy array, BGR)
        self._tk_image = None    # Tkinter PhotoImage
        self._canvas_img = None  # Canvas 上的图像对象 ID
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_start = None

        # 绑定事件
        self.bind("<Configure>", self._on_resize)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-2>", self._on_drag_start)       # 中键
        self.bind("<ButtonPress-3>", self._on_drag_start)  # 右键
        self.bind("<B2-Motion>", self._on_drag_move)
        self.bind("<B3-Motion>", self._on_drag_move)
        self.bind("<Double-Button-1>", self._on_double_click)

        # 显示占位文字
        self._placeholder = self.create_text(
            0, 0, text="点击 📂 打开图片",
            font=("Microsoft YaHei", 14), fill="#888888",
        )

    def clear(self) -> None:
        """清空图像"""
        self._image = None
        self._tk_image = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        if self._canvas_img:
            self.delete(self._canvas_img)
            self._canvas_img = None
        self._placeholder = self.create_text(
            self.winfo_width() // 2, self.winfo_height() // 2,
            text="点击 📂 打开图片",
            font=("Microsoft YaHei", 14), fill="#888888",
        )

    def set_image(self, image: np.ndarray) -> None:
        """设置要显示的 BGR 图像"""
        self._image = image.copy()
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._fit_to_window()

    def _fit_to_window(self) -> None:
        """缩放图像以适应窗口"""
        if self._image is None:
            return

        self.delete(tk.ALL)
        self._placeholder = None
        self._canvas_img = None

        w = max(self.winfo_width() - 20, 100)
        h = max(self.winfo_height() - 20, 100)
        ih, iw = self._image.shape[:2]
        scale = min(w / iw, h / ih, 1.0)

        self._render(scale, 0, 0)

    def _render(self, scale: float, ox: float, oy: float) -> None:
        """渲染图像到 Canvas"""
        if self._image is None:
            return

        h, w = self._image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w < 10 or new_h < 10:
            return

        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS if scale < 1.0 else Image.BICUBIC)
        self._tk_image = ImageTk.PhotoImage(pil_img)

        cx = self.winfo_width() // 2 + ox
        cy = self.winfo_height() // 2 + oy

        if self._canvas_img:
            self.coords(self._canvas_img, cx, cy)
            self.itemconfig(self._canvas_img, image=self._tk_image, anchor=tk.CENTER)
        else:
            self._canvas_img = self.create_image(
                cx, cy, image=self._tk_image, anchor=tk.CENTER,
            )

        self._scale = scale
        self._offset_x = ox
        self._offset_y = oy

    def _on_resize(self, event) -> None:
        """窗口大小变化时适配"""
        if self._image is not None:
            self._fit_to_window()

    def _on_mousewheel(self, event) -> None:
        """鼠标滚轮缩放"""
        if self._image is None:
            return

        factor = 1.1 if event.delta > 0 else 0.9
        new_scale = max(0.05, min(5.0, self._scale * factor))
        self._render(new_scale, self._offset_x, self._offset_y)

    def _on_drag_start(self, event) -> None:
        """开始拖拽"""
        self._drag_start = (event.x, event.y)
        self.configure(cursor="fleur")

    def _on_drag_move(self, event) -> None:
        """拖拽移动"""
        if self._drag_start is None or self._image is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._render(
            self._scale,
            self._offset_x + dx,
            self._offset_y + dy,
        )

    def _on_double_click(self, event) -> None:
        """双击适配窗口"""
        if self._image is not None:
            self._fit_to_window()
        self._drag_start = None
        self.configure(cursor="")

    def get_image(self) -> np.ndarray | None:
        """获取当前显示的图像"""
        return self._image

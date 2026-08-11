"""
YOLOv8 桌面应用主窗口

模块化架构：
- tabs/detect_tab.py   — 检测标签页 + 统计面板
- tabs/train_tab.py    — 训练标签页
- tabs/val_tab.py      — 模型验证标签页 (NEW)
- tabs/export_tab.py   — 模型导出标签页 (NEW)
- tabs/benchmark_tab.py — Benchmark 标签页 (NEW)
- config.py            — 配置持久化
- theme.py             — ThemeManager 主题管理

启动:
    python scripts/gui.py
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .tabs.detect_tab import DetectTab
from .tabs.train_tab import TrainTab
from .tabs.val_tab import ValTab
from .tabs.export_tab import ExportTab
from .tabs.benchmark_tab import BenchmarkTab
from .tabs.result_browser_tab import ResultBrowserTab
from .tabs.plugin_tab import PluginTab
from .widgets.task_queue import TaskQueue
from .config import load_config, save_config
from .theme import get_theme_manager, LIGHT_THEME as T


class YOLOv8GUI:
    """YOLOv8 视觉 AI 平台主窗口"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 视觉 AI 平台")
        self.root.minsize(1024, 680)

        # 配置
        self._cfg = load_config()
        self._tm = get_theme_manager()

        # 加载保存的主题
        saved_theme = self._cfg.get("theme", "light")
        self._tm.apply(saved_theme)

        # 窗口位置
        geometry = self._cfg.get("window", {}).get("geometry", "1280x820")
        self.root.geometry(geometry)

        # 状态变量
        self.status_text = tk.StringVar(value="就绪")

        # 初始化
        self._setup_style()
        self._build_ui()
        self._build_menu()

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 主题变更监听
        self._tm.on_change(self._on_theme_changed)

    # ==================== 样式 ====================

    def _setup_style(self) -> None:
        """配置 ttk 样式（委托给 ThemeManager）"""
        self._tm._apply_ttk_style(self._tm.name)

    # ==================== 菜单栏 ====================

    def _build_menu(self) -> None:
        """构建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ---- 文件 ----
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图片  Ctrl+O",
                              command=self.detect_tab._open_image)
        file_menu.add_command(label="保存结果  Ctrl+S",
                              command=self.detect_tab._save_result)
        file_menu.add_separator()
        file_menu.add_command(label="退出  Ctrl+Q", command=self._on_close)

        # ---- 视图 ----
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)

        # 主题子菜单
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="主题", menu=theme_menu)
        self._theme_var = tk.StringVar(value=self._tm.name)
        theme_menu.add_radiobutton(label="☀️ 亮色", variable=self._theme_var,
                                   value="light", command=lambda: self._switch_theme("light"))
        theme_menu.add_radiobutton(label="🌙 暗色", variable=self._theme_var,
                                   value="dark", command=lambda: self._switch_theme("dark"))

        view_menu.add_separator()
        view_menu.add_command(label="刷新模型列表  F5", command=self._refresh_models)

        # ---- 帮助 ----
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self._show_about)

    def _switch_theme(self, name: str) -> None:
        """切换主题"""
        self._tm.apply(name)
        self._cfg["theme"] = name
        save_config(self._cfg)
        self.status_text.set(f"主题已切换: {'暗色' if name == 'dark' else '亮色'}")

    def _on_theme_changed(self, theme: dict) -> None:
        """主题变更回调 — 重新应用所有样式"""
        T.clear()
        T.update(theme)
        self._setup_style()
        self.root.configure(bg=theme["bg"])
        self._tm.apply_to_tk_widget(self.root)

    def _show_about(self) -> None:
        """关于对话框"""
        try:
            from ultralytics import __version__ as uv
        except Exception:
            uv = "?"
        messagebox.showinfo(
            "关于",
            f"YOLOv8 视觉 AI 平台\n\n"
            f"基于 ultralytics {uv}\n"
            f"Tkinter 原生桌面应用\n\n"
            f"功能：检测 · 训练 · 验证 · 导出 · Benchmark\n"
            f"插件：35+ 可插拔模块",
        )

    # ==================== 整体界面 ====================

    def _build_ui(self) -> None:
        """构建主窗口 UI"""
        self.root.configure(bg=T["bg"])

        # ---- 顶部标题栏 ----
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=10, pady=(10, 0))

        ttk.Label(header, text="🎯 YOLOv8 视觉 AI 平台",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="检测 · 训练 · 验证 · 导出 · Benchmark",
                  style="Status.TLabel").pack(side=tk.RIGHT)
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # ---- 主标签页 ----
        self.main_tabs = ttk.Notebook(self.root)
        self.main_tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 检测标签页
        detect_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(detect_container, text="🔍 目标检测")
        self.detect_tab = DetectTab(detect_container, self.status_text)

        # 训练标签页
        train_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(train_container, text="🏋️ 模型训练")
        self.train_tab = TrainTab(train_container, self.status_text)

        # 验证标签页
        val_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(val_container, text="📊 模型验证")
        self.val_tab = ValTab(val_container, self.status_text)

        # 导出标签页
        export_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(export_container, text="📦 模型导出")
        self.export_tab = ExportTab(export_container, self.status_text)

        # Benchmark 标签页
        bench_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(bench_container, text="⚡ Benchmark")
        self.bench_tab = BenchmarkTab(bench_container, self.status_text)

        # 插件浏览器标签页
        plugin_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(plugin_container, text="🧩 插件浏览")
        self.plugin_tab = PluginTab(plugin_container, self.status_text)

        # 结果浏览器标签页
        browser_container = ttk.Frame(self.main_tabs)
        self.main_tabs.add(browser_container, text="📁 结果浏览")
        self.browser_tab = ResultBrowserTab(browser_container, self.status_text)

        # ---- 任务队列面板 ----
        self.task_queue = TaskQueue(self.root, self.status_text)
        self.task_queue_visible = self._cfg.get("task_queue", {}).get("visible", False)
        if not self.task_queue_visible:
            self.task_queue.pack_forget()

        # ---- 底部状态栏 ----
        self._build_status_bar()

        # ---- 快捷键 ----
        self._bind_shortcuts()

    def _build_status_bar(self) -> None:
        """构建状态栏"""
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(fill=tk.X)
        inner = ttk.Frame(bar)
        inner.pack(fill=tk.X)

        ttk.Label(inner, textvariable=self.status_text,
                  style="Status.TLabel").pack(side=tk.LEFT, padx=5, pady=3)

        # 任务队列切换按钮
        ttk.Button(inner, text="📋 任务队列", command=self._toggle_task_queue,
                   style="Status.TLabel").pack(side=tk.RIGHT, padx=5, pady=1)

        # 主题指示器
        theme_label = "🌙 暗色" if self._tm.name == "dark" else "☀️ 亮色"
        ttk.Label(inner, text=theme_label,
                  style="Status.TLabel").pack(side=tk.RIGHT, padx=5, pady=3)

        # 版本信息
        try:
            from ultralytics import __version__ as ult_ver
            ttk.Label(inner, text=f"ultralytics {ult_ver}",
                      style="Status.TLabel").pack(side=tk.RIGHT, padx=5, pady=3)
        except Exception:
            pass

    # ==================== 快捷键 ====================

    def _bind_shortcuts(self) -> None:
        """绑定快捷键（增强版）"""
        # 全局
        self.root.bind("<Control-o>", lambda e: self.detect_tab._open_image())
        self.root.bind("<Control-d>", lambda e: self._safe_detect())
        self.root.bind("<Control-s>", lambda e: self.detect_tab._save_result())
        self.root.bind("<Control-q>", lambda e: self._on_close())
        self.root.bind("<F5>", lambda e: self._refresh_models())

        # 标签页切换
        self.root.bind("<Control-Key-1>", lambda e: self.main_tabs.select(0))
        self.root.bind("<Control-Key-2>", lambda e: self.main_tabs.select(1))
        self.root.bind("<Control-Key-3>", lambda e: self.main_tabs.select(2))
        self.root.bind("<Control-Key-4>", lambda e: self.main_tabs.select(3))
        self.root.bind("<Control-Key-5>", lambda e: self.main_tabs.select(4))
        self.root.bind("<Control-Key-6>", lambda e: self.main_tabs.select(5))
        self.root.bind("<Control-Key-7>", lambda e: self.main_tabs.select(6))
        self.root.bind("<Control-t>", lambda e: self.main_tabs.select(1))

        # 摄像头
        self.root.bind("<space>", lambda e: self._safe_toggle_camera())

        # 缩放（仅检测标签页）
        self.root.bind("<plus>", lambda e: self._safe_zoom(1.1))
        self.root.bind("<minus>", lambda e: self._safe_zoom(0.9))
        self.root.bind("<equal>", lambda e: self._safe_zoom(1.1))  # Shift+= on some kb

    def _safe_detect(self) -> None:
        """安全触发检测"""
        if self.main_tabs.index(self.main_tabs.select()) == 0:
            self.detect_tab._detect_image()

    def _safe_toggle_camera(self) -> None:
        """安全触发摄像头开关"""
        if self.main_tabs.index(self.main_tabs.select()) == 0:
            self.detect_tab._toggle_camera()

    def _safe_zoom(self, factor: float) -> None:
        """安全缩放（检测标签页）"""
        if self.main_tabs.index(self.main_tabs.select()) == 0:
            tab = self.detect_tab
            viewer = tab.original_viewer
            if tab.result_image is not None:
                viewer = tab.result_viewer
            if viewer._image is not None:
                new_scale = max(0.05, min(5.0, viewer._scale * factor))
                viewer._render(new_scale, viewer._offset_x, viewer._offset_y)

    # ==================== 任务队列 ====================

    def _toggle_task_queue(self) -> None:
        """展开/收起任务队列面板"""
        self.task_queue_visible = not self.task_queue_visible
        if self.task_queue_visible:
            self.task_queue.pack(fill=tk.X, padx=10, pady=(0, 3), after=self.main_tabs)
        else:
            self.task_queue.pack_forget()
        self.status_text.set(f"任务队列: {'显示' if self.task_queue_visible else '隐藏'}")

    def _on_queue_run_task(self, task: dict) -> None:
        """任务队列分发：将任务路由到对应的标签页

        Args:
            task: {"id", "type", "model", "params": {...}}
        """
        task_type = task.get("type", "")
        if task_type == "训练":
            self.main_tabs.select(1)
            self.train_tab._run_queued_task(task)
        elif task_type == "验证":
            self.main_tabs.select(2)
            self.val_tab._run_queued_task(task)
        elif task_type == "导出":
            self.main_tabs.select(3)
            self.export_tab._run_queued_task(task)
        elif task_type == "Benchmark":
            self.main_tabs.select(4)
            self.bench_tab._run_queued_task(task)

    # ==================== 公共方法 ====================

    def _refresh_models(self) -> None:
        """刷新所有模型列表"""
        self.detect_tab.refresh_models()
        self.train_tab.refresh_models()
        self.val_tab.refresh_models()
        self.export_tab.refresh_models()
        self.plugin_tab.refresh_models()
        self.browser_tab.refresh_models()
        self.status_text.set("模型列表已刷新")

    # ==================== 窗口事件 ====================

    def _on_close(self) -> None:
        """窗口关闭事件 — 保存配置"""
        try:
            # 保存窗口几何信息
            self._cfg.setdefault("window", {})["geometry"] = self.root.geometry()

            # 保存主题
            self._cfg["theme"] = self._tm.name

            # 保存各标签页的状态
            self.detect_tab.save_config()
            self.train_tab.save_config()
            self.val_tab.save_config()
            self.export_tab.save_config()
            self.bench_tab.save_config()
            self.plugin_tab.save_config()
            self.browser_tab.save_config()

            # 保存任务队列可见性
            self._cfg.setdefault("task_queue", {})["visible"] = self.task_queue_visible

            save_config(self._cfg)
        except Exception:
            pass

        self.root.destroy()

    # ==================== 启动 ====================

    def run(self) -> None:
        """启动 GUI 主循环"""
        self.root.mainloop()


def main():
    """入口函数"""
    gui = YOLOv8GUI()
    gui.run()


if __name__ == "__main__":
    main()

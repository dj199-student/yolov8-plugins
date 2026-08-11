"""
图表嵌入组件

基于 matplotlib + FigureCanvasTkAgg 的可嵌入图表组件。
支持：
- 训练曲线实时显示（Box/Cls/DFL Loss）
- mAP 指标柱状图
- PR 曲线
- 混淆矩阵
- 自适应暗/亮主题
"""

import tkinter as tk
from tkinter import ttk
from typing import Any
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class ChartPanel(ttk.LabelFrame):
    """可嵌入的 matplotlib 图表面板"""

    def __init__(
        self,
        parent,
        title: str = "图表",
        figsize: tuple = (5, 4),
        dpi: int = 100,
        dark_mode: bool = False,
        show_toolbar: bool = True,
        **kwargs
    ):
        super().__init__(parent, text=title, padding=5, **kwargs)
        self._dark_mode = dark_mode

        # matplotlib Figure
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self._setup_style()

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏（可选）
        if show_toolbar:
            toolbar = NavigationToolbar2Tk(self.canvas, self)
            toolbar.update()
            toolbar.pack(fill=tk.X)

        self._axes: dict[str, Any] = {}  # 缓存子图引用

    def _setup_style(self) -> None:
        """设置 matplotlib 样式"""
        if self._dark_mode:
            self.figure.patch.set_facecolor("#1e1e1e")
            self._fg = "#d4d4d4"
            self._grid = "#3e3e3e"
        else:
            self.figure.patch.set_facecolor("#ffffff")
            self._fg = "#1a1a1a"
            self._grid = "#e0e0e0"

    # ==================== 训练曲线 ====================

    def setup_training_curves(self) -> None:
        """初始化训练曲线子图（3 个 Loss + 1 个 mAP）"""
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        self._axes["box_loss"] = self.figure.add_subplot(gs[0, 0])
        self._axes["cls_loss"] = self.figure.add_subplot(gs[0, 1])
        self._axes["dfl_loss"] = self.figure.add_subplot(gs[1, 0])
        self._axes["metrics"] = self.figure.add_subplot(gs[1, 1])

        for key, ax in self._axes.items():
            ax.set_facecolor(self.figure.patch.get_facecolor())
            ax.tick_params(colors=self._fg, labelsize=8)
            ax.spines["bottom"].set_color(self._grid)
            ax.spines["left"].set_color(self._grid)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.title.set_color(self._fg)
            ax.xaxis.label.set_color(self._fg)
            ax.yaxis.label.set_color(self._fg)
            ax.grid(True, color=self._grid, linewidth=0.5, alpha=0.5)

        self._axes["box_loss"].set_title("Box Loss")
        self._axes["cls_loss"].set_title("Cls Loss")
        self._axes["dfl_loss"].set_title("DFL Loss")
        self._axes["metrics"].set_title("mAP@50")

        self._data = {
            "box_loss": deque(maxlen=500),
            "cls_loss": deque(maxlen=500),
            "dfl_loss": deque(maxlen=500),
            "map50": deque(maxlen=500),
            "epochs": deque(maxlen=500),
        }

    def update_training_curve(self, epoch: int, losses: dict, map50: float | None = None) -> None:
        """更新训练曲线（每 epoch 调用）

        Args:
            epoch: 当前 epoch 编号
            losses: {"box": float, "cls": float, "dfl": float}
            map50: mAP@50 值（验证时才有）
        """
        self._data["epochs"].append(epoch)
        self._data["box_loss"].append(losses.get("box", 0))
        self._data["cls_loss"].append(losses.get("cls", 0))
        self._data["dfl_loss"].append(losses.get("dfl", 0))
        if map50 is not None:
            self._data["map50"].append(map50)

        epochs = list(self._data["epochs"])

        for key in ["box_loss", "cls_loss", "dfl_loss"]:
            ax = self._axes.get(key)
            if ax is None:
                continue
            ax.cla()
            ax.plot(epochs, list(self._data[key]), color="#0078d4", linewidth=1.2)
            ax.set_title(ax.get_title(), fontsize=9)
            ax.grid(True, color=self._grid, linewidth=0.5, alpha=0.5)

        # mAP
        ax_map = self._axes.get("metrics")
        if ax_map is not None:
            ax_map.cla()
            if self._data["map50"]:
                ax_map.plot(
                    epochs[-len(self._data["map50"]):],
                    list(self._data["map50"]),
                    color="#388e3c", linewidth=1.5, marker=".", markersize=3,
                )
            ax_map.set_title("mAP@50", fontsize=9)
            ax_map.grid(True, color=self._grid, linewidth=0.5, alpha=0.5)

        self.canvas.draw_idle()

    # ==================== mAP 指标面板 ====================

    def show_metrics_bar(self, metrics: dict) -> None:
        """显示验证指标柱状图

        Args:
            metrics: {
                "mAP50": float, "mAP50-95": float,
                "precision": float, "recall": float, "f1": float
            }
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.figure.patch.get_facecolor())

        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ["#0078d4", "#106ebe", "#388e3c", "#ff8c00", "#d32f2f"]

        bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, max(1.0, max(values) * 1.15))
        ax.set_ylabel("Value", color=self._fg)
        ax.set_title("Validation Metrics", fontsize=11, color=self._fg, fontweight="bold")

        # 数值标签
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color=self._fg,
            )

        ax.tick_params(colors=self._fg, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self._grid)
        ax.spines["left"].set_color(self._grid)
        ax.grid(axis="y", color=self._grid, linewidth=0.5, alpha=0.5)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ==================== PR 曲线 ====================

    def show_pr_curve(self, recall: list, precision: list, ap: float = 0.0) -> None:
        """显示 PR 曲线

        Args:
            recall: Recall 值列表
            precision: Precision 值列表
            ap: Average Precision
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.figure.patch.get_facecolor())

        ax.plot(recall, precision, color="#0078d4", linewidth=1.5, label=f"AP={ap:.3f}")
        ax.fill_between(recall, precision, alpha=0.15, color="#0078d4")
        ax.set_xlabel("Recall", color=self._fg)
        ax.set_ylabel("Precision", color=self._fg)
        ax.set_title("Precision-Recall Curve", fontsize=11, color=self._fg, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=9)

        ax.tick_params(colors=self._fg, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self._grid)
        ax.spines["left"].set_color(self._grid)
        ax.grid(True, color=self._grid, linewidth=0.5, alpha=0.5)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ==================== Benchmark 对比 ====================

    def show_benchmark_comparison(self, results: list[dict]) -> None:
        """显示 Benchmark 对比图

        Args:
            results: [
                {"model": "yolov8n", "fps": 120, "params_m": 3.2, "flops_g": 8.7, "map50": 0.78},
                ...
            ]
        """
        self.figure.clear()

        # 子图 1: FPS 对比
        ax1 = self.figure.add_subplot(221)
        ax1.set_facecolor(self.figure.patch.get_facecolor())
        models = [r["model"] for r in results]
        fps_vals = [r.get("fps", 0) for r in results]
        bars = ax1.bar(models, fps_vals, color="#0078d4", edgecolor="white")
        ax1.set_title("FPS (越高越好)", fontsize=9, color=self._fg)
        ax1.tick_params(colors=self._fg, labelsize=7)
        for bar, val in zip(bars, fps_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}", ha="center", fontsize=7, color=self._fg)

        # 子图 2: 参数量
        ax2 = self.figure.add_subplot(222)
        ax2.set_facecolor(self.figure.patch.get_facecolor())
        params = [r.get("params_m", 0) for r in results]
        bars = ax2.bar(models, params, color="#388e3c", edgecolor="white")
        ax2.set_title("Params (M)", fontsize=9, color=self._fg)
        ax2.tick_params(colors=self._fg, labelsize=7)
        for bar, val in zip(bars, params):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", fontsize=7, color=self._fg)

        # 子图 3: FLOPs
        ax3 = self.figure.add_subplot(223)
        ax3.set_facecolor(self.figure.patch.get_facecolor())
        flops = [r.get("flops_g", 0) for r in results]
        bars = ax3.bar(models, flops, color="#ff8c00", edgecolor="white")
        ax3.set_title("FLOPs (G)", fontsize=9, color=self._fg)
        ax3.tick_params(colors=self._fg, labelsize=7)
        for bar, val in zip(bars, flops):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", fontsize=7, color=self._fg)

        # 子图 4: mAP
        ax4 = self.figure.add_subplot(224)
        ax4.set_facecolor(self.figure.patch.get_facecolor())
        maps = [r.get("map50", 0) for r in results]
        bars = ax4.bar(models, maps, color="#d32f2f", edgecolor="white")
        ax4.set_title("mAP@50", fontsize=9, color=self._fg)
        ax4.tick_params(colors=self._fg, labelsize=7)
        for bar, val in zip(bars, maps):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=7, color=self._fg)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ==================== 检测统计 ====================

    def show_detection_stats(self, class_counts: dict, confidences: list) -> None:
        """显示检测统计图

        Args:
            class_counts: {"person": 5, "car": 3, ...}
            confidences: [0.95, 0.87, ...]
        """
        self.figure.clear()

        # 子图 1: 类别分布（水平柱状图）
        ax1 = self.figure.add_subplot(121)
        ax1.set_facecolor(self.figure.patch.get_facecolor())
        if class_counts:
            sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [item[0] for item in sorted_items]
            counts = [item[1] for item in sorted_items]
            ax1.barh(names, counts, color="#0078d4", edgecolor="white", height=0.6)
            ax1.set_xlabel("Count", color=self._fg, fontsize=8)
            ax1.set_title("Class Distribution", fontsize=9, color=self._fg, fontweight="bold")
            ax1.tick_params(colors=self._fg, labelsize=7)
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax1.transAxes, color=self._fg)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # 子图 2: 置信度直方图
        ax2 = self.figure.add_subplot(122)
        ax2.set_facecolor(self.figure.patch.get_facecolor())
        if confidences:
            ax2.hist(confidences, bins=20, color="#388e3c", edgecolor="white",
                     alpha=0.85, range=(0, 1))
            ax2.set_xlabel("Confidence", color=self._fg, fontsize=8)
            ax2.set_ylabel("Count", color=self._fg, fontsize=8)
            ax2.set_title("Confidence Distribution", fontsize=9, color=self._fg, fontweight="bold")
            ax2.axvline(x=sum(confidences) / len(confidences), color="#d32f2f",
                        linestyle="--", linewidth=1, label=f"Avg={sum(confidences)/len(confidences):.3f}")
            ax2.legend(fontsize=7, labelcolor=self._fg)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax2.transAxes, color=self._fg)
        ax2.tick_params(colors=self._fg, labelsize=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ==================== 主题切换 ====================

    def set_dark_mode(self, dark: bool) -> None:
        """切换暗色模式"""
        self._dark_mode = dark
        self._setup_style()
        self.canvas.draw_idle()

    def save_figure(self, path: str) -> None:
        """保存图表到文件"""
        self.figure.savefig(path, dpi=150, bbox_inches="tight",
                           facecolor=self.figure.get_facecolor())

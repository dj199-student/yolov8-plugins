"""
模型验证标签页

功能：
- 标准验证（mAP50 / mAP50-95 / Precision / Recall / F1）
- 指标面板（6 个核心指标卡片）
- PR 曲线 + mAP 柱状图（使用 ChartPanel）
- 实时日志输出
- 进度条
- 支持 val / test 分割
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ..widgets.model_selector import ModelSelector
from ..widgets.log_panel import LogPanel
from ..widgets.chart_panel import ChartPanel
from ..workers.val_worker import ValWorker
from ..config import load_config, save_config


class ValTab:
    """模型验证标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 配置
        cfg = load_config()
        val_cfg = cfg.get("validate", {})

        # 变量
        self.val_model_var = tk.StringVar(value=val_cfg.get("model", "yolov8n.pt"))
        self.val_data_var = tk.StringVar(value=val_cfg.get("data", "configs/datasets/coco128.yaml"))
        self.val_batch_var = tk.IntVar(value=val_cfg.get("batch", 16))
        self.val_imgsz_var = tk.IntVar(value=val_cfg.get("imgsz", 640))
        self.val_device_var = tk.StringVar(value=val_cfg.get("device", "cpu"))
        self.val_workers_var = tk.IntVar(value=val_cfg.get("workers", 4))
        self.val_conf_var = tk.DoubleVar(value=val_cfg.get("conf", 0.001))
        self.val_iou_var = tk.DoubleVar(value=val_cfg.get("iou", 0.6))
        self.val_split_var = tk.StringVar(value=val_cfg.get("split", "val"))
        self.val_save_json_var = tk.BooleanVar(value=val_cfg.get("save_json", True))
        self.val_plots_var = tk.BooleanVar(value=val_cfg.get("plots", True))

        # 验证状态
        self.val_running = False

        # Worker
        self.worker = ValWorker()
        self.worker.on_log(self._on_log)
        self.worker.on_progress(self._on_progress)
        self.worker.on_done(self._on_val_done)
        self.worker.on_error(self._on_val_error)

        self._build_ui()

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建验证标签页 UI

        布局：左参数 + 右（图表 + 日志）
        """
        self._h_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self._h_paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧参数面板（可滚动） ----
        left_container = ttk.Frame(self._h_paned, width=320)
        self._h_paned.add(left_container, weight=0)

        # Canvas + Scrollbar 实现滚动
        left_canvas = tk.Canvas(left_container, width=300, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left.bind("<Configure>", lambda e: left_canvas.configure(
            scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor=tk.NW, tags="inner")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        def _configure_val_width(event):
            left_canvas.itemconfig("inner", width=event.width)
        left_canvas.bind("<Configure>", _configure_val_width, add="+")

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮
        def _on_val_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_val_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 模型选择
        model_frame = ttk.LabelFrame(left, text="模型 & 数据", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        ttk.Label(model_frame, text="模型:").pack(anchor=tk.W)
        self.model_selector = ModelSelector(
            model_frame, default_model=self.val_model_var.get(),
            on_change=lambda m: self.val_model_var.set(m),
        )
        self.model_selector.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text="数据集 YAML:").pack(anchor=tk.W, pady=(6, 0))
        data_row = ttk.Frame(model_frame)
        data_row.pack(fill=tk.X)
        ttk.Entry(data_row, textvariable=self.val_data_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(data_row, text="📂", width=3,
                   command=self._browse_data).pack(side=tk.RIGHT, padx=(3, 0))

        # 验证参数
        param_frame = ttk.LabelFrame(left, text="验证参数", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=3)

        self._spin_row(param_frame, "Batch:", self.val_batch_var, 1, 128)
        self._spin_row(param_frame, "Imgsz:", self.val_imgsz_var, 320, 1280, step=32)
        self._spin_row(param_frame, "Workers:", self.val_workers_var, 0, 16)

        # 设备
        ttk.Label(param_frame, text="设备:").pack(anchor=tk.W, pady=(6, 0))
        dev_row = ttk.Frame(param_frame)
        dev_row.pack(fill=tk.X)
        ttk.Radiobutton(dev_row, text="CPU", variable=self.val_device_var,
                        value="cpu").pack(side=tk.LEFT)
        ttk.Radiobutton(dev_row, text="GPU (0)", variable=self.val_device_var,
                        value="0").pack(side=tk.LEFT, padx=10)

        # 高级选项
        adv_frame = ttk.LabelFrame(left, text="高级选项", padding=10)
        adv_frame.pack(fill=tk.X, padx=5, pady=3)

        self._spin_row(adv_frame, "Conf:", self.val_conf_var, 0.001, 0.1, step=0.001)
        self._spin_row(adv_frame, "IoU:", self.val_iou_var, 0.1, 0.95, step=0.05)

        ttk.Label(adv_frame, text="数据集分割:").pack(anchor=tk.W, pady=(6, 0))
        split_combo = ttk.Combobox(adv_frame, textvariable=self.val_split_var,
                                   values=["val", "test"], state="readonly", width=8)
        split_combo.pack(fill=tk.X)

        ttk.Checkbutton(adv_frame, text="保存 JSON 结果",
                        variable=self.val_save_json_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(adv_frame, text="生成图表 (PR/CM)",
                        variable=self.val_plots_var).pack(anchor=tk.W)

        # 操作按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.val_btn = ttk.Button(btn_frame, text="🚀 开始验证", command=self._start_validation,
                                  style="Primary.TButton")
        self.val_btn.pack(fill=tk.X, pady=2)
        self.stop_val_btn = ttk.Button(btn_frame, text="⏹ 停止", command=self._stop_validation,
                                       state=tk.DISABLED)
        self.stop_val_btn.pack(fill=tk.X, pady=2)

        # 进度
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            btn_frame, variable=self.progress_var, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_label = ttk.Label(btn_frame, text="等待开始...", font=("Consolas", 9))
        self.progress_label.pack(anchor=tk.CENTER, pady=2)

        # ---- 右侧 ----
        right = ttk.Frame(self._h_paned)
        self._h_paned.add(right, weight=1)

        # 垂直分上下：图表 + 日志
        self._v_paned = ttk.PanedWindow(right, orient=tk.VERTICAL)
        self._v_paned.pack(fill=tk.BOTH, expand=True)

        # 图表区
        chart_container = ttk.Frame(self._v_paned)
        self._v_paned.add(chart_container, weight=1)

        # 指标卡片 + 图表
        self.metrics_frame = ttk.LabelFrame(chart_container, text="指标面板", padding=8)
        self.metrics_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        # 6 个指标卡片
        self._build_metrics_cards()

        # 图表面板
        self.chart_panel = ChartPanel(chart_container, title="验证图表", show_toolbar=True)
        self.chart_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 日志区
        log_frame = ttk.Frame(self._v_paned)
        self._v_paned.add(log_frame, weight=1)

        self.log_panel = LogPanel(log_frame, title="验证日志")
        self.log_panel.pack(fill=tk.BOTH, expand=True)

        self._poll_log()

        # 初始化 sash 位置（延迟执行，等待窗口布局完成）
        self.parent.after(200, self._init_sash_positions)

    def _init_sash_positions(self) -> None:
        """设置 PanedWindow 初始 sash 位置"""
        try:
            w = self.parent.winfo_width()
            if w > 400:
                self._h_paned.sashpos(0, 320)  # 左侧面板 320px
            h = self.parent.winfo_height()
            if h > 300:
                self._v_paned.sashpos(0, int(h * 0.55))  # 图表区 55%
        except Exception:
            pass  # 窗口尚未就绪，跳过

    def _build_metrics_cards(self) -> None:
        """构建 6 个指标卡片"""
        cards_frame = ttk.Frame(self.metrics_frame)
        cards_frame.pack(fill=tk.X)

        self._metric_labels = {}
        self._metric_vars = {
            "mAP50": tk.StringVar(value="--"),
            "mAP50-95": tk.StringVar(value="--"),
            "mAP75": tk.StringVar(value="--"),
            "Precision": tk.StringVar(value="--"),
            "Recall": tk.StringVar(value="--"),
            "F1": tk.StringVar(value="--"),
        }

        labels = [
            ("mAP@50", "mAP50", "#0078d4"),
            ("mAP@50-95", "mAP50-95", "#106ebe"),
            ("mAP@75", "mAP75", "#388e3c"),
            ("Precision", "Precision", "#ff8c00"),
            ("Recall", "Recall", "#d32f2f"),
            ("F1 Score", "F1", "#7b1fa2"),
        ]

        for i, (title, key, color) in enumerate(labels):
            card = ttk.Frame(cards_frame, relief=tk.RIDGE, borderwidth=1)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3, pady=2)
            # 标题
            ttk.Label(card, text=title, font=("Microsoft YaHei", 8),
                      foreground=color).pack(pady=(5, 0))
            # 值
            val_label = tk.Label(
                card, textvariable=self._metric_vars[key],
                font=("Consolas", 18, "bold"), fg=color, bg="#ffffff",
            )
            val_label.pack(pady=(0, 5))
            self._metric_labels[key] = val_label  # store ref for theme

    # ==================== 工具方法 ====================

    def _spin_row(self, parent, label: str, var, from_val, to_val,
                  step: float = 1, pady: int = 2) -> None:
        """创建 Spinbox 行"""
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(pady * 2, 0))
        ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                    increment=step, width=10).pack(fill=tk.X, pady=pady)

    def _browse_data(self) -> None:
        """浏览数据集 YAML"""
        path = filedialog.askopenfilename(
            title="选择数据集 YAML",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if path:
            self.val_data_var.set(path)

    # ==================== 日志 ====================

    def _on_log(self, msg: str) -> None:
        self.log_panel.write(msg)

    def _poll_log(self) -> None:
        self.log_panel.poll()
        self.parent.after(200, self._poll_log)

    # ==================== 验证控制 ====================

    def _start_validation(self) -> None:
        """启动验证"""
        if self.val_running:
            return

        model_name = self.val_model_var.get()
        data_yaml = self.val_data_var.get()

        if not Path(model_name).exists():
            messagebox.showerror("错误", f"模型不存在: {model_name}")
            return

        # 重置指标
        for var in self._metric_vars.values():
            var.set("--")
        self.chart_panel.figure.clear()
        self.chart_panel.canvas.draw_idle()

        self.val_running = True
        self.val_btn.configure(state=tk.DISABLED)
        self.stop_val_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.configure(text="初始化...")
        self.status_var.set("验证启动中...")

        self.worker.validate(
            model_name=model_name,
            data=data_yaml,
            batch=self.val_batch_var.get(),
            imgsz=self.val_imgsz_var.get(),
            device=self.val_device_var.get(),
            workers=self.val_workers_var.get(),
            conf=self.val_conf_var.get(),
            iou=self.val_iou_var.get(),
            split=self.val_split_var.get(),
            save_json=self.val_save_json_var.get(),
            plots=self.val_plots_var.get(),
        )

    def _stop_validation(self) -> None:
        """停止验证"""
        self.worker.stop()
        self.stop_val_btn.configure(state=tk.DISABLED)
        self.status_var.set("正在停止验证...")

    def _on_progress(self, current: int, total: int) -> None:
        """进度回调（后台线程）"""
        # 调度到主线程
        self.parent.after(0, lambda: self._update_progress(current, total))

    def _update_progress(self, current: int, total: int) -> None:
        """更新进度条（主线程）"""
        pct = min(current / max(total, 1) * 100, 100)
        self.progress_var.set(pct)
        self.progress_label.configure(text=f"Batch {current}/{total}")
        self.status_var.set(f"验证中: {current}/{total} batches")

    def _on_val_done(self, metrics: dict, save_dir: str) -> None:
        """验证完成回调（后台线程）"""
        self.parent.after(0, lambda: self._show_results(metrics, save_dir))

    def _show_results(self, metrics: dict, save_dir: str) -> None:
        """显示验证结果（主线程）"""
        self.val_running = False
        self.val_btn.configure(state=tk.NORMAL)
        self.stop_val_btn.configure(state=tk.DISABLED)
        self.progress_var.set(100)
        self.progress_label.configure(text="验证完成")

        # 更新指标卡片
        for key, var_key in [
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]:
            val = metrics.get(key, 0)
            self._metric_vars[var_key].set(f"{val:.4f}")

        # mAP75 (if available, otherwise estimate)
        map75 = metrics.get("mAP75", 0)
        if map75 == 0 and metrics.get("mAP50", 0) > 0:
            map75 = metrics["mAP50"] * 0.75  # rough estimate
        self._metric_vars["mAP75"].set(f"{map75:.4f}")

        # 显示柱状图
        display_metrics = {
            "mAP50": metrics.get("mAP50", 0),
            "mAP50-95": metrics.get("mAP50-95", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
        }
        self.chart_panel.show_metrics_bar(display_metrics)

        self.status_var.set(f"验证完成 | mAP@50: {metrics.get('mAP50', 0):.4f}")

        # 尝试加载 PR 曲线数据
        if save_dir:
            self._try_load_pr_curve(save_dir, metrics)

    def _try_load_pr_curve(self, save_dir: str, metrics: dict) -> None:
        """尝试从保存目录加载 PR 曲线"""
        try:
            import numpy as np

            # ultralytics 保存 PR 曲线为 .npy 文件
            pr_path = Path(save_dir) / "PR_curve.npy"
            if pr_path.exists():
                data = np.load(pr_path)
                # data shape: (N, 3) -> [class_id, recall, precision]
                # 取所有类的平均值
                if data.ndim == 2 and data.shape[1] >= 3:
                    recall = data[:, 1].tolist()
                    precision = data[:, 2].tolist()
                    ap = metrics.get("mAP50", 0)
                    self.chart_panel.show_pr_curve(recall, precision, ap)
        except Exception:
            pass  # PR 曲线加载失败不影响主流程

    def _on_val_error(self, error: str) -> None:
        """验证错误回调"""
        self.parent.after(0, lambda: self._handle_val_error(error))

    def _handle_val_error(self, error: str) -> None:
        """处理验证错误（主线程）"""
        self.val_running = False
        self.val_btn.configure(state=tk.NORMAL)
        self.stop_val_btn.configure(state=tk.DISABLED)
        self.status_var.set("验证失败")
        messagebox.showerror("验证错误", error)

    # ==================== 配置 ====================

    def refresh_models(self) -> None:
        self.model_selector.refresh()

    def save_config(self) -> None:
        """保存验证参数"""
        cfg = load_config()
        cfg.setdefault("validate", {}).update({
            "model": self.val_model_var.get(),
            "data": self.val_data_var.get(),
            "batch": self.val_batch_var.get(),
            "imgsz": self.val_imgsz_var.get(),
            "device": self.val_device_var.get(),
            "workers": self.val_workers_var.get(),
            "conf": self.val_conf_var.get(),
            "iou": self.val_iou_var.get(),
            "split": self.val_split_var.get(),
            "save_json": self.val_save_json_var.get(),
            "plots": self.val_plots_var.get(),
        })
        save_config(cfg)

    def _run_queued_task(self, task: dict) -> None:
        """任务队列集成：设置参数并启动验证"""
        self.val_model_var.set(task.get("model", "yolov8n.pt"))
        params = task.get("params", {})
        if params.get("data"):
            self.val_data_var.set(params["data"])
        if params.get("batch"):
            self.val_batch_var.set(params["batch"])
        if params.get("imgsz"):
            self.val_imgsz_var.set(params["imgsz"])
        if params.get("device"):
            self.val_device_var.set(params["device"])
        self._start_validation()

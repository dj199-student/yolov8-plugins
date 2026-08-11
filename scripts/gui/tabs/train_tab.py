"""
训练标签页

功能：
- 完整训练参数配置（20+ 参数，分 6 组）
- 可折叠的数据增强参数组
- 实时日志输出（使用 LogPanel）
- 训练进度条 + Epoch 计数
- 可真正停止训练（通过 TrainWorker）
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ultralytics import YOLO

from ..widgets.model_selector import ModelSelector
from ..widgets.log_panel import LogPanel
from ..workers.train_worker import TrainWorker
from ..config import load_config, save_config


class TrainTab:
    """训练标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 配置
        cfg = load_config()
        train_cfg = cfg.get("train", {})
        aug_cfg = train_cfg.get("augment", {})

        # ---- 模型 & 数据 ----
        self.train_model_var = tk.StringVar(value=train_cfg.get("model", "yolov8n.pt"))
        self.data_path_var = tk.StringVar(value=train_cfg.get("data", "configs/datasets/coco128.yaml"))
        self.resume_var = tk.BooleanVar(value=train_cfg.get("resume", False))
        self.pretrained_var = tk.BooleanVar(value=train_cfg.get("pretrained", True))

        # ---- 基础超参数 ----
        self.epochs_var = tk.IntVar(value=train_cfg.get("epochs", 100))
        self.imgsz_var = tk.IntVar(value=train_cfg.get("imgsz", 640))
        self.batch_var = tk.IntVar(value=train_cfg.get("batch", 16))
        self.device_var = tk.StringVar(value=train_cfg.get("device", "cpu"))
        self.workers_var = tk.IntVar(value=train_cfg.get("workers", 4))

        # ---- 优化器 ----
        self.optimizer_var = tk.StringVar(value=train_cfg.get("optimizer", "auto"))
        self.lr0_var = tk.DoubleVar(value=train_cfg.get("lr0", 0.01))
        self.lrf_var = tk.DoubleVar(value=train_cfg.get("lrf", 0.01))
        self.momentum_var = tk.DoubleVar(value=train_cfg.get("momentum", 0.937))
        self.weight_decay_var = tk.DoubleVar(value=train_cfg.get("weight_decay", 0.0005))

        # ---- 学习率调度 ----
        self.cos_lr_var = tk.BooleanVar(value=train_cfg.get("cos_lr", True))
        self.warmup_epochs_var = tk.DoubleVar(value=train_cfg.get("warmup_epochs", 3.0))

        # ---- 高级 ----
        self.amp_var = tk.BooleanVar(value=train_cfg.get("amp", True))
        self.close_mosaic_var = tk.IntVar(value=train_cfg.get("close_mosaic", 10))

        # ---- 数据增强 ----
        self.aug_vars = {
            "hsv_h": tk.DoubleVar(value=aug_cfg.get("hsv_h", 0.015)),
            "hsv_s": tk.DoubleVar(value=aug_cfg.get("hsv_s", 0.7)),
            "hsv_v": tk.DoubleVar(value=aug_cfg.get("hsv_v", 0.4)),
            "degrees": tk.DoubleVar(value=aug_cfg.get("degrees", 0.0)),
            "translate": tk.DoubleVar(value=aug_cfg.get("translate", 0.1)),
            "scale": tk.DoubleVar(value=aug_cfg.get("scale", 0.5)),
            "fliplr": tk.DoubleVar(value=aug_cfg.get("fliplr", 0.5)),
            "mosaic": tk.DoubleVar(value=aug_cfg.get("mosaic", 1.0)),
            "mixup": tk.DoubleVar(value=aug_cfg.get("mixup", 0.0)),
        }
        self._aug_expanded = False  # 增强面板折叠状态

        # 训练状态
        self.train_running = False
        self.worker = TrainWorker()
        self.worker.on_log(self._on_log)
        self.worker.on_epoch(self._on_epoch)
        self.worker.on_done(self._on_train_done)
        self.worker.on_error(self._on_train_error)
        self.worker.on_checkpoint(self._on_checkpoint)

        # 检查是否有中断的训练
        self._pending_resume = train_cfg.get("checkpoint")

        self._build_ui()

        # 延迟检查中断训练（等 UI 就绪）
        if self._pending_resume and self._pending_resume.get("status") == "training":
            self.parent.after(500, self._check_interrupted_training)

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建训练标签页 UI"""
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧参数面板 ----
        left_canvas = tk.Canvas(paned, width=380, highlightthickness=0)
        left_scroll = ttk.Scrollbar(paned, orient=tk.VERTICAL, command=left_canvas.yview)
        left_frame = ttk.Frame(left_canvas)

        left_frame.bind("<Configure>", lambda e: left_canvas.configure(
            scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_frame, anchor=tk.NW)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        paned.add(left_canvas, weight=0)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 1. 模型 & 数据
        self._build_section_model_data(left_frame)
        # 2. 基础超参数
        self._build_section_basic(left_frame)
        # 3. 优化器
        self._build_section_optimizer(left_frame)
        # 4. 学习率调度
        self._build_section_lr_schedule(left_frame)
        # 5. 数据增强（可折叠）
        self._build_section_augment(left_frame)
        # 6. 高级设置
        self._build_section_advanced(left_frame)

        # 训练按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        self.train_btn = ttk.Button(btn_frame, text="🔥 开始训练", command=self._start_training,
                                     style="Primary.TButton")
        self.train_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ 停止训练", command=self._stop_training,
                                    state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        # 进度
        progress_frame = ttk.LabelFrame(left_frame, text="训练进度", padding=10)
        progress_frame.pack(fill=tk.X, pady=(8, 0))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, mode="determinate")
        self.progress_bar.pack(fill=tk.X)
        self.epoch_label = ttk.Label(progress_frame, text="等待开始...")
        self.epoch_label.pack(anchor=tk.CENTER, pady=(4, 0))

        # ---- 右侧日志区 ----
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self.log_panel = LogPanel(right, title="训练日志")
        self.log_panel.pack(fill=tk.BOTH, expand=True)

        # 定时刷新日志
        self._poll_log()

    def _make_section(self, parent, title: str) -> ttk.LabelFrame:
        """创建参数组框架"""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.X, pady=(0, 8))
        return frame

    def _spinbox_row(self, parent, label: str, var, from_val, to_val,
                     step: float = 1, pady: int = 2) -> None:
        """创建标签 + Spinbox 行"""
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        widget = ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                             increment=step, width=10)
        widget.pack(fill=tk.X, pady=pady)

    def _entry_row(self, parent, label: str, var: tk.StringVar, browse_cmd=None, pady: int = 2) -> None:
        """创建标签 + Entry 行（可选浏览按钮）"""
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=pady)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse_cmd:
            ttk.Button(row, text="📂", width=3, command=browse_cmd).pack(side=tk.RIGHT, padx=(3, 0))

    # ---- 各组构建 ----

    def _build_section_model_data(self, parent) -> None:
        frame = self._make_section(parent, "模型 & 数据")
        ttk.Label(frame, text="基础模型:").pack(anchor=tk.W)
        self.model_selector = ModelSelector(
            frame, default_model=self.train_model_var.get(),
            on_change=lambda m: self.train_model_var.set(m),
        )
        self.model_selector.pack(fill=tk.X, pady=2)
        self._entry_row(frame, "数据集 YAML:", self.data_path_var,
                        self._browse_dataset)
        ttk.Checkbutton(frame, text="断点续训 (resume)", variable=self.resume_var).pack(
            anchor=tk.W, pady=2)
        ttk.Checkbutton(frame, text="预训练权重 (pretrained)", variable=self.pretrained_var).pack(
            anchor=tk.W)

    def _build_section_basic(self, parent) -> None:
        frame = self._make_section(parent, "基础超参数")
        self._spinbox_row(frame, "Epochs:", self.epochs_var, 1, 1000)
        self._spinbox_row(frame, "图像尺寸 (imgsz):", self.imgsz_var, 320, 1280, step=32)
        self._spinbox_row(frame, "批量大小 (batch):", self.batch_var, 1, 128)
        self._spinbox_row(frame, "工作线程 (workers):", self.workers_var, 0, 16)

        ttk.Label(frame, text="设备:").pack(anchor=tk.W, pady=(6, 0))
        device_frame = ttk.Frame(frame)
        device_frame.pack(fill=tk.X)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var,
                        value="cpu").pack(side=tk.LEFT)
        ttk.Radiobutton(device_frame, text="GPU (0)", variable=self.device_var,
                        value="0").pack(side=tk.LEFT, padx=10)

    def _build_section_optimizer(self, parent) -> None:
        frame = self._make_section(parent, "优化器")
        ttk.Label(frame, text="优化器:").pack(anchor=tk.W)
        opt_combo = ttk.Combobox(frame, textvariable=self.optimizer_var,
                                 values=["auto", "SGD", "Adam", "AdamW", "RMSProp"],
                                 state="readonly", width=10)
        opt_combo.pack(fill=tk.X, pady=2)

        self._spinbox_row(frame, "初始学习率 (lr0):", self.lr0_var, 0.001, 0.1, step=0.001)
        self._spinbox_row(frame, "最终学习率因子 (lrf):", self.lrf_var, 0.001, 0.1, step=0.001)
        self._spinbox_row(frame, "动量 (momentum):", self.momentum_var, 0.5, 0.999, step=0.001)
        self._spinbox_row(frame, "权重衰减 (weight_decay):", self.weight_decay_var,
                          0.0, 0.01, step=0.0001)

    def _build_section_lr_schedule(self, parent) -> None:
        frame = self._make_section(parent, "学习率调度")
        ttk.Checkbutton(frame, text="余弦退火 (cos_lr)", variable=self.cos_lr_var).pack(
            anchor=tk.W, pady=2)
        self._spinbox_row(frame, "预热轮数 (warmup_epochs):", self.warmup_epochs_var,
                          0.0, 20.0, step=0.5)

    def _build_section_augment(self, parent) -> None:
        """数据增强（可折叠）"""
        self.aug_frame = ttk.LabelFrame(parent, text="数据增强 ▸", padding=10)
        self.aug_frame.pack(fill=tk.X, pady=(0, 8))

        # 标题行 — 点击展开/折叠
        self.aug_header = ttk.Label(
            self.aug_frame,
            text="点击展开 ▼   (HSV / 旋转 / 平移 / 缩放 / Mosaic / MixUp)",
            cursor="hand2", foreground="#888888",
        )
        self.aug_header.pack(anchor=tk.W)
        self.aug_header.bind("<Button-1>", self._toggle_augment)
        self.aug_header.bind("<Enter>", lambda e: self.aug_header.configure(foreground="#0078d4"))
        self.aug_header.bind("<Leave>", lambda e: self.aug_header.configure(foreground="#888888"))

        # 增强参数容器（默认隐藏）
        self.aug_content = ttk.Frame(self.aug_frame)

    def _build_section_advanced(self, parent) -> None:
        frame = self._make_section(parent, "高级设置")
        ttk.Checkbutton(frame, text="AMP 混合精度", variable=self.amp_var).pack(
            anchor=tk.W, pady=2)
        self._spinbox_row(frame, "关闭 Mosaic (最后 N 轮):", self.close_mosaic_var, 0, 50)

    def _toggle_augment(self, event=None) -> None:
        """展开/折叠数据增强面板"""
        if self._aug_expanded:
            self.aug_content.pack_forget()
            self.aug_frame.configure(text="数据增强 ▸")
            self.aug_header.configure(text="点击展开 ▼   (HSV / 旋转 / 平移 / 缩放 / Mosaic / MixUp)")
            self._aug_expanded = False
        else:
            self._build_augment_content()
            self._aug_expanded = True

    def _build_augment_content(self) -> None:
        """构建增强参数内容（首次展开时）"""
        if self._aug_expanded:
            return

        self.aug_content.pack(fill=tk.X, pady=(5, 0))
        self.aug_frame.configure(text="数据增强 ▾")
        self.aug_header.configure(text="点击收起 ▲")

        # HSV
        hsv_frame = ttk.LabelFrame(self.aug_content, text="HSV 颜色增强", padding=5)
        hsv_frame.pack(fill=tk.X, pady=(0, 5))
        for key, label in [("hsv_h", "HSV-H (色相):"), ("hsv_s", "HSV-S (饱和度):"),
                           ("hsv_v", "HSV-V (明度):")]:
            self._spinbox_row(hsv_frame, label, self.aug_vars[key], 0.0, 1.0, step=0.01)

        # 几何变换
        geo_frame = ttk.LabelFrame(self.aug_content, text="几何变换", padding=5)
        geo_frame.pack(fill=tk.X, pady=(0, 5))
        for key, label, lo, hi, step in [
            ("degrees", "旋转角度:", 0.0, 180.0, 1.0),
            ("translate", "平移比例:", 0.0, 0.5, 0.01),
            ("scale", "缩放比例:", 0.0, 1.0, 0.01),
            ("fliplr", "水平翻转概率:", 0.0, 1.0, 0.1),
        ]:
            self._spinbox_row(geo_frame, label, self.aug_vars[key], lo, hi, step=step)

        # 混合增强
        mix_frame = ttk.LabelFrame(self.aug_content, text="混合增强", padding=5)
        mix_frame.pack(fill=tk.X)
        for key, label, lo, hi, step in [
            ("mosaic", "Mosaic 概率:", 0.0, 1.0, 0.1),
            ("mixup", "MixUp 概率:", 0.0, 1.0, 0.1),
        ]:
            self._spinbox_row(mix_frame, label, self.aug_vars[key], lo, hi, step=step)

    # ==================== 日志 ====================

    def _on_log(self, msg: str) -> None:
        """训练日志回调（线程安全）"""
        self.log_panel.write(msg)

    def _poll_log(self) -> None:
        """定时刷新日志"""
        self.log_panel.poll()
        self.parent.after(200, self._poll_log)

    # ==================== 训练控制 ====================

    def _browse_dataset(self) -> None:
        """选择数据集 YAML 文件"""
        path = filedialog.askopenfilename(
            title="选择数据集 YAML",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
        )
        if path:
            self.data_path_var.set(path)

    def _start_training(self) -> None:
        """启动训练"""
        if self.train_running:
            return

        model_name = self.train_model_var.get()
        data_yaml = self.data_path_var.get()

        if not Path(model_name).exists():
            messagebox.showerror("错误", f"模型不存在: {model_name}")
            return

        # 构建训练参数
        train_args = {
            "data": data_yaml,
            "epochs": self.epochs_var.get(),
            "imgsz": self.imgsz_var.get(),
            "batch": self.batch_var.get(),
            "device": self.device_var.get(),
            "workers": self.workers_var.get(),
            "lr0": self.lr0_var.get(),
            "lrf": self.lrf_var.get(),
            "momentum": self.momentum_var.get(),
            "weight_decay": self.weight_decay_var.get(),
            "optimizer": self.optimizer_var.get(),
            "cos_lr": self.cos_lr_var.get(),
            "warmup_epochs": self.warmup_epochs_var.get(),
            "amp": self.amp_var.get(),
            "close_mosaic": self.close_mosaic_var.get(),
            "resume": self.resume_var.get(),
            "pretrained": self.pretrained_var.get(),
            "verbose": False,
        }

        # 添加数据增强参数
        aug_prefixes = {
            "hsv_h": "hsv_h", "hsv_s": "hsv_s", "hsv_v": "hsv_v",
            "degrees": "degrees", "translate": "translate", "scale": "scale",
            "fliplr": "fliplr", "mosaic": "mosaic", "mixup": "mixup",
        }
        for key, param_name in aug_prefixes.items():
            if key in self.aug_vars:
                train_args[param_name] = self.aug_vars[key].get()

        # UI 状态更新
        self.train_running = True
        self.train_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.epoch_label.configure(text="初始化...")
        self.status_var.set("训练启动中...")

        # 后台启动
        self.worker.start(model_name, **train_args)

    def _stop_training(self) -> None:
        """停止训练 — 使用 TrainWorker 的优雅停止机制"""
        self.worker.stop()
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_epoch(self, epoch: int, total: int, losses: dict) -> None:
        """每 epoch 回调（主线程）"""
        pct = epoch / total * 100
        self.progress_var.set(pct)
        self.epoch_label.configure(text=f"Epoch {epoch}/{total}")
        self.status_var.set(f"训练中: Epoch {epoch}/{total}")

    def _on_train_done(self, stopped: bool) -> None:
        """训练完成回调"""
        self.train_running = False
        self.train_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        if stopped:
            self.epoch_label.configure(text="训练已停止")
            self.status_var.set("训练已停止")
        else:
            self.epoch_label.configure(text="训练完成")
            self.status_var.set("训练完成")
            messagebox.showinfo("训练完成", "模型训练已完成！\n请查看日志获取详情。")

    def _on_train_error(self, error: str) -> None:
        """训练错误回调"""
        self.train_running = False
        self.train_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"训练异常: {error[:50]}...")
        messagebox.showerror("训练失败", error)

    # ==================== Checkpoint & Resume ====================

    def _on_checkpoint(self, state: dict) -> None:
        """Checkpoint 回调（后台线程）—— 持久化训练进度"""
        from ..config import save_train_checkpoint, add_train_history

        save_train_checkpoint(state)

        # 如果训练结束，添加到历史记录
        if state.get("status") in ("completed", "stopped", "crashed"):
            add_train_history({
                "model": state.get("model", ""),
                "data": state.get("data", ""),
                "epochs": state.get("total_epochs", 0),
                "best_pt": state.get("best_pt", ""),
                "timestamp": state.get("timestamp", ""),
                "status": state.get("status", ""),
            })

    def _check_interrupted_training(self) -> None:
        """检测中断的训练并询问是否恢复"""
        state = self._pending_resume
        if not state:
            return
        self._pending_resume = None

        last_pt = state.get("last_pt", "")
        if last_pt and not Path(last_pt).exists():
            # checkpoint 文件不存在，清除过期状态
            from ..config import clear_train_checkpoint
            clear_train_checkpoint()
            return

        result = messagebox.askyesno(
            "检测到中断的训练",
            f"上次训练未正常完成！\n\n"
            f"模型: {state.get('model', 'Unknown')}\n"
            f"数据集: {state.get('data', 'Unknown')}\n"
            f"进度: Epoch {state.get('epoch', 0)}/{state.get('total_epochs', 0)}\n"
            f"保存目录: {state.get('save_dir', 'Unknown')}\n"
            f"检查点: {last_pt or 'None'}\n\n"
            f"是否恢复训练？"
        )

        if result:
            self._resume_from_state(state)
        else:
            from ..config import clear_train_checkpoint
            clear_train_checkpoint()
            self.status_var.set("已放弃恢复训练")

    def _resume_from_state(self, state: dict) -> None:
        """从保存的状态恢复训练"""
        last_pt = state.get("last_pt", "")
        if last_pt and Path(last_pt).exists():
            self.train_model_var.set(last_pt)
            self.resume_var.set(True)

        # 恢复训练参数
        train_args = state.get("train_args", {})
        if train_args.get("data"):
            self.data_path_var.set(train_args["data"])
        if train_args.get("epochs"):
            self.epochs_var.set(train_args["epochs"])
        if train_args.get("imgsz"):
            self.imgsz_var.set(train_args["imgsz"])
        if train_args.get("batch"):
            self.batch_var.set(train_args["batch"])
        if train_args.get("device"):
            self.device_var.set(train_args["device"])

        self.status_var.set(
            f"已准备恢复训练 (从 Epoch {state.get('epoch', 0)} 继续) — 点击开始训练"
        )

    def _run_queued_task(self, task: dict) -> None:
        """任务队列集成：设置参数并启动训练

        Args:
            task: {"model", "params": {...}}
        """
        self.train_model_var.set(task.get("model", "yolov8n.pt"))
        params = task.get("params", {})
        if params.get("data"):
            self.data_path_var.set(params["data"])
        if params.get("epochs"):
            self.epochs_var.set(params["epochs"])
        if params.get("imgsz"):
            self.imgsz_var.set(params["imgsz"])
        if params.get("batch"):
            self.batch_var.set(params["batch"])
        if params.get("device"):
            self.device_var.set(params["device"])
        self._start_training()

    # ==================== 配置 ====================

    def refresh_models(self) -> None:
        """刷新模型列表"""
        self.model_selector.refresh()

    def save_config(self) -> None:
        """保存训练参数到配置文件"""
        cfg = load_config()
        cfg.setdefault("train", {}).update({
            "model": self.train_model_var.get(),
            "data": self.data_path_var.get(),
            "epochs": self.epochs_var.get(),
            "imgsz": self.imgsz_var.get(),
            "batch": self.batch_var.get(),
            "device": self.device_var.get(),
            "workers": self.workers_var.get(),
            "lr0": self.lr0_var.get(),
            "lrf": self.lrf_var.get(),
            "momentum": self.momentum_var.get(),
            "weight_decay": self.weight_decay_var.get(),
            "optimizer": self.optimizer_var.get(),
            "cos_lr": self.cos_lr_var.get(),
            "warmup_epochs": self.warmup_epochs_var.get(),
            "amp": self.amp_var.get(),
            "close_mosaic": self.close_mosaic_var.get(),
            "resume": self.resume_var.get(),
            "pretrained": self.pretrained_var.get(),
            "augment": {k: v.get() for k, v in self.aug_vars.items()},
        })
        save_config(cfg)

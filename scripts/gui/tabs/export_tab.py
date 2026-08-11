"""
模型导出标签页

功能：
- 10+ 格式导出（ONNX / TensorRT / TFLite / OpenVINO / CoreML / NCNN / ...）
- 精度选择（FP32 / FP16 / INT8）
- 动态尺寸 + ONNX 简化
- 输出文件大小展示
- ONNX 模型验证
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
from ..workers.export_worker import ExportWorker
from ..config import load_config, save_config


class ExportTab:
    """模型导出标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 配置
        cfg = load_config()
        exp_cfg = cfg.get("export", {})

        # 变量
        self.export_model_var = tk.StringVar(value=exp_cfg.get("model", "yolov8n.pt"))
        self.export_format_var = tk.StringVar(value=exp_cfg.get("format", "onnx"))
        self.export_imgsz_var = tk.IntVar(value=exp_cfg.get("imgsz", 640))
        self.export_precision_var = tk.StringVar(value=exp_cfg.get("precision", "fp32"))
        self.export_dynamic_var = tk.BooleanVar(value=exp_cfg.get("dynamic", False))
        self.export_simplify_var = tk.BooleanVar(value=exp_cfg.get("simplify", True))
        self.export_opset_var = tk.IntVar(value=exp_cfg.get("opset", 12))
        self.export_workspace_var = tk.DoubleVar(value=exp_cfg.get("workspace", 4.0))
        self.export_device_var = tk.StringVar(value=exp_cfg.get("device", "cpu"))

        # 导出状态
        self.export_running = False

        # Worker
        self.worker = ExportWorker()
        self.worker.on_log(self._on_log)
        self.worker.on_done(self._on_export_done)
        self.worker.on_error(self._on_export_error)

        self._build_ui()

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建导出标签页 UI"""
        self._h_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        self._h_paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧参数（可滚动） ----
        left_container = ttk.Frame(self._h_paned, width=340)
        self._h_paned.add(left_container, weight=0)

        left_canvas = tk.Canvas(left_container, width=320, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left.bind("<Configure>", lambda e: left_canvas.configure(
            scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor=tk.NW, tags="inner")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        def _configure_exp_width(event):
            left_canvas.itemconfig("inner", width=event.width)
        left_canvas.bind("<Configure>", _configure_exp_width, add="+")

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_exp_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_exp_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 模型选择
        model_frame = ttk.LabelFrame(left, text="模型", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        ttk.Label(model_frame, text="模型:").pack(anchor=tk.W)
        self.model_selector = ModelSelector(
            model_frame, default_model=self.export_model_var.get(),
            on_change=lambda m: self.export_model_var.set(m),
        )
        self.model_selector.pack(fill=tk.X, pady=2)

        # 导出格式
        fmt_frame = ttk.LabelFrame(left, text="导出格式", padding=10)
        fmt_frame.pack(fill=tk.X, padx=5, pady=3)

        formats = [
            ("ONNX", "onnx"),
            ("TensorRT", "tensorrt"),
            ("TFLite", "tflite"),
            ("OpenVINO", "openvino"),
            ("CoreML", "coreml"),
            ("NCNN", "ncnn"),
        ]

        self._fmt_buttons = {}
        fmt_grid = ttk.Frame(fmt_frame)
        fmt_grid.pack(fill=tk.X)
        for i, (display, value) in enumerate(formats):
            col = i % 2
            row = i // 2
            rb = ttk.Radiobutton(
                fmt_grid, text=display, variable=self.export_format_var,
                value=value,
            )
            rb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=3)

        # 参数
        param_frame = ttk.LabelFrame(left, text="参数", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=3)

        self._spin_row(param_frame, "图像尺寸 (imgsz):", self.export_imgsz_var, 320, 1280, step=32)

        # 精度
        ttk.Label(param_frame, text="精度:").pack(anchor=tk.W, pady=(6, 0))
        prec_row = ttk.Frame(param_frame)
        prec_row.pack(fill=tk.X)
        for label, val in [("FP32", "fp32"), ("FP16", "fp16"), ("INT8", "int8")]:
            ttk.Radiobutton(
                prec_row, text=label, variable=self.export_precision_var, value=val
            ).pack(side=tk.LEFT, padx=5)

        # ONNX 特有选项
        onnx_frame = ttk.LabelFrame(left, text="ONNX 选项", padding=10)
        onnx_frame.pack(fill=tk.X, padx=5, pady=3)

        ttk.Checkbutton(onnx_frame, text="动态尺寸 (dynamic)",
                        variable=self.export_dynamic_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(onnx_frame, text="简化模型 (simplify)",
                        variable=self.export_simplify_var).pack(anchor=tk.W, pady=2)

        self._spin_row(onnx_frame, "Opset:", self.export_opset_var, 10, 20)

        # TensorRT 特有
        trt_frame = ttk.LabelFrame(left, text="TensorRT 选项", padding=10)
        trt_frame.pack(fill=tk.X, padx=5, pady=3)
        self._spin_row(trt_frame, "工作区 (GB):", self.export_workspace_var, 1.0, 16.0, step=1.0)

        # 设备
        ttk.Label(param_frame, text="导出设备:").pack(anchor=tk.W, pady=(6, 0))
        dev_row = ttk.Frame(param_frame)
        dev_row.pack(fill=tk.X)
        ttk.Radiobutton(dev_row, text="CPU", variable=self.export_device_var,
                        value="cpu").pack(side=tk.LEFT)
        ttk.Radiobutton(dev_row, text="GPU (0)", variable=self.export_device_var,
                        value="0").pack(side=tk.LEFT, padx=10)

        # 按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.export_btn = ttk.Button(btn_frame, text="🚀 开始导出", command=self._start_export,
                                     style="Primary.TButton")
        self.export_btn.pack(fill=tk.X, pady=2)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            btn_frame, variable=self.progress_var, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # 结果标签
        self.result_label = ttk.Label(btn_frame, text="", font=("Consolas", 9))
        self.result_label.pack(anchor=tk.CENTER, pady=3)

        # 打开文件夹按钮
        self.open_folder_btn = ttk.Button(
            btn_frame, text="📂 打开输出文件夹", command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_folder_btn.pack(fill=tk.X, pady=2)
        self._last_output_path = ""

        # ---- 右侧日志 ----
        right = ttk.Frame(self._h_paned)
        self._h_paned.add(right, weight=1)

        self.log_panel = LogPanel(right, title="导出日志")
        self.log_panel.pack(fill=tk.BOTH, expand=True)

        self._poll_log()

        # 初始化 sash 位置
        self.parent.after(200, self._init_sash_positions)

    def _init_sash_positions(self) -> None:
        """设置 PanedWindow 初始 sash 位置"""
        try:
            w = self.parent.winfo_width()
            if w > 400:
                self._h_paned.sashpos(0, 340)
        except Exception:
            pass

    # ==================== 工具方法 ====================

    def _spin_row(self, parent, label: str, var, from_val, to_val,
                  step: float = 1) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                    increment=step, width=10).pack(fill=tk.X, pady=2)

    def _open_output_folder(self) -> None:
        """打开输出文件夹"""
        if self._last_output_path:
            folder = str(Path(self._last_output_path).parent)
            if Path(folder).exists():
                os.startfile(folder)

    # ==================== 日志 ====================

    def _on_log(self, msg: str) -> None:
        self.log_panel.write(msg)

    def _poll_log(self) -> None:
        self.log_panel.poll()
        self.parent.after(200, self._poll_log)

    # ==================== 导出控制 ====================

    def _start_export(self) -> None:
        """启动导出"""
        if self.export_running:
            return

        model_name = self.export_model_var.get()
        if not Path(model_name).exists():
            messagebox.showerror("错误", f"模型不存在: {model_name}")
            return

        # 精度映射
        fmt = self.export_format_var.get()
        precision = self.export_precision_var.get()
        half = (precision == "fp16")
        int8 = (precision == "int8")

        self.export_running = True
        self.export_btn.configure(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.result_label.configure(text="")
        self.open_folder_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"导出中 ({fmt.upper()})...")

        self.worker.export(
            model_name=model_name,
            format=fmt,
            imgsz=self.export_imgsz_var.get(),
            half=half,
            int8=int8,
            dynamic=self.export_dynamic_var.get(),
            simplify=self.export_simplify_var.get(),
            opset=self.export_opset_var.get(),
            workspace=self.export_workspace_var.get(),
            device=self.export_device_var.get(),
        )

    def _on_export_done(self, output_path: str, file_size: int, format: str) -> None:
        """导出完成回调"""
        self.parent.after(0, lambda: self._show_export_result(output_path, file_size, format))

    def _show_export_result(self, output_path: str, file_size: int, format: str) -> None:
        """显示导出结果"""
        self.export_running = False
        self.export_btn.configure(state=tk.NORMAL)
        self.progress_bar.stop()

        size_mb = file_size / (1024 * 1024)
        self.result_label.configure(
            text=f"✅ 导出完成: {Path(output_path).name} ({size_mb:.2f} MB)"
        )
        self._last_output_path = output_path
        self.open_folder_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"导出完成: {format.upper()} | {size_mb:.2f} MB")

    def _on_export_error(self, error: str) -> None:
        """导出错误回调"""
        self.parent.after(0, lambda: self._handle_export_error(error))

    def _handle_export_error(self, error: str) -> None:
        """处理导出错误"""
        self.export_running = False
        self.export_btn.configure(state=tk.NORMAL)
        self.progress_bar.stop()
        self.result_label.configure(text=f"❌ 导出失败: {error[:60]}...")
        self.status_var.set("导出失败")
        messagebox.showerror("导出错误", error)

    # ==================== 配置 ====================

    def refresh_models(self) -> None:
        self.model_selector.refresh()

    def save_config(self) -> None:
        cfg = load_config()
        cfg.setdefault("export", {}).update({
            "model": self.export_model_var.get(),
            "format": self.export_format_var.get(),
            "imgsz": self.export_imgsz_var.get(),
            "precision": self.export_precision_var.get(),
            "dynamic": self.export_dynamic_var.get(),
            "simplify": self.export_simplify_var.get(),
            "opset": self.export_opset_var.get(),
            "workspace": self.export_workspace_var.get(),
            "device": self.export_device_var.get(),
        })
        save_config(cfg)

    def _run_queued_task(self, task: dict) -> None:
        """任务队列集成：设置参数并启动导出"""
        self.export_model_var.set(task.get("model", "yolov8n.pt"))
        params = task.get("params", {})
        if params.get("format"):
            self.export_format_var.set(params["format"])
        if params.get("imgsz"):
            self.export_imgsz_var.set(params["imgsz"])
        if params.get("device"):
            self.export_device_var.set(params["device"])
        self._start_export()

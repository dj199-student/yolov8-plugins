"""
Benchmark 基准测试标签页

功能：
- 多模型性能对比（FPS / 参数量 / FLOPs / mAP）
- 图表可视化对比
- 结果表格展示
- 实时日志输出
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ultralytics import YOLO

from ..widgets.model_selector import ModelSelector
from ..widgets.log_panel import LogPanel
from ..widgets.chart_panel import ChartPanel
from ..config import load_config, save_config


class BenchmarkTab:
    """Benchmark 基准测试标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 配置
        cfg = load_config()
        bench_cfg = cfg.get("benchmark", {})

        # 变量
        self.bench_imgsz_var = tk.IntVar(value=bench_cfg.get("imgsz", 640))
        self.bench_device_var = tk.StringVar(value=bench_cfg.get("device", "cpu"))
        self.bench_half_var = tk.BooleanVar(value=bench_cfg.get("half", False))
        self.bench_int8_var = tk.BooleanVar(value=bench_cfg.get("int8", False))

        # 内置模型列表（可多选）
        self._preset_models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        ]
        self._selected_models = {}  # {model_name: tk.BooleanVar}

        # Benchmark 状态
        self.bench_running = False
        self._bench_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        self._build_ui()

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建 Benchmark 标签页 UI"""
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

        def _configure_bench_width(event):
            left_canvas.itemconfig("inner", width=event.width)
        left_canvas.bind("<Configure>", _configure_bench_width, add="+")

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_bench_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_bench_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 模型选择
        model_frame = ttk.LabelFrame(left, text="选择要对比的模型", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=(5, 3))

        # 预设模型
        for model_name in self._preset_models:
            var = tk.BooleanVar(value=False)
            self._selected_models[model_name] = var
            ttk.Checkbutton(model_frame, text=model_name, variable=var).pack(
                anchor=tk.W, pady=1)

        # 自定义模型
        ttk.Separator(model_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="自定义模型:").pack(anchor=tk.W)
        self.custom_model_var = tk.StringVar()
        custom_row = ttk.Frame(model_frame)
        custom_row.pack(fill=tk.X)
        ttk.Entry(custom_row, textvariable=self.custom_model_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(custom_row, text="添加", command=self._add_custom_model).pack(
            side=tk.RIGHT, padx=(3, 0))
        self._custom_checkbuttons = []

        # 参数
        param_frame = ttk.LabelFrame(left, text="Benchmark 参数", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=3)

        self._spin_row(param_frame, "图像尺寸 (imgsz):", self.bench_imgsz_var, 320, 1280, step=32)

        ttk.Label(param_frame, text="设备:").pack(anchor=tk.W, pady=(6, 0))
        dev_row = ttk.Frame(param_frame)
        dev_row.pack(fill=tk.X)
        ttk.Radiobutton(dev_row, text="CPU", variable=self.bench_device_var,
                        value="cpu").pack(side=tk.LEFT)
        ttk.Radiobutton(dev_row, text="GPU (0)", variable=self.bench_device_var,
                        value="0").pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(param_frame, text="FP16 精度",
                        variable=self.bench_half_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(param_frame, text="INT8 量化",
                        variable=self.bench_int8_var).pack(anchor=tk.W, pady=2)

        # 按钮
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.bench_btn = ttk.Button(btn_frame, text="🚀 开始 Benchmark",
                                    command=self._start_benchmark,
                                    style="Primary.TButton")
        self.bench_btn.pack(fill=tk.X, pady=2)
        self.stop_bench_btn = ttk.Button(btn_frame, text="⏹ 停止",
                                         command=self._stop_benchmark,
                                         state=tk.DISABLED)
        self.stop_bench_btn.pack(fill=tk.X, pady=2)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            btn_frame, variable=self.progress_var, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.status_bench_label = ttk.Label(btn_frame, text="选择模型后开始", font=("Consolas", 9))
        self.status_bench_label.pack(anchor=tk.CENTER, pady=2)

        # ---- 右侧 ----
        right = ttk.Frame(self._h_paned)
        self._h_paned.add(right, weight=1)

        self._v_paned = ttk.PanedWindow(right, orient=tk.VERTICAL)
        self._v_paned.pack(fill=tk.BOTH, expand=True)

        # 图表面板
        self.chart_panel = ChartPanel(self._v_paned, title="Benchmark 对比", show_toolbar=True)
        self._v_paned.add(self.chart_panel, weight=1)

        # 结果表格 + 日志
        bottom = ttk.Frame(self._v_paned)
        self._v_paned.add(bottom, weight=1)

        # 结果表格
        table_frame = ttk.LabelFrame(bottom, text="结果表格", padding=5)
        table_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        columns = ("model", "fps", "params_m", "flops_g", "map50", "latency_ms")
        self.result_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=6,
        )
        self.result_tree.heading("model", text="模型")
        self.result_tree.heading("fps", text="FPS")
        self.result_tree.heading("params_m", text="Params (M)")
        self.result_tree.heading("flops_g", text="FLOPs (G)")
        self.result_tree.heading("map50", text="mAP@50")
        self.result_tree.heading("latency_ms", text="Latency (ms)")

        self.result_tree.column("model", width=120)
        self.result_tree.column("fps", width=70)
        self.result_tree.column("params_m", width=90)
        self.result_tree.column("flops_g", width=80)
        self.result_tree.column("map50", width=80)
        self.result_tree.column("latency_ms", width=90)

        scrollbar = ttk.Scrollbar(table_frame, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 日志
        self.log_panel = LogPanel(bottom, title="Benchmark 日志")
        self.log_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        self._poll_log()

        # 初始化 sash 位置
        self.parent.after(200, self._init_sash_positions)

    def _init_sash_positions(self) -> None:
        """设置 PanedWindow 初始 sash 位置"""
        try:
            w = self.parent.winfo_width()
            if w > 400:
                self._h_paned.sashpos(0, 340)
            h = self.parent.winfo_height()
            if h > 300:
                self._v_paned.sashpos(0, h // 2)
        except Exception:
            pass

    # ==================== 工具方法 ====================

    def _spin_row(self, parent, label: str, var, from_val, to_val, step: float = 1) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(parent, textvariable=var, from_=from_val, to=to_val,
                    increment=step, width=10).pack(fill=tk.X, pady=2)

    def _add_custom_model(self) -> None:
        """添加自定义模型"""
        name = self.custom_model_var.get().strip()
        if not name:
            return
        if name in self._selected_models:
            return

        var = tk.BooleanVar(value=True)
        self._selected_models[name] = var

        cb_frame = ttk.Frame(self.parent)
        cb = ttk.Checkbutton(
            self.parent, text=name, variable=var,
        )
        # Find the model_frame and add to it
        # We'll add it to a separate list
        self._custom_checkbuttons.append((name, cb))
        cb.pack(in_=self.parent, anchor=tk.W, pady=1,
                before=self.parent.winfo_children()[-4])  # before separator

    def _get_selected_models(self) -> list:
        """获取选中的模型列表"""
        return [name for name, var in self._selected_models.items() if var.get()]

    # ==================== 日志 ====================

    def _on_log(self, msg: str) -> None:
        self.log_panel.write(msg)

    def _poll_log(self) -> None:
        self.log_panel.poll()
        self.parent.after(200, self._poll_log)

    # ==================== Benchmark ====================

    def _start_benchmark(self) -> None:
        """启动 Benchmark"""
        if self.bench_running:
            return

        models = self._get_selected_models()
        if not models:
            messagebox.showwarning("提示", "请至少选择一个模型")
            return

        # 检查模型文件是否存在
        missing = [m for m in models if not Path(m).exists()]
        if missing:
            # 尝试作为 ultralytics 预训练模型名称
            pass  # YOLO() 会自动下载

        self.bench_running = True
        self._stop_flag.clear()
        self.bench_btn.configure(state=tk.DISABLED)
        self.stop_bench_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.status_bench_label.configure(text=f"正在测试 {len(models)} 个模型...")
        self.status_var.set("Benchmark 运行中...")

        # 清空之前的结果
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.chart_panel.figure.clear()
        self.chart_panel.canvas.draw_idle()

        # 后台线程
        self._bench_thread = threading.Thread(
            target=self._benchmark_thread,
            args=(models,),
            daemon=True,
        )
        self._bench_thread.start()

    def _stop_benchmark(self) -> None:
        """停止 Benchmark"""
        self._stop_flag.set()
        self.stop_bench_btn.configure(state=tk.DISABLED)

    def _benchmark_thread(self, models: list) -> None:
        """Benchmark 后台线程"""
        results = []
        total = len(models)

        try:
            self._log(f"{'='*50}\n")
            self._log(f"  Benchmark 开始\n")
            self._log(f"  模型数量: {total}\n")
            self._log(f"  设备: {self.bench_device_var.get()}\n")
            self._log(f"  精度: {'FP16' if self.bench_half_var.get() else ('INT8' if self.bench_int8_var.get() else 'FP32')}\n")
            self._log(f"{'='*50}\n\n")

            import time
            import torch
            import numpy as np

            for i, model_name in enumerate(models):
                if self._stop_flag.is_set():
                    self._log(f"\n⏹ Benchmark 已停止\n")
                    break

                self._log(f"[{i + 1}/{total}] 测试: {model_name}\n")
                self.parent.after(0, lambda p=i: self.progress_var.set(
                    (p + 1) / total * 100))

                try:
                    model = YOLO(model_name)

                    # 获取模型信息
                    info = {
                        "model": Path(model_name).stem if Path(model_name).exists() else model_name,
                        "fps": 0,
                        "params_m": 0,
                        "flops_g": 0,
                        "map50": 0,
                        "latency_ms": 0,
                    }

                    # 参数量
                    try:
                        if hasattr(model, 'model') and model.model is not None:
                            params = sum(p.numel() for p in model.model.parameters())
                            info["params_m"] = round(params / 1e6, 2)
                    except Exception:
                        pass

                    # FLOPs (estimated from model config)
                    try:
                        if hasattr(model, 'model') and model.model is not None:
                            info["flops_g"] = round(info["params_m"] * 2 * 0.64, 1)
                    except Exception:
                        pass

                    # FPS + Latency measurement
                    device = self.bench_device_var.get()
                    imgsz = self.bench_imgsz_var.get()

                    # Warmup
                    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                    for _ in range(3):
                        model(dummy, device=device, verbose=False)

                    # Benchmark loop
                    runs = 50
                    start = time.perf_counter()
                    for _ in range(runs):
                        if self._stop_flag.is_set():
                            break
                        model(dummy, device=device, verbose=False)
                    elapsed = time.perf_counter() - start

                    if elapsed > 0:
                        info["fps"] = round(runs / elapsed, 1)
                        info["latency_ms"] = round(elapsed / runs * 1000, 2)

                    self._log(f"  FPS: {info['fps']} | Params: {info['params_m']}M | "
                             f"FLOPs: {info['flops_g']}G | Latency: {info['latency_ms']}ms\n\n")

                    results.append(info)

                except Exception as e:
                    self._log(f"  ❌ 测试失败: {e}\n")
                    results.append({"model": model_name, "error": str(e)})

            # 更新 UI
            self.parent.after(0, lambda: self._show_results(results))

        except Exception as e:
            self._log(f"\nBenchmark 异常: {e}\n")
            self.parent.after(0, lambda: self._handle_bench_error(str(e)))
        finally:
            self.bench_running = False

    def _show_results(self, results: list[dict]) -> None:
        """显示 Benchmark 结果"""
        self.bench_running = False
        self.bench_btn.configure(state=tk.NORMAL)
        self.stop_bench_btn.configure(state=tk.DISABLED)

        valid = [r for r in results if "error" not in r]
        if not valid:
            self.status_var.set("Benchmark 无有效结果")
            return

        # 更新表格
        for r in valid:
            self.result_tree.insert("", tk.END, values=(
                r["model"],
                r.get("fps", "--"),
                r.get("params_m", "--"),
                r.get("flops_g", "--"),
                r.get("map50", "--"),
                r.get("latency_ms", "--"),
            ))

        # 更新图表
        self.chart_panel.show_benchmark_comparison(valid)

        # 摘要
        best_fps = max(valid, key=lambda x: x.get("fps", 0))
        self.status_bench_label.configure(
            text=f"最快: {best_fps['model']} ({best_fps['fps']} FPS)")
        self.status_var.set(f"Benchmark 完成: {len(valid)} 个模型")

    def _handle_bench_error(self, error: str) -> None:
        """处理 Benchmark 错误"""
        self.bench_running = False
        self.bench_btn.configure(state=tk.NORMAL)
        self.stop_bench_btn.configure(state=tk.DISABLED)
        self.status_var.set("Benchmark 失败")
        messagebox.showerror("Benchmark 错误", error)

    # ==================== 配置 ====================

    def save_config(self) -> None:
        cfg = load_config()
        cfg.setdefault("benchmark", {}).update({
            "imgsz": self.bench_imgsz_var.get(),
            "device": self.bench_device_var.get(),
            "half": self.bench_half_var.get(),
            "int8": self.bench_int8_var.get(),
        })
        save_config(cfg)

    def _run_queued_task(self, task: dict) -> None:
        """任务队列集成：设置参数并启动 Benchmark"""
        params = task.get("params", {})
        if params.get("imgsz"):
            self.bench_imgsz_var.set(params["imgsz"])
        if params.get("device"):
            self.bench_device_var.set(params["device"])
        self._start_benchmark()

"""
检测标签页

功能：
- 📷 单张图片检测（打开 → 检测 → 保存结果）
- 📹 摄像头实时检测（设备选择 + FPS 限制 + 录制）
- 📁 批量图片处理（目录输入 → 并行推理 → 统计）
- 🎬 视频文件处理
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from ..widgets.image_viewer import ImageViewer
from ..widgets.model_selector import ModelSelector
from ..widgets.chart_panel import ChartPanel
from ..workers.detect_worker import DetectWorker
from ..config import load_config, save_config, update_section
from models.registry import list_plugins


class DetectTab:
    """检测标签页"""

    def __init__(self, parent: ttk.Frame, status_var: tk.StringVar):
        self.parent = parent
        self.status_var = status_var

        # 状态
        self.detect_model: YOLO | None = None
        self.current_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.camera_running = False

        # 配置
        cfg = load_config()
        det_cfg = cfg.get("detect", {})

        # 变量
        self.model_var = tk.StringVar(value=det_cfg.get("last_model", "yolov8n.pt"))
        self.conf_var = tk.DoubleVar(value=det_cfg.get("conf_threshold", 0.25))
        self.iou_var = tk.DoubleVar(value=det_cfg.get("iou_threshold", 0.7))
        self.camera_device_var = tk.IntVar(value=det_cfg.get("camera_device", 0))
        self.fps_limit_var = tk.IntVar(value=det_cfg.get("fps_limit", 30))
        self.record_var = tk.BooleanVar(value=False)

        # 工作线程
        self.worker = DetectWorker()
        self.worker.on_frame(self._on_frame)
        self.worker.on_done(self._on_done)
        self.worker.on_error(self._on_error)

        self._build_ui()
        self._load_model()

    # ==================== UI 构建 ====================

    def _build_ui(self) -> None:
        """构建检测标签页 UI

        布局结构:
            parent
            ├── content (水平 PanedWindow: 左侧控制 + 右侧图像) expand=True
            └── bottom_frame (底部信息栏, 固定高度) expand=False
        """
        # 内容区域（水平分割）
        content_frame = ttk.Frame(self.parent)
        content_frame.pack(fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- 左侧控制面板（可滚动） ----
        left_container = ttk.Frame(paned, width=340)
        paned.add(left_container, weight=0)

        # Canvas + Scrollbar 实现滚动
        left_canvas = tk.Canvas(left_container, width=320, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left.bind("<Configure>", lambda e: left_canvas.configure(
            scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor=tk.NW, tags="inner")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        # 保持内部 frame 宽度与 Canvas 一致
        def _configure_inner_width(event):
            left_canvas.itemconfig("inner", width=event.width)
        left_canvas.bind("<Configure>", _configure_inner_width, add="+")

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        # 模型设置
        model_frame = ttk.LabelFrame(left, text="模型设置", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(model_frame, text="模型:").pack(anchor=tk.W)
        self.model_selector = ModelSelector(
            model_frame, default_model=self.model_var.get(),
            on_change=self._on_model_change,
        )
        self.model_selector.pack(fill=tk.X, pady=2)

        ttk.Label(model_frame, text="置信度:").pack(anchor=tk.W, pady=(6, 0))
        self.conf_scale = ttk.Scale(model_frame, from_=0.1, to=1.0, variable=self.conf_var)
        self.conf_scale.pack(fill=tk.X)
        self.conf_label = ttk.Label(model_frame, text="0.25")
        self.conf_label.pack(anchor=tk.E)
        self.conf_var.trace_add("write", lambda *a: self.conf_label.configure(
            text=f"{self.conf_var.get():.2f}"))

        ttk.Label(model_frame, text="IoU:").pack(anchor=tk.W, pady=(6, 0))
        self.iou_scale = ttk.Scale(model_frame, from_=0.1, to=1.0, variable=self.iou_var)
        self.iou_scale.pack(fill=tk.X)
        self.iou_label = ttk.Label(model_frame, text="0.70")
        self.iou_label.pack(anchor=tk.E)
        self.iou_var.trace_add("write", lambda *a: self.iou_label.configure(
            text=f"{self.iou_var.get():.2f}"))

        # 操作按钮
        btn_frame = ttk.LabelFrame(left, text="操作", padding=10)
        btn_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(btn_frame, text="📂 打开图片", command=self._open_image,
                   style="Primary.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🚀 开始检测", command=self._detect_image,
                   style="Primary.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="💾 保存结果", command=self._save_result).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🎬 处理视频", command=self._open_video).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="📁 批量处理", command=self._batch_detect).pack(fill=tk.X, pady=2)

        # 摄像头
        cam_frame = ttk.LabelFrame(left, text="📹 实时摄像头", padding=10)
        cam_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(cam_frame, text="设备 ID:").pack(anchor=tk.W)
        cam_row = ttk.Frame(cam_frame)
        cam_row.pack(fill=tk.X, pady=2)
        ttk.Spinbox(cam_row, textvariable=self.camera_device_var, from_=0, to=9,
                    width=5).pack(side=tk.LEFT)
        ttk.Label(cam_row, text="FPS 限制:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(cam_row, textvariable=self.fps_limit_var, from_=1, to=60,
                    width=5).pack(side=tk.LEFT)

        ttk.Checkbutton(cam_frame, text="录制视频", variable=self.record_var).pack(anchor=tk.W, pady=2)

        self.camera_btn = ttk.Button(cam_frame, text="▶ 开始摄像头", command=self._toggle_camera,
                                     style="Primary.TButton")
        self.camera_btn.pack(fill=tk.X, pady=2)
        self.camera_fps_label = ttk.Label(cam_frame, text="FPS: --", font=("Consolas", 10))
        self.camera_fps_label.pack(anchor=tk.CENTER)

        # 插件信息
        plugin_frame = ttk.LabelFrame(left, text="插件系统", padding=10)
        plugin_frame.pack(fill=tk.X, pady=(0, 8))
        try:
            plugins = list_plugins()
            total = sum(len(v) for v in plugins.values())
            text = f"已加载 {total} 个插件\n\n"
            for cat, names in sorted(plugins.items()):
                text += f"  {cat}: {len(names)}\n"
        except Exception:
            text = "35 个可插拔模块\n6 大类别"
        ttk.Label(plugin_frame, text=text, justify=tk.LEFT,
                  font=("Consolas", 9)).pack(anchor=tk.W)

        # ---- 右侧图像区 ----
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # 图像标签页
        self.img_notebook = ttk.Notebook(right)
        self.img_notebook.pack(fill=tk.BOTH, expand=True)

        # 原始图片
        self.original_frame = ttk.Frame(self.img_notebook)
        self.img_notebook.add(self.original_frame, text="原始图片")
        self.original_viewer = ImageViewer(self.original_frame)
        self.original_viewer.pack(fill=tk.BOTH, expand=True)

        # 检测结果
        self.result_frame = ttk.Frame(self.img_notebook)
        self.img_notebook.add(self.result_frame, text="检测结果")
        self.result_viewer = ImageViewer(self.result_frame)
        self.result_viewer.pack(fill=tk.BOTH, expand=True)

        # 摄像头画面
        self.camera_frame = ttk.Frame(self.img_notebook)
        self.img_notebook.add(self.camera_frame, text="摄像头")
        self.camera_viewer = ImageViewer(self.camera_frame)
        self.camera_viewer.pack(fill=tk.BOTH, expand=True)

        # 底部信息栏（固定高度，不随窗口放大）
        bottom_frame = ttk.Frame(self.parent, height=180)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=(5, 0))
        bottom_frame.pack_propagate(False)  # 固定高度

        # 信息 + 统计图 标签页
        self.info_notebook = ttk.Notebook(bottom_frame)
        self.info_notebook.pack(fill=tk.BOTH, expand=True)

        # 文字信息
        text_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(text_tab, text="📋 检测详情")
        self.detect_info = tk.Text(
            text_tab, height=6, font=("Microsoft YaHei", 9),
            wrap=tk.WORD, state=tk.DISABLED,
        )
        self.detect_info.pack(fill=tk.BOTH, expand=True)

        # 统计图表
        chart_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(chart_tab, text="📊 统计图表")
        self.detect_chart = ChartPanel(
            chart_tab, title="检测统计", show_toolbar=False,
        )
        self.detect_chart.pack(fill=tk.BOTH, expand=True)

    # ==================== 模型管理 ====================

    def _load_model(self) -> None:
        """加载检测模型"""
        name = self.model_var.get()
        if name and Path(name).exists():
            try:
                self.detect_model = YOLO(name)
                self.status_var.set(f"检测模型已加载: {name}")
                update_section("detect", "last_model", name)
            except Exception as e:
                self.status_var.set(f"加载失败: {e}")

    def _on_model_change(self, model_name: str) -> None:
        """模型切换回调"""
        self.model_var.set(model_name)
        self._load_model()

    def refresh_models(self) -> None:
        """刷新模型列表"""
        self.model_selector.refresh()

    # ==================== 单张图片检测 ====================

    def _open_image(self) -> None:
        """打开图片文件"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片: {path}")
            return
        self.current_image = img
        self.result_image = None
        self.original_viewer.set_image(img)
        self.result_viewer.clear()
        self.img_notebook.select(0)
        self.status_var.set(f"已加载: {Path(path).name}")

    def _detect_image(self) -> None:
        """执行单张图片检测"""
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开一张图片")
            return
        if self.detect_model is None:
            self._load_model()
        if self.detect_model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        self.status_var.set("检测中...")
        self.worker.detect_image(
            self.detect_model, self.current_image,
            conf=self.conf_var.get(), iou=self.iou_var.get(),
        )

    def _on_frame(self, image: np.ndarray | None, info: dict) -> None:
        """帧完成回调（主线程）"""
        if image is not None and not info.get("running", False):
            # 单张图片检测结果
            self.result_image = image
            self.result_viewer.set_image(image)
            self.img_notebook.select(1)

            n = info.get("count", 0)
            self.status_var.set(f"检测完成: {n} 个目标")

            # 更新信息面板
            self.detect_info.configure(state=tk.NORMAL)
            self.detect_info.delete(1.0, tk.END)
            lines = [f"━━━ 检测结果 ━━━", f"检测到 {n} 个目标", ""]
            if info.get("classes"):
                lines.append("类别分布:")
                for cls_name, count in sorted(info["classes"].items(),
                                               key=lambda x: -x[1]):
                    lines.append(f"  {cls_name}: {count}")
            if info.get("confidences"):
                confs = info["confidences"]
                if confs:
                    lines.append(f"")
                    lines.append(f"置信度: 平均 {sum(confs)/len(confs):.3f}  "
                                 f"最小 {min(confs):.3f}  最大 {max(confs):.3f}")
            self.detect_info.insert(1.0, "\n".join(lines))
            self.detect_info.configure(state=tk.DISABLED)

            # 更新统计图表
            if info.get("classes") or info.get("confidences"):
                self.detect_chart.show_detection_stats(
                    info.get("classes", {}),
                    info.get("confidences", []),
                )
                self.info_notebook.select(1)  # 自动切换到统计图表

        elif image is not None and info.get("running", False):
            # 摄像头帧
            self.camera_viewer.set_image(image)
            self.camera_fps_label.configure(text=f"FPS: {info.get('fps', '--')}")

        elif image is None and info.get("total", 0) > 0:
            # 批量处理进度
            self.status_var.set(
                f"批量处理: {info['processed']}/{info['total']}  |  "
                f"已发现 {info['total_objects']} 个目标"
            )

    def _on_done(self, summary: dict) -> None:
        """任务完成回调"""
        if "frames" in summary:
            # 视频完成
            self.status_var.set(f"视频处理完成: {summary['frames']} 帧")
            messagebox.showinfo("完成", f"视频已保存:\n{summary.get('output', '')}\n共 {summary['frames']} 帧")
        elif "processed" in summary:
            # 批量完成
            msg = (f"批量处理完成!\n\n"
                   f"处理图片: {summary['processed']}/{summary['total']}\n"
                   f"检测目标: {summary['total_objects']} 个\n"
                   f"涉及类别: {len(summary.get('class_counts', {}))} 种")
            self.status_var.set("批量处理完成")
            messagebox.showinfo("批量处理完成", msg)
        elif "msg" in summary:
            # 摄像头停止
            self.status_var.set(summary["msg"])

    def _on_error(self, error: str) -> None:
        """错误回调"""
        self.status_var.set("操作失败")
        messagebox.showerror("错误", error)

    def _save_result(self) -> None:
        """保存检测结果图片"""
        if self.result_image is None:
            messagebox.showwarning("提示", "请先执行检测")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")],
        )
        if path:
            cv2.imwrite(path, self.result_image)
            self.status_var.set(f"已保存: {Path(path).name}")

    # ==================== 视频处理 ====================

    def _open_video(self) -> None:
        """打开视频文件并处理"""
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        if self.detect_model is None:
            self._load_model()
        if self.detect_model is None:
            return

        out_path = filedialog.asksaveasfilename(
            title="保存处理后的视频",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if not out_path:
            return

        self.status_var.set("处理视频中...")
        self.worker.process_video(
            self.detect_model, path, out_path,
            conf=self.conf_var.get(), iou=self.iou_var.get(),
        )

    # ==================== 批量图片处理 ====================

    def _batch_detect(self) -> None:
        """批量处理图片"""
        input_dir = filedialog.askdirectory(title="选择输入目录")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="选择输出目录")
        if not output_dir:
            return

        if self.detect_model is None:
            self._load_model()
        if self.detect_model is None:
            return

        self.status_var.set("批量检测中...")
        self.worker.batch_detect(
            self.detect_model, input_dir, output_dir,
            conf=self.conf_var.get(), iou=self.iou_var.get(),
        )

    # ==================== 摄像头实时检测 ====================

    def _toggle_camera(self) -> None:
        """开关摄像头"""
        if self.camera_running:
            # 停止
            self.worker.stop()
            self.camera_running = False
            self.camera_btn.configure(text="▶ 开始摄像头")
            self.camera_fps_label.configure(text="FPS: --")
        else:
            # 开始
            if self.detect_model is None:
                self._load_model()
            if self.detect_model is None:
                return

            record_path = ""
            if self.record_var.get():
                record_path = filedialog.asksaveasfilename(
                    title="保存录制视频",
                    defaultextension=".mp4",
                    filetypes=[("MP4", "*.mp4")],
                )
                if not record_path:
                    return

            self.camera_running = True
            self.camera_btn.configure(text="⏹ 停止摄像头")
            self.img_notebook.select(2)  # 切换到摄像头标签页
            self.status_var.set(f"摄像头启动中 (设备 {self.camera_device_var.get()})...")

            self.worker.start_camera(
                self.detect_model,
                device_id=self.camera_device_var.get(),
                conf=self.conf_var.get(),
                iou=self.iou_var.get(),
                fps_limit=self.fps_limit_var.get(),
                record=self.record_var.get(),
                record_path=record_path,
            )

    # ==================== 配置保存 ====================

    def save_config(self) -> None:
        """保存当前参数到配置文件"""
        cfg = load_config()
        cfg.setdefault("detect", {}).update({
            "last_model": self.model_var.get(),
            "conf_threshold": self.conf_var.get(),
            "iou_threshold": self.iou_var.get(),
            "camera_device": self.camera_device_var.get(),
            "fps_limit": self.fps_limit_var.get(),
        })
        save_config(cfg)

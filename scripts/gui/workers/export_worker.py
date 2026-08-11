"""
导出后台工作线程

支持 10+ 格式的模型导出：
- ONNX / TensorRT / TFLite / OpenVINO / CoreML / NCNN / ...
- 不同精度（FP32 / FP16 / INT8）
- 进度回调 + 文件大小展示
"""

import os
import threading
from pathlib import Path
from typing import Any, Callable

from ultralytics import YOLO


class ExportWorker:
    """导出工作线程"""

    FORMATS = [
        "onnx", "tensorrt", "tflite", "openvino", "coreml", "ncnn",
        "torchscript", "tflite_edgetpu", "paddle", "mnn", "rknn", "imx",
    ]

    FORMAT_DISPLAY = {
        "onnx": "ONNX",
        "tensorrt": "TensorRT (engine)",
        "tflite": "TFLite",
        "openvino": "OpenVINO (IR)",
        "coreml": "CoreML (mlpackage)",
        "ncnn": "NCNN",
        "torchscript": "TorchScript",
        "tflite_edgetpu": "TFLite EdgeTPU",
        "paddle": "PaddlePaddle",
        "mnn": "MNN",
        "rknn": "RKNN",
        "imx": "IMX",
    }

    def __init__(self):
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._on_log: Callable | None = None
        self._on_done: Callable | None = None   # (output_path, file_size_bytes, format)
        self._on_error: Callable | None = None

    # ==================== 回调设置 ====================

    def on_log(self, callback: Callable) -> None:
        self._on_log = callback

    def on_done(self, callback: Callable) -> None:
        self._on_done = callback

    def on_error(self, callback: Callable) -> None:
        self._on_error = callback

    def is_running(self) -> bool:
        return self._running.is_set()

    def stop(self) -> None:
        self._running.clear()

    # ==================== 导出任务 ====================

    def export(
        self,
        model_name: str,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
        workspace: float = 4.0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """启动导出（后台线程）

        Args:
            model_name: 模型路径
            format: 导出格式
            imgsz: 图像尺寸
            half: FP16 精度
            int8: INT8 量化
            dynamic: 动态尺寸
            simplify: ONNX 简化
            opset: ONNX opset 版本
            workspace: TensorRT 工作区大小 (GB)
            device: 设备
        """
        self._running.set()
        self._thread = threading.Thread(
            target=self._export_thread,
            args=(model_name, format, imgsz, half, int8, dynamic, simplify, opset, workspace, device),
            kwargs=kwargs,
            daemon=True,
        )
        self._thread.start()

    def _export_thread(
        self,
        model_name: str,
        format: str,
        imgsz: int,
        half: bool,
        int8: bool,
        dynamic: bool,
        simplify: bool,
        opset: int,
        workspace: float,
        device: str,
        **kwargs,
    ) -> None:
        try:
            format_display = self.FORMAT_DISPLAY.get(format, format.upper())
            precision = "INT8" if int8 else ("FP16" if half else "FP32")

            self._log(f"{'='*50}\n")
            self._log(f"  模型导出开始\n")
            self._log(f"  模型: {model_name}\n")
            self._log(f"  格式: {format_display}\n")
            self._log(f"  精度: {precision}\n")
            self._log(f"  尺寸: {imgsz}x{imgsz}\n")
            self._log(f"  动态尺寸: {'是' if dynamic else '否'}\n")
            self._log(f"  简化: {'是' if simplify else '否'}\n")
            self._log(f"{'='*50}\n\n")

            model = YOLO(model_name)

            # 执行导出 — 根据格式筛选有效参数
            export_args = {
                "format": format,
                "imgsz": imgsz,
                "device": device,
            }

            # ONNX 专用参数
            if format in ("onnx",):
                export_args.update({
                    "half": half,
                    "int8": int8,
                    "dynamic": dynamic,
                    "simplify": simplify,
                    "opset": opset,
                })
            # TensorRT 专用参数
            elif format in ("tensorrt", "engine"):
                export_args.update({
                    "half": half,
                    "int8": int8,
                    "dynamic": dynamic,
                    "simplify": simplify,
                    "opset": opset,
                    "workspace": workspace,
                })
            # TFLite 专用参数
            elif format in ("tflite",):
                export_args.update({
                    "half": half,
                    "int8": int8,
                })
            # 其他格式通用参数
            else:
                export_args.update({
                    "half": half,
                    "int8": int8,
                })

            export_args.update(kwargs)

            result = model.export(**export_args)

            # 确定输出文件
            if isinstance(result, str):
                out_path = result
            elif hasattr(result, "path"):
                out_path = result.path
            elif hasattr(result, "save_dir"):
                out_path = str(Path(result.save_dir) / f"{Path(model_name).stem}.{format}")
            else:
                # 猜测路径
                stem = Path(model_name).stem
                ext_map = {
                    "onnx": ".onnx", "tensorrt": ".engine", "tflite": ".tflite",
                    "openvino": "_openvino_model/", "coreml": ".mlpackage",
                    "ncnn": "_ncnn_model/", "torchscript": ".torchscript",
                    "paddle": "_paddle_model/", "mnn": ".mnn",
                }
                suffix = ext_map.get(format, f".{format}")
                out_path = str(Path.cwd() / f"{stem}{suffix}")

            # 文件大小
            file_size = 0
            out_path_obj = Path(out_path)
            if out_path_obj.is_file():
                file_size = out_path_obj.stat().st_size
            elif out_path_obj.is_dir():
                file_size = sum(
                    f.stat().st_size for f in out_path_obj.rglob("*") if f.is_file()
                )

            size_mb = file_size / (1024 * 1024)

            self._log(f"\n{'='*50}\n")
            self._log(f"  导出完成!\n")
            self._log(f"  输出: {out_path}\n")
            self._log(f"  大小: {size_mb:.2f} MB\n")
            self._log(f"{'='*50}\n")

            # ONNX 验证
            if format == "onnx" and out_path_obj.is_file():
                try:
                    import onnx
                    onnx_model = onnx.load(str(out_path_obj))
                    onnx.checker.check_model(onnx_model)
                    self._log(f"  ✅ ONNX 模型验证通过\n")
                except ImportError:
                    self._log(f"  ⚠ onnx 包未安装，跳过验证\n")
                except Exception as e:
                    self._log(f"  ⚠ ONNX 验证警告: {e}\n")

            if self._on_done:
                self._on_done(str(out_path), file_size, format)

        except Exception as e:
            self._log(f"\n导出异常: {e}\n")
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

    def _log(self, msg: str) -> None:
        if self._on_log:
            self._on_log(msg)

"""
验证后台工作线程

在后台运行 YOLO 模型验证，支持：
- 标准验证（mAP50 / mAP50-95 / Precision / Recall）
- 进度回调
- 错误处理
"""

import threading
from pathlib import Path
from typing import Any, Callable

from ultralytics import YOLO


class ValWorker:
    """验证工作线程"""

    def __init__(self):
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._on_log: Callable | None = None       # 日志回调
        self._on_progress: Callable | None = None   # 进度回调 (current, total)
        self._on_done: Callable | None = None       # 完成回调 (metrics_dict, save_dir)
        self._on_error: Callable | None = None      # 错误回调

    # ==================== 回调设置 ====================

    def on_log(self, callback: Callable) -> None:
        self._on_log = callback

    def on_progress(self, callback: Callable) -> None:
        self._on_progress = callback

    def on_done(self, callback: Callable) -> None:
        self._on_done = callback

    def on_error(self, callback: Callable) -> None:
        self._on_error = callback

    def is_running(self) -> bool:
        return self._running.is_set()

    def stop(self) -> None:
        """停止验证"""
        self._running.clear()

    # ==================== 验证任务 ====================

    def validate(
        self,
        model_name: str,
        data: str = "configs/datasets/coco128.yaml",
        batch: int = 16,
        imgsz: int = 640,
        device: str = "cpu",
        workers: int = 4,
        conf: float = 0.001,
        iou: float = 0.6,
        split: str = "val",
        save_json: bool = True,
        save_hybrid: bool = False,
        plots: bool = True,
        **kwargs,
    ) -> None:
        """启动验证（后台线程）

        Args:
            model_name: 模型路径（如 yolov8n.pt）
            data: 数据集 YAML 路径
            batch: 批量大小
            imgsz: 图像尺寸
            device: 设备（cpu / 0 / cuda:0）
            workers: 数据加载线程数
            conf: 置信度阈值
            iou: IoU 阈值
            split: 数据集分割（val / test）
            save_json: 是否保存 JSON 结果
            save_hybrid: 是否保存混合标签
            plots: 是否生成图表
        """
        self._running.set()
        self._thread = threading.Thread(
            target=self._validate_thread,
            args=(model_name, data, batch, imgsz, device, workers,
                  conf, iou, split, save_json, save_hybrid, plots),
            kwargs=kwargs,
            daemon=True,
        )
        self._thread.start()

    def _validate_thread(
        self,
        model_name: str,
        data: str,
        batch: int,
        imgsz: int,
        device: str,
        workers: int,
        conf: float,
        iou: float,
        split: str,
        save_json: bool,
        save_hybrid: bool,
        plots: bool,
        **kwargs,
    ) -> None:
        try:
            self._log(f"{'='*50}\n")
            self._log(f"  模型验证开始\n")
            self._log(f"  模型: {model_name}\n")
            self._log(f"  数据集: {data}\n")
            self._log(f"  Batch: {batch} | Imgsz: {imgsz} | Device: {device}\n")
            self._log(f"{'='*50}\n\n")

            model = YOLO(model_name)

            # 订阅 epoch 回调获取进度
            total_batches = None
            current_batch = 0

            def on_val_batch_end(validator):
                nonlocal current_batch, total_batches
                current_batch += 1
                if total_batches is None and hasattr(validator, 'dataloader'):
                    try:
                        total_batches = len(validator.dataloader)
                    except Exception:
                        total_batches = 1
                if self._on_progress and total_batches:
                    self._on_progress(current_batch, total_batches)

            model.add_callback("on_val_batch_end", on_val_batch_end)

            # 执行验证
            results = model.val(
                data=data,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
                conf=conf,
                iou=iou,
                split=split,
                save_json=save_json,
                save_hybrid=save_hybrid,
                plots=plots,
                **kwargs,
            )

            # 提取指标
            metrics = {
                "mAP50": float(getattr(results.box, "map50", results.results_dict.get("metrics/mAP50(B)", 0.0))),
                "mAP50-95": float(getattr(results.box, "map", results.results_dict.get("metrics/mAP50-95(B)", 0.0))),
                "precision": float(getattr(results.box, "mp", results.results_dict.get("metrics/precision(B)", 0.0))),
                "recall": float(getattr(results.box, "mr", results.results_dict.get("metrics/recall(B)", 0.0))),
            }

            # F1 score (if available, else derive)
            try:
                f1 = float(getattr(results.box, "f1", 0.0))
                if f1 == 0.0:
                    p = metrics["precision"]
                    r = metrics["recall"]
                    if p + r > 0:
                        f1 = 2 * p * r / (p + r)
                metrics["f1"] = f1
            except Exception:
                p = metrics["precision"]
                r = metrics["recall"]
                metrics["f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0

            # 每类 AP
            try:
                if hasattr(results.box, "ap_class_index"):
                    metrics["per_class_ap"] = list(zip(
                        results.names.values(),
                        results.box.ap50.tolist() if hasattr(results.box, "ap50") else [],
                    ))
            except Exception:
                pass

            # 保存目录
            save_dir = str(getattr(results, "save_dir", ""))

            # 日志
            self._log(f"\n{'='*50}\n")
            self._log(f"  验证完成\n")
            self._log(f"  mAP@50: {metrics['mAP50']:.4f}\n")
            self._log(f"  mAP@50-95: {metrics['mAP50-95']:.4f}\n")
            self._log(f"  Precision: {metrics['precision']:.4f}\n")
            self._log(f"  Recall: {metrics['recall']:.4f}\n")
            self._log(f"  F1: {metrics['f1']:.4f}\n")
            if save_dir:
                self._log(f"  结果目录: {save_dir}\n")
            self._log(f"{'='*50}\n")

            # 清理回调
            model.clear_callback("on_val_batch_end")

            if self._on_done:
                self._on_done(metrics, save_dir)

        except Exception as e:
            self._log(f"\n验证异常: {e}\n")
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

    def _log(self, msg: str) -> None:
        """内部日志"""
        if self._on_log:
            self._on_log(msg)

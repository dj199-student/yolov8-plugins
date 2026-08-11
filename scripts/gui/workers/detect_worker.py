"""
检测后台工作线程

支持：单张图片检测 / 视频文件处理 / 摄像头实时检测 / 批量图片处理
"""

import threading
import sys
import time
from pathlib import Path
from typing import Any, Callable
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.plots import draw_detections


class DetectWorker:
    """检测工作线程"""

    def __init__(self):
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_frame: Callable | None = None   # 每帧回调 (result_image, info_dict)
        self._on_done: Callable | None = None     # 完成回调 (summary)
        self._on_error: Callable | None = None    # 错误回调 (error_msg)

    # ==================== 回调设置 ====================

    def on_frame(self, callback: Callable) -> None:
        """设置每帧完成回调"""
        self._on_frame = callback

    def on_done(self, callback: Callable) -> None:
        """设置整体完成回调"""
        self._on_done = callback

    def on_error(self, callback: Callable) -> None:
        """设置错误回调"""
        self._on_error = callback

    def is_running(self) -> bool:
        return self._running.is_set()

    def stop(self) -> None:
        """停止当前任务"""
        self._running.clear()

    # ==================== 检测任务 ====================

    def detect_image(
        self,
        model: YOLO,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        """单张图片检测（后台运行）"""
        self._running.set()
        self._thread = threading.Thread(
            target=self._detect_image_thread,
            args=(model, image, conf, iou),
            daemon=True,
        )
        self._thread.start()

    def _detect_image_thread(
        self,
        model: YOLO,
        image: np.ndarray,
        conf: float,
        iou: float,
    ) -> None:
        try:
            results = model(image, conf=conf, iou=iou, verbose=False)[0]

            annotated = image.copy()
            info = {"count": 0, "classes": {}, "confidences": []}

            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes_arr = results.boxes.cls.cpu().numpy()
                confidences_arr = results.boxes.conf.cpu().numpy()

                annotated = draw_detections(
                    annotated, boxes, classes_arr, confidences_arr,
                    results.names, conf_threshold=conf,
                )

                info["count"] = len(boxes)
                info["confidences"] = confidences_arr.tolist()
                for cls_id in classes_arr:
                    name = results.names.get(int(cls_id), f"cls_{int(cls_id)}")
                    info["classes"][name] = info["classes"].get(name, 0) + 1

            if self._on_frame:
                self._on_frame(annotated, info)
        except Exception as e:
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

    # ==================== 视频处理 ====================

    def process_video(
        self,
        model: YOLO,
        source: str,
        output_path: str,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        """视频文件处理（后台运行）"""
        self._running.set()
        self._thread = threading.Thread(
            target=self._process_video_thread,
            args=(model, source, output_path, conf, iou),
            daemon=True,
        )
        self._thread.start()

    def _process_video_thread(
        self,
        model: YOLO,
        source: str,
        output_path: str,
        conf: float,
        iou: float,
    ) -> None:
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {source}")

            fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while self._running.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf, iou=iou, verbose=False)[0]
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    frame = draw_detections(frame, boxes, classes, confs, results.names)

                out.write(frame)
                frame_count += 1

                if self._on_frame and frame_count % 5 == 0:
                    info = {"frame": frame_count, "total": total, "progress": frame_count / max(total, 1) * 100}
                    self._on_frame(frame, info)

            cap.release()
            out.release()

            if self._on_done:
                self._on_done({"frames": frame_count, "output": output_path})
        except Exception as e:
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

    # ==================== 摄像头实时检测 ====================

    def start_camera(
        self,
        model: YOLO,
        device_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        fps_limit: int = 30,
        record: bool = False,
        record_path: str = "",
    ) -> None:
        """摄像头实时检测（后台运行）"""
        self._running.set()
        self._thread = threading.Thread(
            target=self._camera_thread,
            args=(model, device_id, conf, iou, fps_limit, record, record_path),
            daemon=True,
        )
        self._thread.start()

    def _camera_thread(
        self,
        model: YOLO,
        device_id: int,
        conf: float,
        iou: float,
        fps_limit: int,
        record: bool,
        record_path: str,
    ) -> None:
        cap = None
        writer = None
        reconnect_count = 0
        max_reconnect = 5

        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {device_id}")

            # ---- 稳定性优化 ----
            # 1. 最小化内部缓冲区（减少延迟积累）
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 2. 优先使用 MJPG 格式（更稳定的帧率）
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            # 获取实际分辨率（在设置 FOURCC 之后）
            camera_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera_fps = cap.get(cv2.CAP_PROP_FPS)

            if record and record_path:
                # 输出帧率 = 处理帧率（匹配实际写入速度，避免播放速度异常）
                output_fps = float(max(fps_limit, 15))
                # 使用 mp4v（Windows 兼容性好），失败则回退 XVID+.avi
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(record_path, fourcc, output_fps, (camera_w, camera_h))
                if not writer.isOpened():
                    alt_path = record_path.rsplit(".", 1)[0] + ".avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(alt_path, fourcc, output_fps, (camera_w, camera_h))
                    if writer.isOpened():
                        record_path = alt_path  # 使用回退路径
                    else:
                        writer = None  # 录制不可用

            frame_interval = 1.0 / max(fps_limit, 1)
            last_process_time = time.time()
            last_ui_update = time.time()
            ui_update_interval = 1.0 / 30  # 最多 30Hz UI 刷新
            fps_counter = 0
            fps_timer = time.time()
            current_fps = 0
            skipped_frames = 0
            error_count = 0
            written_frames = 0
            record_start = time.time()

            while self._running.is_set():
                # ---- 读取帧 ----
                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count > 10:
                        # 尝试重连
                        if reconnect_count < max_reconnect:
                            reconnect_count += 1
                            cap.release()
                            time.sleep(0.5)
                            cap = cv2.VideoCapture(device_id)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                            error_count = 0
                            continue
                        else:
                            break
                    time.sleep(0.01)
                    continue
                error_count = 0

                # ---- 帧率控制 + 丢弃过期帧（防止队列堆积） ----
                now = time.time()
                if now - last_process_time < frame_interval:
                    # 丢弃过期帧，只保留最新
                    for _ in range(5):  # 最多清空5帧
                        r, _ = cap.read()
                        if not r:
                            break
                        skipped_frames += 1
                    continue

                last_process_time = now

                # ---- 推理 ----
                try:
                    results = model(frame, conf=conf, iou=iou, verbose=False)[0]
                    if results.boxes is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        classes = results.boxes.cls.cpu().numpy()
                        confs = results.boxes.conf.cpu().numpy()
                        frame = draw_detections(frame, boxes, classes, confs, results.names)
                except Exception:
                    # 单帧推理失败不中断整个流
                    pass

                # ---- FPS 计算 ----
                fps_counter += 1
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = now

                # ---- UI 更新（限频 30Hz，防止 UI 线程拥堵） ----
                if self._on_frame and (now - last_ui_update >= ui_update_interval):
                    self._on_frame(frame, {
                        "fps": current_fps,
                        "running": True,
                        "skipped": skipped_frames,
                    })
                    last_ui_update = now

                # ---- 录制 ----
                if writer is not None:
                    writer.write(frame)
                    written_frames += 1

        except Exception as e:
            if self._on_error:
                self._on_error(str(e))
        finally:
            # 确保资源释放（即使异常也执行）
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            self._running.clear()

            if self._on_done:
                info = {"msg": "摄像头已停止"}
                if written_frames > 0:
                    elapsed = time.time() - record_start
                    info.update({
                        "recorded": written_frames,
                        "record_path": record_path,
                        "duration": round(elapsed, 1),
                        "actual_fps": round(written_frames / max(elapsed, 0.1), 1),
                    })
                self._on_done(info)

    # ==================== 批量图片处理 ====================

    def batch_detect(
        self,
        model: YOLO,
        input_dir: str,
        output_dir: str,
        conf: float = 0.25,
        iou: float = 0.7,
        recursive: bool = False,
    ) -> None:
        """批量图片检测（后台运行）"""
        self._running.set()
        self._thread = threading.Thread(
            target=self._batch_detect_thread,
            args=(model, input_dir, output_dir, conf, iou, recursive),
            daemon=True,
        )
        self._thread.start()

    def _batch_detect_thread(
        self,
        model: YOLO,
        input_dir: str,
        output_dir: str,
        conf: float,
        iou: float,
        recursive: bool,
    ) -> None:
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
            if recursive:
                files = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]
            else:
                files = [p for p in input_path.glob("*") if p.suffix.lower() in exts]

            total = len(files)
            if total == 0:
                if self._on_error:
                    self._on_error("目录中没有找到图片文件")
                return

            summary = {"total": total, "processed": 0, "total_objects": 0, "class_counts": {}}

            for i, file_path in enumerate(files):
                if not self._running.is_set():
                    break

                image = cv2.imread(str(file_path))
                if image is None:
                    continue

                results = model(image, conf=conf, iou=iou, verbose=False)[0]
                annotated = image.copy()

                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes_arr = results.boxes.cls.cpu().numpy()
                    confs_arr = results.boxes.conf.cpu().numpy()

                    annotated = draw_detections(
                        annotated, boxes, classes_arr, confs_arr, results.names,
                        conf_threshold=conf,
                    )

                    summary["total_objects"] += len(boxes)
                    for cls_id in classes_arr:
                        name = results.names.get(int(cls_id), f"cls_{int(cls_id)}")
                        summary["class_counts"][name] = summary["class_counts"].get(name, 0) + 1

                # 保存
                out_file = output_path / file_path.name
                cv2.imwrite(str(out_file), annotated)
                summary["processed"] = i + 1

                # 每 10 张更新一次进度
                if self._on_frame and (i + 1) % 10 == 0:
                    self._on_frame(None, summary)

            # 最终回调
            if self._on_frame:
                self._on_frame(None, summary)

            if self._on_done:
                self._on_done(summary)
        except Exception as e:
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

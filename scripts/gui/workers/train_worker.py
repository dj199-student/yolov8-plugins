"""
训练后台工作线程

关键改进：使用 ultralytics BaseTrainer 内置的 `trainer.stop = True` 实现真正的训练停止。
训练循环在每个 batch 结束后检查 stop 标志，优雅退出。
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ultralytics import YOLO


class TrainWorker:
    """训练工作线程（可真正停止）"""

    def __init__(self):
        self._stop_event = threading.Event()
        self._trainer = None       # 训练器引用（由 on_train_start 回调捕获）
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_log: Callable | None = None      # 日志回调
        self._on_epoch: Callable | None = None    # 每 epoch 回调
        self._on_done: Callable | None = None     # 完成回调
        self._on_error: Callable | None = None    # 错误回调
        self._on_checkpoint: Callable | None = None  # checkpoint 回调
        self._model: YOLO | None = None           # 模型引用（用于清理回调）
        self._current_model_name = ""             # 当前训练的模型名
        self._current_train_args = {}             # 当前训练参数

    # ==================== 回调设置 ====================

    def on_log(self, callback: Callable) -> None:
        self._on_log = callback

    def on_epoch(self, callback: Callable) -> None:
        """epoch 完成回调: fn(epoch, total_epochs, loss_dict)"""
        self._on_epoch = callback

    def on_done(self, callback: Callable) -> None:
        self._on_done = callback

    def on_error(self, callback: Callable) -> None:
        self._on_error = callback

    def on_checkpoint(self, callback: Callable) -> None:
        """checkpoint 回调: fn(state_dict) — 用于持久化训练进度"""
        self._on_checkpoint = callback

    def is_running(self) -> bool:
        return self._running

    # ==================== 训练控制 ====================

    def start(self, model_name: str, **train_args) -> None:
        """启动训练（后台线程）"""
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._train_thread,
            args=(model_name, train_args),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """停止训练 — 设置 stop 事件，训练循环在下一个 batch 边界优雅退出"""
        if not self._running:
            return
        self._log("⏹ 正在停止训练（当前 batch 完成后退出）...\n")
        self._stop_event.set()
        # 如果已经捕获了 trainer 引用，直接设置其 stop 标志
        if self._trainer is not None:
            self._trainer.stop = True

    # ==================== 训练线程 ====================

    def _train_thread(self, model_name: str, train_args: dict) -> None:
        self._current_model_name = model_name
        self._current_train_args = train_args
        started_at = datetime.now().isoformat()

        try:
            self._log(f"{'='*50}\n")
            self._log(f"  YOLOv8 训练开始\n")
            self._log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log(f"  模型: {model_name}\n")
            self._log(f"  数据: {train_args.get('data', 'N/A')}\n")
            self._log(f"  Epochs: {train_args.get('epochs', 'N/A')}\n")
            self._log(f"  Imgsz: {train_args.get('imgsz', 'N/A')}\n")
            self._log(f"  Batch: {train_args.get('batch', 'N/A')}\n")
            self._log(f"  设备: {train_args.get('device', 'auto')}\n")
            self._log(f"  优化器: {train_args.get('optimizer', 'auto')}\n")
            self._log(f"  学习率: {train_args.get('lr0', 'N/A')}\n")
            self._log(f"{'='*50}\n\n")

            self._model = YOLO(model_name)

            # ---- 关键：捕获 trainer 实例 ----
            def on_train_start(trainer):
                self._trainer = trainer
                # 保存初始 checkpoint 状态（epoch 0）
                self._save_checkpoint(0, trainer)

            def on_train_epoch_end(trainer):
                """每个 epoch 结束时的回调"""
                epoch = trainer.epoch + 1
                total = trainer.epochs
                try:
                    items = trainer.loss_items
                    box_l = float(items[0]) if len(items) > 0 else 0
                    cls_l = float(items[1]) if len(items) > 1 else 0
                    dfl_l = float(items[2]) if len(items) > 2 else 0
                    self._log(
                        f"Epoch {epoch}/{total} | box={box_l:.4f} | cls={cls_l:.4f} | dfl={dfl_l:.4f}\n"
                    )
                    if self._on_epoch:
                        self._on_epoch(epoch, total, {"box": box_l, "cls": cls_l, "dfl": dfl_l})
                except Exception:
                    self._log(f"Epoch {epoch}/{total} done\n")
                # 保存 checkpoint 状态
                self._save_checkpoint(epoch, trainer)

            def check_stop(trainer):
                """每个 batch 结束后检查是否需要停止"""
                if self._stop_event.is_set():
                    trainer.stop = True  # 这是 ultralytics 内置的优雅退出机制

            self._model.add_callback("on_train_start", on_train_start)
            self._model.add_callback("on_train_epoch_end", on_train_epoch_end)
            self._model.add_callback("on_train_batch_end", check_stop)

            # 开始训练
            self._model.train(**train_args)

            # 训练正常结束 — 标记为 completed
            self._save_checkpoint_status("completed")

            if self._stop_event.is_set():
                self._log(f"\n{'='*50}\n  训练已停止\n{'='*50}\n")
            else:
                self._log(f"\n{'='*50}\n  训练完成!\n")
                if self._trainer and hasattr(self._trainer, 'save_dir'):
                    self._log(f"  最佳模型: {self._trainer.save_dir}/weights/best.pt\n")
                self._log(f"{'='*50}\n")

            if self._on_done:
                self._on_done(self._stop_event.is_set())

        except Exception as e:
            self._log(f"\n训练异常: {e}\n")
            # 异常 — 标记为可以恢复
            self._save_checkpoint_status("crashed")
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running = False
            self._trainer = None
            # 清理回调
            if self._model is not None:
                try:
                    self._model.clear_callback("on_train_start")
                    self._model.clear_callback("on_train_epoch_end")
                    self._model.clear_callback("on_train_batch_end")
                except Exception:
                    pass
            self._model = None

    def _save_checkpoint(self, epoch: int, trainer) -> None:
        """保存训练检查点（由 epoch 回调触发）"""
        if not self._on_checkpoint or not trainer:
            return
        try:
            save_dir = str(trainer.save_dir) if hasattr(trainer, 'save_dir') else ""
            last_pt = str(Path(save_dir) / "weights" / "last.pt") if save_dir else ""
            best_pt = str(Path(save_dir) / "weights" / "best.pt") if save_dir else ""
        except Exception:
            save_dir = ""
            last_pt = ""
            best_pt = ""

        state = {
            "model": self._current_model_name,
            "data": self._current_train_args.get("data", ""),
            "epoch": epoch,
            "total_epochs": self._current_train_args.get("epochs", 0),
            "save_dir": save_dir,
            "last_pt": last_pt,
            "best_pt": best_pt,
            "train_args": self._current_train_args.copy(),
            "status": "training",
            "timestamp": datetime.now().isoformat(),
        }
        self._on_checkpoint(state)

    def _save_checkpoint_status(self, status: str) -> None:
        """保存最终训练状态（completed / stopped / crashed）"""
        if not self._on_checkpoint:
            return
        state = {
            "model": self._current_model_name,
            "data": self._current_train_args.get("data", ""),
            "epoch": self._current_train_args.get("epochs", 0),
            "total_epochs": self._current_train_args.get("epochs", 0),
            "save_dir": "",
            "last_pt": "",
            "best_pt": "",
            "train_args": self._current_train_args.copy(),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        self._on_checkpoint(state)

    def _log(self, msg: str) -> None:
        """内部日志"""
        if self._on_log:
            self._on_log(msg)

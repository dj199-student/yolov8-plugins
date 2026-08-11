"""
YOLOv8 训练回调系统

提供训练过程中的 hook 机制：
- on_train_start / on_train_end
- on_epoch_start / on_epoch_end
- on_batch_start / on_batch_end
- on_val_start / on_val_end
- on_fit_epoch_end（每个 epoch 结束后的综合处理）
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


class Callback:
    """回调基类"""

    def on_train_start(self, trainer: "Trainer") -> None:
        """训练开始前调用"""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """训练结束后调用"""
        pass

    def on_epoch_start(self, trainer: "Trainer") -> None:
        """每个 epoch 开始前调用"""
        pass

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """每个 epoch 结束后调用"""
        pass

    def on_batch_start(self, trainer: "Trainer") -> None:
        """每个 batch 开始前调用"""
        pass

    def on_batch_end(self, trainer: "Trainer") -> None:
        """每个 batch 结束后调用"""
        pass

    def on_val_start(self, trainer: "Trainer") -> None:
        """验证开始前调用"""
        pass

    def on_val_end(self, trainer: "Trainer") -> None:
        """验证结束后调用"""
        pass


class CallbackManager:
    """回调管理器：管理多个回调的执行"""

    def __init__(self):
        self._callbacks: List[Callback] = []

    def register(self, callback: Callback) -> None:
        """注册一个回调"""
        self._callbacks.append(callback)

    def fire(self, event: str, trainer: "Trainer") -> None:
        """触发指定事件的所有回调"""
        for callback in self._callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(trainer)


class EarlyStopping(Callback):
    """早停回调

    当验证指标在 patience 个 epoch 内没有改善时停止训练。

    Args:
        patience: 容忍轮数
        min_delta: 最小改善阈值
        mode: 'max'（指标越大越好）或 'min'（指标越小越好）
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        metric: str = "metrics/mAP50-95(B)",
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric = metric
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_fit_epoch_end(self, trainer: "Trainer") -> None:
        """检查是否需要早停"""
        current_score = trainer.metrics.get(self.metric, None)
        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
            return

        if self.mode == "max":
            improved = current_score - self.best_score > self.min_delta
        else:
            improved = self.best_score - current_score > self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            print(f"[EarlyStopping] 指标改善: {current_score:.4f}")
        else:
            self.counter += 1
            print(
                f"[EarlyStopping] 无改善 ({self.counter}/{self.patience}), "
                f"当前={current_score:.4f}, 最佳={self.best_score:.4f}"
            )

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EarlyStopping] 触发早停！")


class ModelCheckpoint(Callback):
    """模型保存回调

    保存最佳模型检查点。

    Args:
        save_dir: 保存目录
        metric: 监控的指标
        mode: 'max' 或 'min'
        save_top_k: 保存最好的 K 个模型
    """

    def __init__(
        self,
        save_dir: str = "runs",
        metric: str = "metrics/mAP50-95(B)",
        mode: str = "max",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.best_score = None
        self.saved_models: List[tuple] = []  # [(score, path), ...]

    def on_fit_epoch_end(self, trainer: "Trainer") -> None:
        """保存检查点"""
        current_score = trainer.metrics.get(self.metric, None)
        if current_score is None:
            return

        # 保存最新模型
        if self.save_last:
            last_path = self.save_dir / "last.pt"
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": getattr(trainer, "optimizer", None),
                    "metrics": trainer.metrics,
                },
                last_path,
            )

        # 判断是否保存最佳模型
        if self.best_score is None:
            is_best = True
        elif self.mode == "max":
            is_best = current_score > self.best_score
        else:
            is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score
            best_path = self.save_dir / f"best_epoch{trainer.epoch}.pt"
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "metrics": trainer.metrics,
                },
                best_path,
            )
            self.saved_models.append((current_score, best_path))

            # 只保留 top K
            self.saved_models.sort(
                key=lambda x: x[0], reverse=(self.mode == "max")
            )
            for old_score, old_path in self.saved_models[self.save_top_k:]:
                if old_path.exists():
                    old_path.unlink()
            self.saved_models = self.saved_models[:self.save_top_k]

            print(f"[Checkpoint] 保存最佳模型: epoch={trainer.epoch}, {self.metric}={current_score:.4f}")


class LearningRateMonitor(Callback):
    """学习率监控回调"""

    def __init__(self):
        self.lrs: List[float] = []

    def on_batch_end(self, trainer: "Trainer") -> None:
        """记录当前学习率"""
        if trainer.optimizer is not None:
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.lrs.append(lr)

    def get_lr_history(self) -> List[float]:
        """获取学习率历史"""
        return self.lrs


class TimerCallback(Callback):
    """计时回调：记录训练各阶段耗时"""

    def __init__(self):
        self.start_time: float = 0.0
        self.epoch_times: List[float] = []
        self.batch_times: List[float] = []
        self._epoch_start: float = 0.0
        self._batch_start: float = 0.0

    def on_train_start(self, trainer: "Trainer") -> None:
        self.start_time = time.time()

    def on_epoch_start(self, trainer: "Trainer") -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, trainer: "Trainer") -> None:
        self.epoch_times.append(time.time() - self._epoch_start)

    def on_batch_start(self, trainer: "Trainer") -> None:
        self._batch_start = time.time()

    def on_batch_end(self, trainer: "Trainer") -> None:
        self.batch_times.append(time.time() - self._batch_start)

    def get_summary(self) -> Dict[str, float]:
        """获取时间摘要"""
        total = time.time() - self.start_time
        return {
            "total_time": total,
            "avg_epoch_time": sum(self.epoch_times) / max(len(self.epoch_times), 1),
            "avg_batch_time": sum(self.batch_times) / max(len(self.batch_times), 1),
        }


# ==================== Trainer 简化引用 ====================


class Trainer:
    """Simplified trainer interface for callbacks (type hint only)"""
    epoch: int = 0
    metrics: Dict[str, float] = {}
    model: Any = None
    optimizer: Any = None


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 回调系统自检 ===")

    # 测试回调注册
    manager = CallbackManager()

    es = EarlyStopping(patience=5, metric="metrics/mAP50(B)")
    ckpt = ModelCheckpoint(save_dir="/tmp/test_ckpt")
    lrm = LearningRateMonitor()
    timer = TimerCallback()

    manager.register(es)
    manager.register(ckpt)
    manager.register(lrm)
    manager.register(timer)

    print(f"已注册 {len(manager._callbacks)} 个回调")
    print(f"  - EarlyStopping (patience={es.patience})")
    print(f"  - ModelCheckpoint (metric={ckpt.metric})")
    print(f"  - LearningRateMonitor")
    print(f"  - TimerCallback")

    print("回调系统就绪 ✅")

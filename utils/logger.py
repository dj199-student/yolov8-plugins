"""
YOLOv8 日志工具

统一的日志记录接口，支持：
- 控制台输出（彩色）
- TensorBoard 写入
- 文件日志
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ==================== 颜色终端输出 ====================


class Colors:
    """ANSI 颜色代码"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color_print(text: str, color: str = "", bold: bool = False) -> None:
    """彩色打印

    Args:
        text: 文本内容
        color: 颜色名 ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'gray')
        bold: 是否加粗
    """
    color_code = getattr(Colors, color.upper(), "")
    bold_code = Colors.BOLD if bold else ""
    print(f"{bold_code}{color_code}{text}{Colors.RESET}")


# ==================== 日志器 ====================


class Logger:
    """YOLOv8 日志器

    同时输出到控制台、文件、TensorBoard。

    Args:
        save_dir: 日志保存目录
        project: 项目名称
        name: 实验名称
        tensorboard: 是否启用 TensorBoard
        exist_ok: 是否允许覆盖已有目录
    """

    def __init__(
        self,
        save_dir: str = "runs",
        project: str = "yolov8_plugins",
        name: str = "exp",
        tensorboard: bool = True,
        exist_ok: bool = False,
    ):
        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / project / f"{name}_{timestamp}"

        if exist_ok and self.save_dir.exists():
            import shutil
            shutil.rmtree(self.save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_file = self.save_dir / "train.log"

        # 文件日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("YOLOv8")
        self.logger.setLevel(logging.INFO)

        # TensorBoard
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.save_dir / "tensorboard"))
                self.info(f"TensorBoard 日志目录: {self.save_dir / 'tensorboard'}")
            except ImportError:
                print("[WARN] TensorBoard 未安装，跳过 TensorBoard 日志")

        self._metrics: Dict[str, Any] = {}

    def info(self, msg: str) -> None:
        """Info 级别日志"""
        self.logger.info(msg)

    def warn(self, msg: str) -> None:
        """Warning 级别日志"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Error 级别日志"""
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """Debug 级别日志"""
        self.logger.debug(msg)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """记录指标到 TensorBoard 和内存

        Args:
            metrics: 指标字典
            step: 全局步数（epoch 或 batch）
            prefix: 指标前缀（如 'train/' 或 'val/'）
        """
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            self._metrics[full_key] = value

            if self.writer is not None:
                self.writer.add_scalar(full_key, value, step)

        # 控制台输出
        metric_str = "  ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.info(f"[Step {step}] {metric_str}")

    def log_image(
        self, tag: str, image, step: int, dataformats: str = "HWC"
    ) -> None:
        """记录图像到 TensorBoard"""
        if self.writer is not None:
            import numpy as np
            if isinstance(image, np.ndarray) and dataformats == "HWC":
                image = image.transpose(2, 0, 1)  # HWC → CHW
            self.writer.add_image(tag, image, step)

    def log_model_graph(self, model, input_tensor) -> None:
        """记录模型图到 TensorBoard"""
        if self.writer is not None:
            try:
                self.writer.add_graph(model, input_tensor)
            except Exception as e:
                self.warn(f"无法记录模型图: {e}")

    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        """记录超参数"""
        if self.writer is not None:
            self.writer.add_text("hyperparameters", str(hparams))
        self.info(f"超参数: {hparams}")

    def close(self) -> None:
        """关闭日志器"""
        if self.writer is not None:
            self.writer.close()
        self.info(f"日志已保存到: {self.save_dir}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前记录的指标"""
        return self._metrics.copy()


# ==================== 进度条 ====================


def progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 30,
) -> str:
    """生成进度条字符串

    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文字
        suffix: 后缀文字
        length: 进度条长度

    Returns:
        进度条字符串，可直接 print
    """
    percent = float(current) / max(total, 1)
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    return f"\r{prefix} |{bar}| {percent:.1%} {suffix}"


# ==================== 训练状态表格 ====================


def print_training_header() -> None:
    """打印训练状态表头"""
    header = (
        f"{'Epoch':>6s}  "
        f"{'GPU_mem':>8s}  "
        f"{'box_loss':>9s}  "
        f"{'cls_loss':>9s}  "
        f"{'Instances':>10s}  "
        f"{'Size':>6s}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))


def print_training_step(
    epoch: int,
    gpu_mem: float,
    box_loss: float,
    cls_loss: float,
    instances: int,
    img_size: int,
) -> None:
    """打印单步训练状态"""
    print(
        f"{epoch:>6d}  "
        f"{gpu_mem:>7.1f}G  "
        f"{box_loss:>9.4f}  "
        f"{cls_loss:>9.4f}  "
        f"{instances:>10d}  "
        f"{img_size:>5d} ",
        end="\n",
    )


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 日志工具自检 ===")

    # 彩色打印测试
    color_print("红色文字", "red")
    color_print("绿色文字", "green")
    color_print("黄色文字", "yellow")
    color_print("蓝色加粗文字", "blue", bold=True)

    # 日志器测试
    logger = Logger(save_dir="/tmp/test_logs", tensorboard=False)
    logger.info("测试 info 日志")
    logger.warn("测试 warn 日志")
    logger.log_metrics({"loss": 0.123, "mAP": 0.75}, step=0)

    # 进度条测试
    print(progress_bar(25, 100, prefix="Training", suffix="25/100"))
    print(progress_bar(75, 100, prefix="Training", suffix="75/100"))

    # 清理
    logger.close()
    import shutil
    if Path("/tmp/test_logs").exists():
        shutil.rmtree("/tmp/test_logs")

    print("日志工具就绪 ✅")

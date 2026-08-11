"""
YOLOv8 配置解析器

支持从 YAML 文件加载和验证配置，并自动合并默认值。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ==================== 默认配置 ====================

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- 模型配置 ----
    "model": {
        "base": "yolov8n.pt",  # 基础模型 (n/s/m/l/x)
        "plugins": {
            "backbone": [],
            "neck": [],
            "head": [],
        },
    },
    # ---- 训练配置 ----
    "train": {
        "data": "coco128.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": "",  # 空 = 自动选择
        "workers": 8,
        "optimizer": "auto",  # auto / SGD / Adam / AdamW
        "lr0": 0.01,          # 初始学习率
        "lrf": 0.01,          # 最终学习率因子 (lr0 * lrf)
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": True,       # 余弦退火
        "close_mosaic": 10,   # 最后 N 轮关闭 Mosaic 增强
        "amp": True,          # 自动混合精度
        "resume": False,
        "pretrained": True,
    },
    # ---- 数据增强配置 ----
    "augment": {
        "hsv_h": 0.015,       # HSV-Hue 扰动
        "hsv_s": 0.7,         # HSV-Saturation 扰动
        "hsv_v": 0.4,         # HSV-Value 扰动
        "degrees": 0.0,       # 旋转角度
        "translate": 0.1,     # 平移比例
        "scale": 0.5,         # 缩放比例
        "shear": 0.0,         # 剪切角度
        "perspective": 0.0,   # 透视变换
        "flipud": 0.0,        # 上下翻转概率
        "fliplr": 0.5,        # 左右翻转概率
        "mosaic": 1.0,        # Mosaic 增强概率
        "mixup": 0.0,         # MixUp 增强概率
        "copy_paste": 0.0,    # Copy-Paste 增强概率
    },
    # ---- 验证配置 ----
    "val": {
        "data": "coco128.yaml",
        "imgsz": 640,
        "batch": 16,
        "device": "",
        "workers": 8,
        "conf": 0.001,         # 置信度阈值
        "iou": 0.6,            # NMS IoU 阈值
        "max_det": 300,        # 最大检测数
        "half": True,          # FP16 推理
        "dnn": False,          # OpenCV DNN
        "plots": True,         # 是否绘制结果图
        "save_json": False,    # 保存 COCO JSON
    },
    # ---- 导出配置 ----
    "export": {
        "format": "onnx",     # onnx / engine / tflite / openvino / coreml
        "imgsz": 640,
        "half": False,        # FP16
        "int8": False,        # INT8 量化
        "dynamic": False,     # 动态 batch / 尺寸
        "simplify": True,     # ONNX 简化
        "opset": 12,          # ONNX opset 版本
        "workspace": 4,       # TensorRT 工作区 (GB)
    },
    # ---- 日志配置 ----
    "log": {
        "project": "yolov8_plugins",
        "name": "exp",
        "exist_ok": False,
        "save_dir": "runs",
        "tensorboard": True,
    },
}


# ==================== 配置加载与合并 ====================


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """从 YAML 文件加载配置并合并默认值

    Args:
        config_path: YAML 配置文件路径

    Returns:
        合并后的完整配置字典

    Example:
        >>> cfg = load_config("configs/default.yaml")
        >>> print(cfg["train"]["epochs"])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        user_config = {}

    # 深度合并默认值和用户配置
    merged = deep_merge(DEFAULT_CONFIG.copy(), user_config)
    return merged


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """保存配置到 YAML 文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def deep_merge(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """深度合并两个字典，override 的值覆盖 base

    Args:
        base: 基础字典（默认值）
        override: 覆盖字典（用户指定值）

    Returns:
        合并后的新字典
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取模型相关配置"""
    return config.get("model", {})


def get_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取训练相关配置"""
    return config.get("train", {})


def get_val_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取验证相关配置"""
    return config.get("val", {})


def get_export_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """提取导出相关配置"""
    return config.get("export", {})


# ==================== YAML 工具 ====================


class ConfigDict(dict):
    """支持点号访问的字典

    Example:
        >>> cfg = ConfigDict({"a": {"b": 1}})
        >>> cfg.a.b
        1
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' 没有属性 '{name}'")
        if isinstance(value, dict):
            value = ConfigDict(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载任意 YAML 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 测试配置加载 ===")

    # 测试默认配置
    cfg = load_config("configs/default.yaml") if Path("configs/default.yaml").exists() else DEFAULT_CONFIG
    print(f"默认模型: {cfg['model']['base']}")
    print(f"训练轮数: {cfg['train']['epochs']}")
    print(f"图像尺寸: {cfg['train']['imgsz']}")

    # 测试深度合并
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}, f"合并失败: {merged}"
    print("深度合并测试通过 ✅")
    print("配置模块就绪 ✅")

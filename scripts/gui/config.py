"""
GUI 配置持久化

自动保存和恢复用户设置（窗口位置、上次使用的模型、参数等）。
配置文件路径: ~/.yolo_gui_config.json
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path.home() / ".yolo_gui_config.json"

# 默认配置
DEFAULT_GUI_CONFIG: Dict[str, Any] = {
    "window": {
        "geometry": "1280x820",
        "state": "normal",  # normal / maximized
    },
    "detect": {
        "last_model": "yolov8n.pt",
        "conf_threshold": 0.25,
        "iou_threshold": 0.7,
        "camera_device": 0,
        "fps_limit": 30,
    },
    "train": {
        "model": "yolov8n.pt",
        "data": "configs/datasets/coco128.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": "cpu",
        "workers": 4,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "optimizer": "auto",
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "amp": True,
        "close_mosaic": 10,
        "resume": False,
        "pretrained": True,
        "augment": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
        },
        "checkpoint": None,  # 训练断点信息 {"model","data","epoch","total_epochs","save_dir","last_pt","status"}
        "history": [],       # 历史训练记录（最近 10 条）
    },
    "theme": "light",
    "validate": {
        "model": "yolov8n.pt",
        "data": "configs/datasets/coco128.yaml",
        "batch": 16,
        "imgsz": 640,
        "device": "cpu",
        "workers": 4,
        "conf": 0.001,
        "iou": 0.6,
        "split": "val",
        "save_json": True,
        "plots": True,
    },
    "export": {
        "model": "yolov8n.pt",
        "format": "onnx",
        "imgsz": 640,
        "precision": "fp32",
        "dynamic": False,
        "simplify": True,
        "opset": 12,
        "workspace": 4.0,
        "device": "cpu",
    },
    "benchmark": {
        "imgsz": 640,
        "device": "cpu",
        "half": False,
        "int8": False,
    },
    "plugins": {
        "base_model": "yolov8n.pt",
        "last_plugins": {
            "backbone": [],
            "neck": [],
            "head": [],
        },
    },
    "task_queue": {
        "visible": False,
    },
}


def load_config() -> Dict[str, Any]:
    """加载用户配置，合并默认值"""
    if not CONFIG_PATH.exists():
        return DEFAULT_GUI_CONFIG.copy()

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_GUI_CONFIG.copy()

    # 深度合并
    merged = DEFAULT_GUI_CONFIG.copy()
    _deep_merge(merged, saved)
    return merged


def save_config(config: Dict[str, Any]) -> None:
    """保存配置到文件"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[Config] 保存配置失败: {e}")


def update_section(section: str, key: str, value: Any) -> None:
    """更新某个配置段的某个字段"""
    config = load_config()
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config(config)


def update_nested(section: str, subsection: str, key: str, value: Any) -> None:
    """更新嵌套配置（如 train.augment.hsv_h）"""
    config = load_config()
    if section not in config:
        config[section] = {}
    if subsection not in config[section]:
        config[section][subsection] = {}
    config[section][subsection][key] = value
    save_config(config)


def _deep_merge(base: Dict, override: Dict) -> None:
    """深度合并 override 到 base（原地修改）"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_train_checkpoint(checkpoint: dict) -> None:
    """保存训练检查点（用于断点续训）

    Args:
        checkpoint: {"model","data","epoch","total_epochs","save_dir","last_pt","best_pt","status"}
    """
    config = load_config()
    config.setdefault("train", {})["checkpoint"] = checkpoint
    save_config(config)


def clear_train_checkpoint() -> None:
    """清除训练检查点"""
    config = load_config()
    config.setdefault("train", {})["checkpoint"] = None
    save_config(config)


def get_train_checkpoint() -> dict | None:
    """获取上次训练检查点"""
    config = load_config()
    return config.get("train", {}).get("checkpoint")


def add_train_history(record: dict) -> None:
    """添加训练历史记录（保留最近 10 条）

    Args:
        record: {"model","data","epochs","best_pt","timestamp","status"}
    """
    config = load_config()
    history = config.setdefault("train", {}).setdefault("history", [])
    history.insert(0, record)
    # 保留最近 10 条
    config["train"]["history"] = history[:10]
    save_config(config)

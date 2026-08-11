#!/usr/bin/env python3
"""
YOLOv8 训练脚本

使用方式:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 300 --batch 32
    python scripts/train.py --model yolov8s.pt --data coco128.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from utils.config import load_config, get_train_config, get_model_config
from utils.logger import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 训练脚本（支持插件配置）"
    )
    # 配置文件模式
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    # 直接参数模式（覆盖配置）
    parser.add_argument("--model", type=str, default=None, help="模型文件（.pt）")
    parser.add_argument("--data", type=str, default=None, help="数据集 YAML")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=None, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=None, help="批量大小")
    parser.add_argument("--device", type=str, default=None, help="设备 (0, 1, cpu)")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--lr0", type=float, default=None, help="初始学习率")
    parser.add_argument("--resume", action="store_true", help="从中断处恢复训练")
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--project", type=str, default=None, help="项目名称")
    return parser.parse_args()


def build_train_args(config: dict, args: argparse.Namespace) -> dict:
    """从配置文件和命令行参数构建训练参数"""
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(config)
    log_cfg = config.get("log", {})
    augment_cfg = config.get("augment", {})

    # 命令行参数覆盖配置文件
    train_args = {
        "data": args.data or train_cfg.get("data", "coco128.yaml"),
        "epochs": args.epochs or train_cfg.get("epochs", 100),
        "imgsz": args.imgsz or train_cfg.get("imgsz", 640),
        "batch": args.batch or train_cfg.get("batch", 16),
        "device": args.device or train_cfg.get("device", ""),
        "workers": args.workers or train_cfg.get("workers", 8),
        "lr0": args.lr0 or train_cfg.get("lr0", 0.01),
        "lrf": train_cfg.get("lrf", 0.01),
        "momentum": train_cfg.get("momentum", 0.937),
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
        "cos_lr": train_cfg.get("cos_lr", True),
        "close_mosaic": train_cfg.get("close_mosaic", 10),
        "amp": train_cfg.get("amp", True),
        "resume": args.resume or train_cfg.get("resume", False),
        "pretrained": train_cfg.get("pretrained", True),
        "optimizer": train_cfg.get("optimizer", "auto"),
        # 项目设置
        "project": args.project or log_cfg.get("project", "yolov8_plugins"),
        "name": args.name or log_cfg.get("name", "exp"),
        "exist_ok": log_cfg.get("exist_ok", False),
        # 数据增强参数
        **{f"hsv_{k}": v for k, v in augment_cfg.items()
           if k.startswith("h")},
        **{k: v for k, v in augment_cfg.items()
           if not k.startswith("h")},
    }

    return train_args


def main():
    args = parse_args()

    # 加载配置
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # 尝试在 configs/ 目录下查找
            config_path = Path("configs") / args.config
        if not config_path.exists():
            print(f"[ERROR] 配置文件不存在: {args.config}")
            sys.exit(1)
        config = load_config(config_path)
        print(f"[INFO] 加载配置: {config_path}")
    else:
        from utils.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
        print("[INFO] 使用默认配置")

    # 确定模型
    model_path = args.model or config.get("model", {}).get("base", "yolov8n.pt")
    print(f"[INFO] 加载模型: {model_path}")

    # 检查插件配置
    plugins_cfg = config.get("model", {}).get("plugins", {})
    if any(plugins_cfg.values()):
        print(f"[INFO] 检测到插件配置:")
        for location, plugins in plugins_cfg.items():
            if plugins:
                for p in plugins:
                    print(f"  - [{location}] {p.get('type', 'unknown')}")

    # 加载模型
    model = YOLO(model_path)

    # 构建训练参数
    train_args = build_train_args(config, args)

    # 打印训练参数摘要
    print("\n" + "=" * 50)
    print("          训练参数摘要")
    print("=" * 50)
    for k, v in train_args.items():
        print(f"  {k:<20s}: {v}")
    print("=" * 50 + "\n")

    # 初始化日志
    logger = Logger(
        save_dir=train_args.get("project", "runs"),
        project=train_args.get("project", "yolov8_plugins"),
        name=train_args.get("name", "exp"),
        tensorboard=True,
    )

    try:
        # 开始训练
        results = model.train(**train_args)
        print(f"\n[SUCCESS] 训练完成！")
        print(f"[INFO] 最佳模型: {results.save_dir / 'weights' / 'best.pt'}")

    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()

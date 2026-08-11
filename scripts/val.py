#!/usr/bin/env python3
"""
YOLOv8 验证脚本

使用方式:
    python scripts/val.py --model yolov8n.pt --data coco128.yaml
    python scripts/val.py --model runs/train/exp/weights/best.pt --data custom.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from utils.config import load_config, get_val_config
from utils.metrics import compute_map, bbox_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 验证脚本")
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="模型权重路径 (.pt)",
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="数据集 YAML 配置文件",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="图像尺寸",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批量大小",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="设备 (cpu / 0)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou", type=float, default=0.6,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--max-det", type=int, default=300,
        help="最大检测数",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 推理",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="绘制验证结果图",
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="保存 COCO 格式的 JSON 结果",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["val", "test", "train"],
        help="评估数据集划分",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] 图像尺寸: {args.imgsz}, 批量: {args.batch}")

    # 开始验证
    try:
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            half=args.half,
            plots=args.plots,
            save_json=args.save_json,
            split=args.split,
        )

        # 打印关键指标
        print("\n" + "=" * 50)
        print("          验证结果")
        print("=" * 50)
        key_metrics = [
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(B)", "metrics/recall(B)",
        ]
        for key in key_metrics:
            if hasattr(metrics, key.replace("metrics/", "").replace("(B)", "")):
                val = getattr(
                    metrics,
                    key.replace("metrics/", "").replace("(B)", ""),
                    None,
                )
                if val is not None:
                    print(f"  {key}: {val:.4f}")
            elif hasattr(metrics, "results_dict"):
                val = metrics.results_dict.get(key)
                if val is not None:
                    print(f"  {key}: {val:.4f}")

        print("=" * 50 + "\n")
        print("[SUCCESS] 验证完成！")

    except Exception as e:
        print(f"[ERROR] 验证失败: {e}")
        raise


if __name__ == "__main__":
    main()

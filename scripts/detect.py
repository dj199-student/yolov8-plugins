#!/usr/bin/env python3
"""
YOLOv8 推理脚本

使用方式:
    python scripts/detect.py --source image.jpg
    python scripts/detect.py --source video.mp4 --conf 0.5
    python scripts/detect.py --source 0  # 摄像头
    python scripts/detect.py --source path/to/images/ --save-txt
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from ultralytics import YOLO

from utils.plots import draw_detections, COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 推理脚本")
    parser.add_argument(
        "--source", "-s", type=str, required=True,
        help="输入源 (图片/视频/目录/摄像头ID)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="yolov8n.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="推理图像尺寸",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="设备 (cpu / 0 / 0,1)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="保存结果",
    )
    parser.add_argument(
        "--save-txt", action="store_true",
        help="保存检测结果为 txt",
    )
    parser.add_argument(
        "--nosave", action="store_true",
        help="不保存结果",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="实时显示结果",
    )
    parser.add_argument(
        "--project", type=str, default="runs/detect",
        help="结果保存目录",
    )
    parser.add_argument(
        "--name", type=str, default="exp",
        help="实验名称",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 推理",
    )
    return parser.parse_args()


def process_image(model: YOLO, source: str, args: argparse.Namespace):
    """处理单张图片"""
    image = cv2.imread(source)
    if image is None:
        print(f"[ERROR] 无法读取图片: {source}")
        return

    results = model(
        image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
    )[0]

    # 绘制检测结果
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        annotated = draw_detections(
            image,
            boxes,
            classes,
            confidences,
            class_names=results.names,
        )
    else:
        annotated = image

    # 保存
    if args.save and not args.nosave:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / Path(source).name
        cv2.imwrite(str(save_path), annotated)
        print(f"[INFO] 结果已保存: {save_path}")

    # 显示
    if args.show:
        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def process_video(model: YOLO, source: str, args: argparse.Namespace):
    """处理视频"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {source}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 视频写入器
    writer = None
    if args.save and not args.nosave:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(source).stem}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            verbose=False,
        )[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()

            frame = draw_detections(
                frame, boxes, classes, confidences, results.names
            )

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"\r进度: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)", end="")

    print(f"\n[INFO] 处理完成: {frame_count} 帧")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # 加载模型
    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(args.model)

    # 判断输入源类型
    source = args.source

    # 如果是摄像头
    if source.isdigit():
        source = int(source)
        print(f"[INFO] 使用摄像头: {source}")
        cap = cv2.VideoCapture(source)
        process_video_with_cap(model, cap, args)
        return

    source_path = Path(source)

    # 单张图片
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if source_path.is_file() and source_path.suffix.lower() in image_exts:
        print(f"[INFO] 处理图片: {source}")
        process_image(model, str(source_path), args)

    # 视频文件
    elif source_path.is_file() and source_path.suffix.lower() in {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"
    }:
        print(f"[INFO] 处理视频: {source}")
        process_video(model, str(source_path), args)

    # 目录
    elif source_path.is_dir():
        image_files = []
        for ext in image_exts:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        print(f"[INFO] 在目录中找到 {len(image_files)} 张图片")
        for img_path in image_files:
            process_image(model, str(img_path), args)

    else:
        # 尝试作为视频流/摄像头
        print(f"[INFO] 尝试作为视频流打开: {source}")
        process_video(model, source, args)


def process_video_with_cap(model: YOLO, cap: cv2.VideoCapture, args: argparse.Namespace):
    """使用 cv2.VideoCapture 对象处理视频"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            frame = draw_detections(frame, boxes, classes, confidences, results.names)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

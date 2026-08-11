#!/usr/bin/env python3
"""
YOLOv8 模型导出脚本

支持导出格式:
    - ONNX (默认): 通用开放格式
    - TensorRT (engine): NVIDIA GPU 加速
    - OpenVINO: Intel 加速
    - CoreML: Apple 设备
    - TFLite: 移动端/嵌入式
    - NCNN: 移动端/嵌入式（腾讯）

使用方式:
    python scripts/export.py --model yolov8n.pt --format onnx
    python scripts/export.py --model best.pt --format engine --half --imgsz 640
    python scripts/export.py --model best.pt --format tflite --int8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


SUPPORTED_FORMATS = {
    "onnx": "ONNX (*.onnx)",
    "engine": "TensorRT (*.engine)",
    "openvino": "OpenVINO (*.xml)",
    "coreml": "CoreML (*.mlpackage)",
    "tflite": "TFLite (*.tflite)",
    "tflite_int8": "TFLite INT8 量化",
    "ncnn": "NCNN (*.param + *.bin)",
    "torchscript": "TorchScript (*.torchscript)",
    "pb": "TensorFlow SavedModel (*.pb)",
    "saved_model": "TensorFlow SavedModel",
    "edgetpu": "Edge TPU (*.tflite)",
}

# 跨平台文件大小
def get_size_str(path: Path) -> str:
    """获取文件大小字符串"""
    size = path.stat().st_size if path.exists() else 0
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 模型导出脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="支持的格式:\n  " + "\n  ".join(
            f"{k}: {v}" for k, v in SUPPORTED_FORMATS.items()
        ),
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="模型权重路径 (.pt)",
    )
    parser.add_argument(
        "--format", "-f", type=str, default="onnx",
        choices=list(SUPPORTED_FORMATS.keys()),
        help="导出格式",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="导出输入尺寸",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 半精度",
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="INT8 量化（仅 TFLite/Engine）",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="动态 batch / 尺寸（仅 ONNX）",
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="ONNX 模型简化（默认启用）",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset 版本",
    )
    parser.add_argument(
        "--workspace", type=float, default=4.0,
        help="TensorRT 工作区大小 (GB)",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="导出设备 (0 / cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(args.model)

    format_name = SUPPORTED_FORMATS[args.format]
    print(f"[INFO] 导出格式: {args.format} ({format_name})")
    print(f"[INFO] 输入尺寸: {args.imgsz}")

    # 构建导出参数
    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "opset": args.opset,
        "workspace": args.workspace,
        "device": args.device,
    }

    # 特定格式的处理
    if args.format == "tflite_int8":
        export_kwargs["format"] = "tflite"
        export_kwargs["int8"] = True

    try:
        # 导出模型
        print(f"\n[INFO] 开始导出...")
        export_path = model.export(**{k: v for k, v in export_kwargs.items() if v is not None})

        # 显示导出文件信息
        if isinstance(export_path, str):
            export_path = Path(export_path)

        if export_path and Path(export_path).exists():
            file_size = get_size_str(Path(export_path))
            print(f"\n[SUCCESS] 导出成功！")
            print(f"[INFO] 输出文件: {export_path}")
            print(f"[INFO] 文件大小: {file_size}")

            # 验证导出结果
            if args.format in ("onnx", "engine"):
                try:
                    import onnx
                    onnx_model = onnx.load(str(export_path))
                    onnx.checker.check_model(onnx_model)
                    print(f"[INFO] ONNX 模型验证通过 ✅")
                except Exception as e:
                    print(f"[WARN] 模型验证失败: {e}")
        else:
            print(f"[SUCCESS] 导出完成！")

    except Exception as e:
        print(f"[ERROR] 导出失败: {e}")
        print(f"[TIP] 请确保已安装对应格式的依赖：")
        if args.format == "engine":
            print("  TensorRT: pip install tensorrt")
        elif args.format == "openvino":
            print("  OpenVINO: pip install openvino")
        elif args.format == "coreml":
            print("  CoreML: pip install coremltools")
        elif args.format == "ncnn":
            print("  NCNN: 需要编译 ncnn 和 pnnx")
        raise


if __name__ == "__main__":
    main()

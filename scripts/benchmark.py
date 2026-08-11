#!/usr/bin/env python3
"""
YOLOv8 性能基准测试脚本

测试模型在不同硬件/精度下的推理速度。

使用方式:
    python scripts/benchmark.py --model yolov8n.pt
    python scripts/benchmark.py --model best.pt --imgsz 640 1280 --half
    python scripts/benchmark.py --model yolov8n.pt --compare  # 对比所有规模
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ultralytics import YOLO


MODEL_VARIANTS = [
    ("yolov8n.pt", "YOLOv8 Nano"),
    ("yolov8s.pt", "YOLOv8 Small"),
    ("yolov8m.pt", "YOLOv8 Medium"),
    ("yolov8l.pt", "YOLOv8 Large"),
    ("yolov8x.pt", "YOLOv8 X-Large"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 基准测试")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--half", action="store_true", help="FP16 测试")
    parser.add_argument("--int8", action="store_true", help="INT8 测试")
    parser.add_argument("--device", type=str, default="", help="设备")
    parser.add_argument("--warmup", type=int, default=3, help="预热轮数")
    parser.add_argument("--runs", type=int, default=100, help="测试轮数")
    parser.add_argument("--compare", action="store_true", help="对比所有规模")
    return parser.parse_args()


def benchmark_model(
    model_path: str,
    imgsz: int,
    batch: int,
    half: bool,
    device: str,
    warmup: int,
    runs: int,
) -> dict:
    """对单个模型进行基准测试"""
    model = YOLO(model_path)

    # 创建随机输入
    dummy_input = torch.randn(batch, 3, imgsz, imgsz)

    if device:
        dummy_input = dummy_input.to(device)

    # 预热
    print(f"  预热 ({warmup} 轮)...", end=" ")
    for _ in range(warmup):
        _ = model(dummy_input, verbose=False)
    print("done")

    # 正式测试
    print(f"  测试 ({runs} 轮)...", end=" ")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input, verbose=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input, verbose=False)
        elapsed = time.perf_counter() - start
    print("done")

    avg_time = elapsed / runs * 1000  # ms
    fps = 1000 / avg_time * batch

    # 获取参数和 FLOPs
    try:
        from thop import profile
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        flops_str = f"{flops/1e9:.2f} GFLOPs"
        params_str = f"{params/1e6:.2f}M"
    except ImportError:
        flops_str = "N/A"
        params_str = "N/A"

    return {
        "model": Path(model_path).stem,
        "imgsz": imgsz,
        "batch": batch,
        "avg_time_ms": avg_time,
        "fps": fps,
        "params": params_str,
        "flops": flops_str,
        "half": half,
    }


def print_results(results: list):
    """打印基准测试结果表格"""
    print("\n" + "=" * 90)
    print(f"{'Model':<20s} {'Size':>6s} {'Batch':>5s} "
          f"{'Latency(ms)':>12s} {'FPS':>10s} "
          f"{'Params':>10s} {'FLOPs':>12s}  {'Precision':>10s}")
    print("=" * 90)

    for r in results:
        precision = "FP16" if r["half"] else "FP32"
        print(
            f"{r['model']:<20s} {r['imgsz']:>5d}  {r['batch']:>5d}  "
            f"{r['avg_time_ms']:>10.2f}  {r['fps']:>9.1f}  "
            f"{r['params']:>10s} {r['flops']:>12s}  {precision:>10s}"
        )
    print("=" * 90)


def main():
    args = parse_args()

    all_results = []

    if args.compare:
        print("[INFO] 对比所有 YOLOv8 变体\n")
        for model_path, name in MODEL_VARIANTS:
            print(f"\n--- {name} ---")
            for imgsz in args.imgsz:
                for batch in args.batch:
                    result = benchmark_model(
                        model_path, imgsz, batch, args.half,
                        args.device, args.warmup, args.runs,
                    )
                    all_results.append(result)
                    print(f"    Latency: {result['avg_time_ms']:.2f}ms, "
                          f"FPS: {result['fps']:.1f}")
    else:
        print(f"[INFO] 基准测试: {args.model}\n")
        for imgsz in args.imgsz:
            for batch in args.batch:
                # FP32
                print(f"\n--- imgsz={imgsz}, batch={batch}, FP32 ---")
                result = benchmark_model(
                    args.model, imgsz, batch, False,
                    args.device, args.warmup, min(args.runs, 50),
                )
                all_results.append(result)
                print(f"    Latency: {result['avg_time_ms']:.2f}ms, "
                      f"FPS: {result['fps']:.1f}")

                # FP16 (如果启用)
                if args.half:
                    print(f"\n--- imgsz={imgsz}, batch={batch}, FP16 ---")
                    result_half = benchmark_model(
                        args.model, imgsz, batch, True,
                        args.device, args.warmup, min(args.runs, 50),
                    )
                    all_results.append(result_half)
                    print(f"    Latency: {result_half['avg_time_ms']:.2f}ms, "
                          f"FPS: {result_half['fps']:.1f}")

    # 打印汇总表格
    print_results(all_results)

    # 保存到 CSV
    csv_path = Path("runs/benchmark") / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[INFO] 结果已保存: {csv_path}")


if __name__ == "__main__":
    main()

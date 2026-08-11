#!/usr/bin/env python3
"""
Phase 2 功能测试脚本

测试所有新增功能：
1. ChartPanel — 5 种图表类型
2. ValWorker — 验证后台线程
3. ExportWorker — 导出后台线程
4. ValTab / ExportTab / BenchmarkTab — UI 构建
5. DetectTab 统计面板
6. 配置持久化
7. 完整 GUI 导入
"""

import sys
import os
import json
import time
import tempfile
import threading
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ==================== 测试工具 ====================

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  — {detail}")
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def summary():
    print(f"\n{'='*60}")
    total = PASS + FAIL
    print(f"  结果: {PASS} 通过 / {FAIL} 失败 / {total} 总计")
    print(f"{'='*60}")
    return FAIL == 0


# ==================== Test 1: ChartPanel ====================

def test_chart_panel():
    section("Test 1: ChartPanel 图表组件")

    import matplotlib
    matplotlib.use("Agg")  # 无头模式

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()

    from scripts.gui.widgets.chart_panel import ChartPanel

    # 1a: 创建面板
    panel = ChartPanel(root, title="Test Chart", show_toolbar=False)
    check("ChartPanel 创建", panel is not None)

    # 1b: 训练曲线
    panel.setup_training_curves()
    check("setup_training_curves", len(panel._axes) == 4,
          f"期望 4 个子图, 实际 {len(panel._axes)}")

    for epoch in range(1, 11):
        panel.update_training_curve(
            epoch,
            {"box": 1.0 - epoch * 0.08, "cls": 0.8 - epoch * 0.06, "dfl": 0.5 - epoch * 0.02},
            map50=0.3 + epoch * 0.05,
        )
    check("update_training_curve", panel._data["box_loss"] is not None)
    check("训练数据量", len(panel._data["epochs"]) == 10,
          f"期望 10 epochs, 实际 {len(panel._data['epochs'])}")

    # 1c: mAP 柱状图
    metrics = {"mAP50": 0.782, "mAP50-95": 0.587, "Precision": 0.813, "Recall": 0.754, "F1": 0.782}
    panel.show_metrics_bar(metrics)
    check("show_metrics_bar", panel.figure.axes is not None)

    # 1d: PR 曲线
    import numpy as np
    recall = np.linspace(0, 1, 100).tolist()
    precision = [1.0 - r * 0.5 for r in recall]
    panel.show_pr_curve(recall, precision, ap=0.85)
    check("show_pr_curve", panel.figure.axes is not None)

    # 1e: Benchmark 对比
    bench_results = [
        {"model": "yolov8n", "fps": 120, "params_m": 3.2, "flops_g": 8.7, "map50": 0.78},
        {"model": "yolov8s", "fps": 80, "params_m": 11.2, "flops_g": 28.6, "map50": 0.82},
        {"model": "yolov8m", "fps": 50, "params_m": 25.9, "flops_g": 78.9, "map50": 0.85},
    ]
    panel.show_benchmark_comparison(bench_results)
    check("show_benchmark_comparison", len(panel.figure.axes) == 4,
          f"期望 4 个子图, 实际 {len(panel.figure.axes)}")

    # 1f: 检测统计
    class_counts = {"person": 8, "car": 5, "bicycle": 3, "dog": 2, "cat": 1}
    confidences = [0.95, 0.87, 0.92, 0.78, 0.65, 0.88, 0.91, 0.72]
    panel.show_detection_stats(class_counts, confidences)
    check("show_detection_stats", len(panel.figure.axes) == 2,
          f"期望 2 个子图, 实际 {len(panel.figure.axes)}")

    # 1g: 保存图表
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    panel.save_figure(tmp)
    check("save_figure", Path(tmp).exists() and Path(tmp).stat().st_size > 0,
          f"文件大小: {Path(tmp).stat().st_size if Path(tmp).exists() else 0}")
    Path(tmp).unlink(missing_ok=True)

    # 1h: 暗色模式
    panel.set_dark_mode(True)
    check("set_dark_mode (dark)", panel._dark_mode is True)
    panel.set_dark_mode(False)
    check("set_dark_mode (light)", panel._dark_mode is False)

    root.destroy()


# ==================== Test 2: Workers ====================

def test_val_worker():
    section("Test 2: ValWorker 验证线程")

    from scripts.gui.workers.val_worker import ValWorker

    w = ValWorker()
    check("ValWorker 创建", w is not None)
    check("is_running (初始)", w.is_running() is False)

    # 回调测试
    logs = []
    done_data = []
    errors = []

    w.on_log(lambda msg: logs.append(msg))
    w.on_done(lambda metrics, save_dir: done_data.append((metrics, save_dir)))
    w.on_error(lambda err: errors.append(err))

    # 启动验证（使用 coco128 快速测试）
    w.validate(
        model_name="yolov8n.pt",
        data="coco128.yaml",
        batch=4,
        imgsz=320,
        device="cpu",
        workers=0,
        epochs=1,  # 快速测试
    )

    # 等待完成
    timeout = 120
    waited = 0
    while w.is_running() and waited < timeout:
        time.sleep(1)
        waited += 1

    check("验证未超时", waited < timeout, f"等待 {waited}s")
    check("日志有内容", len(logs) > 0, f"日志条数: {len(logs)}")
    check("验证完成回调", len(done_data) > 0, f"done_data: {len(done_data)}")

    if done_data:
        metrics, save_dir = done_data[0]
        check("有 mAP50", "mAP50" in metrics, str(metrics))
        check("mAP50 有效", 0 < metrics.get("mAP50", 0) <= 1,
              f"mAP50={metrics.get('mAP50', 'N/A')}")
        check("有 Precision", "precision" in metrics)
        check("有 Recall", "recall" in metrics)

    check("无错误", len(errors) == 0, str(errors[:1]) if errors else "")

    # 停止测试
    w2 = ValWorker()
    w2.validate(model_name="yolov8n.pt", data="coco128.yaml", batch=1, imgsz=320, device="cpu", workers=0)
    time.sleep(0.5)
    w2.stop()
    check("stop() 不报错", True)


def test_export_worker():
    section("Test 3: ExportWorker 导出线程")

    from scripts.gui.workers.export_worker import ExportWorker

    w = ExportWorker()
    check("ExportWorker 创建", w is not None)
    check("12 种格式注册", len(w.FORMAT_DISPLAY) >= 10,
          f"格式数: {len(w.FORMAT_DISPLAY)}")

    # 回调测试
    logs = []
    done_data = []
    errors = []

    w.on_log(lambda msg: logs.append(msg))
    w.on_done(lambda path, size, fmt: done_data.append((path, size, fmt)))
    w.on_error(lambda err: errors.append(err))

    # 导出 ONNX
    w.export(
        model_name="yolov8n.pt",
        format="onnx",
        imgsz=320,
        half=False,
        simplify=True,
        opset=12,
        device="cpu",
    )

    timeout = 120
    waited = 0
    while w.is_running() and waited < timeout:
        time.sleep(1)
        waited += 1

    check("导出未超时", waited < timeout, f"等待 {waited}s")
    check("日志有内容", len(logs) > 0, f"日志条数: {len(logs)}")

    if done_data:
        path, size, fmt = done_data[0]
        check("输出文件存在", Path(path).exists(), str(path))
        check("文件大小 > 0", size > 0, f"size={size}")
        check("格式为 onnx", fmt == "onnx", fmt)
        # 清理
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    check("无错误", len(errors) == 0, str(errors[:1]) if errors else "")


# ==================== Test 3: Tab UI 构建 ====================

def test_tabs_ui():
    section("Test 4: 标签页 UI 构建")

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()

    # 清空测试配置
    try:
        Path.home().joinpath(".yolo_gui_config_test.json")
    except Exception:
        pass

    status_var = tk.StringVar(value="测试中...")

    # ---- ValTab ----
    from scripts.gui.tabs.val_tab import ValTab

    val_frame = tk.Frame(root)
    val_tab = ValTab(val_frame, status_var)
    check("ValTab 创建", val_tab is not None)
    check("ValTab model 变量", val_tab.val_model_var.get() == "yolov8n.pt")
    check("ValTab 指标卡片", len(val_tab._metric_vars) == 6,
          f"指标数: {len(val_tab._metric_vars)}")
    check("ValTab chart_panel 存在", val_tab.chart_panel is not None)
    check("ValTab log_panel 存在", val_tab.log_panel is not None)

    # 测试指标更新
    test_metrics = {"mAP50": 0.782, "mAP50-95": 0.587, "precision": 0.813, "recall": 0.754, "f1": 0.782}
    val_tab._show_results(test_metrics, "")
    check("mAP50 卡片更新", val_tab._metric_vars["mAP50"].get() == "0.7820")
    check("Precision 卡片更新", val_tab._metric_vars["Precision"].get() == "0.8130")

    # ---- ExportTab ----
    from scripts.gui.tabs.export_tab import ExportTab

    export_frame = tk.Frame(root)
    export_tab = ExportTab(export_frame, status_var)
    check("ExportTab 创建", export_tab is not None)
    check("ExportTab format var", export_tab.export_format_var.get() == "onnx")
    check("ExportTab precision", export_tab.export_precision_var.get() == "fp32")
    check("ExportTab log_panel", export_tab.log_panel is not None)

    # ---- BenchmarkTab ----
    from scripts.gui.tabs.benchmark_tab import BenchmarkTab

    bench_frame = tk.Frame(root)
    bench_tab = BenchmarkTab(bench_frame, status_var)
    check("BenchmarkTab 创建", bench_tab is not None)
    check("BenchmarkTab 预设模型", len(bench_tab._preset_models) == 5,
          f"预设数: {len(bench_tab._preset_models)}")
    check("BenchmarkTab chart_panel", bench_tab.chart_panel is not None)
    check("BenchmarkTab result_tree", bench_tab.result_tree is not None)
    check("BenchmarkTab log_panel", bench_tab.log_panel is not None)

    # 测试模型选择
    for model in bench_tab._preset_models:
        bench_tab._selected_models[model].set(True)
    selected = bench_tab._get_selected_models()
    check("Benchmark 模型选择", len(selected) == 5, f"选中: {len(selected)}")

    # ---- DetectTab statistics ----
    from scripts.gui.tabs.detect_tab import DetectTab

    detect_frame = tk.Frame(root)
    detect_tab = DetectTab(detect_frame, status_var)
    check("DetectTab detect_chart 存在", detect_tab.detect_chart is not None)
    check("DetectTab info_notebook 存在", detect_tab.info_notebook is not None)

    # 测试统计图表更新
    class_counts = {"person": 8, "car": 5, "bicycle": 3}
    confidences = [0.95, 0.87, 0.92, 0.78, 0.65, 0.88, 0.91, 0.72]
    chart_tab = DetectTab(detect_frame, status_var)  # second instance for chart test
    chart_tab.detect_chart.show_detection_stats(class_counts, confidences)
    check("检测统计图表渲染", chart_tab.detect_chart.figure.axes is not None)

    # ---- Config save/load ----
    from scripts.gui.config import load_config, save_config
    test_cfg_path = Path.home() / ".yolo_gui_config_test.json"

    cfg = load_config()
    check("load_config 有 validate 段", "validate" in cfg, str(cfg.keys()))
    check("load_config 有 export 段", "export" in cfg)
    check("load_config 有 benchmark 段", "benchmark" in cfg)
    check("validate.model 默认值", cfg["validate"]["model"] == "yolov8n.pt")
    check("export.format 默认值", cfg["export"]["format"] == "onnx")

    val_tab.save_config()
    export_tab.save_config()
    bench_tab.save_config()
    detect_tab.save_config()
    check("save_config 不报错", True)

    root.destroy()


# ==================== Test 4: Full GUI import ====================

def test_full_gui():
    section("Test 5: 完整 GUI 导入")

    # 测试所有导入
    modules = [
        ("scripts.gui", "主包"),
        ("scripts.gui.app", "YOLOv8GUI"),
        ("scripts.gui.config", "配置"),
        ("scripts.gui.theme", "主题"),
        ("scripts.gui.tabs.detect_tab", "DetectTab"),
        ("scripts.gui.tabs.train_tab", "TrainTab"),
        ("scripts.gui.tabs.val_tab", "ValTab"),
        ("scripts.gui.tabs.export_tab", "ExportTab"),
        ("scripts.gui.tabs.benchmark_tab", "BenchmarkTab"),
        ("scripts.gui.widgets.chart_panel", "ChartPanel"),
        ("scripts.gui.widgets.image_viewer", "ImageViewer"),
        ("scripts.gui.widgets.log_panel", "LogPanel"),
        ("scripts.gui.widgets.model_selector", "ModelSelector"),
        ("scripts.gui.workers.detect_worker", "DetectWorker"),
        ("scripts.gui.workers.train_worker", "TrainWorker"),
        ("scripts.gui.workers.val_worker", "ValWorker"),
        ("scripts.gui.workers.export_worker", "ExportWorker"),
    ]

    import importlib
    for mod_name, desc in modules:
        try:
            mod = importlib.import_module(mod_name)
            check(f"导入 {desc}", True)
        except Exception as e:
            check(f"导入 {desc}", False, str(e))

    # 测试 YOLOv8GUI 可实例化（不启动 mainloop）
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from scripts.gui.app import YOLOv8GUI
        gui = YOLOv8GUI()
        check("YOLOv8GUI 实例化", gui is not None)
        check("5 个标签页注册", len(gui.main_tabs.tabs()) == 5,
              f"实际: {len(gui.main_tabs.tabs())}")

        # 验证标签页文本
        tab_texts = [gui.main_tabs.tab(i, "text") for i in range(5)]
        check("标签页 🔍 检测", "目标检测" in tab_texts[0], tab_texts[0])
        check("标签页 🏋️ 训练", "模型训练" in tab_texts[1], tab_texts[1])
        check("标签页 📊 验证", "模型验证" in tab_texts[2], tab_texts[2])
        check("标签页 📦 导出", "模型导出" in tab_texts[3], tab_texts[3])
        check("标签页 ⚡ Benchmark", "Benchmark" in tab_texts[4], tab_texts[4])

        gui.root.destroy()
    except Exception as e:
        check("YOLOv8GUI 实例化", False, str(e))


# ==================== Main ====================

def main():
    print("🧪 YOLOv8 GUI Phase 2 功能测试")
    print(f"   Python: {sys.version}")
    print(f"   CWD: {os.getcwd()}")

    test_chart_panel()
    test_val_worker()
    test_export_worker()
    test_tabs_ui()
    test_full_gui()

    return summary()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

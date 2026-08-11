#!/usr/bin/env python3
"""
YOLOv8 桌面图形界面 — Tkinter 原生桌面应用
零浏览器依赖，不受 JS 注入影响

启动:
    python scripts/gui.py

Phase 1 改进 (2026-08-10):
    ✅ 模块化拆分 (gui/ 包)
    ✅ 训练可真正停止 (trainer.stop)
    ✅ 摄像头实时检测
    ✅ 批量图片处理
    ✅ 完整训练参数 (20+ 项)
    ✅ 配置持久化
    ✅ 快捷键系统
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gui import main

if __name__ == "__main__":
    main()

"""
YOLOv8 GUI 桌面应用

启动方式:
    python scripts/gui.py
    或
    python -m scripts.gui
"""

from .app import YOLOv8GUI


def main():
    """GUI 入口函数"""
    gui = YOLOv8GUI()
    gui.run()

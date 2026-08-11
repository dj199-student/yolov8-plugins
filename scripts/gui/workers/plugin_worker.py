"""
插件构建后台工作线程

使用 PluginBuilder 在后台构建带插件的 YOLO 模型，
支持日志输出和错误处理。
"""

import threading
import io
import sys
from typing import Any, Callable


class PluginWorker:
    """插件模型构建工作线程"""

    def __init__(self):
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._on_log: Callable | None = None
        self._on_done: Callable | None = None
        self._on_error: Callable | None = None

    # ==================== 回调设置 ====================

    def on_log(self, callback: Callable) -> None:
        self._on_log = callback

    def on_done(self, callback: Callable) -> None:
        self._on_done = callback

    def on_error(self, callback: Callable) -> None:
        self._on_error = callback

    def is_running(self) -> bool:
        return self._running.is_set()

    def stop(self) -> None:
        self._running.clear()

    # ==================== 构建任务 ====================

    def build_model(self, config: dict) -> None:
        """启动模型构建（后台线程）

        Args:
            config: 插件配置字典，格式：
                {
                    "base": "yolov8n.pt",
                    "plugins": {
                        "backbone": [{"type": "se_attention", "params": {...}}],
                        "neck": [],
                        "head": []
                    }
                }
        """
        if self._running.is_set():
            self._log("⚠ 已有构建任务在运行中\n")
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._build_thread,
            args=(config,),
            daemon=True,
        )
        self._thread.start()

    def _build_thread(self, config: dict) -> None:
        try:
            from models.plugin_builder import PluginBuilder

            self._log(f"{'='*50}\n")
            self._log(f"  插件模型构建开始\n")
            self._log(f"  基础模型: {config.get('base', 'yolov8n.pt')}\n")

            plugins = config.get("plugins", {})
            for section in ("backbone", "neck", "head"):
                items = plugins.get(section, [])
                if items:
                    names = [p["type"] for p in items]
                    self._log(f"  {section}: {', '.join(names)}\n")
            self._log(f"{'='*50}\n\n")

            # 捕获 print 输出作为日志
            old_stdout = sys.stdout
            captured = io.StringIO()
            sys.stdout = captured

            try:
                builder = PluginBuilder(config)
                model = builder.build()
            finally:
                sys.stdout = old_stdout

            # 输出捕获的日志
            captured_output = captured.getvalue()
            if captured_output:
                for line in captured_output.splitlines():
                    self._log(f"  {line}\n")

            # 摘要
            summary = builder.summary()
            self._log(f"\n{'='*50}\n")
            self._log(f"  构建完成\n")
            self._log(f"  基础模型: {summary['base_model']}\n")
            self._log(f"  插件总数: {summary['total_plugins']}\n")
            for p in summary["plugins"]:
                self._log(f"    - {p['name']} ({p['params']:,} params)\n")
            self._log(f"{'='*50}\n")

            if self._on_done:
                self._on_done(summary, model)

        except Exception as e:
            self._log(f"\n构建异常: {e}\n")
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._running.clear()

    def _log(self, msg: str) -> None:
        if self._on_log:
            self._on_log(msg)

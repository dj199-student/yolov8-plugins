"""
日志面板组件

可复用的日志显示面板（Text + Scrollbar），支持：
- 线程安全的日志写入（通过 Queue）
- 暗色背景风格
- 自动滚动到底部
- 导出到文件
"""

import tkinter as tk
from tkinter import ttk, filedialog
import queue
from datetime import datetime


class LogPanel(ttk.LabelFrame):
    """日志面板"""

    def __init__(self, parent, title: str = "日志", max_lines: int = 5000, **kwargs):
        super().__init__(parent, text=title, padding=5, **kwargs)
        self._max_lines = max_lines
        self._log_queue = queue.Queue()

        # 工具栏
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 3))

        ttk.Button(toolbar, text="📋 复制全部", command=self._copy_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 导出", command=self._export).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗑 清空", command=self._clear).pack(side=tk.LEFT, padx=2)
        self._line_count_var = tk.StringVar(value="0 行")
        ttk.Label(toolbar, textvariable=self._line_count_var, font=("Consolas", 8)).pack(
            side=tk.RIGHT, padx=5,
        )

        # 日志文本区
        text_frame = ttk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self._text = tk.Text(
            text_frame,
            font=("Consolas", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            relief=tk.FLAT,
            borderwidth=3,
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

        self._line_count = 0

    def write(self, msg: str) -> None:
        """线程安全的日志写入"""
        self._log_queue.put(msg)

    def poll(self) -> None:
        """将队列中的日志刷新到 UI（需在 tk 定时器中调用）"""
        while True:
            try:
                msg = self._log_queue.get_nowait()
                self._append(msg)
            except queue.Empty:
                break

    def _append(self, msg: str) -> None:
        """实际写入 Text 组件"""
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, msg)
        self._text.see(tk.END)

        # 限制行数，超出时删除头部
        self._line_count += msg.count("\n")
        if self._line_count > self._max_lines:
            # 删除前 20% 的行
            total_lines = int(self._text.index("end-1c").split(".")[0])
            remove_to = total_lines // 5
            self._text.delete("1.0", f"{remove_to}.0")
            self._line_count = int(self._text.index("end-1c").split(".")[0])

        self._text.configure(state=tk.DISABLED)
        self._line_count_var.set(f"{self._line_count} 行")

    def write_now(self, msg: str) -> None:
        """直接写入（主线程中调用）"""
        self._append(msg)

    def _copy_all(self) -> None:
        """复制全部日志"""
        text = self._text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)

    def _export(self) -> None:
        """导出日志到文件"""
        path = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".log",
            filetypes=[("Log 文件", "*.log"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialfile=f"yolo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._text.get("1.0", tk.END))
            except IOError as e:
                print(f"[LogPanel] 导出失败: {e}")

    def _clear(self) -> None:
        """清空日志"""
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
        self._line_count = 0
        self._line_count_var.set("0 行")

    def set_bg(self, bg: str, fg: str) -> None:
        """设置日志区背景色（用于主题切换）"""
        self._text.configure(bg=bg, fg=fg)

"""
任务队列组件

支持顺序执行多任务（训练→验证→导出→Benchmark）。
- 使用 ttk.Treeview 显示任务列表
- 顺序执行：前一个完成后自动启动下一个
- 支持取消等待中的任务
- 可折叠面板

集成方式：
    task_queue = TaskQueue(app, status_var)
    task_queue.pack(fill=tk.X)
    task_queue.add_task("训练", "yolov8n.pt", train_fn)
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import uuid
from typing import Any, Callable


class TaskQueue(ttk.LabelFrame):
    """任务队列组件"""

    # 状态常量
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    STATUS_ICONS = {
        PENDING: "⏳",
        RUNNING: "🔄",
        COMPLETED: "✅",
        FAILED: "❌",
        CANCELLED: "🚫",
    }

    def __init__(self, parent, status_var: tk.StringVar = None, **kwargs):
        super().__init__(
            parent,
            text="📋 任务队列",
            padding=5,
            **kwargs,
        )
        self._status_var = status_var

        # 任务列表
        self._tasks: list[dict] = []
        self._task_lock = threading.Lock()
        self._queue_thread: threading.Thread | None = None
        self._running = False

        # 当前执行的任务 ID
        self._current_task_id: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """构建 UI"""
        # 工具栏
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 3))

        self._collapse_btn = ttk.Button(
            toolbar, text="▼ 收起", width=8,
            command=self._toggle_collapse,
        )
        self._collapse_btn.pack(side=tk.LEFT)

        self._queue_count_var = tk.StringVar(value="0 个任务")
        ttk.Label(toolbar, textvariable=self._queue_count_var,
                  font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=10)

        ttk.Button(toolbar, text="🗑 清除已完成",
                   command=self._clear_completed, width=12).pack(side=tk.RIGHT, padx=2)

        # Treeview
        columns = ("task_type", "model", "status", "progress")
        self._tree = ttk.Treeview(
            self, columns=columns, show="headings",
            height=4, selectmode="browse",
        )
        self._tree.heading("task_type", text="类型", anchor=tk.W)
        self._tree.heading("model", text="模型/任务", anchor=tk.W)
        self._tree.heading("status", text="状态", anchor=tk.CENTER)
        self._tree.heading("progress", text="进度", anchor=tk.CENTER)

        self._tree.column("task_type", width=80, anchor=tk.W, stretch=False)
        self._tree.column("model", width=160, anchor=tk.W)
        self._tree.column("status", width=70, anchor=tk.CENTER, stretch=False)
        self._tree.column("progress", width=80, anchor=tk.CENTER, stretch=False)

        tree_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 操作列（通过右键菜单）
        self._tree_menu = tk.Menu(self._tree, tearoff=0)
        self._tree_menu.add_command(label="❌ 取消任务", command=self._cancel_selected)
        self._tree_menu.add_command(label="🗑 移除任务", command=self._remove_selected)
        self._tree.bind("<Button-3>", self._on_right_click)

        self._collapsed = False

    # ==================== 公共 API ====================

    def add_task(
        self,
        task_type: str,
        model_name: str,
        execute_fn: Callable,
        on_progress: Callable = None,
        **kwargs,
    ) -> str:
        """添加任务到队列

        Args:
            task_type: 任务类型（训练/验证/导出/Benchmark）
            model_name: 模型名称
            execute_fn: 执行函数 fn(task_id) — 必须在线程中调用，完成时调用 self._on_task_done(task_id, success, result)
            on_progress: 进度回调 fn(progress_pct: float)
            **kwargs: 附加数据

        Returns:
            task_id: 任务 ID
        """
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "type": task_type,
            "model": model_name,
            "status": self.PENDING,
            "progress": 0.0,
            "execute_fn": execute_fn,
            "on_progress": on_progress,
            "kwargs": kwargs,
            "result": None,
            "added_at": time.time(),
        }

        with self._task_lock:
            self._tasks.append(task)

        self._refresh_display()
        self._start_queue()

        if self._status_var:
            self._status_var.set(f"任务已加入队列: {task_type} - {model_name}")

        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消指定任务

        Args:
            task_id: 任务 ID

        Returns:
            是否成功取消
        """
        with self._task_lock:
            for task in self._tasks:
                if task["id"] == task_id:
                    if task["status"] == self.PENDING:
                        task["status"] = self.CANCELLED
                        self._refresh_display()
                        return True
                    elif task["status"] == self.RUNNING:
                        # 运行中的任务不能直接取消
                        return False
        return False

    def update_progress(self, task_id: str, progress: float) -> None:
        """更新任务进度

        Args:
            task_id: 任务 ID
            progress: 进度百分比 (0-100)
        """
        with self._task_lock:
            for task in self._tasks:
                if task["id"] == task_id:
                    task["progress"] = min(progress, 100)
                    break
        self._refresh_display()

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """标记任务完成"""
        with self._task_lock:
            for task in self._tasks:
                if task["id"] == task_id:
                    task["status"] = self.COMPLETED
                    task["progress"] = 100
                    task["result"] = result
                    break
        self._current_task_id = None
        self._refresh_display()
        self._start_queue()  # 启动下一个

    def mark_failed(self, task_id: str, error: str = "") -> None:
        """标记任务失败"""
        with self._task_lock:
            for task in self._tasks:
                if task["id"] == task_id:
                    task["status"] = self.FAILED
                    task["result"] = error
                    break
        self._current_task_id = None
        self._refresh_display()
        self._start_queue()  # 启动下一个（不因一个失败而中断）

    def get_queue_status(self) -> dict:
        """返回队列状态摘要"""
        with self._task_lock:
            counts = {
                self.PENDING: 0,
                self.RUNNING: 0,
                self.COMPLETED: 0,
                self.FAILED: 0,
                self.CANCELLED: 0,
            }
            for task in self._tasks:
                counts[task["status"]] = counts.get(task["status"], 0) + 1
            return {
                "total": len(self._tasks),
                "counts": counts,
                "current_task": self._current_task_id,
            }

    # ==================== 内部逻辑 ====================

    def _start_queue(self) -> None:
        """启动队列处理线程"""
        if self._current_task_id is not None:
            return  # 已有任务在运行

        next_task = None
        with self._task_lock:
            for task in self._tasks:
                if task["status"] == self.PENDING:
                    next_task = task
                    break

        if next_task is None:
            return

        self._current_task_id = next_task["id"]
        next_task["status"] = self.RUNNING
        self._refresh_display()

        # 在后台线程执行
        thread = threading.Thread(
            target=self._execute_task,
            args=(next_task,),
            daemon=True,
        )
        thread.start()

    def _execute_task(self, task: dict) -> None:
        """在后台线程执行任务"""
        task_id = task["id"]
        try:
            task["execute_fn"](task_id)
            # 如果 execute_fn 没有调用 mark_completed，自动标记
            if task["status"] == self.RUNNING:
                self.mark_completed(task_id)
        except Exception as e:
            self.mark_failed(task_id, str(e))

    def _refresh_display(self) -> None:
        """刷新 Treeview 显示（主线程安全）"""
        # 保存展开状态
        selected = self._tree.selection()

        # 清空
        for item in self._tree.get_children():
            self._tree.delete(item)

        with self._task_lock:
            for i, task in enumerate(self._tasks):
                icon = self.STATUS_ICONS.get(task["status"], "❓")
                status_text = f"{icon} {task['status']}"

                progress_text = "--"
                if task["status"] == self.RUNNING:
                    progress_text = f"{task['progress']:.0f}%"
                elif task["status"] == self.COMPLETED:
                    progress_text = "100%"
                elif task["status"] == self.FAILED:
                    progress_text = "失败"

                item_id = self._tree.insert(
                    "", "end",
                    iid=task["id"],
                    values=(
                        task["type"],
                        task["model"],
                        status_text,
                        progress_text,
                    ),
                )

                # 颜色标记
                if task["status"] == self.RUNNING:
                    self._tree.tag_configure("running", foreground="#0078d4")
                    self._tree.item(item_id, tags=("running",))
                elif task["status"] == self.COMPLETED:
                    self._tree.tag_configure("completed", foreground="#388e3c")
                    self._tree.item(item_id, tags=("completed",))
                elif task["status"] == self.FAILED:
                    self._tree.tag_configure("failed", foreground="#d32f2f")
                    self._tree.item(item_id, tags=("failed",))

        # 更新计数
        pending_count = sum(1 for t in self._tasks if t["status"] in (self.PENDING, self.RUNNING))
        total = len(self._tasks)
        self._queue_count_var.set(f"{total} 个任务 ({pending_count} 待处理)")

    def _cancel_selected(self) -> None:
        """取消选中的任务"""
        selection = self._tree.selection()
        if selection:
            task_id = selection[0]
            if self.cancel_task(task_id):
                if self._status_var:
                    self._status_var.set("任务已取消")
            else:
                if self._status_var:
                    self._status_var.set("无法取消运行中的任务")

    def _remove_selected(self) -> None:
        """移除选中的任务"""
        selection = self._tree.selection()
        if not selection:
            return
        task_id = selection[0]
        with self._task_lock:
            self._tasks = [t for t in self._tasks if t["id"] != task_id]
        if task_id == self._current_task_id:
            self._current_task_id = None
        self._refresh_display()

    def _clear_completed(self) -> None:
        """清除已完成/失败/取消的任务"""
        with self._task_lock:
            self._tasks = [
                t for t in self._tasks
                if t["status"] in (self.PENDING, self.RUNNING)
            ]
        self._refresh_display()

    def _toggle_collapse(self) -> None:
        """展开/收起"""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._tree.pack_forget()
            for child in self.winfo_children():
                if isinstance(child, ttk.Scrollbar):
                    child.pack_forget()
            self._collapse_btn.configure(text="▶ 展开")
        else:
            # 重新显示 tree
            tree_scroll = None
            for child in self.winfo_children():
                if isinstance(child, ttk.Scrollbar):
                    tree_scroll = child
                    break
            self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            if tree_scroll:
                tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self._collapse_btn.configure(text="▼ 收起")

    def _on_right_click(self, event) -> None:
        """右键菜单"""
        item = self._tree.identify_row(event.y)
        if item:
            self._tree.selection_set(item)
            self._tree_menu.post(event.x_root, event.y_root)
